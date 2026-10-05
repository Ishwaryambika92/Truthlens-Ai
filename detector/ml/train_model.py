"""
Train the TruthLens AI fake-news classifier on the real-world dataset.

Usage (from the TruthLensAi/ project root, venv active):

    python detector/ml/prepare_dataset.py   # if not already run
    python detector/ml/train_model.py

WHAT THIS DOES (and why):
  1. Loads detector/ml/news_dataset.csv (text,label - produced by
     prepare_dataset.py from a real-world source, see raw/DATASET_SOURCE.md).
  2. Cleans text with the SAME preprocessing used at inference time
     (detector.preprocessing.clean_text), so train/inference never drift.
  3. Splits into TRAIN / VALIDATION / TEST (70/15/15, stratified by label).
     The raw dataset has no article dates, so a true time-aware split isn't
     possible (documented honestly in dataset_report.json); duplicate/
     near-duplicate removal in prepare_dataset.py is the leakage safeguard
     instead, so the same story can't appear in two different splits.
  4. Trains THREE candidate models on TRAIN only:
       - TF-IDF + Logistic Regression
       - TF-IDF + Calibrated LinearSVC (CalibratedClassifierCV gives it a
         genuine predict_proba, so all three candidates are compared on
         equal footing and any of them can produce an honest confidence)
       - TF-IDF + Multinomial Naive Bayes
     Each candidate is scored on VALIDATION only (accuracy / macro-F1) -
     the test set is never touched during model selection.
  5. Picks the best candidate by validation macro-F1, then evaluates that
     ONE model exactly once on the held-out TEST set - the number you
     should trust as the honest generalization estimate.
  6. Derives HIGH / MEDIUM / LOW confidence thresholds from the VALIDATION
     set (not guessed, not hard-coded): finds the probability level below
     which validation accuracy meaningfully drops, and uses that as the
     UNCERTAIN cutoff.
  7. Runs error analysis on TEST set mistakes (saved to error_analysis.json)
     so failure patterns are visible instead of hidden behind one accuracy
     number.
  8. Saves detector/ml/model_pipeline.pkl (single TF-IDF + classifier
     Pipeline - keeps the vectorizer and classifier bundled together,
     which is what fixed the earlier scikit-learn version-compatibility
     bug) plus metrics.json, error_analysis.json, and train_env.json.

HONESTY NOTE: this prints and saves real, computed numbers only. It does
not claim 100% real-world accuracy, and the saved metrics explicitly
distinguish "held-out test" performance (the number that matters) from
validation performance (used only for model selection).
"""
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detector.preprocessing import clean_text  # noqa: E402

DATASET_PATH = ML_DIR / "news_dataset.csv"
PIPELINE_PATH = ML_DIR / "model_pipeline.pkl"
TRAIN_ENV_PATH = ML_DIR / "train_env.json"
METRICS_PATH = ML_DIR / "metrics.json"
ERROR_ANALYSIS_PATH = ML_DIR / "error_analysis.json"

RANDOM_SEED = 42
POS_LABEL = "FAKE"  # the label precision/recall/F1 report focuses on


def load_dataset():
    texts, labels = [], []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().upper()
            if not text or label not in ("REAL", "FAKE"):
                continue
            texts.append(text)
            labels.append(label)
    return texts, labels


def build_candidates():
    """Candidate models to compare on VALIDATION only. Several C values are
    included for the two linear models because C=1.0 (an sklearn default)
    tends to over-fit to rare, dataset/entity-specific tokens on a TF-IDF
    feature space this wide - the fix is chosen by validation performance,
    not by tuning against any specific example text."""
    candidates = {}
    tfidf_kwargs = dict(max_features=30000, ngram_range=(1, 2), min_df=5, max_df=0.85, sublinear_tf=True)

    for c in (0.1, 0.3, 1.0):
        candidates[f"logistic_regression_C{c}"] = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=c)),
        ])
        candidates[f"calibrated_linear_svc_C{c}"] = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf", CalibratedClassifierCV(
                LinearSVC(class_weight="balanced", C=c, max_iter=5000), cv=3
            )),
        ])

    candidates["multinomial_nb"] = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
        ("clf", MultinomialNB(alpha=0.5)),
    ])
    return candidates


def evaluate(pipeline, X, y, pos_label=POS_LABEL):
    y_pred = pipeline.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, pos_label=pos_label),
        "recall": recall_score(y, y_pred, pos_label=pos_label),
        "f1_score": f1_score(y, y_pred, pos_label=pos_label),
        "f1_macro": f1_score(y, y_pred, average="macro"),
    }
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        classes = list(pipeline.classes_)
        pos_idx = classes.index(pos_label)
        try:
            metrics["roc_auc"] = roc_auc_score((np.array(y) == pos_label).astype(int), proba[:, pos_idx])
        except ValueError:
            metrics["roc_auc"] = None
    return metrics, y_pred


def derive_confidence_thresholds(pipeline, X_val, y_val):
    """Determine HIGH/MEDIUM/LOW confidence cutoffs from VALIDATION data
    (not guessed): bucket predictions by their confidence and find where
    accuracy starts dropping meaningfully."""
    if not hasattr(pipeline, "predict_proba"):
        return {"high": 0.80, "medium": 0.60, "note": "Model has no predict_proba; using conservative defaults."}

    proba = pipeline.predict_proba(X_val)
    preds = pipeline.classes_[np.argmax(proba, axis=1)]
    confidences = np.max(proba, axis=1)
    correct = (preds == np.array(y_val))

    buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    bucket_stats = []
    for lo, hi in buckets:
        mask = (confidences >= lo) & (confidences < hi)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n > 0 else None
        bucket_stats.append({"range": f"{lo:.1f}-{hi:.1f}", "n": n, "accuracy": acc})

    qualifying_high = [float(b["range"].split("-")[0]) for b in bucket_stats
                        if b["accuracy"] is not None and b["accuracy"] >= 0.85 and b["n"] >= 10]
    qualifying_medium = [float(b["range"].split("-")[0]) for b in bucket_stats
                          if b["accuracy"] is not None and b["accuracy"] >= 0.70 and b["n"] >= 10]
    high_threshold = min(qualifying_high) if qualifying_high else 0.85
    medium_threshold = min(qualifying_medium) if qualifying_medium else 0.65
    medium_threshold = min(medium_threshold, high_threshold - 0.01)

    return {
        "high": round(high_threshold, 2),
        "medium": round(medium_threshold, 2),
        "validation_confidence_buckets": bucket_stats,
    }


def run_error_analysis(pipeline, texts_test, X_test, y_test, y_pred, max_examples=15):
    errors = []
    for text, true_label, pred_label in zip(texts_test, y_test, y_pred):
        if true_label != pred_label:
            errors.append({
                "true_label": true_label,
                "predicted_label": pred_label,
                "text_excerpt": text[:300],
                "text_length": len(text),
            })

    false_positives = [e for e in errors if e["true_label"] == "REAL" and e["predicted_label"] == "FAKE"]
    false_negatives = [e for e in errors if e["true_label"] == "FAKE" and e["predicted_label"] == "REAL"]

    def length_bucket_stats(err_list):
        if not err_list:
            return {}
        lengths = [e["text_length"] for e in err_list]
        return {"count": len(err_list), "avg_length": round(sum(lengths) / len(lengths), 1),
                "min_length": min(lengths), "max_length": max(lengths)}

    return {
        "total_test_examples": len(y_test),
        "total_errors": len(errors),
        "error_rate": round(len(errors) / len(y_test), 4) if y_test else None,
        "false_positives_real_predicted_fake": {
            "count": len(false_positives),
            "stats": length_bucket_stats(false_positives),
            "examples": false_positives[:max_examples],
        },
        "false_negatives_fake_predicted_real": {
            "count": len(false_negatives),
            "stats": length_bucket_stats(false_negatives),
            "examples": false_negatives[:max_examples],
        },
        "interpretation": (
            "false_positives (legitimate REAL articles predicted FAKE) are the most "
            "user-visible failure mode for a tool meant to check real news - review the "
            "examples above for common patterns (e.g. short length, quote-heavy text, "
            "unusual formatting) before relying on this model for production decisions."
        ),
    }


def main():
    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found at {DATASET_PATH}. Run prepare_dataset.py first.")

    texts, labels = load_dataset()
    if len(texts) < 200:
        raise SystemExit("Dataset too small to train/validate/test reliably.")

    cleaned = [clean_text(t) for t in texts]

    # 70 / 15 / 15 stratified split. Split twice to get three disjoint sets.
    X_temp, X_test, y_temp, y_test, raw_temp, raw_test = train_test_split(
        cleaned, labels, texts, test_size=0.15, random_state=RANDOM_SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=RANDOM_SEED, stratify=y_temp
    )  # 0.1765 * 0.85 ~= 0.15 of the original total -> ~70/15/15 overall

    print(f"Split sizes -> train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}\n")

    candidates = build_candidates()
    validation_results = {}

    print("Training and validating candidate models (test set NOT touched yet)...\n")
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        val_metrics, _ = evaluate(pipeline, X_val, y_val)
        validation_results[name] = val_metrics
        print(f"  [{name}] validation: accuracy={val_metrics['accuracy']:.4f} "
              f"f1_macro={val_metrics['f1_macro']:.4f} "
              f"precision(FAKE)={val_metrics['precision']:.4f} recall(FAKE)={val_metrics['recall']:.4f}")

    best_name = max(validation_results, key=lambda n: validation_results[n]["f1_macro"])
    best_pipeline = candidates[best_name]
    print(f"\nSelected model based on VALIDATION macro-F1: {best_name}\n")

    # ---- Evaluate the selected model EXACTLY ONCE on the held-out test set ----
    test_metrics, y_pred_test = evaluate(best_pipeline, X_test, y_test)
    cm = confusion_matrix(y_test, y_pred_test, labels=["REAL", "FAKE"]).tolist()
    class_report = classification_report(y_test, y_pred_test, output_dict=True)

    print("Held-out TEST metrics (the number that matters for real-world generalization):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    thresholds = derive_confidence_thresholds(best_pipeline, X_val, y_val)
    print(f"\nConfidence thresholds derived from validation data: {thresholds}")

    error_analysis = run_error_analysis(best_pipeline, raw_test, X_test, y_test, y_pred_test)
    with open(ERROR_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2)
    print(f"\nTest-set errors: {error_analysis['total_errors']} / {error_analysis['total_test_examples']} "
          f"({error_analysis['error_rate']:.2%})")
    print(f"  False positives (REAL predicted FAKE): {error_analysis['false_positives_real_predicted_fake']['count']}")
    print(f"  False negatives (FAKE predicted REAL): {error_analysis['false_negatives_fake_predicted_real']['count']}")

    metrics = {
        "model_selected": best_name,
        "selection_criterion": "highest validation macro-F1 (test set untouched during selection)",
        "random_seed": RANDOM_SEED,
        "dataset_size": len(texts),
        "train_size": len(X_train),
        "validation_size": len(X_val),
        "test_size": len(X_test),
        "preprocessing": "detector.preprocessing.clean_text (shared with inference)",
        "validation_metrics_all_candidates": validation_results,
        "test_metrics_selected_model": test_metrics,
        "test_confusion_matrix_labels": ["REAL", "FAKE"],
        "test_confusion_matrix": cm,
        "test_classification_report": class_report,
        "confidence_thresholds": thresholds,
        "honest_disclaimer": (
            "These metrics describe performance on a held-out split of ONE real-world "
            "dataset (see raw/DATASET_SOURCE.md), whose topic coverage skews toward "
            "U.S. political news circa 2015-2016. They are NOT a guarantee of accuracy "
            "on arbitrary future real-world articles, especially on topics/styles "
            "underrepresented in the training data. 100% accuracy, if ever observed, "
            "would reflect this test split - not real-world infallibility."
        ),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Refit the selected model on TRAIN + VALIDATION (standard practice once
    # test evaluation is done and the model won't be re-selected again),
    # so the deployed model benefits from every labeled example except the
    # untouched test set.
    final_pipeline = candidates[best_name]
    final_pipeline.fit(X_train + X_val, y_train + y_val)
    joblib.dump(final_pipeline, PIPELINE_PATH)

    train_env = {"scikit_learn_version": sklearn.__version__, "python_version": platform.python_version()}
    with open(TRAIN_ENV_PATH, "w", encoding="utf-8") as f:
        json.dump(train_env, f, indent=2)

    print(f"\nSaved deployed pipeline (trained on train+val, evaluated once on held-out test) -> {PIPELINE_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved error analysis -> {ERROR_ANALYSIS_PATH}")
    print(f"Saved training environment info -> {TRAIN_ENV_PATH}")


if __name__ == "__main__":
    main()
