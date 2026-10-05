"""
Test the trained TruthLens AI model on a mix of curated examples AND your
own pasted text - including completely unseen, real-world articles that
were never part of training/validation/test.

Usage (from the TruthLensAi/ project root, venv active):

    python detector/ml/test_model.py            # curated examples only
    python detector/ml/test_model.py --interactive   # also prompts for your own text

This script loads detector/ml/model_pipeline.pkl directly (the same
artifact detector/services.py uses in the Django app) so what you see here
is exactly what the web app would predict for the same input.

IMPORTANT: none of the expected predictions below are hard-coded checks
that force a "pass" - this script only PRINTS what the model actually
predicts, so you can judge the results yourself. It intentionally includes
examples the model is likely to get wrong (e.g. opinion/editorial pieces),
because honest testing means showing failures too.
"""
import argparse
import sys
from pathlib import Path

import joblib

ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detector.preprocessing import clean_text  # noqa: E402

PIPELINE_PATH = ML_DIR / "model_pipeline.pkl"


# Curated test cases spanning the categories the project spec asked for.
# All text below was written for this test script (not copied from any
# copyrighted news source) so it's genuinely unseen by the model, while
# still being representative of each category's typical style.
CURATED_CASES = [
    ("Neutral, real-world-style news", (
        "The city council voted 6-3 on Tuesday to approve a new budget for road "
        "maintenance in the coming fiscal year. The plan allocates funding for "
        "repaving several major intersections and repairing storm-damaged "
        "sidewalks. Officials said the work is expected to begin in early spring "
        "and will be funded through a combination of state grants and municipal "
        "bonds. Residents can review the full budget proposal on the city's "
        "website starting next week."
    )),
    ("Clearly fabricated / sensational", (
        "SHOCKING: Secret leaked documents reveal that a hidden group of world "
        "leaders has been controlling the global weather for decades!!! Doctors "
        "HATE this one simple trick that exposes the truth they don't want you "
        "to know. Share this before it gets DELETED by the mainstream media!!! "
        "You won't believe what happens next."
    )),
    ("Short news snippet", (
        "Local authorities confirmed a water main break on Fifth Avenue this "
        "morning, temporarily disrupting service to nearby homes. Repair crews "
        "are on site and expect service to be restored by this evening."
    )),
    ("Long, detailed real-world-style article", (
        "Researchers at a regional university published new findings this week "
        "on the effects of urban tree cover on summer temperatures. The study, "
        "which tracked surface temperatures across twelve neighborhoods over "
        "three summers, found that areas with denser tree canopy were on "
        "average four degrees cooler than comparable areas with minimal "
        "vegetation. The team says the results could inform future zoning and "
        "public health decisions, particularly in lower-income neighborhoods "
        "that historically have less tree cover. City planners have already "
        "requested a copy of the full report to consider for the next update "
        "to the municipal green-space plan. The researchers noted several "
        "limitations, including a relatively short three-year window and the "
        "difficulty of isolating tree cover from other variables like building "
        "density and proximity to water. They recommend a longer-term followup "
        "study before any policy is finalized, and emphasized that correlation "
        "between tree cover and cooling, while consistent with prior research "
        "elsewhere, does not by itself prove causation in this specific dataset."
    )),
    ("Opinion/editorial (known hard case - see error_analysis.json)", (
        "It's time we had an honest conversation about how our leaders have "
        "failed to address the housing crisis. For years, promises have been "
        "made and broken, while ordinary families struggle to find affordable "
        "places to live. I believe the solution starts with holding elected "
        "officials accountable at the ballot box, not with more empty rhetoric."
    )),
    ("Neutral technology news, different topic than training data", (
        "A software company announced Wednesday that it will roll out a "
        "security patch for its widely used messaging application after "
        "researchers identified a vulnerability that could expose user "
        "metadata. The company said no evidence has been found that the flaw "
        "was exploited, and the patch will be delivered automatically to all "
        "users over the coming week."
    )),
]


def load_pipeline():
    if not PIPELINE_PATH.exists():
        raise SystemExit(
            f"No trained model found at {PIPELINE_PATH}.\n"
            f"Run: python detector/ml/prepare_dataset.py && python detector/ml/train_model.py"
        )
    return joblib.load(PIPELINE_PATH)


def predict_and_report(pipeline, text: str, label: str = ""):
    cleaned = clean_text(text)
    prediction = pipeline.predict([cleaned])[0]
    has_proba = hasattr(pipeline, "predict_proba")
    confidence = None
    if has_proba:
        proba = pipeline.predict_proba([cleaned])[0]
        classes = list(pipeline.classes_)
        confidence = round(float(proba[classes.index(prediction)]) * 100, 1)

    print(f"\n{'=' * 70}")
    if label:
        print(f"CASE: {label}")
    print(f"TEXT (excerpt): {text[:160]}{'...' if len(text) > 160 else ''}")
    print(f"PREDICTION:          {prediction}")
    print(f"CONFIDENCE:          {confidence if confidence is not None else 'N/A'}%")
    print(f"PROBABILITY AVAILABLE: {has_proba}")
    return prediction, confidence


def run_curated():
    pipeline = load_pipeline()
    print(f"Loaded model from {PIPELINE_PATH}\n")
    print("Running curated test cases (categories: neutral/real-world-style, "
          "fabricated/sensational, short, long, opinion/editorial, unseen topic).")
    print("None of these are hard-coded to 'pass' - results are printed as-is.\n")

    results = []
    for label, text in CURATED_CASES:
        pred, conf = predict_and_report(pipeline, text, label)
        results.append((label, pred, conf))

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    for label, pred, conf in results:
        print(f"  {label:55s} -> {pred:10s} ({conf}%)" if conf is not None else f"  {label:55s} -> {pred}")
    print(
        "\nReminder: this is a small, illustrative set of examples, not a formal "
        "evaluation. See detector/ml/metrics.json for the real held-out test-set "
        "numbers, and detector/ml/error_analysis.json for known failure patterns "
        "(e.g. opinion/editorial pieces are the hardest category)."
    )
    return pipeline


def run_interactive(pipeline):
    print(f"\n{'=' * 70}")
    print("INTERACTIVE MODE - paste any article text (not from the training set),")
    print("then press Enter twice to run the prediction. Type 'quit' to exit.\n")
    while True:
        print("Paste article text:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                return
            if line.strip().lower() == "quit":
                return
            if line == "" and lines:
                break
            lines.append(line)
        text = "\n".join(lines).strip()
        if not text:
            continue
        predict_and_report(pipeline, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Also prompt for your own article text")
    args = parser.parse_args()

    pipeline = run_curated()
    if args.interactive:
        run_interactive(pipeline)
