"""
Core analysis services for TruthLens AI.

Every function here is a REAL implementation - no random numbers, no
hard-coded results. Where a feature genuinely requires an external service
that is not configured (fact verification, LLM-based summarization/chatbot),
the function returns a clearly-labeled "unavailable" result instead of
fabricating one.
"""
import os
import re
import json
import platform
from pathlib import Path
from functools import lru_cache

import numpy as np
import sklearn
import joblib
import requests
from typing import Any, Dict, List, Optional, Tuple

from .preprocessing import clean_text, word_count

ML_DIR = Path(__file__).resolve().parent / "ml"
PIPELINE_PATH = ML_DIR / "model_pipeline.pkl"
TRAIN_ENV_PATH = ML_DIR / "train_env.json"
METRICS_PATH = ML_DIR / "metrics.json"

# Default confidence thresholds, used only if metrics.json (produced by
# detector/ml/train_model.py) doesn't have validation-derived ones yet.
DEFAULT_HIGH_THRESHOLD = 0.80
DEFAULT_MEDIUM_THRESHOLD = 0.65

# Legacy two-file artifacts from an earlier version of the training script.
# Only used as a fallback if a combined pipeline hasn't been trained yet.
LEGACY_MODEL_PATH = ML_DIR / "model.pkl"
LEGACY_VECTORIZER_PATH = ML_DIR / "vectorizer.pkl"

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip() or (GOOGLE_API_KEY.strip() if GOOGLE_API_KEY else "")
GOOGLE_FACTCHECK_API_KEY = os.environ.get("FACT_CHECK_API_KEY", "").strip()
GOOGLE_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", os.environ.get("SARVAM_AI_API_KEY", "")).strip()

from . import ai_services


class ModelUnavailableError(Exception):
    pass


def _retrain_hint():
    return (
        "Run `python detector/ml/train_model.py` from the project root "
        "(with your virtual environment active) to train and save a fresh model."
    )


@lru_cache(maxsize=1)
def _load_model():
    """Load the trained model, verify it's actually usable in THIS
    environment, and fail fast with a clear, actionable message rather than
    letting a version-mismatch AttributeError surface mid-request.

    Root cause this guards against: a scikit-learn estimator pickle is only
    reliably loadable with the SAME (or very close) scikit-learn version it
    was trained with. Internals of LogisticRegression (e.g. the
    `multi_class` attribute) have changed across releases, so a pickle
    trained on one version and loaded on another can raise
    `AttributeError: 'LogisticRegression' object has no attribute
    'multi_class'` the first time an incompatible code path (like
    predict_proba) runs. The fix is to retrain locally with the currently
    installed scikit-learn - never copy a .pkl between environments with
    different scikit-learn versions.
    """
    pipeline = None

    if PIPELINE_PATH.exists():
        try:
            pipeline = joblib.load(PIPELINE_PATH)
        except Exception as e:
            raise ModelUnavailableError(
                f"The trained model file could not be loaded ({type(e).__name__}: {e}). "
                f"It is likely incompatible with the installed scikit-learn version "
                f"({sklearn.__version__}). {_retrain_hint()}"
            )
    elif LEGACY_MODEL_PATH.exists() and LEGACY_VECTORIZER_PATH.exists():
        # Backward-compatible fallback for projects still using the older
        # two-file (model.pkl + vectorizer.pkl) format.
        try:
            clf = joblib.load(LEGACY_MODEL_PATH)
            vectorizer = joblib.load(LEGACY_VECTORIZER_PATH)
            from sklearn.pipeline import Pipeline as _Pipeline
            pipeline = _Pipeline([("tfidf", vectorizer), ("clf", clf)])
        except Exception as e:
            raise ModelUnavailableError(
                f"The legacy model/vectorizer files could not be loaded ({type(e).__name__}: {e}). "
                f"{_retrain_hint()}"
            )
    else:
        raise ModelUnavailableError(
            f"No trained ML model was found in {ML_DIR}. {_retrain_hint()}"
        )

    # Warn (non-fatally) on a training/runtime scikit-learn version mismatch,
    # since that's the most common root cause of subtle prediction bugs.
    if TRAIN_ENV_PATH.exists():
        try:
            trained_with = json.loads(TRAIN_ENV_PATH.read_text()).get("scikit_learn_version")
            if trained_with and trained_with != sklearn.__version__:
                print(
                    f"[TruthLens AI] WARNING: model was trained with scikit-learn "
                    f"{trained_with}, but {sklearn.__version__} is installed. "
                    f"If predictions fail, {_retrain_hint()}"
                )
        except Exception:
            pass

    # Self-test: actually exercise predict()/predict_proba() once at load
    # time on a throwaway input, so an incompatible pickle is caught HERE
    # (with a clear error) instead of failing deep inside a user's request.
    try:
        probe_vec = pipeline.named_steps["tfidf"].transform(["self test probe text"])
        pipeline.named_steps["clf"].predict(probe_vec)
        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            pipeline.named_steps["clf"].predict_proba(probe_vec)
    except Exception as e:
        raise ModelUnavailableError(
            f"The trained model failed a self-test ({type(e).__name__}: {e}). This "
            f"usually means it was trained with a different scikit-learn version than "
            f"the one installed now ({sklearn.__version__}). {_retrain_hint()}"
        )

    return pipeline.named_steps["clf"], pipeline.named_steps["tfidf"]


# ---------------------------------------------------------------------------
# Confidence thresholds & generic weight extraction
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _confidence_thresholds():
    """Load HIGH/MEDIUM confidence cutoffs derived from validation data by
    train_model.py. Below MEDIUM, predict_news() reports UNCERTAIN instead
    of forcing a REAL/FAKE call the model isn't actually confident about."""
    if METRICS_PATH.exists():
        try:
            data = json.loads(METRICS_PATH.read_text())
            thresholds = data.get("confidence_thresholds", {})
            high = float(thresholds.get("high", DEFAULT_HIGH_THRESHOLD))
            medium = float(thresholds.get("medium", DEFAULT_MEDIUM_THRESHOLD))
            return high, medium
        except Exception:
            pass
    return DEFAULT_HIGH_THRESHOLD, DEFAULT_MEDIUM_THRESHOLD


def _extract_class_weights(clf):
    """Return a per-feature weight array pointing toward classes_[1], plus
    the classes list, for whichever model type is currently deployed - or
    None if the model type doesn't expose interpretable linear weights.
    Supports: LogisticRegression / LinearSVC (coef_), CalibratedClassifierCV
    wrapping either of those, and MultinomialNB (feature_log_prob_)."""
    classes = list(getattr(clf, "classes_", []))

    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_[0]), classes

    # CalibratedClassifierCV: average the underlying estimator's coef_
    # across its internal cross-validated calibrators.
    calibrated = getattr(clf, "calibrated_classifiers_", None)
    if calibrated:
        weights = []
        for c in calibrated:
            base = getattr(c, "estimator", None) or getattr(c, "base_estimator", None)
            if base is not None and hasattr(base, "coef_"):
                weights.append(np.asarray(base.coef_[0]))
        if weights:
            return np.mean(weights, axis=0), classes

    # MultinomialNB: log-probability difference between classes acts as an
    # analogous "weight toward class 1" for each feature.
    if hasattr(clf, "feature_log_prob_") and len(classes) == 2:
        return np.asarray(clf.feature_log_prob_[1] - clf.feature_log_prob_[0]), classes

    return None, classes


# ---------------------------------------------------------------------------
# Core ML prediction
# ---------------------------------------------------------------------------

def predict_news(raw_text: str) -> dict:
    """Run the real ML pipeline: preprocess -> vectorize -> predict.

    Returns dict with prediction ("REAL" / "FAKE" / "UNCERTAIN"), a
    confidence_status ("HIGH" / "MEDIUM" / "LOW"), confidence (0-100 or
    None), and top contributing terms for explainability.

    IMPORTANT: "prediction" reflects which historical writing-style
    patterns (real vs. hyperpartisan/fake sources) this text resembles per
    the trained classifier - it is NOT a factual truth verdict. When the
    model's own probability estimate is low, this function reports
    UNCERTAIN rather than forcing a REAL/FAKE call, and the caller should
    recommend the user verify via the fact-check layer / trusted sources.
    """
    try:
        clf, vectorizer = _load_model()
    except ModelUnavailableError as e:
        return {
            "prediction": None,
            "confidence": None,
            "confidence_display": "Model unavailable.",
            "confidence_status": None,
            "raw_model_label": None,
            "top_terms": [],
            "error": str(e),
        }

    cleaned = clean_text(raw_text)
    if not cleaned:
        return {
            "prediction": None,
            "confidence": None,
            "confidence_display": "No usable text after preprocessing.",
            "confidence_status": None,
            "raw_model_label": None,
            "top_terms": [],
            "error": "empty_text",
        }

    vec = vectorizer.transform([cleaned])

    try:
        raw_model_label = clf.predict(vec)[0]

        confidence = None
        confidence_display = "Model confidence information unavailable."
        confidence_status = None
        prediction = raw_model_label

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(vec)[0]
            classes = list(clf.classes_)
            idx = classes.index(raw_model_label)
            confidence_fraction = float(proba[idx])
            confidence = round(confidence_fraction * 100, 1)
            confidence_display = f"{confidence}%"

            high, medium = _confidence_thresholds()
            if confidence_fraction >= high:
                confidence_status = "HIGH"
            elif confidence_fraction >= medium:
                confidence_status = "MEDIUM"
            else:
                confidence_status = "LOW"
                # Below the validation-derived MEDIUM threshold, the model's
                # own probability estimate is close to a coin flip - report
                # UNCERTAIN rather than a falsely confident REAL/FAKE label.
                prediction = "UNCERTAIN"
    except AttributeError as e:
        # Defense in depth: _load_model() already self-tests the model, but
        # if an incompatible pickle somehow slipped through, fail cleanly
        # here instead of a raw 500 error reaching the user.
        return {
            "prediction": None,
            "confidence": None,
            "confidence_display": "Model unavailable.",
            "confidence_status": None,
            "raw_model_label": None,
            "top_terms": [],
            "error": (
                f"The ML model raised an unexpected error ({e}). This usually means it "
                f"was trained with a different scikit-learn version than what's "
                f"currently installed. Retrain it with `python detector/ml/train_model.py`."
            ),
        }

    is_ood, ood_reason, ood_metrics = check_distribution_shift(cleaned, vectorizer, vec)
    if is_ood and prediction != "UNCERTAIN":
        confidence_status = "LOW"
        prediction = "UNCERTAIN"
        confidence_display = f"{confidence}% (Out-of-Distribution)"

    top_terms = _top_contributing_terms(clf, vectorizer, vec, raw_model_label)

    return {
        "prediction": prediction,  # "REAL" / "FAKE" / "UNCERTAIN"
        "raw_model_label": raw_model_label,  # the model's raw REAL/FAKE call, pre-threshold
        "confidence": confidence,
        "confidence_display": confidence_display,
        "confidence_status": confidence_status,  # "HIGH" / "MEDIUM" / "LOW" / None
        "is_ood": is_ood,
        "ood_reason": ood_reason,
        "ood_metrics": ood_metrics,
        "top_terms": top_terms,
        "error": None,
    }


def check_distribution_shift(cleaned_text: str, vectorizer, vec) -> tuple[bool, str, dict]:
    """Out-of-distribution (OOD) detection.
    Evaluates whether an article falls significantly outside the historical training
    distribution based on vocabulary overlap and active feature density.
    """
    tokens = [w for w in cleaned_text.lower().split() if len(w) > 2]
    if len(tokens) < 25:
        return False, "", {"tokens": len(tokens), "overlap_ratio": 1.0, "active_features": vec.nnz}

    unique_tokens = set(tokens)
    vocab = getattr(vectorizer, "vocabulary_", {})
    in_vocab = [w for w in unique_tokens if w in vocab]
    overlap_ratio = len(in_vocab) / len(unique_tokens) if unique_tokens else 0.0

    # Substantial distribution shift: novel vocabulary (< 35% in-vocab) or sparse non-zero features
    is_ood = (overlap_ratio < 0.35) or (len(tokens) >= 30 and vec.nnz < 4)
    reason = (
        "Article vocabulary exhibits significant distribution shift from historical training data."
        if is_ood
        else ""
    )
    return is_ood, reason, {
        "tokens": len(tokens),
        "unique_tokens": len(unique_tokens),
        "in_vocab": len(in_vocab),
        "overlap_ratio": round(overlap_ratio, 3),
        "active_features": vec.nnz,
    }


def _top_contributing_terms(clf, vectorizer, vec, prediction, top_n=8):
    """Explainability: which terms present in THIS text pushed the model
    toward its decision. Uses the model's actual per-feature weights
    (works across LogisticRegression, CalibratedClassifierCV-wrapped
    LinearSVC, and MultinomialNB - see _extract_class_weights), restricted
    to terms that actually appear in the input (no invented signals)."""
    try:
        weights, classes = _extract_class_weights(clf)
        if weights is None or not classes:
            return []
        sign = 1 if prediction == classes[1] else -1

        feature_names = vectorizer.get_feature_names_out()
        nz = vec.nonzero()[1]
        scored = []
        for idx in nz:
            weight = weights[idx] * sign
            if weight > 0:
                scored.append((feature_names[idx], round(float(weight), 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Clickbait analysis - deterministic, rule-based
# ---------------------------------------------------------------------------

CLICKBAIT_PHRASES = [
    "you won't believe", "shocking", "won't believe what", "doctors hate",
    "this one trick", "what happens next", "will blow your mind", "the truth about",
    "they don't want you to know", "exposed", "revealed", "breaking",
    "gone wrong", "you need to see", "changes everything", "secret",
    "before it's too late", "wake up", "mainstream media won't tell you",
    "click here", "share this", "won't believe", "goes viral",
]


def analyze_clickbait(raw_text: str) -> dict:
    """Deterministic clickbait scoring based on real, explainable signals:
    sensational phrase matches, ALL-CAPS words, excessive punctuation,
    and exclamation density. No randomness."""
    text = raw_text or ""
    lower = text.lower()

    score = 0
    signals = []

    phrase_hits = [p for p in CLICKBAIT_PHRASES if p in lower]
    if phrase_hits:
        pts = min(40, len(phrase_hits) * 10)
        score += pts
        signals.append(f"Sensational phrase(s) detected: {', '.join(phrase_hits[:5])}")

    caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
    if caps_words:
        pts = min(20, len(caps_words) * 4)
        score += pts
        signals.append(f"{len(caps_words)} ALL-CAPS word(s) found (e.g. {', '.join(sorted(set(caps_words))[:3])})")

    exclam_count = text.count("!")
    if exclam_count:
        pts = min(20, exclam_count * 3)
        score += pts
        signals.append(f"{exclam_count} exclamation mark(s) used")

    question_bait = len(re.findall(r"\?\?+", text))
    if question_bait:
        score += min(10, question_bait * 5)
        signals.append("Excessive/rhetorical question punctuation detected")

    ellipsis_count = text.count("...")
    if ellipsis_count:
        score += min(10, ellipsis_count * 2)
        signals.append(f"{ellipsis_count} curiosity-gap ellipsis pattern(s) detected")

    score = max(0, min(100, score))

    if score >= 60:
        classification = "HIGH"
    elif score >= 25:
        classification = "MEDIUM"
    else:
        classification = "LOW"

    if not signals:
        signals.append("No strong clickbait indicators detected in the text.")

    return {"score": score, "classification": classification, "signals": signals}


# ---------------------------------------------------------------------------
# AI-generated text analysis - EXPERIMENTAL heuristic (no fabricated %)
# ---------------------------------------------------------------------------

def analyze_ai_generated(raw_text: str) -> dict:
    """Heuristic, experimental estimate of whether text may be AI-generated.

    Uses real, computable linguistic-uniformity signals (sentence-length
    variance, repeated-phrase rate, vocabulary diversity). This is NOT a
    calibrated AI-detection model and is explicitly labeled experimental,
    with no fabricated confidence percentage - only a coarse label plus the
    raw signal values that produced it, so the user can judge for themselves.
    """
    text = (raw_text or "").strip()
    if len(text.split()) < 40:
        return {
            "label": "Uncertain",
            "reason": "Text is too short for a meaningful heuristic estimate.",
            "signals": {},
            "disclaimer": "AI-generated text detection is experimental and may produce incorrect results.",
        }

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    lengths = [len(s.split()) for s in sentences] or [0]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)

    words = re.findall(r"[a-zA-Z']+", text.lower())
    vocab_diversity = (len(set(words)) / len(words)) if words else 0

    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    repeat_rate = 0.0
    if trigrams:
        repeat_rate = 1 - (len(set(trigrams)) / len(trigrams))

    signals = {
        "avg_sentence_length": round(mean_len, 1),
        "sentence_length_variance": round(variance, 1),
        "vocabulary_diversity": round(vocab_diversity, 3),
        "trigram_repetition_rate": round(repeat_rate, 3),
    }

    uniform_score = 0
    if variance < 12:
        uniform_score += 1
    if repeat_rate < 0.03:
        uniform_score += 1
    if 0.35 < vocab_diversity < 0.65:
        uniform_score += 1

    if uniform_score >= 3:
        label = "Likely AI Generated"
    elif uniform_score <= 1:
        label = "Likely Human Written"
    else:
        label = "Uncertain"

    return {
        "label": label,
        "signals": signals,
        "disclaimer": "AI-generated text detection is experimental and may produce incorrect results.",
    }


# ---------------------------------------------------------------------------
# Explanation assembly (ML + linguistic signals, kept separate from
# external fact verification per spec section 14)
# ---------------------------------------------------------------------------

def build_explanation(prediction_result: dict, clickbait_result: dict) -> list:
    explanation = []
    if prediction_result.get("error"):
        explanation.append("The ML model could not produce a prediction for this input.")
        return explanation

    if prediction_result.get("prediction") == "UNCERTAIN":
        explanation.append(
            f"The model's probability estimate for this text was too close to a coin flip "
            f"(confidence status: {prediction_result.get('confidence_status')}) to responsibly "
            f"call it REAL or FAKE, so TruthLens AI is reporting UNCERTAIN. Its underlying "
            f"raw label lean was '{prediction_result.get('raw_model_label')}', but you should "
            f"verify this article through trusted sources or the fact-check results below."
        )

    terms = prediction_result.get("top_terms") or []
    if terms:
        term_list = ", ".join(f"'{t}'" for t, _ in terms)
        explanation.append(
            f"The model's prediction was most influenced by these terms present in the "
            f"text: {term_list}. Note this reflects writing-style patterns learned from "
            f"historical training articles, not a factual verification of this article's claims."
        )
    else:
        explanation.append(
            "The model made a prediction, but no individual terms stood out as strong "
            "contributors for this specific input."
        )

    if clickbait_result["classification"] in ("HIGH", "MEDIUM"):
        explanation.append(
            f"Separately, the text shows {clickbait_result['classification'].lower()} "
            "clickbait characteristics (see Clickbait Analysis below). This is a "
            "linguistic-style signal only and is not, by itself, proof that the article "
            "is fake."
        )

    return explanation


# ---------------------------------------------------------------------------
# Summarization - real LLM call via Anthropic API if configured, else
# graceful fallback (extractive summary) so the feature still works.
# ---------------------------------------------------------------------------

def summarize_text(raw_text: str, length: str = "medium") -> dict:
    length = length if length in ("short", "medium", "detailed") else "medium"
    target = {
        "short": "2-3 sentences",
        "medium": "1 short paragraph (4-6 sentences)",
        "detailed": "3-4 paragraphs covering all major points",
    }[length]

    # Try active LLM summarization (Google Gemini priority, then Anthropic)
    prompt = (
        f"Summarize the following news article factually and neutrally in {target}. "
        "Only use information present in the article. Do not add opinions or outside facts.\n\n"
        f"ARTICLE:\n{raw_text[:12000]}"
    )
    res = ai_services.call_llm(prompt, max_tokens=700)
    if res.get("success") and res.get("text"):
        provider_name = res.get("provider", "ai")
        return {"summary": res["text"].strip(), "source": f"ai ({provider_name})", "error": None}

    # Deterministic extractive fallback
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text.strip()) if s.strip()]
    n = {"short": 2, "medium": 4, "detailed": 8}[length]
    fallback = " ".join(sentences[:n]) or "Not enough text to summarize."
    has_llm_key = bool(GEMINI_API_KEY or ANTHROPIC_API_KEY)
    note = " (AI summarization API call failed; showing extractive fallback.)" if has_llm_key else ""
    return {
        "summary": fallback,
        "source": "extractive_fallback",
        "error": (
            "AI-powered summarization is not configured (set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env) - "
            "showing a basic extractive summary instead." + note
        ) if not has_llm_key else None,
    }


# ---------------------------------------------------------------------------
# Fact verification - NEVER invents sources. Uses Google Fact Check Tools
# API if configured; otherwise returns a clearly-labeled unavailable result.
# ---------------------------------------------------------------------------

def verify_facts(raw_text: str, claims: Optional[List[str]] = None) -> dict:
    return ai_services.verify_google_factcheck(raw_text, claims_list=claims)


# ---------------------------------------------------------------------------
# Explanation assistant / chatbot - grounded in TruthLens verification
# architecture. Real LLM call (Gemini priority, Anthropic fallback) with
# comprehensive platform knowledge and grounded fallback.
# ---------------------------------------------------------------------------

def answer_assistant_question(
    question: str,
    analysis_context: Optional[dict] = None,
    history: Optional[List[dict]] = None,
) -> str:
    """Provides natural, accurate, and grounded explanations as TruthLens Assistant.
    Understands TruthLens architecture (ML style classifier vs live evidence
    verification vs final assessment), handles general questions and specific
    analysis results, and maintains conversational context."""
    has_analysis = bool(analysis_context and isinstance(analysis_context, dict))

    # Build system instructions
    system_instruction = (
        "You are TruthLens Assistant, a helpful, intelligent, and friendly AI verification guide "
        "for the TruthLens AI platform.\n\n"
        "TRUTHLENS CORE ARCHITECTURE & PLATFORM KNOWLEDGE:\n"
        "1. Mission: Help users verify news, spot misinformation, and understand credibility signals.\n"
        "2. Multi-Layer Verification Pipeline:\n"
        "   - Layer 1 (ML Stylistic Classifier): Examines writing style, sensationalism, and linguistic "
        "patterns based on historical training data (ISOT dataset). Stylistic patterns are NOT proof of factual truth.\n"
        "   - Layer 2 (External Evidence Verification): Retrieves live reporting and accredited fact-checks "
        "using Google Fact Check Tools API and Tavily Real-Time Web Search. Never fabricates evidence.\n"
        "   - Layer 3 (Temporal Anchoring): Identifies whether news is contemporary/current or historical "
        "to prevent modern events from being misjudged by older training data.\n"
        "   - Layer 4 (Final Assessment): Combines signals into LIKELY REAL, LIKELY FAKE, or UNCERTAIN. "
        "Verified independent reporting overrides the ML stylistic prediction (e.g., if ML predicted Fake "
        "due to style, but live sources corroborate the facts, the Final Assessment is LIKELY REAL).\n"
        "   - 'UNCERTAIN' Result: Means TruthLens does not have enough reliable evidence or external sources "
        "disagree. It is safer to remain uncertain and review sources than to guess.\n"
        "3. Platform Features:\n"
        "   - Detect News: Paste text (30-50,000 characters), upload PDF/TXT (up to 10MB), or enter an article URL.\n"
        "   - Dashboard: Tracks personal verification metrics, real vs fake analytics, and recent analyses.\n"
        "   - History: Comprehensive log of all past analyses.\n"
        "   - Saved Articles: Bookmark verified articles for research.\n"
        "   - Profile Settings: Edit handle, email, name, avatar, and bio.\n\n"
        "RESPONSE RULES:\n"
        "- Tone: Professional, friendly, concise, and natural.\n"
        "- DO NOT start answers with repetitive filler phrases like 'Sure!', 'Certainly!', or 'Absolutely!'.\n"
        "- Explain technical concepts in plain, accessible language.\n"
        "- If an API is unavailable or unconfigured, clearly explain that external verification is unavailable.\n"
        "- Never fabricate evidence, URLs, or fact-check verdicts.\n"
        "- If the user asks something completely unrelated to news, truth, or TruthLens, politely steer "
        "them back to news verification and TruthLens features.\n"
    )

    if has_analysis:
        system_instruction += (
            f"\nCURRENT ANALYSIS CONTEXT (Ground your answer strictly in this data):\n"
            f"{json.dumps(analysis_context, indent=2, default=str)[:7000]}"
        )
    else:
        system_instruction += (
            "\nNote: The user is currently browsing the general site without an active analysis selected. "
            "Help them with general questions, navigation, feature explanations, or verification guidance."
        )

    # Format multi-turn conversation
    formatted_prompt = ""
    if history and isinstance(history, list):
        for turn in history[-6:]:
            role = "User" if turn.get("role") in ("user", "human") else "Assistant"
            text = (turn.get("content") or turn.get("text") or "").strip()
            if text:
                formatted_prompt += f"{role}: {text}\n\n"
    formatted_prompt += f"User: {question}\n\nAssistant:"

    res = ai_services.call_llm(formatted_prompt, system_instruction=system_instruction, max_tokens=600)
    if res.get("success") and res.get("text"):
        return res["text"].strip()

    # -----------------------------------------------------------------------
    # Intelligent Grounded Fallback (when LLM API is unavailable or offline)
    # -----------------------------------------------------------------------
    q = question.lower()
    ctx = analysis_context or {}

    # 1. UNCERTAIN meaning
    if "uncertain" in q or "what does uncertain" in q:
        return (
            "Uncertain means TruthLens doesn't have enough reliable evidence to confidently classify "
            "the news as real or fake. When sources are scarce or conflicting, it's safer to review the "
            "available evidence instead of guessing."
        )

    # 2. How to check / analyze news
    if any(k in q for k in ("how do i check", "how to check", "how to analyze", "how do i analyze", "how to detect", "upload", "url")):
        return (
            "To check news on TruthLens, head over to the **Detect News** page. You can paste article text "
            "(minimum 30 characters), upload a PDF or TXT file (up to 10MB), or provide an article URL. "
            "TruthLens will analyze linguistic patterns, search live web evidence, and deliver a comprehensive verification result."
        )

    # 3. Why this result / Explain result
    if any(k in q for k in ("why did my article", "why was this", "explain my result", "explain this result", "why fake", "why real")):
        if has_analysis:
            final = ctx.get("final_assessment") or "Not evaluated"
            pred = ctx.get("ml_prediction") or ctx.get("prediction") or "N/A"
            v_conf = ctx.get("verification_confidence_score") or ctx.get("verification_confidence") or "N/A"
            reasons = ctx.get("verification_reasoning") or ctx.get("explanation") or []
            reason_text = " ".join(reasons[:2]) if reasons else "No specific reasoning details recorded."

            if final == "LIKELY REAL" and pred == "FAKE":
                return (
                    f"This article received a Final Assessment of **LIKELY REAL** (Confidence: {v_conf}). "
                    f"While the stylistic ML model initially flagged the writing style as suspicious (ML Style: {pred}), "
                    f"independent real-world reporting and live web evidence corroborated the central claims, "
                    f"overriding the historical style classifier.\n\nKey finding: {reason_text}"
                )
            elif final == "LIKELY FAKE" and pred == "REAL":
                return (
                    f"This article received a Final Assessment of **LIKELY FAKE** (Confidence: {v_conf}). "
                    f"Although the writing style resembles conventional journalistic reporting, external verification "
                    f"and fact-checks contradicted central claims.\n\nKey finding: {reason_text}"
                )
            else:
                return (
                    f"This article received a Final Assessment of **{final}** (Confidence: {v_conf}). "
                    f"ML stylistic classifier: {pred}. {reason_text}"
                )
        return (
            "TruthLens uses two different signals to reach a result: the ML model looks at language patterns "
            "learned from historical training data, while external verification checks live web reporting and accredited fact-checks. "
            "These are separate signals, and verified real-world evidence overrides stylistic predictions."
        )

    # 4. How TruthLens works / verification process
    if any(k in q for k in ("how does truthlens work", "how is news verified", "how does it work", "pipeline")):
        return (
            "TruthLens verifies news through a multi-layer pipeline:\n\n"
            "1. **Linguistic ML Analysis**: Examines writing patterns and sensationalist phrasing.\n"
            "2. **Live Evidence Verification**: Searches Google Fact Check Tools and live web sources via Tavily.\n"
            "3. **Temporal Anchoring**: Detects whether news is contemporary or historical.\n"
            "4. **Final Assessment**: Combines evidence and style signals into LIKELY REAL, LIKELY FAKE, or UNCERTAIN, "
            "prioritizing verified external facts over writing style alone."
        )

    # 5. ML vs Evidence / Difference
    if any(k in q for k in ("difference", "ml vs", "evidence vs", "style vs")):
        return (
            "Here is the key distinction in TruthLens:\n\n"
            "• **ML Model Prediction**: Evaluates writing style, clickbait patterns, and syntax learned from historical "
            "training articles. It is NOT proof of factual truth.\n"
            "• **External Evidence Verification**: Retrieves live articles from credible newsrooms and accredited "
            "fact-checkers to confirm whether reported events actually happened."
        )

    # 6. Confidence scores
    if "confidence" in q:
        if has_analysis:
            v_score = ctx.get("verification_confidence_score") or "N/A"
            m_score = ctx.get("ml_confidence_display") or f"{ctx.get('ml_confidence')}%" if ctx.get("ml_confidence") else "N/A"
            return (
                f"For this article:\n"
                f"• **Verification Confidence**: {v_score} (based on corroborated independent web evidence).\n"
                f"• **ML Style Confidence**: {m_score} (model's probability estimate based on writing style)."
            )
        return (
            "TruthLens provides two distinct confidence scores:\n"
            "• **Verification Confidence**: Measures how strongly independent news sources and fact-checkers corroborate or contradict claims.\n"
            "• **ML Confidence**: Reflects the machine learning model's statistical certainty regarding writing-style patterns."
        )

    # 7. Clickbait
    if "clickbait" in q:
        cb = ctx.get("clickbait", {})
        if cb.get("score") is not None:
            return (
                f"Clickbait score for this article is {cb.get('score')}% ({cb.get('classification')}). "
                f"Signals: {', '.join(cb.get('signals', [])) or 'None detected'}."
            )
        return (
            "Clickbait analysis measures sensationalism, emotional exaggeration, and misleading hooks. "
            "A high clickbait score indicates manipulative headline style, but does not by itself prove an article is false."
        )

    # 8. Summary
    if "summar" in q:
        if ctx.get("summary"):
            return f"**Article Summary:**\n\n{ctx['summary']}"
        return (
            "TruthLens automatically generates concise AI or extractive summaries for articles over 150 characters "
            "so you can quickly understand the key claims."
        )

    # 9. Claims and facts
    if any(k in q for k in ("claim", "fact", "source", "evidence")):
        claims = ctx.get("claims", [])
        if claims:
            return "Extracted claims from this article:\n" + "\n".join(f"• {c}" for c in claims[:5])
        return (
            "TruthLens extracts central factual claims from news articles and verifies them against "
            "accredited fact-checkers (PolitiFact, Snopes, Reuters) and live authoritative news reporting."
        )

    # Default greeting / general assistance
    return (
        "Hi! I'm TruthLens Assistant. I can help you verify news, understand your analysis results, "
        "explain evidence vs ML signals, and guide you through TruthLens features. "
        "How can I help you today?"
    )


