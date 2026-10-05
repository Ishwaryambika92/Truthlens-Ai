"""
The verification pipeline orchestrator - wires Layers 1-10 together in
order. Each layer stays in its own modular component:
  Layer 1: Input extraction & SSRF protection (detector/extract_text.py)
  Layer 2: Article quality & ambiguity gate (detector/verification/quality.py)
  Layer 3: Source & domain analysis (detector/verification/source_analysis.py)
  Layer 4: ML fake-news classifier [inference-only] (detector/services.py)
  Layer 5: Claim extraction (detector/verification/claims.py)
  Layer 6: Fact-check tools API (detector/services.py)
  Layer 7-8: Web evidence search & multi-source corroboration (detector/verification/web_evidence.py)
  Layer 9-10: Temporal context analysis, signal aggregation, and final assessment
             (detector/verification/temporal.py & aggregation.py)
"""
from .. import services
from . import quality, temporal, source_analysis, claims as claims_module, web_evidence, aggregation


def run_verification(article_text: str, metadata: dict | None = None) -> dict:
    """Runs the full multi-layer verification pipeline on an article.

    metadata: output of detector.extract_text._extract_metadata, or dict
    optionally containing 'temporal_preference', 'published_date', 'domain', etc.

    Returns a comprehensive dict with every layer's output plus the final
    aggregated assessment and verification confidence.
    """
    metadata = metadata or {}

    # --- Layer 2: quality check & ambiguity gate ---
    quality_result = quality.check_article_quality(article_text, metadata)

    # --- Temporal context analysis (detects historical vs current news) ---
    temporal_info = temporal.detect_temporal_context(
        article_text, metadata=metadata, explicit_choice=metadata.get("temporal_preference")
    )

    # --- Layer 3: source analysis (independent of quality result) ---
    source_result = source_analysis.analyze_source(metadata)

    if not quality_result["usable"]:
        # Unusable, short, or ambiguous content -> return UNCERTAIN without guessing
        empty_prediction = {
            "prediction": None, "confidence": None, "confidence_display": "N/A",
            "confidence_status": None, "raw_model_label": None, "top_terms": [],
            "error": "insufficient_content",
        }
        empty_clickbait = {"score": 0, "classification": "LOW", "signals": []}
        empty_ai = {"label": "Uncertain", "signals": {}, "disclaimer": ""}
        empty_factcheck = {"available": False, "claims": [], "message": "Not run - article quality check failed."}
        empty_evidence = {"available": False, "claim_evidence": [], "message": "Not run - article quality check failed."}
        final = aggregation.aggregate_signals(
            quality_result=quality_result, prediction_result=empty_prediction,
            clickbait_result=empty_clickbait, ai_text_result=empty_ai,
            source_result=source_result, fact_check_result=empty_factcheck,
            evidence_result=empty_evidence, temporal_info=temporal_info,
        )
        return {
            "quality": quality_result,
            "temporal": temporal_info,
            "source": source_result,
            "prediction": empty_prediction,
            "clickbait": empty_clickbait,
            "ai_text": empty_ai,
            "claims": [],
            "fact_check": empty_factcheck,
            "evidence": empty_evidence,
            "final": final,
        }

    # --- Layer 4: ML classification (strictly inference-only) ---
    prediction_result = services.predict_news(article_text)

    # --- Linguistic signals (clickbait score is excluded from final verdict) ---
    clickbait_result = services.analyze_clickbait(article_text)
    ai_text_result = services.analyze_ai_generated(article_text)

    # --- Layer 5: claim extraction (deterministic heuristic + AI fallback) ---
    claim_result = claims_module.extract_claims(article_text)
    extracted_claims = claim_result.get("claims") or []
    if not extracted_claims:
        extracted_claims = claims_module.extract_claims_heuristic(article_text)

    # --- Layer 6: fact-check database (external API if configured) ---
    fact_check_result = services.verify_facts(article_text, claims=extracted_claims)

    # --- Layers 7-8: independent web evidence + cross-source check ---
    evidence_result = web_evidence.gather_evidence(extracted_claims)

    # --- Layers 9-10: aggregation + final assessment ---
    final = aggregation.aggregate_signals(
        quality_result=quality_result,
        prediction_result=prediction_result,
        clickbait_result=clickbait_result,
        ai_text_result=ai_text_result,
        source_result=source_result,
        fact_check_result=fact_check_result,
        evidence_result=evidence_result,
        temporal_info=temporal_info,
    )

    return {
        "quality": quality_result,
        "temporal": temporal_info,
        "source": source_result,
        "prediction": prediction_result,
        "clickbait": clickbait_result,
        "ai_text": ai_text_result,
        "claims": extracted_claims,
        "claim_extraction_message": claim_result.get("message"),
        "fact_check": fact_check_result,
        "evidence": evidence_result,
        "final": final,
    }
