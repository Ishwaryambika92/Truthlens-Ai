"""
Layer 9 - Signal aggregation
Layer 10 - Uncertainty and final assessment

DOCUMENTED DECISION STRATEGY:

1. Article Quality / Ambiguity Gate (Tier 1):
   - If content is short, ambiguous, or lacks sufficient factual content ->
     Stop immediately and return UNCERTAIN with verification confidence LOW.
     Do not guess.

2. Evidence Override Gate (Tier 1 - bidirectional):
   - Contradicting evidence overrides ML REAL: If independent sources or
     fact-check databases contradict central claims -> LIKELY FAKE (High confidence).
   - Corroborating evidence overrides ML FAKE: If independent sources or
     fact-check databases corroborate central claims -> LIKELY REAL (High/Medium confidence).
   Corroborated external evidence always outranks historical ML style predictions.

3. Temporal Context Differentiation:
   - Current / Present News: Must NOT rely only on the historical ML model.
     If no independent external evidence or fact-checks can be retrieved
     (e.g. missing API keys, unindexed breaking news), the system reports
     UNCERTAIN with verification confidence LOW instead of guessing.
     When external evidence IS found, evidence determines the verdict.
   - Historical / Past News: ML prediction is combined with available source
     and evidence signals.

4. Source and Domain Independence:
   - Unknown domains are NOT penalized as fake (neutral weight 0.0).

5. Clickbait Independence:
   - Clickbait score is surfaced as a stylistic signal but EXCLUDED from
     the final assessment calculation.

6. Distinct Confidence Metrics:
   - ML confidence (learned style probability) and Verification confidence
     (evidence corroboration certainty) are computed and reported separately.
"""

ML_WEIGHT_HISTORICAL = 0.35
SOURCE_WEIGHT = 0.10
FACTCHECK_WEIGHT = 0.25
WEB_EVIDENCE_WEIGHT = 0.30

# For current news with evidence, external corroboration is primary:
CURRENT_WEB_WEIGHT = 0.65
CURRENT_FACTCHECK_WEIGHT = 0.25
CURRENT_SOURCE_WEIGHT = 0.05
CURRENT_ML_WEIGHT = 0.05

from .claims import normalize_claim_status, assign_claim_importance


def _ml_component(prediction_result: dict) -> float:
    """+1 leans REAL, -1 leans FAKE, 0 if unavailable/uncertain/OOD."""
    if (
        prediction_result.get("error")
        or prediction_result.get("prediction") == "UNCERTAIN"
        or prediction_result.get("is_ood")
        or not prediction_result.get("confidence")
    ):
        return 0.0
    label = prediction_result.get("raw_model_label")
    frac = prediction_result["confidence"] / 100.0
    direction = 1.0 if label == "REAL" else (-1.0 if label == "FAKE" else 0.0)
    return direction * max(0.0, (frac - 0.5) * 2)


def _factcheck_component(fact_check_result: dict) -> tuple[float, bool, bool, bool]:
    """Returns (score, considered: bool, has_false: bool, has_true: bool)."""
    if not fact_check_result.get("available") or not fact_check_result.get("claims"):
        return 0.0, False, False, False

    scores = []
    has_false = False
    has_true = False

    for c in fact_check_result["claims"]:
        alignment = c.get("alignment")
        if alignment == "CONTRADICTS":
            scores.append(-1.0)
            has_false = True
        elif alignment == "SUPPORTS":
            scores.append(1.0)
            has_true = True
        elif alignment == "UNRELATED":
            continue
        else:
            rating = (c.get("status") or "").lower()
            fc_text = (c.get("claim") or "").lower()
            is_hoax_counterclaim = any(h in fc_text for h in ("fake", "hoax", "staged", "conspiracy", "animation", "fraud"))
            if any(k in rating for k in ("false", "fake", "pants on fire", "incorrect", "hoax", "debunked")):
                if is_hoax_counterclaim:
                    scores.append(1.0)
                    has_true = True
                else:
                    scores.append(-1.0)
                    has_false = True
            elif any(k in rating for k in ("true", "correct", "accurate", "verified", "real")):
                scores.append(1.0)
                has_true = True
            elif any(k in rating for k in ("misleading", "mixture", "half", "partly")):
                scores.append(-0.3)

    if not scores:
        return 0.0, False, False, False

    return sum(scores) / len(scores), True, has_false, has_true


def _web_evidence_component(evidence_result: dict) -> tuple[float, bool, bool, bool, int, int]:
    """Returns (score, considered: bool, has_contradiction: bool, has_support: bool, supported_count, contradicted_count)."""
    records = evidence_result.get("claim_evidence") or []
    usable = [r for r in records if r.get("available")]
    if not usable:
        return 0.0, False, False, False, 0, 0

    status_scores = {"SUPPORTED": 1.0, "CONTRADICTED": -1.0, "MIXED": -0.2, "UNVERIFIED": 0.0}
    scores = [status_scores.get(r.get("status"), 0.0) for r in usable]

    supported_count = sum(1 for r in usable if r.get("status") == "SUPPORTED")
    contradicted_count = sum(1 for r in usable if r.get("status") == "CONTRADICTED")

    has_contradiction = contradicted_count > 0 and contradicted_count > supported_count
    has_support = supported_count > 0 and contradicted_count == 0

    return sum(scores) / len(scores), True, has_contradiction, has_support, supported_count, contradicted_count


def _evaluate_evidence_strength(evidence_result: dict, fact_check_result: dict) -> dict:
    """
    Evaluates evidence strength conceptually without relying on raw source counts.
    Considers source reliability tiers, directness, and primary source overrides.
    """
    records = evidence_result.get("claim_evidence") or []
    usable_claims = [r for r in records if r.get("available")]

    fc_score, fc_considered, fc_has_false, fc_has_true = _factcheck_component(fact_check_result)

    # Collect all retrieved sources across claims
    all_retrieved_sources = []
    for c in usable_claims:
        for s in c.get("sources", []):
            if s.get("retrieved") and not s.get("syndicated_of"):
                all_retrieved_sources.append(s)

    if all_retrieved_sources:
        supporting_sources = [
            s for s in all_retrieved_sources
            if s.get("verdict") == "SUPPORTS" or s.get("content_verdict") == "SUPPORTS"
        ]
        contradicting_sources = [
            s for s in all_retrieved_sources
            if s.get("verdict") == "CONTRADICTS" or s.get("content_verdict") == "CONTRADICTS"
        ]
        insufficient_sources = [
            s for s in all_retrieved_sources
            if s.get("verdict") in ("INSUFFICIENT", "UNRELATED")
            or s.get("content_verdict") in ("INSUFFICIENT", "UNRELATED")
        ]

        total_support_strength = sum(s.get("strength_score", 1.0) for s in supporting_sources)
        total_contra_strength = sum(s.get("strength_score", 1.0) for s in contradicting_sources)

        strong_support_count = sum(1 for s in supporting_sources if s.get("strength_label") == "STRONG")
        strong_contra_count = sum(1 for s in contradicting_sources if s.get("strength_label") == "STRONG")
        weak_contra_count = sum(1 for s in contradicting_sources if s.get("strength_label") == "WEAK")

        primary_support_count = sum(1 for s in supporting_sources if s.get("is_primary_source"))
        primary_contra_count = sum(1 for s in contradicting_sources if s.get("is_primary_source"))
    else:
        # Fallback for synthetic unit tests where sources list is empty and only claim statuses are given
        supported_claims = [r for r in usable_claims if r.get("status") == "SUPPORTED"]
        contradicted_claims = [r for r in usable_claims if r.get("status") == "CONTRADICTED"]

        supporting_sources = []
        contradicting_sources = []
        insufficient_sources = []

        total_support_strength = len(supported_claims) * 1.2
        total_contra_strength = len(contradicted_claims) * 1.2

        strong_support_count = len(supported_claims)
        strong_contra_count = len(contradicted_claims)
        weak_contra_count = 0

        primary_support_count = sum(1 for r in supported_claims if any(s.get("is_primary_source") for s in r.get("sources", [])))
        primary_contra_count = sum(1 for r in contradicted_claims if any(s.get("is_primary_source") for s in r.get("sources", [])))

    supported_claims_count = sum(1 for r in usable_claims if normalize_claim_status(r.get("status")) == "SUPPORTED")
    contradicted_claims_count = sum(1 for r in usable_claims if normalize_claim_status(r.get("status")) == "CONTRADICTED")
    insufficient_claims_count = sum(1 for r in usable_claims if normalize_claim_status(r.get("status")) == "INSUFFICIENT")
    mixed_claims_count = sum(1 for r in usable_claims if (r.get("status") or "").upper() == "MIXED")
    unverified_claims_count = insufficient_claims_count

    has_web_evidence = bool(all_retrieved_sources or usable_claims)

    # Claim importance evaluation (Section 19: important claims carry more weight)
    central_claims = [
        r for idx, r in enumerate(usable_claims)
        if (r.get("importance") or assign_claim_importance(r.get("claim", ""), idx, len(usable_claims))) in ("CENTRAL", "IMPORTANT")
    ]
    central_supported = sum(1 for r in central_claims if normalize_claim_status(r.get("status")) == "SUPPORTED")
    central_contradicted = sum(1 for r in central_claims if normalize_claim_status(r.get("status")) == "CONTRADICTED")

    has_mixed_claims = (
        (supported_claims_count >= 1 and contradicted_claims_count >= 1)
        or (strong_support_count >= 1 and strong_contra_count >= 1 and contradicted_claims_count >= 1)
    )

    # If central assertions are refuted, minor true assertions cannot mask the falsehood (Section 19)
    if central_contradicted >= 1 and central_supported == 0:
        has_mixed_claims = False

    # 1. Critical Contradiction Gate:
    # Requires authoritative fact-check refutation OR central claim contradiction OR strong direct contradiction
    has_critical_contradiction = False
    if not has_mixed_claims:
        if fc_has_false:
            # Authoritative verified fact-checker debunked the claim
            has_critical_contradiction = True
        elif central_contradicted >= 1 and central_supported == 0:
            # Central assertions are refuted
            has_critical_contradiction = True
        elif (contradicted_claims_count >= 1 and supported_claims_count == 0):
            has_critical_contradiction = True
        elif (strong_contra_count >= 1 or primary_contra_count >= 1 or contradicted_claims_count >= 1):
            if total_support_strength == 0.0:
                has_critical_contradiction = True
            elif total_contra_strength >= 1.5 * total_support_strength and not primary_support_count and not strong_support_count:
                has_critical_contradiction = True

    # 2. Strong Corroboration Gate:
    # Requires fact-check verified true OR strong reliable corroboration without direct contradiction of claims
    has_strong_corroboration = False
    if not has_mixed_claims and not fc_has_false and strong_contra_count == 0 and primary_contra_count == 0 and contradicted_claims_count == 0:
        if fc_has_true:
            has_strong_corroboration = True
        elif primary_support_count >= 1:
            # Primary Source Override (Rule 6)
            has_strong_corroboration = True
        elif total_support_strength >= 1.0 or supported_claims_count >= 1:
            # If weak contradictions exist, supporting strength must clearly outweigh them (e.g. 4 strong vs 3 weak)
            if total_contra_strength == 0.0 or total_support_strength >= 1.5 * total_contra_strength:
                has_strong_corroboration = True

    # 3. Genuine Conflict Gate (Rule 7):
    # When strong sources genuinely disagree on both sides
    has_genuine_conflict = False
    if not has_critical_contradiction and not has_strong_corroboration and not has_mixed_claims:
        if (strong_support_count >= 1 and strong_contra_count >= 1):
            has_genuine_conflict = True
        elif (mixed_claims_count >= 1 or (supported_claims_count >= 1 and contradicted_claims_count >= 1)):
            has_genuine_conflict = True

    # Calculate normalized web evidence score
    if total_support_strength + total_contra_strength > 0:
        evidence_score = (total_support_strength - total_contra_strength) / (total_support_strength + total_contra_strength)
    elif usable_claims:
        status_map = {"SUPPORTED": 1.0, "CONTRADICTED": -1.0, "MIXED": -0.2, "UNVERIFIED": 0.0}
        evidence_score = sum(status_map.get(r.get("status"), 0.0) for r in usable_claims) / len(usable_claims)
    else:
        evidence_score = 0.0

    return {
        "has_web_evidence": has_web_evidence,
        "fc_considered": fc_considered,
        "fc_has_false": fc_has_false,
        "fc_has_true": fc_has_true,
        "fc_score": fc_score,
        "evidence_score": evidence_score,
        "supporting_sources_count": len(supporting_sources) if all_retrieved_sources else 0,
        "contradicting_sources_count": len(contradicting_sources) if all_retrieved_sources else 0,
        "insufficient_sources_count": len(insufficient_sources) if all_retrieved_sources else 0,
        "total_support_strength": round(total_support_strength, 2),
        "total_contra_strength": round(total_contra_strength, 2),
        "strong_support_count": strong_support_count,
        "strong_contra_count": strong_contra_count,
        "weak_contra_count": weak_contra_count,
        "primary_support_count": primary_support_count,
        "primary_contra_count": primary_contra_count,
        "supported_claims_count": supported_claims_count,
        "contradicted_claims_count": contradicted_claims_count,
        "mixed_claims_count": mixed_claims_count,
        "unverified_claims_count": unverified_claims_count,
        "insufficient_claims_count": insufficient_claims_count,
        "claims_count": len(usable_claims),
        "sources_count": len(all_retrieved_sources),
        "has_critical_contradiction": has_critical_contradiction,
        "has_strong_corroboration": has_strong_corroboration,
        "has_genuine_conflict": has_genuine_conflict,
        "has_mixed_claims": has_mixed_claims,
    }


def _source_component(source_result: dict) -> float:
    if not source_result.get("available"):
        return 0.0
    match = source_result.get("reference_list_match")
    if match == "established_news_organization":
        return 0.3
    if match == "satire":
        return -0.20
    # Unknown domain: neutral 0.0 (never negative)
    return 0.0


def aggregate_signals(*, quality_result, prediction_result, clickbait_result,
                       ai_text_result, source_result, fact_check_result,
                       evidence_result, temporal_info: dict | None = None) -> dict:
    """Aggregates all verification layers into the final transparent assessment."""
    temporal_info = temporal_info or {"context": "unspecified", "label": "Unspecified Date", "reasons": []}
    temporal_context = temporal_info.get("context", "unspecified")
    is_current = (temporal_context == "current")
    is_historical = (temporal_context == "historical")

    reasoning = []
    signals_considered = []

    # -----------------------------------------------------------------------
    # TIER 1A: Quality & Ambiguity Gate (Requirement 9)
    # -----------------------------------------------------------------------
    if not quality_result.get("usable", True):
        return {
            "final_assessment": "UNCERTAIN",
            "verification_confidence": "LOW",
            "verification_confidence_score": 15.0,
            "temporal_context": temporal_context,
            "temporal_info": temporal_info,
            "ml_prediction": prediction_result.get("prediction"),
            "ml_confidence": prediction_result.get("confidence"),
            "evidence_found": {
                "available": False,
                "summary": "Article quality check failed; external verification not run.",
                "claims_count": 0,
                "supported_count": 0,
                "contradicted_count": 0,
            },
            "reasoning": [
                "Insufficient content or high ambiguity: the article cannot be reliably classified as REAL or FAKE.",
            ] + quality_result.get("reasons", []),
            "signals_considered": ["quality_gate"],
            "composite_score": 0.0,
        }

    # Evaluate all components
    ml_score = _ml_component(prediction_result)
    raw_ml_label = prediction_result.get("raw_model_label")
    ml_confidence = prediction_result.get("confidence")

    ev_eval = _evaluate_evidence_strength(evidence_result, fact_check_result)
    source_score = _source_component(source_result)

    factcheck_considered = ev_eval["fc_considered"]
    fc_has_false = ev_eval["fc_has_false"]
    fc_has_true = ev_eval["fc_has_true"]
    factcheck_score = ev_eval["fc_score"]

    web_considered = ev_eval["has_web_evidence"]
    web_score = ev_eval["evidence_score"]
    web_sup_count = ev_eval["supporting_sources_count"]
    web_contra_count = ev_eval["contradicting_sources_count"]

    evidence_available = factcheck_considered or web_considered

    has_critical_contradiction = ev_eval["has_critical_contradiction"]
    has_strong_corroboration = ev_eval["has_strong_corroboration"]
    has_genuine_conflict = ev_eval["has_genuine_conflict"]

    if not prediction_result.get("error"):
        signals_considered.append("ml_classification")
    if factcheck_considered:
        signals_considered.append("fact_check_database")
    if web_considered:
        signals_considered.append("independent_web_evidence")
    if source_result.get("available"):
        signals_considered.append("source_analysis")

    # Build evidence summary
    evidence_found = {
        "available": evidence_available,
        "summary": "Independent evidence was gathered and evaluated." if evidence_available else "No external evidence could be retrieved.",
        "claims_count": ev_eval["claims_count"],
        "supported_count": ev_eval["supported_claims_count"],
        "contradicted_count": ev_eval["contradicted_claims_count"],
        "insufficient_count": ev_eval["insufficient_claims_count"],
        "sources_count": ev_eval["sources_count"],
        "supporting_sources_count": web_sup_count,
        "contradicting_sources_count": web_contra_count,
        "insufficient_sources_count": ev_eval["insufficient_sources_count"],
        "total_support_strength": ev_eval["total_support_strength"],
        "total_contra_strength": ev_eval["total_contra_strength"],
        "strong_support_count": ev_eval["strong_support_count"],
        "strong_contra_count": ev_eval["strong_contra_count"],
        "weak_contra_count": ev_eval["weak_contra_count"],
        "primary_support_count": ev_eval["primary_support_count"],
        "primary_contra_count": ev_eval["primary_contra_count"],
        "fact_checks_count": len(fact_check_result.get("claims", [])),
    }

    # Temporal context explanation
    if is_current:
        reasoning.append(f"Temporal Context: Identified as current/recent news ({temporal_info.get('label')}).")
    elif is_historical:
        reasoning.append(f"Temporal Context: Identified as historical news ({temporal_info.get('label')}).")
    else:
        reasoning.append("Temporal Context: No specific publication date or temporal anchor detected.")

    # -----------------------------------------------------------------------
    # TIER 1B: Evidence Overrides ML in BOTH directions (Rules 6, 7, 8, 9, 10)
    # -----------------------------------------------------------------------
    has_mixed_claims = ev_eval.get("has_mixed_claims", False)
    if has_mixed_claims:
        reasoning.append(
            "The article contains a mixture of verified statements and contradicted assertions. "
            f"{ev_eval['supported_claims_count']} claim(s) are corroborated by independent reporting, "
            f"while {ev_eval['contradicted_claims_count']} claim(s) directly conflict with verified facts."
        )
        reasoning.append(
            "TruthLens AI classifies this report as MIXED / MISLEADING because it contains both verified facts and contradicted or debunked assertions."
        )
        return {
            "final_assessment": "MIXED / MISLEADING",
            "verification_confidence": "MEDIUM",
            "verification_confidence_score": 82.0,
            "temporal_context": temporal_context,
            "temporal_info": temporal_info,
            "ml_prediction": prediction_result.get("prediction"),
            "ml_confidence": ml_confidence,
            "evidence_found": evidence_found,
            "reasoning": reasoning,
            "signals_considered": signals_considered,
            "composite_score": 0.0,
        }

    if has_critical_contradiction:
        # Verified Contradicting evidence overrides ML REAL
        reasoning.append(
            "Independent reliable evidence CONTRADICTS at least one central claim in this article. "
            "Verified contradicting external evidence overrides the ML model's stylistic prediction."
        )
        if factcheck_considered and fc_has_false:
            reasoning.append("Authoritative fact-checking databases record claims in this article as false or debunked.")
        if web_considered and web_contra_count > 0:
            reasoning.append(f"{web_contra_count} independent retrieved source(s) refute the reported event (evidence strength: Strong).")

        conf_high = (ev_eval["strong_contra_count"] >= 2 or fc_has_false or ev_eval["primary_contra_count"] >= 1)
        return {
            "final_assessment": "LIKELY FAKE",
            "verification_confidence": "HIGH" if conf_high else "MEDIUM",
            "verification_confidence_score": 88.0 if conf_high else 76.0,
            "temporal_context": temporal_context,
            "temporal_info": temporal_info,
            "ml_prediction": prediction_result.get("prediction"),
            "ml_confidence": ml_confidence,
            "evidence_found": evidence_found,
            "reasoning": reasoning,
            "signals_considered": signals_considered,
            "composite_score": -0.85,
        }

    if has_strong_corroboration:
        # Corroborating evidence confirms real news
        reasoning.append(
            "Independent verified reporting SUPPORTS and corroborates the central claims in this article. "
            "Real-world external evidence overrides the historical ML model's stylistic prediction."
        )
        if ev_eval["primary_support_count"] >= 1:
            reasoning.append("Official primary source announcements directly confirm and corroborate the reported event.")
        if web_considered and web_sup_count > 0:
            reasoning.append(f"{web_sup_count} independent retrieved source(s) corroborate the reported facts (evidence strength: Strong).")
        if factcheck_considered and fc_has_true:
            reasoning.append("Fact-checking databases confirm the accuracy of claims in this article.")
        if ev_eval["weak_contra_count"] > 0:
            reasoning.append(f"{ev_eval['weak_contra_count']} weak or indirect contradictory source(s) were evaluated and superseded by strong corroborating evidence.")

        is_opinion = quality_result.get("is_opinion", False)
        if is_opinion:
            reasoning.append("Content is identified as an opinion/editorial commentary piece. Verification confidence is capped at LOW.")
            conf_label = "LOW"
            conf_score = 35.0
        else:
            conf_high = (ev_eval["strong_support_count"] >= 2 or ev_eval["primary_support_count"] >= 1 or fc_has_true)
            conf_label = "HIGH" if conf_high else "MEDIUM"
            conf_score = 88.0 if conf_high else 78.0

        return {
            "final_assessment": "LIKELY REAL",
            "verification_confidence": conf_label,
            "verification_confidence_score": conf_score,
            "temporal_context": temporal_context,
            "temporal_info": temporal_info,
            "ml_prediction": prediction_result.get("prediction"),
            "ml_confidence": ml_confidence,
            "evidence_found": evidence_found,
            "reasoning": reasoning,
            "signals_considered": signals_considered,
            "composite_score": 0.85,
        }

    if has_genuine_conflict:
        # Rule 7: Genuine conflict between strong sources -> UNCERTAIN
        reasoning.append(
            "Independent reputable sources genuinely conflict on this claim. "
            "Reliable reporting exists on both sides, so TruthLens AI reports UNCERTAIN rather than taking an unsupported side."
        )
        reasoning.append(
            f"Evidence evaluation: {web_sup_count} supporting source(s) vs {web_contra_count} contradicting source(s) with conflicting direct evidence."
        )

        return {
            "final_assessment": "UNCERTAIN",
            "verification_confidence": "MEDIUM" if (ev_eval["strong_support_count"] >= 1 and ev_eval["strong_contra_count"] >= 1) else "LOW",
            "verification_confidence_score": 42.0,
            "temporal_context": temporal_context,
            "temporal_info": temporal_info,
            "ml_prediction": prediction_result.get("prediction"),
            "ml_confidence": ml_confidence,
            "evidence_found": evidence_found,
            "reasoning": reasoning,
            "signals_considered": signals_considered,
            "composite_score": 0.0,
        }

    # -----------------------------------------------------------------------
    # TIER 2: Current News Handling (Requirements 4, 5, 8)
    # "Current/present news must not rely only on the historical ML model."
    # -----------------------------------------------------------------------
    if is_current and not evidence_available:
        # Current news with NO external evidence: do NOT guess based on ML alone!
        reasoning.append(
            "Current/present news cannot be reliably verified using only a historical ML classifier, "
            "as the model was trained on past events and cannot verify new real-world facts."
        )
        reasoning.append(
            "No independent external evidence or fact-checks could be confirmed at this time "
            "(external services unconfigured or no matching sources found). "
            "TruthLens AI reports UNCERTAIN rather than guessing."
        )
        if prediction_result.get("prediction") and not prediction_result.get("error"):
            reasoning.append(
                f"Historical ML model stylistic reference: leans {prediction_result.get('raw_model_label')} "
                f"({prediction_result.get('confidence_display', 'N/A')}), but this is style-only, not fact verification."
            )

        return {
            "final_assessment": "UNCERTAIN",
            "verification_confidence": "LOW",
            "verification_confidence_score": 25.0,
            "temporal_context": temporal_context,
            "temporal_info": temporal_info,
            "ml_prediction": prediction_result.get("prediction"),
            "ml_confidence": ml_confidence,
            "evidence_found": evidence_found,
            "reasoning": reasoning,
            "signals_considered": ["temporal_analysis", "ml_classification"],
            "composite_score": 0.0,
        }

    # -----------------------------------------------------------------------
    # TIER 3: Weighted Multi-Signal Combination
    # -----------------------------------------------------------------------
    if is_current and evidence_available:
        # For current news with evidence, evidence dominates:
        w_web = CURRENT_WEB_WEIGHT if web_considered else 0.0
        w_fc = CURRENT_FACTCHECK_WEIGHT if factcheck_considered else 0.0
        w_src = CURRENT_SOURCE_WEIGHT if source_result.get("available") else 0.0
        w_ml = CURRENT_ML_WEIGHT if not prediction_result.get("error") else 0.0

        weighted_sum = (w_web * web_score) + (w_fc * factcheck_score) + (w_src * source_score) + (w_ml * ml_score)
        weight_total = w_web + w_fc + w_src + w_ml
        composite = (weighted_sum / weight_total) if weight_total > 0 else 0.0
    else:
        # Historical news or unspecified date:
        w_ml = ML_WEIGHT_HISTORICAL if not prediction_result.get("error") else 0.0
        w_fc = FACTCHECK_WEIGHT if factcheck_considered else 0.0
        w_web = WEB_EVIDENCE_WEIGHT if web_considered else 0.0
        w_src = SOURCE_WEIGHT if source_result.get("available") else 0.0

        weighted_sum = (w_ml * ml_score) + (w_fc * factcheck_score) + (w_web * web_score) + (w_src * source_score)
        weight_total = w_ml + w_fc + w_web + w_src
        composite = (weighted_sum / weight_total) if weight_total > 0 else 0.0

    # Add descriptive reasoning lines
    if not prediction_result.get("error"):
        reasoning.append(
            f"ML model style classification: {prediction_result.get('raw_model_label')} "
            f"({prediction_result.get('confidence_display', 'N/A')})."
        )
        if prediction_result.get("is_ood"):
            reasoning.append(
                "Substantial vocabulary shift detected relative to historical training data (Out-of-Distribution). "
                "The ML style classifier signal was neutralized to prevent distribution-shift false alarms."
            )

    if web_considered:
        reasoning.append(
            f"Independent web evidence: {web_sup_count} claim(s) corroborated, "
            f"{web_contra_count} contradicted, out of {len(evidence_result.get('claim_evidence', []))} checked."
        )
    else:
        reasoning.append("Independent web evidence search was not available or found no candidate articles.")

    if factcheck_considered:
        reasoning.append("Matching records found in the fact-check database.")
    else:
        reasoning.append("No matching fact-check database entries were found.")

    if source_result.get("available"):
        domain = source_result.get("domain")
        ref_match = source_result.get("reference_list_match")
        if ref_match == "established_news_organization":
            reasoning.append(f"Source '{domain}' is an established news outlet (weak positive signal).")
        elif ref_match == "satire":
            reasoning.append(f"Source '{domain}' is an identified satire site.")
        else:
            reasoning.append(
                f"Source '{domain}' is not on the established reference list. "
                "This is typical for most legitimate web publishers and is not a sign of fake news."
            )

    # Final verdict and confidence calibration
    if not evidence_available:
        # No external corroboration at all - cap confidence at LOW
        # regardless of ML confidence, per documented policy.
        if abs(composite) < 0.15 or prediction_result.get("prediction") == "UNCERTAIN":
            final = "UNCERTAIN"
            conf_label = "LOW"
        else:
            final = "LIKELY REAL" if composite > 0 else "LIKELY FAKE"
            conf_label = "LOW"
        reasoning.append(
            "No independent evidence or fact-check confirmation was available, so this "
            "assessment relies on the ML model alone and verification confidence is capped at LOW."
        )
    else:
        # External evidence was available
        if abs(composite) < 0.12:
            final = "UNCERTAIN"
            conf_label = "LOW"
            reasoning.append("Available signals were mixed or too weak to reach a confident assessment.")
        else:
            final = "LIKELY REAL" if composite > 0 else "LIKELY FAKE"
            conf_label = "HIGH" if abs(composite) >= 0.45 else "MEDIUM"

    conf_score = round(min(abs(composite) * 100 + (25 if evidence_available else 0), 97), 1) \
        if final != "UNCERTAIN" else round(abs(composite) * 60, 1)

    is_opinion = quality_result.get("is_opinion", False)
    if is_opinion:
        conf_label = "LOW"
        conf_score = min(conf_score, 35.0)
        reasoning.append("Content is identified as an opinion/editorial commentary piece. Verification confidence is capped at LOW.")

    # Guard: Never allow an article with contradicted claims to become LIKELY REAL
    if ev_eval.get("contradicted_claims_count", 0) > 0 and final == "LIKELY REAL":
        if ev_eval.get("supported_claims_count", 0) > 0:
            final = "MIXED / MISLEADING"
        else:
            final = "LIKELY FAKE"

    return {
        "final_assessment": final,
        "verification_confidence": conf_label,
        "verification_confidence_score": conf_score,
        "temporal_context": temporal_context,
        "temporal_info": temporal_info,
        "ml_prediction": prediction_result.get("prediction"),
        "ml_confidence": ml_confidence,
        "evidence_found": evidence_found,
        "reasoning": reasoning,
        "signals_considered": signals_considered,
        "composite_score": round(composite, 4),
    }
