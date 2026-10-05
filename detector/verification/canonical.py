import logging
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Set

from .claims import (
    normalize_claim_text,
    assign_claim_importance,
    parse_structured_claim,
    classify_claim_modality,
    normalize_claim_status,
)

logger = logging.getLogger("truthlens.verification.canonical")

CANONICAL_CLAIM_STATUSES = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT"}


def canonicalize_url(raw_url: str) -> str:
    """
    Normalizes and canonicalizes URLs to prevent duplicate counting:
    - Lowercases scheme and netloc (strips 'www.')
    - Removes tracking query parameters (utm_*, ref, etc.)
    - Removes fragments and trailing slashes
    """
    if not raw_url or not isinstance(raw_url, str):
        return ""
    raw_url = raw_url.strip()
    try:
        parsed = urllib.parse.urlsplit(raw_url)
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/")
        # Filter tracking parameters
        tracking_keys = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "ref", "source", "fbclid", "gclid", "msclkid", "ocid"
        }
        query_pairs = urllib.parse.parse_qsl(parsed.query)
        filtered_query = [(k, v) for k, v in query_pairs if k.lower() not in tracking_keys]
        query_str = urllib.parse.urlencode(filtered_query)
        clean = urllib.parse.urlunsplit((scheme, netloc, path, query_str, ""))
        return clean.rstrip("/")
    except Exception:
        return raw_url.lower().rstrip("/")


def validate_analysis_consistency(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evidence Consistency Validator (Section 23).
    Enforces all 10 consistency checks across backend data, claims, sources, and UI summaries:
    1. Sum of claim statuses == total claims
    2. Supporting claim count matches actual claims
    3. Contradicting claim count matches actual claims
    4. Insufficient count matches actual claims
    5. Every evidence item references a valid claim
    6. No source is counted twice as independent evidence
    7. Final verdict is compatible with claim statuses
    8. Confidence is compatible with evidence completeness
    9. Historical ML does not override final evidence-based verdict
    10. UI summary values equal backend canonical values
    """
    errors = []
    claims = result.get("claims") or result.get("processed_claims") or []
    claim_counts = result.get("claim_counts") or {}
    source_counts = result.get("source_counts") or {}
    summary = result.get("summary") or {}
    final_assessment = result.get("final_assessment") or {}
    ml_signal = result.get("ml_signal") or {}

    total_claims = len(claims)
    sup_cnt = sum(1 for c in claims if c.get("status") == "SUPPORTED")
    contra_cnt = sum(1 for c in claims if c.get("status") == "CONTRADICTED")
    insuf_cnt = sum(1 for c in claims if c.get("status") == "INSUFFICIENT")

    # Check 1: Sum of claim statuses == total claims
    if (sup_cnt + contra_cnt + insuf_cnt) != total_claims:
        errors.append(f"Claim status sum ({sup_cnt + contra_cnt + insuf_cnt}) != total claims ({total_claims})")

    # Checks 2-4: Claim counts match actual claims
    if claim_counts.get("supported") != sup_cnt:
        errors.append(f"Claim count supported ({claim_counts.get('supported')}) != actual ({sup_cnt})")
    if claim_counts.get("contradicted") != contra_cnt:
        errors.append(f"Claim count contradicted ({claim_counts.get('contradicted')}) != actual ({contra_cnt})")
    if claim_counts.get("insufficient") != insuf_cnt:
        errors.append(f"Claim count insufficient ({claim_counts.get('insufficient')}) != actual ({insuf_cnt})")

    # Check 5: Every evidence item references a valid claim
    claim_ids = {c.get("id") or c.get("claim_id") for c in claims if c.get("id") or c.get("claim_id")}
    for c in claims:
        for ev in c.get("evidence", []):
            if ev.get("claim_id") and ev.get("claim_id") not in claim_ids:
                errors.append(f"Evidence item references unknown claim_id: {ev.get('claim_id')}")

    # Check 6: No duplicate source counted twice as independent evidence
    unique_sources = result.get("sources") or result.get("all_sources") or []
    seen_ids = set()
    for s in unique_sources:
        sid = s.get("source_id") or s.get("canonical_url") or s.get("url")
        if sid:
            if sid in seen_ids:
                errors.append(f"Duplicate independent source found in unique list: {sid}")
            seen_ids.add(sid)

    # Check 7: Final verdict compatibility
    verdict = final_assessment.get("verdict") or final_assessment.get("label") or summary.get("final_verdict") or ""
    if contra_cnt >= 1 and sup_cnt == 0 and "REAL" in verdict:
        errors.append(f"Verdict {verdict} contradicts claim statuses (0 supported, {contra_cnt} contradicted)")
    if contra_cnt >= 1 and sup_cnt >= 1 and verdict == "LIKELY REAL":
        errors.append(f"Verdict {verdict} cannot be LIKELY REAL when contradictions exist ({sup_cnt} sup, {contra_cnt} contra)")
    if total_claims > 0 and insuf_cnt == total_claims and ("REAL" in verdict or "FAKE" in verdict):
        errors.append(f"Verdict {verdict} cannot be conclusive when all claims are INSUFFICIENT")

    # Check 8: Confidence compatibility
    conf_score = final_assessment.get("confidence_score") or (final_assessment.get("confidence", 0) * 100) or summary.get("confidence", 0)
    conf_label = final_assessment.get("confidence_label") or summary.get("confidence_level") or ""
    if verdict == "UNCERTAIN" and (conf_label == "HIGH" or conf_score > 55.0):
        errors.append(f"Confidence cannot be HIGH for UNCERTAIN verdict (got {conf_label}, {conf_score})")
    if total_claims > 0 and insuf_cnt == total_claims and conf_label == "HIGH":
        errors.append(f"Confidence cannot be HIGH when evidence is insufficient for all claims")

    # Check 9: Historical ML does not override evidence verdict
    ml_pred = ml_signal.get("status") or (result.get("historical_ml") or {}).get("prediction") or ""
    if ml_pred == "FAKE" and sup_cnt >= 1 and contra_cnt == 0 and verdict != "LIKELY REAL":
        errors.append("Historical ML FAKE signal improperly overrode strong factual corroboration")
    if ml_pred == "REAL" and contra_cnt >= 1 and sup_cnt == 0 and verdict != "LIKELY FAKE":
        errors.append("Historical ML REAL signal improperly overrode factual contradiction")

    # Check 10: UI summary values equal canonical values
    ev_summary = result.get("evidence_summary") or {}
    if ev_summary.get("supporting") != sup_cnt:
        errors.append(f"Evidence summary supporting ({ev_summary.get('supporting')}) != claim count ({sup_cnt})")
    if ev_summary.get("contradicting") != contra_cnt:
        errors.append(f"Evidence summary contradicting ({ev_summary.get('contradicting')}) != claim count ({contra_cnt})")

    if errors:
        logger.warning("CONSISTENCY VALIDATION ERRORS: %s", errors)
        return {
            "passed": False,
            "errors": errors,
            "checks_run": 10,
        }

    return {
        "passed": True,
        "errors": [],
        "checks_run": 10,
    }


def aggregate_verification(claims: list, evidence: dict | None = None) -> Dict[str, Any]:
    """
    Convenience alias for build_canonical_verification_result.
    Satisfies Section 16 requirements.
    """
    return build_canonical_verification_result(claims_list=claims, evidence_data=evidence)


def build_canonical_verification_result(
    analysis=None,
    *,
    claims_list: Optional[List[Any]] = None,
    evidence_data: Optional[Dict[str, Any]] = None,
    fact_checks: Optional[Any] = None,
    prediction_result: Optional[Dict[str, Any]] = None,
    input_type: str = "Pasted Text",
    temporal_context: Optional[str] = None,
    temporal_detected_year: Optional[int] = None,
    stored_final_assessment: Optional[str] = None,
    stored_verification_confidence: Optional[str] = None,
    stored_confidence_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Creates ONE authoritative canonical verification result object for every analysis.
    Satisfies TruthLens AI Single Source of Truth architecture:
    - Claim counts are exact and derived directly from verified claims.
    - Source counts are calculated independently from unique, deduplicated sources.
    - Contradicted claims and contradicting evidence are accurately represented.
    - Evaluates to MIXED / MISLEADING when both supported and contradicted claims exist.
    - Separates Historical ML signal from real-world claim verification.
    - Dynamically generates timeline, why points, and key takeaways.
    - Enforces automated consistency validations.
    """
    # 1. Unpack from Analysis model if provided
    is_opinion = False
    if analysis is not None:
        claims_list = claims_list if claims_list is not None else getattr(analysis, "claims", [])
        evidence_data = evidence_data if evidence_data is not None else getattr(analysis, "evidence", {})
        if fact_checks is None and hasattr(analysis, "fact_checks"):
            try:
                fact_checks = list(analysis.fact_checks.all())
            except Exception:
                fact_checks = []
        if hasattr(analysis, "get_input_type_display"):
            input_type = analysis.get_input_type_display()
        temporal_context = temporal_context or getattr(analysis, "temporal_context", "unspecified")
        temporal_detected_year = temporal_detected_year or getattr(analysis, "temporal_detected_year", None)
        stored_final_assessment = stored_final_assessment or getattr(analysis, "final_assessment", None)
        stored_verification_confidence = stored_verification_confidence or getattr(analysis, "verification_confidence", None)
        stored_confidence_score = stored_confidence_score if stored_confidence_score is not None else getattr(analysis, "verification_confidence_score", None)

        raw_article_text = getattr(analysis, "extracted_text", "") or ""
        if raw_article_text:
            from .quality import check_article_quality
            q_res = check_article_quality(raw_article_text)
            is_opinion = q_res.get("is_opinion", False)

        if prediction_result is None:
            prediction_result = {
                "prediction": getattr(analysis, "prediction", None) or "UNCERTAIN",
                "confidence": getattr(analysis, "confidence", None) if getattr(analysis, "confidence", None) is not None else 50.0,
                "confidence_status": getattr(analysis, "confidence_status", None) or "LOW",
                "raw_model_label": getattr(analysis, "prediction", None) or "UNCERTAIN",
            }

    claims_list = claims_list or []
    evidence_data = evidence_data or {}
    claim_evidence_list = evidence_data.get("claim_evidence", []) if isinstance(evidence_data, dict) else []

    # Map evidence items by lowercase claim string, cleaned text, and index
    ev_by_claim = {}
    ev_by_clean = {}
    ev_by_index = {}
    for idx, ev in enumerate(claim_evidence_list):
        if not isinstance(ev, dict):
            continue
        c_text = (ev.get("claim") or "").strip()
        if c_text:
            ev_by_claim[c_text.lower()] = ev
            c_clean = re.sub(r"[^\w\s]", "", c_text.lower()).strip()
            if c_clean:
                ev_by_clean[c_clean] = ev
        ev_by_index[idx] = ev

    # Extract source claim statements
    raw_source_claims = []
    if claims_list:
        raw_source_claims = [c.strip() if isinstance(c, str) else str(c.get("claim", "")).strip() for c in claims_list]
    elif claim_evidence_list:
        raw_source_claims = [ev.get("claim", "").strip() for ev in claim_evidence_list if ev.get("claim")]

    # Deduplicate while preserving order
    seen_claims = set()
    source_claims = []
    for c in raw_source_claims:
        if c and c.lower() not in seen_claims:
            seen_claims.add(c.lower())
            source_claims.append(c)

    processed_claims = []
    unique_sources_map: Dict[str, Dict[str, Any]] = {}
    supporting_source_ids: Set[str] = set()
    contradicting_source_ids: Set[str] = set()
    neutral_source_ids: Set[str] = set()
    unrelated_source_ids: Set[str] = set()
    has_verified_factcheck = False

    evidence_counts = {
        "supporting": 0,
        "contradicting": 0,
        "insufficient": 0,
        "unrelated": 0,
    }

    # Match external fact-checks to claims if available
    factcheck_by_claim = {}
    if fact_checks:
        for fc in fact_checks:
            fc_text = getattr(fc, "claim", "") or ""
            if fc_text:
                factcheck_by_claim[fc_text.lower()] = fc

    # Process each claim
    for idx, c_str in enumerate(source_claims, start=1):
        clean_key = re.sub(r"[^\w\s]", "", c_str.lower()).strip()
        ev = ev_by_claim.get(c_str.lower()) or ev_by_clean.get(clean_key) or ev_by_index.get(idx - 1, {})

        # Fact check matching
        matching_fc = factcheck_by_claim.get(c_str.lower()) or factcheck_by_claim.get(clean_key)
        raw_status = None
        if matching_fc:
            fc_stat = (getattr(matching_fc, "status", "") or "").upper()
            if any(k in fc_stat for k in ("FALSE", "FAKE", "PANTS", "INCORRECT", "DEBUNKED", "CONTRADICT")):
                raw_status = "CONTRADICTED"
                has_verified_factcheck = True
            elif any(k in fc_stat for k in ("TRUE", "CORRECT", "VERIFIED", "ACCURATE", "SUPPORT")):
                raw_status = "SUPPORTED"
                has_verified_factcheck = True

        if not raw_status and isinstance(ev, dict):
            raw_status = ev.get("status")

        canonical_status = normalize_claim_status(raw_status)

        summary = ev.get("summary") if isinstance(ev, dict) and ev.get("summary") else (
            f"Fact-check database records this claim as {raw_status}." if matching_fc else "No independent verification summary available for this claim."
        )
        raw_sources = ev.get("sources", []) if isinstance(ev, dict) else []

        claim_id = f"claim_{idx}"
        claim_sources = []

        for s in raw_sources:
            if not isinstance(s, dict):
                continue
            url = s.get("url") or ""
            domain = s.get("domain") or (url.split("//")[-1].split("/")[0] if url else "Unknown Source")
            publisher = s.get("publisher") or s.get("title") or domain
            published_date = s.get("published_date")

            verdict_raw = (s.get("content_verdict") or s.get("verdict") or s.get("relationship") or "").upper()

            if verdict_raw in ("SUPPORTS", "SUPPORTED", "TRUE", "CORROBORATED"):
                verdict = "SUPPORTED"
                rel = "SUPPORTS"
                verdict_class = "status-supported"
                evidence_counts["supporting"] += 1
            elif verdict_raw in ("CONTRADICTS", "CONTRADICTED", "FALSE", "REFUTED", "DEBUNKED"):
                verdict = "CONTRADICTED"
                rel = "CONTRADICTS"
                verdict_class = "status-contradicted"
                evidence_counts["contradicting"] += 1
            elif verdict_raw == "UNRELATED":
                verdict = "UNRELATED"
                rel = "UNRELATED"
                verdict_class = "badge-neutral"
                evidence_counts["unrelated"] += 1
            else:
                verdict = "INSUFFICIENT"
                rel = "NEUTRAL"
                verdict_class = "status-insufficient"
                evidence_counts["insufficient"] += 1

            is_primary = bool(s.get("is_primary_source"))
            source_rel = s.get("source_reliability") or ("VERY_HIGH" if is_primary else "LOW")
            source_rel_label = s.get("source_reliability_label") or ("Very High (Primary Source)" if is_primary else "General Web")
            strength_label = s.get("strength_label") or ("STRONG" if is_primary else ("MODERATE" if verdict in ("SUPPORTED", "CONTRADICTED") else "WEAK"))
            strength_score = s.get("strength_score", 1.0 if is_primary else 0.5)
            reason = s.get("reason") or "Retrieved for claim verification."
            directness = s.get("directness") or ("DIRECT" if is_primary else "INDIRECT")

            raw_excerpt = (s.get("evidence_text") or s.get("text_excerpt") or s.get("excerpt") or reason).strip()
            excerpt = raw_excerpt[:250] + "..." if len(raw_excerpt) > 250 else raw_excerpt

            is_syndicated = bool(s.get("syndicated_of"))

            src_obj = {
                "claim_id": claim_id,
                "claim_addressed": c_str,
                "url": url,
                "source_url": url,
                "domain": domain,
                "publisher": publisher,
                "source_title": publisher,
                "published_date": published_date,
                "verdict": verdict,
                "relationship": rel,
                "verdict_class": verdict_class,
                "is_primary": is_primary,
                "is_syndicated": is_syndicated,
                "syndicated_of": s.get("syndicated_of"),
                "source_reliability": source_rel,
                "source_reliability_label": source_rel_label,
                "strength_label": strength_label,
                "strength_score": strength_score,
                "evidence_strength": strength_label,
                "source_quality": round(float(strength_score) / 2.0, 2),
                "relevance_score": 1.0 if rel in ("SUPPORTS", "CONTRADICTS") else (0.5 if rel == "NEUTRAL" else 0.0),
                "directness": directness,
                "reason": reason,
                "excerpt": excerpt,
                "evidence_text": raw_excerpt,
                "retrieved": s.get("retrieved", False),
                "validation_error": s.get("validation_error"),
            }
            claim_sources.append(src_obj)

            # Unique source tracking via canonicalized identifier
            clean_url = canonicalize_url(url)
            source_id = clean_url if clean_url else f"{domain.lower()}:{publisher.lower()}"

            # Only independent non-syndicated sources count toward independent corroboration (Section 7)
            if not is_syndicated and s.get("retrieved") is not False:
                if verdict == "SUPPORTED":
                    supporting_source_ids.add(source_id)
                elif verdict == "CONTRADICTED":
                    contradicting_source_ids.add(source_id)
                elif verdict == "UNRELATED":
                    unrelated_source_ids.add(source_id)
                else:
                    neutral_source_ids.add(source_id)

                if source_id not in unique_sources_map:
                    unique_sources_map[source_id] = {
                        "source_id": source_id,
                        "url": url,
                        "canonical_url": clean_url,
                        "domain": domain,
                        "publisher": publisher,
                        "published_date": published_date,
                        "is_primary": is_primary,
                        "source_reliability": source_rel,
                        "source_reliability_label": source_rel_label,
                        "strength_label": strength_label,
                        "strength_score": strength_score,
                        "directness": directness,
                        "reason": reason,
                        "excerpt": excerpt,
                        "claims_addressed": [c_str],
                        "verdicts": {verdict},
                        "retrieved": s.get("retrieved", False),
                        "validation_error": s.get("validation_error"),
                    }
                else:
                    entry = unique_sources_map[source_id]
                    if c_str not in entry["claims_addressed"]:
                        entry["claims_addressed"].append(c_str)
                    entry["verdicts"].add(verdict)
                    if is_primary:
                        entry["is_primary"] = True
                        entry["source_reliability"] = source_rel
                        entry["source_reliability_label"] = source_rel_label
                    if strength_score > entry.get("strength_score", 0):
                        entry["strength_score"] = strength_score
                        entry["strength_label"] = strength_label

        if canonical_status == "SUPPORTED":
            badge_class = "status-supported"
            status_icon = "✓"
        elif canonical_status == "CONTRADICTED":
            badge_class = "status-contradicted"
            status_icon = "✕"
        else:
            badge_class = "status-insufficient"
            status_icon = "?"

        norm_text = normalize_claim_text(c_str)
        importance = assign_claim_importance(c_str, idx - 1, len(source_claims))
        structured = parse_structured_claim(c_str, idx, len(source_claims))

        sup_in_claim = sum(1 for s in claim_sources if s["verdict"] == "SUPPORTED" and not s.get("is_syndicated"))
        contra_in_claim = sum(1 for s in claim_sources if s["verdict"] == "CONTRADICTED" and not s.get("is_syndicated"))

        # Claim confidence is grounded in evidence
        if canonical_status == "SUPPORTED":
            if sup_in_claim >= 2 or any(s.get("is_primary") for s in claim_sources):
                c_conf = 0.94
            elif sup_in_claim == 1:
                c_conf = 0.82
            else:
                c_conf = 0.70
        elif canonical_status == "CONTRADICTED":
            if contra_in_claim >= 2 or any(s.get("is_primary") for s in claim_sources) or has_verified_factcheck:
                c_conf = 0.92
            else:
                c_conf = 0.80
        else:
            c_conf = 0.35

        if isinstance(ev, dict) and ev.get("confidence") is not None:
            c_conf = float(ev.get("confidence"))

        c_reason = ev.get("reason") if isinstance(ev, dict) and ev.get("reason") else summary

        claim_evidence_items = []
        for s in claim_sources:
            claim_evidence_items.append({
                "claim_id": claim_id,
                "source_id": s.get("url") or s.get("domain", ""),
                "source_url": s.get("url", ""),
                "source_title": s.get("publisher") or s.get("domain", ""),
                "relationship": s["relationship"],
                "relevance_score": s["relevance_score"],
                "source_quality": s["source_quality"],
                "evidence_strength": s["evidence_strength"],
                "evidence_text": s["evidence_text"],
                "strength": float(s.get("strength_score", 0.0)),
                "directness": s.get("directness", "DIRECT"),
                "reason": s.get("reason", ""),
            })

        processed_claims.append({
            "id": claim_id,
            "claim_id": claim_id,
            "index": idx,
            "text": c_str,
            "original_text": c_str,
            "normalized_text": norm_text,
            "normalized_claim": norm_text,
            "subject": structured["subject"],
            "action": structured["action"],
            "object": structured["object"],
            "time": structured["time"],
            "location": structured["location"],
            "quantity": structured["quantity"],
            "attribution": structured["attribution"],
            "modality": structured["modality"],
            "is_checkable": structured["is_checkable"],
            "modality_reason": structured["modality_reason"],
            "importance": importance,
            "claim": c_str,
            "claim_text": c_str,
            "status": canonical_status,
            "confidence": round(c_conf * 100, 1) if c_conf <= 1.0 else round(c_conf, 1),
            "reason": c_reason,
            "raw_status": raw_status,
            "status_icon": status_icon,
            "badge_class": badge_class,
            "summary": summary,
            "evidence": claim_evidence_items,
            "sources": claim_sources,
            "sources_count": len(claim_sources),
            "supporting_sources_count": sup_in_claim,
            "contradicting_sources_count": contra_in_claim,
            "supporting_evidence_count": sup_in_claim,
            "contradicting_evidence_count": contra_in_claim,
        })

    # Integrate verified fact-check records into unique sources
    if fact_checks:
        for fc in fact_checks:
            url = getattr(fc, "url", "") or ""
            clean_url = canonicalize_url(url)
            fc_status = (getattr(fc, "status", "") or "").upper()
            fc_claim_text = getattr(fc, "claim", "") or ""
            fc_source = getattr(fc, "source", "") or "Fact-Check Archive"

            if any(k in fc_status for k in ("FALSE", "FAKE", "PANTS", "INCORRECT", "DEBUNKED")):
                v = "CONTRADICTED"
                v_class = "status-contradicted"
                has_verified_factcheck = True
                evidence_counts["contradicting"] += 1
            elif any(k in fc_status for k in ("TRUE", "CORRECT")):
                v = "SUPPORTED"
                v_class = "status-supported"
                evidence_counts["supporting"] += 1
            else:
                v = "INSUFFICIENT"
                v_class = "status-insufficient"
                evidence_counts["insufficient"] += 1

            source_id = clean_url if clean_url else f"factcheck:{fc_source.lower()}:{fc_claim_text[:30].lower()}"

            if v == "SUPPORTED":
                supporting_source_ids.add(source_id)
            elif v == "CONTRADICTED":
                contradicting_source_ids.add(source_id)
            else:
                neutral_source_ids.add(source_id)

            if source_id not in unique_sources_map:
                unique_sources_map[source_id] = {
                    "source_id": source_id,
                    "url": url,
                    "canonical_url": clean_url,
                    "domain": fc_source,
                    "publisher": fc_source,
                    "published_date": None,
                    "is_primary": False,
                    "source_reliability": "HIGH",
                    "source_reliability_label": "High (Verified Fact-Checker)",
                    "strength_label": "STRONG",
                    "strength_score": 1.2,
                    "directness": "DIRECT",
                    "reason": f"Fact Check Rating: {fc_status}",
                    "excerpt": f"Fact Check: {fc_claim_text}"[:200],
                    "claims_addressed": [fc_claim_text] if fc_claim_text else [],
                    "verdicts": {v},
                    "retrieved": True,
                    "validation_error": None,
                }

    # Finalize unique sources list and primary verdict per unique source
    all_unique_sources = []
    for s_id, s_data in unique_sources_map.items():
        verdicts = s_data["verdicts"]
        if "CONTRADICTED" in verdicts:
            s_verdict = "CONTRADICTED"
            s_verdict_class = "status-contradicted"
        elif "SUPPORTED" in verdicts:
            s_verdict = "SUPPORTED"
            s_verdict_class = "status-supported"
        elif "INSUFFICIENT" in verdicts:
            s_verdict = "INSUFFICIENT"
            s_verdict_class = "status-insufficient"
        else:
            s_verdict = "UNRELATED"
            s_verdict_class = "badge-neutral"

        s_data["verdict"] = s_verdict
        s_data["verdict_class"] = s_verdict_class
        all_unique_sources.append(s_data)

    # 2. Exact Claim Counts (Derived directly from processed_claims - Section 3)
    supported_cnt = sum(1 for c in processed_claims if c["status"] == "SUPPORTED")
    contradicted_cnt = sum(1 for c in processed_claims if c["status"] == "CONTRADICTED")
    insufficient_cnt = sum(1 for c in processed_claims if c["status"] == "INSUFFICIENT")
    total_cnt = len(processed_claims)

    claim_counts = {
        "total": total_cnt,
        "supported": supported_cnt,
        "contradicted": contradicted_cnt,
        "insufficient": insufficient_cnt,
        "unrelated": 0,
    }

    # 3. Source Counts (Derived directly from deduplicated unique sources - Section 4)
    unique_neutral = len(neutral_source_ids - supporting_source_ids - contradicting_source_ids)
    unique_unrelated = len(unrelated_source_ids - supporting_source_ids - contradicting_source_ids)

    source_counts = {
        "unique": len(unique_sources_map),
        "total": len(unique_sources_map),
        "supporting": len(supporting_source_ids),
        "contradicting": len(contradicting_source_ids),
        "neutral": unique_neutral,
        "insufficient": unique_neutral,
        "unrelated": unique_unrelated,
    }

    # Qualitative evidence chips
    strong_sup_count = sum(
        1 for s in all_unique_sources
        if s.get("verdict") == "SUPPORTED" and (s.get("strength_label") == "STRONG" or s.get("is_primary"))
    )
    strong_contra_count = sum(
        1 for s in all_unique_sources
        if s.get("verdict") == "CONTRADICTED" and (s.get("strength_label") == "STRONG" or s.get("is_primary"))
    )
    primary_support_count = sum(1 for s in all_unique_sources if s.get("verdict") == "SUPPORTED" and s.get("is_primary"))
    primary_contra_count = sum(1 for s in all_unique_sources if s.get("verdict") == "CONTRADICTED" and s.get("is_primary"))

    if supported_cnt >= 1:
        sup_strength = "Strong" if strong_sup_count >= 1 else "Moderate"
        sup_detail = "High confidence" if strong_sup_count >= 1 else "Corroborated reporting"
    else:
        sup_strength = "None"
        sup_detail = "No corroborated claims"

    if contradicted_cnt >= 1:
        contra_strength = "Strong" if (strong_contra_count >= 1 or has_verified_factcheck) else "Moderate / Weak"
        contra_detail = "High confidence refutation" if (strong_contra_count >= 1 or has_verified_factcheck) else "Disputed claims"
    else:
        contra_strength = "None"
        contra_detail = "No refutations"

    evidence_summary = {
        "supporting": supported_cnt,
        "supporting_strength": sup_strength,
        "supporting_detail": sup_detail,
        "contradicting": contradicted_cnt,
        "contradicting_strength": contra_strength,
        "contradicting_detail": contra_detail,
        "insufficient": insufficient_cnt,
        "insufficient_strength": "Neutral",
        "insufficient_detail": "Limited evidence" if insufficient_cnt > 0 else "None",
    }

    # 4. Final Assessment Determination (Deterministic Rules with Claim Importance - Sections 10, 11, 12, 13)
    central_claims = [c for c in processed_claims if c.get("importance") in ("CENTRAL", "IMPORTANT")]
    central_supported = sum(1 for c in central_claims if c["status"] == "SUPPORTED")
    central_contradicted = sum(1 for c in central_claims if c["status"] == "CONTRADICTED")

    if total_cnt == 0 or (supported_cnt == 0 and contradicted_cnt == 0):
        # Case 0: No claims or all claims insufficient
        verdict = "UNCERTAIN"
        conf_label = "LOW"
        conf_score = 30.0
        subtitle = "Available external evidence is insufficient to verify or refute with certainty."
    elif contradicted_cnt >= 1 and supported_cnt == 0:
        # Case 1: All verified claims contradicted
        verdict = "LIKELY FAKE"
        if strong_contra_count >= 2 or has_verified_factcheck or primary_contra_count >= 1:
            conf_label = "HIGH"
            conf_score = 90.0
        elif strong_contra_count >= 1 or contradicted_cnt >= 2:
            conf_label = "HIGH"
            conf_score = 86.0
        else:
            conf_label = "MEDIUM"
            conf_score = 75.0
        subtitle = "Central factual assertions are refuted by independent external evidence."
    elif supported_cnt >= 1 and contradicted_cnt >= 1:
        # Case 2: Mixed claims (both supported and contradicted)
        verdict = "MIXED / MISLEADING"
        conf_label = "MEDIUM"
        conf_score = 80.0
        subtitle = "The article contains both corroborated factual statements and refuted assertions."
    elif supported_cnt >= 1 and contradicted_cnt == 0:
        # Case 3: Supported claims with NO contradictions
        if insufficient_cnt > 0:
            # Some claims supported, some unverified
            if supported_cnt >= insufficient_cnt and central_supported >= 1 and (strong_sup_count >= 1 or source_counts["supporting"] >= 2):
                verdict = "LIKELY REAL"
                conf_label = "MEDIUM"  # Never HIGH when evidence is incomplete!
                conf_score = 62.0
                subtitle = "Key claims are corroborated by external reporting, though some details remain unverified."
            else:
                verdict = "UNCERTAIN"
                conf_label = "LOW"
                conf_score = 42.0
                subtitle = "Available external evidence is incomplete and insufficient to reach a high-confidence assessment."
        else:
            # 100% of claims supported
            verdict = "LIKELY REAL"
            if total_cnt == 1 and strong_sup_count == 0 and primary_support_count == 0:
                conf_label = "LOW"
                conf_score = 44.0
                subtitle = "Single claim supported by limited external reporting."
            elif strong_sup_count >= 2 or primary_support_count >= 1 or has_verified_factcheck:
                conf_label = "HIGH"
                conf_score = 92.0
                subtitle = "Based on current external evidence and corroborated independent reporting."
            elif source_counts["supporting"] >= 2 or strong_sup_count >= 1:
                conf_label = "HIGH"
                conf_score = 86.0
                subtitle = "Based on current external evidence and corroborated independent reporting."
            else:
                conf_label = "MEDIUM"
                conf_score = 74.0
                subtitle = "Based on corroborated independent reporting."
    else:
        verdict = "UNCERTAIN"
        conf_label = "LOW"
        conf_score = 35.0
        subtitle = "Available external evidence is insufficient to verify or refute with certainty."

    # Opinion pieces must never have high verification confidence (Section 15)
    if is_opinion:
        conf_label = "LOW"
        conf_score = min(conf_score, 38.0)

    # 5. Dynamic Why Points (Section 17)
    why_points = []
    if verdict == "MIXED / MISLEADING":
        why_points.append({
            "icon": "✓",
            "icon_type": "success",
            "title": "Supported claims",
            "desc": f"{supported_cnt} of {total_cnt} claims are supported by independent evidence.",
        })
        why_points.append({
            "icon": "✕",
            "icon_type": "danger",
            "title": "Contradicted claims",
            "desc": f"{contradicted_cnt} claim{'s contain' if contradicted_cnt != 1 else ' contains'} evidence that conflicts with the article.",
        })
        why_points.append({
            "icon": "⚠",
            "icon_type": "warn",
            "title": "Evidence conflict",
            "desc": "The article contains both verified and disputed statements.",
        })
    elif verdict == "LIKELY REAL":
        if temporal_context == "current":
            why_points.append({
                "icon": "✓", "icon_type": "success", "title": "Current event",
                "desc": "The article refers to a recent event evaluated against live independent reporting.",
            })
        elif temporal_context == "historical":
            why_points.append({
                "icon": "✓", "icon_type": "success", "title": "Historical event",
                "desc": f"The article documents an event from {temporal_detected_year or 'past records'}, verified against established archives.",
            })
        else:
            why_points.append({
                "icon": "✓", "icon_type": "success", "title": "General reporting",
                "desc": "Analyzed for factual claim veracity across independent sources.",
            })
        why_points.append({
            "icon": "✓", "icon_type": "success", "title": "Independent corroboration",
            "desc": f"{source_counts['supporting']} independent source(s) support verified claims.",
        })
        why_points.append({
            "icon": "✓", "icon_type": "success", "title": "Evidence agreement",
            "desc": "The retrieved evidence supports the central claims without major discrepancies.",
        })
        why_points.append({
            "icon": "✓", "icon_type": "success", "title": "No contradictory evidence",
            "desc": "No material contradictory evidence was identified.",
        })
    elif verdict == "LIKELY FAKE":
        why_points.append({
            "icon": "✕", "icon_type": "danger", "title": "Contradictory evidence",
            "desc": f"{contradicted_cnt} claim{'s contain' if contradicted_cnt != 1 else ' contains'} evidence that conflicts with the article.",
        })
        if has_verified_factcheck:
            why_points.append({
                "icon": "✕", "icon_type": "danger", "title": "Fact-checker refutation",
                "desc": "Verified fact-checking records explicitly flag claims in this story as false or debunked.",
            })
        else:
            why_points.append({
                "icon": "✕", "icon_type": "danger", "title": "Refuted by independent reporting",
                "desc": f"{source_counts['contradicting']} independent source(s) dispute key assertions.",
            })
    else:  # UNCERTAIN
        why_points.append({
            "icon": "?", "icon_type": "warn", "title": "Unverified report",
            "desc": "No reliable independent reporting has corroborated the central claims yet.",
        })
        why_points.append({
            "icon": "?", "icon_type": "warn", "title": "Insufficient evidence",
            "desc": "Available records do not contain enough specific corroboration for the claims.",
        })

    # 6. Dynamic Key Takeaways (Section 18)
    key_takeaways = []
    if verdict == "MIXED / MISLEADING":
        key_takeaways.append({
            "icon": "✓", "type": "real",
            "text": f"{supported_cnt} claim{'s are' if supported_cnt != 1 else ' is'} supported by retrieved evidence.",
        })
        key_takeaways.append({
            "icon": "✕", "type": "fake",
            "text": f"{contradicted_cnt} claim{'s conflict' if contradicted_cnt != 1 else ' conflicts'} with retrieved evidence.",
        })
        key_takeaways.append({
            "icon": "⚠", "type": "warn",
            "text": "The article contains material disputed statements.",
        })
        key_takeaways.append({
            "icon": "ℹ", "type": "info",
            "text": "Do not treat the article as fully verified.",
        })
    elif verdict == "LIKELY REAL":
        key_takeaways.append({
            "icon": "✓", "type": "real",
            "text": "Independent reporting corroborates the central claims.",
        })
        key_takeaways.append({
            "icon": "✓", "type": "real",
            "text": "No material contradictory evidence was identified.",
        })
        if insufficient_cnt > 0:
            key_takeaways.append({
                "icon": "?", "type": "warn",
                "text": f"{insufficient_cnt} detail(s) remain unverified by external records.",
            })
    elif verdict == "LIKELY FAKE":
        key_takeaways.append({
            "icon": "✕", "type": "fake",
            "text": "Core claims contradict verified records or known fact-checks.",
        })
        key_takeaways.append({
            "icon": "✕", "type": "fake",
            "text": "Independent news outlets do not corroborate the reported event.",
        })
        key_takeaways.append({
            "icon": "✕", "type": "fake",
            "text": "Key assertions exhibit significant discrepancy with established facts.",
        })
    else:  # UNCERTAIN
        key_takeaways.append({
            "icon": "?", "type": "warn",
            "text": "Available evidence is insufficient to verify or refute key claims.",
        })
        key_takeaways.append({
            "icon": "?", "type": "warn",
            "text": "No authoritative corroborating news report or fact-check indexed yet.",
        })
        key_takeaways.append({
            "icon": "?", "type": "warn",
            "text": "Independent verification is strongly recommended before sharing.",
        })

    # 7. Separated Historical ML Signal (Section 14)
    prediction_result = prediction_result or {}
    raw_ml_pred = (prediction_result.get("prediction") or "UNCERTAIN").upper()
    ml_conf = prediction_result.get("confidence")
    ml_conf_val = float(ml_conf) if ml_conf is not None else 50.0

    if ml_conf_val < 70.0 or prediction_result.get("confidence_status") == "LOW" or raw_ml_pred == "UNCERTAIN":
        ml_label = "UNCERTAIN"
        ml_badge = "warn"
    elif raw_ml_pred == "REAL":
        ml_label = "LEANS REAL"
        ml_badge = "real"
    elif raw_ml_pred == "FAKE":
        ml_label = "LEANS FAKE"
        ml_badge = "fake"
    else:
        ml_label = "UNCERTAIN"
        ml_badge = "warn"

    ml_signal = {
        "label": ml_label,
        "confidence": round(ml_conf_val / 100.0, 2),
        "status": raw_ml_pred,
        "score": ml_conf_val,
        "score_display": f"{ml_conf_val:.1f}%",
        "strength_label": ml_label,
        "label_display": f"{ml_label} — {ml_conf_val:.1f}% signal strength",
        "badge_class": ml_badge,
        "raw_label": prediction_result.get("raw_model_label") or raw_ml_pred,
        "role": "AUXILIARY_ONLY",
        "description": "This is an auxiliary ML signal based on patterns learned from historical training data. It does not determine whether the current real-world claims are true.",
        "why_different": "The ML model evaluates writing patterns learned from historical training data. The Final Assessment evaluates real-world evidence retrieved for the claims. When current evidence is available, factual evidence has priority over historical writing-style patterns.",
    }

    # 8. Dynamic Timeline (Section 19)
    timeline = [
        {
            "step": 1,
            "title": "Input Received",
            "detail": input_type or "Pasted Text",
        },
        {
            "step": 2,
            "title": "Claims Extracted",
            "detail": f"{total_cnt} claims" if total_cnt != 1 else "1 claim",
            "count": total_cnt,
        },
        {
            "step": 3,
            "title": "Evidence Retrieved",
            "detail": f"{source_counts['total']} unique sources" if source_counts['total'] != 1 else "1 unique source",
            "count": source_counts["total"],
        },
        {
            "step": 4,
            "title": "Claims Verified",
            "detail": f"{supported_cnt} Supported, {contradicted_cnt} Contradicted, {insufficient_cnt} Insufficient",
            "counts": {
                "supported": supported_cnt,
                "contradicted": contradicted_cnt,
                "insufficient": insufficient_cnt,
            },
        },
        {
            "step": 5,
            "title": "Final Assessment",
            "detail": verdict,
        },
    ]

    final_assessment_obj = {
        "label": verdict,
        "confidence": round(conf_score / 100.0, 2),
        "reason": subtitle,
        "verdict": verdict,
        "confidence_label": conf_label,
        "confidence_score": conf_score,
        "subtitle": subtitle,
        "why_points": why_points,
        "key_takeaways": key_takeaways,
    }

    # Clean Canonical API Summary (Section 28)
    summary_obj = {
        "final_verdict": verdict.replace(" / ", "_").replace(" ", "_"),
        "verdict_display": verdict,
        "confidence": round(conf_score),
        "confidence_level": conf_label,
        "total_claims": total_cnt,
        "supported_claims": supported_cnt,
        "contradicted_claims": contradicted_cnt,
        "insufficient_claims": insufficient_cnt,
    }

    historical_ml_api = {
        "prediction": raw_ml_pred,
        "probability": round(ml_conf_val / 100.0, 4),
        "role": "AUXILIARY_ONLY",
        "signal_display": ml_signal["label_display"],
    }

    canonical_result = {
        "summary": summary_obj,
        "claims": processed_claims,
        "sources": all_unique_sources,
        "source_counts": source_counts,
        "claim_counts": claim_counts,
        "evidence_counts": evidence_counts,
        "evidence_summary": evidence_summary,
        "final_assessment": final_assessment_obj,
        "ml_signal": ml_signal,
        "historical_ml": historical_ml_api,
        "timeline": timeline,
        "article": {
            "input_type": input_type,
            "temporal_context": temporal_context,
            "temporal_detected_year": temporal_detected_year,
        },
        "has_verified_factcheck": has_verified_factcheck,
        "processed_claims": processed_claims,
        "all_sources": all_unique_sources,
    }

    # Validate analysis consistency (Section 23)
    val_report = validate_analysis_consistency(canonical_result)
    canonical_result["consistency_check"] = val_report

    return canonical_result
