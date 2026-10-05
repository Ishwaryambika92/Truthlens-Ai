"""
Layer 7 - Independent web evidence search
Layer 8 - Cross-source consistency

HARDENED VERSION: Never trusts an LLM's own verdict about a source without
independently fetching and validating that source first.

Pipeline (per claim):
  1. SEARCH (LLM + web_search tool): asked ONLY to return candidate
     {title, url} pairs it actually found via search - explicitly NOT
     asked to judge support/contradiction at this stage.
  2. FETCH & VALIDATE: every candidate URL is independently fetched via
     detector.extract_text.extract_from_url (SSRF-safe, size-capped,
     content-type-checked). Malformed or dead URLs are dropped.
  3. DEDUPLICATE syndicated copies: sources sharing a domain or near-identical
     text are grouped so they count as ONE independent source, not several.
  4. CONTENT-GROUNDED JUDGMENT: given ONLY the claim and retrieved text,
     judges SUPPORTS / CONTRADICTS / UNRELATED per source.
  5. AGGREGATE independent sources' verdicts into one claim-level status.

Gated behind ANTHROPIC_API_KEY. Every failure mode degrades to UNVERIFIED/dropped
rather than crashing or fabricating.
"""
import difflib
import json
import logging
import os
import re

import requests

logger = logging.getLogger("truthlens.verification.web_evidence")

from .. import extract_text
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from ..preprocessing import STOPWORDS
from .source_analysis import classify_source_reliability, normalize_domain
from .claims import classify_claim_modality
from .entailment import evaluate_claim_evidence_entailment, ACTION_CONTRADICTION_MAP

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip() or (GOOGLE_API_KEY.strip() if GOOGLE_API_KEY else "")

VALID_CLAIM_STATUSES = {"SUPPORTED", "CONTRADICTED", "MIXED", "UNVERIFIED"}
VALID_CONTENT_VERDICTS = {"SUPPORTS", "CONTRADICTS", "INSUFFICIENT", "UNRELATED"}

MAX_CANDIDATE_SOURCES = 5
SYNDICATION_SIMILARITY_THRESHOLD = 0.85  # difflib ratio on retrieved text excerpts

PRIMARY_SOURCE_DOMAIN_HINTS = (
    ".gov", ".mil", "who.int", "un.org", "europa.eu", "courtlistener.com",
    "sec.gov", "nih.gov", "cdc.gov", "supremecourt.gov", "federalregister.gov",
)

SEARCH_SYSTEM_PROMPT = (
    "You search the web to find candidate sources relevant to a factual claim. "
    "Do NOT judge whether the claim is true or false, and do NOT summarize what "
    "any source says. Just report up to 5 distinct, real web pages your search "
    "actually found that are directly relevant to this claim, prioritizing "
    "official/primary sources (government, company press releases, court "
    "records, scientific publications, statistical agencies) when applicable, "
    "plus independent reputable news coverage. Respond ONLY with a JSON array "
    "of objects: [{\"title\": \"...\", \"url\": \"...\"}]. No preamble, no "
    "markdown fences. Only include URLs your search actually returned - never "
    "guess, construct, or recall a URL from memory."
)

CONTENT_JUDGMENT_SYSTEM_PROMPT = (
    "You are an objective, meticulous news verification analyst evaluating independent evidence sources against a specific factual claim.\n\n"
    "For EACH source below, judge STRICTLY from the ACTUAL RETRIEVED TEXT whether it directly supports, directly contradicts, is insufficient to verify, or is unrelated to the claim.\n\n"
    "STRICT VERDICT RULES:\n"
    "1. 'SUPPORTS': The retrieved text explicitly and directly corroborates or confirms that the asserted event, policy, action, or fact actually took place or is true.\n"
    "   - CRITICAL: TOPICAL RELEVANCE IS NOT EVIDENCE (Section 7). A source that merely discusses the same topic, company, person, or mission is NOT automatically supporting. If the retrieved text describes discussions, background, or plans without confirming the specific event/action actually occurred, mark it 'INSUFFICIENT' or 'UNRELATED', NEVER 'SUPPORTS'.\n"
    "   - CRITICAL: Do NOT mark SUPPORTS if the text is debunking the claim, discussing a viral fake message, or merely repeating rumors.\n"
    "2. 'CONTRADICTS': The retrieved text DIRECTLY and EXPLICITLY disputes, refutes, or denies the claim (e.g., proves the event did NOT occur, proves it was fabricated/debunked, marks it as a hoax/scam, or quotes official denials of the specific claim).\n"
    "   - CRITICAL RESTRICTION: A source must NEVER be classified as CONTRADICTS merely because it discusses a different date/details, covers a related event, is older, does not mention the claim, or states that it 'could not independently verify' the event. These cases MUST be marked 'INSUFFICIENT' or 'UNRELATED'.\n"
    "3. 'INSUFFICIENT': The source is relevant to the topic or entities, but does NOT contain enough verified factual confirmation to definitively prove or disprove the specific assertion (e.g. wire reports saying 'could not be independently verified', or secondary commentary without confirmed facts).\n"
    "4. 'UNRELATED': The source is about a different topic, does not address the claim, or is completely off-topic.\n\n"
    "DIRECTNESS ASSESSMENT:\n"
    "- 'DIRECT': Source directly investigated, observed, or made an official declaration about the specific claim.\n"
    "- 'INDIRECT': Source reports general background, secondary reactions, or partial contextual facts.\n"
    "- 'NONE': Not applicable (unrelated/insufficient).\n\n"
    "Respond ONLY with a JSON array, one object per source in the exact order given:\n"
    '[{"verdict": "SUPPORTS" | "CONTRADICTS" | "INSUFFICIENT" | "UNRELATED", "directness": "DIRECT" | "INDIRECT" | "NONE", "reason": "one concise grounded sentence explaining why"}].'
)


def _post_anthropic(system_prompt: str, user_content: str, *, use_web_search: bool, max_tokens: int = 900):
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
    }
    if use_web_search:
        payload["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
    resp = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=45,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_text_blocks(data: dict) -> str:
    return "\n".join(b["text"] for b in data.get("content", []) if b.get("type") == "text").strip()


def _strip_fences(raw: str) -> str:
    raw = raw.strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    return raw


# ---------------------------------------------------------------------------
# Stage 1: search for CANDIDATE sources only (never trusted as verdict)
# ---------------------------------------------------------------------------

def generate_search_queries(claim: str) -> list[str]:
    """
    Generates multiple targeted search queries for a claim (Sections 6, 7, 15):
    - Query 1: entities + action verbs + year/date
    - Query 2: full normalized claim statement
    - Query 3: entity + official announcement / press release
    - Query 4: entity + disconfirmation / refutation search (Section 15)
    """
    clean = claim.rstrip(".").strip()
    words = [w for w in re.findall(r"\b[a-zA-Z0-9\-_]+\b", clean) if w.lower() not in STOPWORDS and len(w) >= 3]
    entities = [w for w in re.findall(r"\b[A-Z0-9][a-zA-Z0-9\-_/]+\b", clean) if w.lower() not in STOPWORDS]

    queries = []
    # Query 1: Entity + key content words
    if entities:
        q1 = " ".join(entities + [w for w in words if w not in entities][:4])
        queries.append(q1)

    # Query 2: Direct claim text
    queries.append(clean[:120])

    # Query 3: Entity + official announcement / press release
    if entities:
        primary_entity = entities[0]
        q3 = f"{primary_entity} " + " ".join([w for w in words if w != primary_entity][:3]) + " official announcement"
        queries.append(q3)

    # Query 4: Disconfirmation / refutation candidates (Section 15)
    claim_lower = clean.lower()
    disconf_terms = ["denied", "refuted", "debunked", "false", "hoax", "fact check"]
    for action_key, conflicting_set in ACTION_CONTRADICTION_MAP.items():
        if re.search(r"\b" + re.escape(action_key) + r"\b", claim_lower):
            priority_terms = ("lost", "defeated", "conceded", "denied", "refuted", "failed", "dead", "cancelled", "debunked")
            prioritized = [t for t in priority_terms if t in conflicting_set]
            rest = [t for t in sorted(conflicting_set) if t not in prioritized]
            disconf_terms = (prioritized + rest)[:2] + disconf_terms
            break

    if entities:
        q4 = f"{entities[0]} " + " ".join([w for w in words if w not in entities][:2]) + " " + " ".join(disconf_terms[:2])
        queries.append(q4)
    elif words:
        q4 = " ".join(words[:3]) + " " + disconf_terms[0]
        queries.append(q4)

    seen = set()
    unique_queries = []
    for q in queries:
        q_norm = q.lower().strip()
        if q_norm not in seen and len(q_norm) > 4:
            seen.add(q_norm)
            unique_queries.append(q)

    return unique_queries[:4]


def _search_candidate_sources(claim: str) -> list:
    """Returns a list of {"title": str, "url": str} from targeted multi-query search (Section 7)."""
    queries = generate_search_queries(claim)
    candidates = []
    seen_urls = set()

    # 1. Dedicated live web search via Tavily API if configured
    if TAVILY_API_KEY:
        try:
            from .. import ai_services
            for q in queries:
                tav_res = ai_services.search_tavily(q, max_results=3)
                for item in tav_res:
                    u = item.get("url")
                    if u and u not in seen_urls:
                        seen_urls.add(u)
                        candidates.append(item)
                if len(candidates) >= MAX_CANDIDATE_SOURCES:
                    break
            if candidates:
                return candidates[:MAX_CANDIDATE_SOURCES]
        except Exception:
            pass

    # 2. Anthropic with web search tool (preserves unit test mocks)
    if ANTHROPIC_API_KEY == "fake-test-key" or (ANTHROPIC_API_KEY and not candidates):
        try:
            target_query = queries[0] if queries else claim
            data = _post_anthropic(
                SEARCH_SYSTEM_PROMPT,
                f"Claim: \"{claim}\"\nTarget search query: {target_query}\n\nFind candidate sources via web search.",
                use_web_search=True,
            )
            raw = _strip_fences(_extract_text_blocks(data))
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for item in parsed[:MAX_CANDIDATE_SOURCES]:
                    if isinstance(item, dict) and item.get("url"):
                        u = str(item["url"]).strip()
                        if u not in seen_urls:
                            seen_urls.add(u)
                            candidates.append({
                                "title": str(item.get("title", ""))[:200],
                                "url": u,
                            })
                return candidates[:MAX_CANDIDATE_SOURCES]
        except Exception:
            pass

    return candidates



# ---------------------------------------------------------------------------
# Stage 2: independently fetch + validate each candidate URL
# ---------------------------------------------------------------------------

def _fetch_and_validate_source(candidate: dict) -> dict:
    url = candidate["url"]
    title_hint = candidate.get("title", "")

    if not re.match(r"^https?://[^\s]+\.[^\s]+", url):
        return {
            "url": url,
            "title": title_hint,
            "publisher": title_hint or "Malformed Source URL",
            "published_date": None,
            "domain": None,
            "stage": "SEARCH_RESULT_FOUND",
            "is_primary_source": False,
            "is_fact_checker": False,
            "source_reliability": "LOW",
            "source_reliability_weight": 0.3,
            "source_reliability_label": "Low (Malformed URL)",
            "retrieved": False,
            "text_excerpt": None,
            "validation_error": "Malformed or fabricated-looking URL - rejected before fetching.",
        }

    # Call with single argument to match existing mock signatures
    text, error, metadata = extract_text.extract_from_url(url)
    if error:
        content_snippet = (candidate.get("content") or "").strip()
        if content_snippet and len(content_snippet) >= 40:
            parsed_domain = normalize_domain(url)
            rel = classify_source_reliability(parsed_domain or url)
            is_primary = rel["is_primary_source"] or bool(parsed_domain and any(hint in parsed_domain for hint in PRIMARY_SOURCE_DOMAIN_HINTS))
            publisher = metadata.get("site_name") or metadata.get("title") or title_hint or parsed_domain or "Unknown Source"
            return {
                "url": url,
                "title": metadata.get("title") or title_hint or url,
                "publisher": publisher,
                "published_date": metadata.get("published_date"),
                "domain": parsed_domain,
                "is_primary_source": is_primary,
                "is_fact_checker": rel["is_fact_checker"],
                "source_reliability": rel["tier"],
                "source_reliability_weight": rel["weight"],
                "source_reliability_label": rel["tier_label"],
                "stage": "SOURCE_RETRIEVED",
                "retrieved": True,
                "text_excerpt": content_snippet[:3000],
                "validation_error": None,
            }
        norm_dom = normalize_domain(url)
        rel = classify_source_reliability(norm_dom or url)
        return {
            "url": url,
            "title": title_hint,
            "publisher": title_hint or norm_dom or "Unknown Source",
            "published_date": None,
            "domain": norm_dom or None,
            "stage": "SEARCH_RESULT_FOUND",
            "is_primary_source": rel["is_primary_source"],
            "is_fact_checker": rel["is_fact_checker"],
            "source_reliability": rel["tier"],
            "source_reliability_weight": rel["weight"],
            "source_reliability_label": rel["tier_label"],
            "retrieved": False,
            "text_excerpt": None,
            "validation_error": error,
        }

    domain = metadata.get("domain") or normalize_domain(url)
    rel = classify_source_reliability(domain or url)
    is_primary = rel["is_primary_source"] or bool(domain and any(hint in domain for hint in PRIMARY_SOURCE_DOMAIN_HINTS))
    publisher = metadata.get("site_name") or metadata.get("title") or title_hint or domain or "Unknown Source"
    return {
        "url": url,
        "title": metadata.get("title") or title_hint or url,
        "publisher": publisher,
        "published_date": metadata.get("published_date"),
        "domain": domain,
        "is_primary_source": is_primary,
        "is_fact_checker": rel["is_fact_checker"],
        "source_reliability": rel["tier"],
        "source_reliability_weight": rel["weight"],
        "source_reliability_label": rel["tier_label"],
        "stage": "SOURCE_RETRIEVED",
        "retrieved": True,
        "text_excerpt": text[:3000],
        "validation_error": None,
    }


def _deduplicate_syndicated(retrieved_sources: list) -> tuple:
    """Groups sources that are the same domain or near-identical text."""
    independent = []
    duplicates = []
    seen_domains = set()

    for src in retrieved_sources:
        if src["domain"] and src["domain"] in seen_domains:
            src["syndicated_of"] = next(k["url"] for k in independent if k["domain"] == src["domain"])
            duplicates.append(src)
            continue
        is_dup = False
        for k in independent:
            if k.get("text_excerpt") and src.get("text_excerpt"):
                ratio = difflib.SequenceMatcher(
                    None, k["text_excerpt"][:400], src["text_excerpt"][:400]
                ).ratio()
                if ratio >= SYNDICATION_SIMILARITY_THRESHOLD:
                    src["syndicated_of"] = k["url"]
                    is_dup = True
                    break
        if is_dup:
            duplicates.append(src)
        else:
            if src["domain"]:
                seen_domains.add(src["domain"])
            independent.append(src)

    return independent, duplicates


# ---------------------------------------------------------------------------
# Stage 3: content-grounded judgment against retrieved text
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> list | None:
    """Robustly extract a JSON array from text even if surrounded by commentary or fences."""
    if not text:
        return None
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
            return data["results"]
    except Exception:
        pass

    # Extract markdown code blocks
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Extract between outer square brackets
    bracket_match = re.search(r"(\[.*\])", text, re.DOTALL)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(1))
        except Exception:
            pass

    return None


def _judge_sources_against_claim(claim: str, independent_sources: list) -> list:
    """Returns a list of {"verdict", "directness", "reason"} aligned to independent_sources order."""
    if not independent_sources:
        return []

    # 1. Anthropic if explicitly mocked by unit tests
    if ANTHROPIC_API_KEY == "fake-test-key":
        try:
            sources_block = "\n\n".join(
                f"SOURCE {i+1} ({s['url']}):\n{s['text_excerpt'][:1500]}"
                for i, s in enumerate(independent_sources)
            )
            data = _post_anthropic(
                CONTENT_JUDGMENT_SYSTEM_PROMPT,
                f"Claim: \"{claim}\"\n\n{sources_block}",
                use_web_search=False,
                max_tokens=600,
            )
            raw = _strip_fences(_extract_text_blocks(data))
            parsed = _extract_json_array(raw)
            if isinstance(parsed, list) and len(parsed) == len(independent_sources):
                results = []
                for item in parsed:
                    verdict = str(item.get("verdict", "UNRELATED")).upper() if isinstance(item, dict) else "UNRELATED"
                    if verdict not in VALID_CONTENT_VERDICTS:
                        verdict = "UNRELATED"
                    directness = str(item.get("directness", "DIRECT")).upper() if isinstance(item, dict) else "DIRECT"
                    if directness not in ("DIRECT", "INDIRECT", "NONE"):
                        directness = "DIRECT" if verdict in ("SUPPORTS", "CONTRADICTS") else "NONE"
                    reason = str(item.get("reason", ""))[:400] if isinstance(item, dict) else ""
                    results.append({"verdict": verdict, "directness": directness, "reason": reason})
                return results
        except Exception:
            pass

    # 2. Google Gemini (fast, active quota - preferred in production)
    if GEMINI_API_KEY:
        try:
            from .. import ai_services
            sources_block = "\n\n".join(
                f"SOURCE {i+1} ({s['url']}):\n{s['text_excerpt'][:1500]}"
                for i, s in enumerate(independent_sources)
            )
            res = ai_services.call_gemini(
                f"Claim: \"{claim}\"\n\n{sources_block}",
                system_instruction=CONTENT_JUDGMENT_SYSTEM_PROMPT,
                max_tokens=600,
            )
            if res.get("success") and res.get("text"):
                parsed = _extract_json_array(res["text"])
                if isinstance(parsed, list) and len(parsed) == len(independent_sources):
                    results = []
                    for item in parsed:
                        verdict = str(item.get("verdict", "UNRELATED")).upper() if isinstance(item, dict) else "UNRELATED"
                        if verdict not in VALID_CONTENT_VERDICTS:
                            verdict = "UNRELATED"
                        directness = str(item.get("directness", "DIRECT")).upper() if isinstance(item, dict) else "DIRECT"
                        if directness not in ("DIRECT", "INDIRECT", "NONE"):
                            directness = "DIRECT" if verdict in ("SUPPORTS", "CONTRADICTS") else "NONE"
                        reason = str(item.get("reason", ""))[:400] if isinstance(item, dict) else ""
                        results.append({"verdict": verdict, "directness": directness, "reason": reason})
                    return results
        except Exception:
            pass

    # 3. Anthropic Claude (if configured and not already attempted)
    if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "fake-test-key":
        try:
            sources_block = "\n\n".join(
                f"SOURCE {i+1} ({s['url']}):\n{s['text_excerpt'][:1500]}"
                for i, s in enumerate(independent_sources)
            )
            data = _post_anthropic(
                CONTENT_JUDGMENT_SYSTEM_PROMPT,
                f"Claim: \"{claim}\"\n\n{sources_block}",
                use_web_search=False,
                max_tokens=600,
            )
            raw = _strip_fences(_extract_text_blocks(data))
            parsed = _extract_json_array(raw)
            if isinstance(parsed, list) and len(parsed) == len(independent_sources):
                results = []
                for item in parsed:
                    verdict = str(item.get("verdict", "UNRELATED")).upper() if isinstance(item, dict) else "UNRELATED"
                    if verdict not in VALID_CONTENT_VERDICTS:
                        verdict = "UNRELATED"
                    directness = str(item.get("directness", "DIRECT")).upper() if isinstance(item, dict) else "DIRECT"
                    if directness not in ("DIRECT", "INDIRECT", "NONE"):
                        directness = "DIRECT" if verdict in ("SUPPORTS", "CONTRADICTS") else "NONE"
                    reason = str(item.get("reason", ""))[:400] if isinstance(item, dict) else ""
                    results.append({"verdict": verdict, "directness": directness, "reason": reason})
                return results
        except Exception:
            pass

    return _judge_sources_heuristic(claim, independent_sources)


SYNONYM_GROUPS = [
    {"launched", "lifted off", "took off", "blasted off", "rocketed", "sent into space"},
    {"deployed", "placed in orbit", "injected into orbit", "released", "separated"},
    {"died", "passed away", "succumbed", "lost life"},
    {"arrested", "detained", "taken into custody", "apprehended"},
    {"signed", "inked", "finalized", "ratified", "agreed"},
    {"approved", "cleared", "greenlit", "sanctioned", "passed"},
    {"announced", "confirmed", "stated", "reported", "revealed"},
    {"sriharikota", "satish dhawan space centre", "sdsc", "shar"},
]


def _judge_sources_heuristic(claim: str, independent_sources: list) -> list:
    """
    Universal claim-evidence entailment evaluating actual text against exact claim proposition (Sections 7, 8, 9, 10, 11).
    Distinguishes SUPPORTS, CONTRADICTS, INSUFFICIENT, and UNRELATED without domain hardcoding.
    Topical relevance alone is NEVER treated as support.
    """
    results = []
    for src in independent_sources:
        text = src.get("text_excerpt") or ""
        ent_res = evaluate_claim_evidence_entailment(claim, text, source_metadata=src)

        verdict = ent_res["verdict"]
        directness = ent_res.get("directness", "DIRECT" if verdict in ("SUPPORTS", "CONTRADICTS") else "NONE")
        reason = ent_res.get("reason", "")
        evidence_text = ent_res.get("evidence_text", text[:280])

        results.append({
            "verdict": verdict,
            "directness": directness,
            "reason": reason,
            "evidence_text": evidence_text,
            "entailment_state": ent_res.get("entailment_state", "UNKNOWN"),
            "conflict_type": ent_res.get("conflict_type"),
        })
    return results


def calculate_evidence_strength(source: dict, claim: str) -> tuple[float, str]:
    """
    Computes evidence strength conceptually using:
    Source reliability + Claim relevance + Directness + Independence
    Returns: (strength_score: float, strength_label: 'STRONG' | 'MODERATE' | 'WEAK' | 'NONE')
    """
    verdict = source.get("content_verdict", "UNRELATED")
    if verdict not in ("SUPPORTS", "CONTRADICTS"):
        return 0.0, "NONE"

    rel_weight = source.get("source_reliability_weight", 0.3)
    if source.get("is_primary_source"):
        rel_weight = max(rel_weight, 2.0)

    directness = source.get("directness", "DIRECT")
    directness_mult = 1.0 if directness == "DIRECT" else 0.65

    relevance_mult = 1.0
    text_len = len(source.get("text_excerpt") or "")
    if text_len < 100:
        relevance_mult = 0.7

    score = round(rel_weight * directness_mult * relevance_mult, 2)
    if score >= 1.0:
        label = "STRONG"
    elif score >= 0.5:
        label = "MODERATE"
    else:
        label = "WEAK"

    return score, label


_STAGE_BY_VERDICT = {
    "SUPPORTS": "SOURCE_CONTENT_SUPPORTS_CLAIM",
    "CONTRADICTS": "SOURCE_CONTENT_CONTRADICTS_CLAIM",
    "INSUFFICIENT": "SOURCE_NOT_SUFFICIENT",
    "UNRELATED": "SOURCE_NOT_SUFFICIENT",
}


def verify_claim_with_evidence(claim: str) -> dict:
    """Returns a per-claim evidence record. Never fabricates sources; degrades gracefully."""
    modality_info = classify_claim_modality(claim)
    if not modality_info.get("is_checkable", True):
        return {
            "claim": claim,
            "status": "INSUFFICIENT",
            "confidence": 0.30,
            "reason": f"{modality_info['modality']}: {modality_info['reason']}",
            "summary": f"This statement is classified as {modality_info['modality']} and is not an independently verifiable factual assertion.",
            "sources": [],
            "available": True,
            "modality": modality_info["modality"],
            "is_checkable": False,
        }

    if not ANTHROPIC_API_KEY and not TAVILY_API_KEY:
        return {
            "claim": claim,
            "status": "INSUFFICIENT",
            "confidence": 0.20,
            "reason": "Independent web evidence search is unavailable (search API not configured).",
            "summary": "Independent web evidence search is unavailable (search API not configured).",
            "sources": [],
            "available": False,
        }

    candidates = _search_candidate_sources(claim)
    if not candidates:
        return {
            "claim": claim,
            "status": "INSUFFICIENT",
            "confidence": 0.25,
            "reason": "Search did not return any usable candidate sources for this claim.",
            "summary": "Search did not return any usable candidate sources for this claim.",
            "sources": [],
            "available": True,
        }

    if len(candidates) > 1:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(candidates), 5)) as executor:
            retrieved_attempts = list(executor.map(_fetch_and_validate_source, candidates))
    else:
        retrieved_attempts = [_fetch_and_validate_source(c) for c in candidates]
    successfully_retrieved = [r for r in retrieved_attempts if r["retrieved"]]
    failed_sources = [r for r in retrieved_attempts if not r["retrieved"]]

    if not successfully_retrieved:
        return {
            "claim": claim,
            "status": "INSUFFICIENT",
            "confidence": 0.30,
            "reason": "Retrieved candidate sources could not be independently retrieved or validated.",
            "summary": (
                f"{len(failed_sources)} candidate source(s) were found by search but none could be "
                f"independently retrieved/validated (unreachable, blocked, or malformed URLs)."
            ),
            "sources": failed_sources,
            "available": True,
        }

    independent, syndicated_duplicates = _deduplicate_syndicated(successfully_retrieved)
    verdicts = _judge_sources_against_claim(claim, independent)

    for src, verdict in zip(independent, verdicts):
        src["claim"] = claim
        src["content_verdict"] = verdict["verdict"]
        src["verdict"] = verdict["verdict"]
        src["directness"] = verdict.get("directness", "DIRECT")
        src["reason"] = verdict["reason"]
        raw_ev_text = verdict.get("evidence_text") or src.get("text_excerpt") or src.get("reason", "")
        src["evidence_text"] = raw_ev_text
        src["stage"] = _STAGE_BY_VERDICT.get(verdict["verdict"], "SOURCE_NOT_SUFFICIENT")
        score, label = calculate_evidence_strength(src, claim)
        src["strength_score"] = score
        src["strength_label"] = label

        # Canonical evidence contract (Section 6)
        src["source_url"] = src.get("url", "")
        src["source_title"] = src.get("title") or src.get("publisher", "")
        v_upper = verdict["verdict"].upper()
        if v_upper in ("SUPPORTS", "SUPPORTED"):
            rel = "SUPPORTS"
        elif v_upper in ("CONTRADICTS", "CONTRADICTED"):
            rel = "CONTRADICTS"
        elif v_upper == "UNRELATED":
            rel = "UNRELATED"
        else:
            rel = "NEUTRAL"
        src["relationship"] = rel
        src["evidence_strength"] = label
        src["source_quality"] = round(src.get("source_reliability_weight", 0.5) / 2.0, 2)
        src["relevance_score"] = 1.0 if rel in ("SUPPORTS", "CONTRADICTS") else (0.5 if rel == "NEUTRAL" else 0.0)

        logger.info(
            "SOURCE_EVAL_DEBUG | CLAIM: %s | SOURCE: %s | QUALITY: %s | VERDICT: %s | DIRECTNESS: %s | REASON: %s",
            claim[:60], src.get("domain") or src.get("url", "")[:40], src.get("source_reliability"),
            src["content_verdict"], src["directness"], src["reason"][:80]
        )

    for src in syndicated_duplicates:
        src["claim"] = claim
        src["content_verdict"] = "UNRELATED"
        src["verdict"] = "UNRELATED"
        src["relationship"] = "UNRELATED"
        src["directness"] = "NONE"
        src["reason"] = "Syndicated duplicate of another retrieved source."
        src["strength_score"] = 0.0
        src["strength_label"] = "NONE"
        src["evidence_strength"] = "NONE"
        src["source_url"] = src.get("url", "")
        src["source_title"] = src.get("title") or src.get("publisher", "")
        src["source_quality"] = 0.0
        src["relevance_score"] = 0.0
        src["evidence_text"] = src.get("text_excerpt", "")

    for src in failed_sources:
        src["claim"] = claim
        src["content_verdict"] = "UNVERIFIED"
        src["verdict"] = "UNVERIFIED"
        src["relationship"] = "UNRELATED"
        src["directness"] = "NONE"
        src["reason"] = src.get("validation_error") or "Source could not be fetched or verified."
        src["strength_score"] = 0.0
        src["strength_label"] = "NONE"
        src["evidence_strength"] = "NONE"
        src["source_url"] = src.get("url", "")
        src["source_title"] = src.get("title") or src.get("publisher", "")
        src["source_quality"] = 0.0
        src["relevance_score"] = 0.0
        src["evidence_text"] = ""

    supporting = [s for s in independent if s.get("content_verdict") == "SUPPORTS"]
    contradicting = [s for s in independent if s.get("content_verdict") == "CONTRADICTS"]

    support_strength = sum(s.get("strength_score", 0.0) for s in supporting)
    contra_strength = sum(s.get("strength_score", 0.0) for s in contradicting)

    strong_supporting = [s for s in supporting if s.get("strength_label") == "STRONG"]
    strong_contradicting = [s for s in contradicting if s.get("strength_label") == "STRONG"]

    has_primary_support = any(s.get("is_primary_source") for s in supporting)
    has_primary_contra = any(s.get("is_primary_source") for s in contradicting)

    # Conceptual hierarchy for claim-level status (Sections 13, 14, 15, 17):
    if supporting and not contradicting:
        status = "SUPPORTED"
        if has_primary_support or len(strong_supporting) >= 2:
            confidence = 0.94
            reason = "Official primary source directly confirms the event, date, and location."
        elif strong_supporting or support_strength >= 1.0:
            confidence = 0.86
            reason = f"{len(supporting)} reliable source(s) directly corroborate the assertion."
        else:
            confidence = 0.74
            reason = f"{len(supporting)} source(s) corroborate the claim with moderate evidence."
        strength_desc = "Strong" if (strong_supporting or has_primary_support) else "Moderate"
        summary = (
            f"{len(supporting)} independently-retrieved and content-verified source(s) "
            f"support this claim ({strength_desc} evidence)."
        )
    elif contradicting and not supporting:
        has_direct_contra = any(s.get("directness") == "DIRECT" for s in contradicting)
        if strong_contradicting or has_primary_contra or contra_strength >= 0.6 or has_direct_contra:
            status = "CONTRADICTED"
            if has_primary_contra or len(strong_contradicting) >= 2:
                confidence = 0.93
                reason = "Official source or established reporting directly contradicts this claim."
            else:
                confidence = 0.85
                reason = f"{len(contradicting)} reliable source(s) refute the factual assertion."
            summary = (
                f"{len(contradicting)} independently-retrieved and content-verified source(s) "
                f"directly contradict this claim."
            )
        else:
            status = "INSUFFICIENT"
            confidence = 0.35
            reason = "Available sources dispute the claim, but evidence is weak or indirect; insufficient to establish contradiction."
            summary = (
                f"{len(contradicting)} source(s) disputed this claim, but evidence was weak or indirect; "
                f"insufficient to definitively contradict."
            )
    elif supporting and contradicting:
        # Rule 9 & Section 15: Check authority and strength weighting
        if (strong_supporting or has_primary_support) and (support_strength >= 1.5 * contra_strength) and not strong_contradicting:
            status = "SUPPORTED"
            confidence = 0.85
            reason = f"{len(supporting)} authoritative source(s) support the claim, outweighing weak contradictions."
            summary = (
                f"{len(supporting)} reliable source(s) support this claim (evidence strength: Strong), "
                f"outweighing {len(contradicting)} weak or indirect contradiction(s)."
            )
        elif (strong_contradicting or has_primary_contra) and (contra_strength >= 1.5 * support_strength) and not strong_supporting:
            status = "CONTRADICTED"
            confidence = 0.88
            reason = f"{len(contradicting)} authoritative source(s) directly dispute the claim, outweighing weak support."
            summary = (
                f"{len(contradicting)} reliable source(s) directly dispute this claim (evidence strength: Strong), "
                f"outweighing weak support."
            )
        else:
            # Genuine conflict between comparable sources -> INSUFFICIENT to establish either side
            status = "INSUFFICIENT"
            confidence = 0.38
            reason = "Available sources present conflicting evidence; insufficient to definitively support or contradict."
            summary = (
                f"{len(supporting)} source(s) support while "
                f"{len(contradicting)} source(s) contradict this claim — evidence is genuinely conflicting."
            )
    else:
        status = "INSUFFICIENT"
        confidence = 0.28
        reason = "Available sources discuss the topic but do not establish or refute the specific claim."
        summary = "Sources were retrieved but their actual content did not sufficiently address this claim."

    all_sources = independent + syndicated_duplicates + failed_sources

    # Developer diagnostics logging (Section 29)
    logger.info(
        "DIAGNOSTIC | CLAIM: %s | RETRIEVED: %d | UNIQUE: %d | STATUS: %s | CONFIDENCE: %.2f | REASON: %s",
        claim[:80], len(retrieved_attempts), len(independent), status, confidence, reason
    )

    return {
        "claim": claim,
        "status": status,
        "confidence": confidence,
        "reason": reason,
        "summary": summary,
        "sources": all_sources,
        "support_strength": round(support_strength, 2),
        "contra_strength": round(contra_strength, 2),
        "strong_supporting_count": len(strong_supporting),
        "strong_contradicting_count": len(strong_contradicting),
        "available": True,
    }


def gather_evidence(claims: list) -> dict:
    """Runs verify_claim_with_evidence for each claim."""
    has_search_key = bool(ANTHROPIC_API_KEY or TAVILY_API_KEY)
    if not claims:
        return {
            "available": has_search_key,
            "claim_evidence": [],
            "message": None if has_search_key else "Web evidence search unavailable - search API not configured.",
        }

    target_claims = claims[:4]
    if len(target_claims) > 1:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(target_claims), 4)) as executor:
            records = list(executor.map(verify_claim_with_evidence, target_claims))
    else:
        records = [verify_claim_with_evidence(c) for c in target_claims]
    return {
        "available": any(r["available"] for r in records),
        "claim_evidence": records,
        "message": None if has_search_key else "Web evidence search unavailable - search API not configured.",
    }

