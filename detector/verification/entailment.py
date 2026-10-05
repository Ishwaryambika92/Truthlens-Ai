"""
Universal Evidence-Grounded Claim Entailment and Contradiction Engine (Layers 7-8).

Core Principle (Section 1 & 7):
TruthLens must never answer "Does this article look real?" or "Does this source talk about the same topic?"
It must answer: "Does the reliable evidence logically entail, contradict, or remain insufficient to establish the exact proposition?"

Topical relevance is NOT evidence. A source discussing an entity or background without confirming
the specific predicate/action must receive NEUTRAL, never SUPPORTS.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("truthlens.verification.entailment")

# ---------------------------------------------------------------------------
# Universal Action / Outcome Antonyms & Conflicting States (Section 9)
# Domain-independent pairs where asserting one contradicts the other.
# ---------------------------------------------------------------------------
ACTION_CONTRADICTION_MAP = {
    # Movement & Space & Aviation
    "landed": {"flyby", "fly by", "lunar orbit only", "orbital only", "will not land", "did not land", "crashed", "missed", "aborted landing"},
    "landing": {"flyby", "fly by", "orbit only", "will not land", "did not land", "no landing"},
    "launched": {"cancelled", "canceled", "postponed", "delayed", "scrubbed", "aborted", "grounded", "rescheduled"},
    "launch": {"cancellation", "postponement", "delayed", "scrubbed", "aborted", "cancelled", "canceled"},
    "arrived": {"never arrived", "departed", "delayed", "stranded", "diverted", "failed to arrive"},
    
    # Outcomes, Contests & Elections
    "won": {"lost", "defeated", "conceded", "runner-up", "disqualified", "failed to win"},
    "winner": {"loser", "runner-up", "defeated", "conceded"},
    "elected": {"defeated", "lost election", "failed to win", "rejected by voters", "conceded"},
    "passed": {"failed", "rejected", "defeated", "vetoed", "blocked", "struck down"},
    
    # Decisions & Approvals
    "approved": {"rejected", "vetoed", "dismissed", "denied", "turned down", "disapproved", "blocked"},
    "confirmed": {"denied", "refuted", "disputed", "dismissed", "debunked", "unconfirmed"},
    "agreed": {"disagreed", "rejected", "refused", "declined", "walked away"},
    "signed": {"refused to sign", "rejected", "vetoed", "cancelled", "scrapped"},
    
    # Status & Life
    "alive": {"dead", "died", "passed away", "killed", "succumbed", "fatal"},
    "died": {"alive", "survived", "recovering", "in stable condition", "hoax death"},
    "killed": {"survived", "unharmed", "alive"},
    "arrested": {"released without charge", "cleared", "acquitted", "on the run", "denied arrest"},
    "opened": {"closed", "shut", "halted", "suspended"},
    "closed": {"opened", "operating", "remains open", "active"},
    
    # Quantities & Trends
    "increased": {"decreased", "dropped", "fell", "declined", "plunged", "reduced", "slumped"},
    "grew": {"shrank", "contracted", "decreased", "fell", "dropped"},
    "profit": {"loss", "net loss", "deficit", "unprofitable"},
    "surplus": {"deficit", "shortfall"},
    
    # Corporate & Transactions
    "acquired": {"sold", "divested", "rejected acquisition", "terminated merger", "cancelled takeover"},
    "bought": {"sold", "rejected offer"},
    "resigned": {"denied resignation", "remains in office", "appointed", "retained", "refused to step down"},
    "hired": {"fired", "dismissed", "terminated", "rejected applicant"},
}

# Universal Synonyms for Supported Entailment
UNIVERSAL_SYNONYMS = [
    {"launched", "lifted off", "took off", "blasted off", "rocketed", "sent into space", "deployed"},
    {"deployed", "placed in orbit", "injected into orbit", "released", "separated"},
    {"died", "passed away", "succumbed", "lost life"},
    {"arrested", "detained", "taken into custody", "apprehended"},
    {"signed", "inked", "finalized", "ratified", "agreed to"},
    {"approved", "cleared", "greenlit", "sanctioned", "passed", "authorized"},
    {"announced", "confirmed", "stated", "reported", "revealed", "declared"},
    {"won", "victorious", "secured victory", "triumphed", "elected", "claimed victory"},
    {"resigned", "stepped down", "quit", "vacated office", "relinquished post"},
    {"discovered", "found", "detected", "uncovered", "identified"},
    {"hit", "struck", "slammed into", "impacted"},
    {"cancelled", "canceled", "called off", "scrapped", "abandoned"},
    {"increased", "rose", "grew", "surged", "climbed", "jumped"},
    {"decreased", "fell", "dropped", "declined", "slumped", "plunged"},
]

# Explicit Refutation Phrases (Direct contradictions)
EXPLICIT_REFUTATION_PHRASES = [
    "claim is false", "claims are false", "claim is untrue", "is a hoax",
    "is completely false", "is entirely false", "is fake", "has been debunked",
    "have been debunked", "falsely claimed", "falsely claims", "falsely reporting",
    "false rumor", "false rumors", "no evidence that", "proven false", "proved false",
    "widely debunked", "debunked by", "fabricated story", "fabricated claim",
    "debunking the claim", "fact check: false", "fact-check: false",
    "rated false", "verdict: false", "pants on fire", "denied the claim",
    "officially denied the claim", "officially denied reports", "officially denied",
    "denied reports that", "refuted the claim", "baseless rumor", "baseless claim",
    "refuted reports that", "refuted claims that", "debunked reports that",
    "there is no truth to", "is inaccurate", "is not true", "did not happen",
    "never took place", "never happened"
]

# Phrases indicating that source does NOT confirm or deny the assertion
NON_CONFIRMATION_PHRASES = [
    "could not be independently verified", "could not independently verify",
    "cannot independently verify", "cannot independently confirm",
    "unable to independently verify", "not immediately verifiable",
    "unverified reports", "reports could not be verified", "has not been verified",
    "did not independently verify", "has not independently confirmed",
    "pending verification", "unconfirmed reports", "claims remain unverified"
]

# Speculative / Hypothetical markers that prevent a source from entailing a completed fact
HYPOTHETICAL_MARKERS = [
    "could happen", "may happen", "might happen", "is being considered",
    "under discussion", "was discussed", "talks were held", "proposed",
    "considering whether", "weighed the possibility", "hypothetically",
    "scientists warn that", "experts speculate that", "allegations that",
    "unconfirmed rumor", "speculation that", "concept design", "plans suggest"
]

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "can", "will", "just", "should",
    "now", "be", "was", "were", "been", "being", "have", "has", "had", "do",
    "does", "did", "at", "by", "with", "into", "over", "after"
}


# ---------------------------------------------------------------------------
# Structured Extractor Helpers
# ---------------------------------------------------------------------------

def extract_numbers_and_metrics(text: str) -> List[Tuple[float, str]]:
    """
    Extracts numbers, percentages, currencies, and quantities from text.
    Returns: list of (value: float, unit: str)
    Examples:
    '40%' -> (40.0, '%')
    '$5 billion' -> (5000000000.0, '$')
    '100 deaths' -> (100.0, 'deaths')
    '100,000 units' -> (100000.0, 'units')
    """
    results = []
    if not text:
        return results

    # 1. Percentages
    pct_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    for p in pct_matches:
        try:
            results.append((float(p), "%"))
        except ValueError:
            pass

    # 2. Currencies with scale ($5 billion, 5 million dollars, etc.)
    curr_matches = re.findall(r"([$€£₹])\s*(\d+(?:\.\d+)?)\s*(billion|million|crore|lakh)?", text, re.IGNORECASE)
    for sym, val, scale in curr_matches:
        try:
            v = float(val)
            s_lower = scale.lower() if scale else ""
            if "billion" in s_lower:
                v *= 1_000_000_000
            elif "million" in s_lower:
                v *= 1_000_000
            elif "crore" in s_lower:
                v *= 10_000_000
            elif "lakh" in s_lower:
                v *= 100_000
            results.append((v, sym))
        except ValueError:
            pass

    # 3. Quantities with units (e.g. 100 people, 10 deaths, 100,000 units)
    qty_matches = re.findall(
        r"\b(\d[\d,.]*)\s+(people|persons|attendees|deaths|killed|injured|dead|units|votes|seats|copies|satellites|astronauts)\b",
        text,
        re.IGNORECASE,
    )
    for val_str, unit in qty_matches:
        try:
            clean_v = float(val_str.replace(",", ""))
            results.append((clean_v, unit.lower()))
        except ValueError:
            pass

    return results


def extract_dates_and_years(text: str) -> Dict[str, Set[str]]:
    """
    Extracts dates, years, and months from text.
    Returns: {"years": set, "months": set, "full_dates": set}
    """
    text_lower = text.lower()
    years = set(re.findall(r"\b(19\d\d|20\d\d)\b", text))
    
    months_list = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    found_months = {m for m in months_list if re.search(r"\b" + m + r"\b", text_lower)}
    
    # Specific dates: 'October 4', 'January 10, 2026', '4 September'
    date_patterns = re.findall(
        r"\b((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?)\b",
        text_lower,
    )
    full_dates = {re.sub(r"(st|nd|rd|th)", "", d).strip() for d in date_patterns}
    
    return {
        "years": years,
        "months": found_months,
        "full_dates": full_dates,
    }


def extract_locations(text: str) -> Set[str]:
    """
    Extracts geographic locations from prepositional phrases: 'in Delhi', 'in Mumbai', etc.
    """
    matches = re.findall(r"\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text)
    locations = set()
    non_locations = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    for m in matches:
        if m not in non_locations and len(m) > 2:
            locations.add(m.lower())
    return locations


# ---------------------------------------------------------------------------
# Exact Universal Entailment Evaluator (Sections 7, 8, 9, 10, 11)
# ---------------------------------------------------------------------------

def evaluate_claim_evidence_entailment(claim_input: Any, evidence_text: str, source_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Performs universal, domain-independent claim-evidence entailment.
    Distinguishes:
      - SUPPORTS (evidence logically entails the exact proposition)
      - CONTRADICTS (evidence directly refutes or asserts conflicting facts)
      - NEUTRAL (evidence mentions related entities/topic but does NOT establish the action)
      - UNRELATED (evidence does not discuss the subject/entities)
    """
    source_metadata = source_metadata or {}
    
    # Normalize claim string
    if isinstance(claim_input, dict):
        claim_str = claim_input.get("normalized_claim") or claim_input.get("original_text") or claim_input.get("claim", "")
        c_modality = claim_input.get("modality", "FACTUAL")
        c_is_checkable = claim_input.get("is_checkable", True)
    else:
        claim_str = str(claim_input or "")
        c_modality = "FACTUAL"
        c_is_checkable = True

    claim_clean = claim_str.strip()
    claim_lower = claim_clean.lower()
    text_clean = (evidence_text or "").strip()
    text_lower = text_clean.lower()

    if not text_clean:
        return {
            "relationship": "UNRELATED",
            "verdict": "UNRELATED",
            "entailment_state": "UNKNOWN",
            "directness": "NONE",
            "confidence": 0.2,
            "reason": "Retrieved source text was empty or inaccessible.",
            "conflict_type": None,
            "evidence_text": "",
        }

    # Extract words and entities from claim
    claim_words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9\-_]{3,}\b", claim_clean) if w.lower() not in STOPWORDS]
    claim_entities = [w for w in re.findall(r"\b[A-Z][a-zA-Z0-9\-_/]+\b", claim_clean) if w.lower() not in STOPWORDS]
    claim_entities_lower = [e.lower() for e in claim_entities]

    # Snippet selection: first 280 chars
    evidence_snippet = text_clean[:280] + ("..." if len(text_clean) > 280 else "")

    # 1. Subject / Entity Presence Gate
    # If the source does not even mention the entities/subject or core nouns, it is UNRELATED.
    matched_entities = sum(1 for e in claim_entities_lower if e in text_lower)
    core_nouns = [w for w in claim_words if len(w) >= 4]
    matched_core_nouns = sum(1 for w in core_nouns if w in text_lower)

    if claim_entities_lower and matched_entities == 0 and matched_core_nouns < 2:
        return {
            "relationship": "UNRELATED",
            "verdict": "UNRELATED",
            "entailment_state": "UNKNOWN",
            "directness": "NONE",
            "confidence": 0.2,
            "reason": "Source text does not discuss the subject or core entities of the claim.",
            "conflict_type": None,
            "evidence_text": evidence_snippet,
        }

    # 2. Explicit Refutations Gate (Section 9)
    for phrase in EXPLICIT_REFUTATION_PHRASES:
        if phrase in text_lower:
            return {
                "relationship": "CONTRADICTS",
                "verdict": "CONTRADICTS",
                "entailment_state": "CONTRADICTS",
                "directness": "DIRECT",
                "confidence": 0.95,
                "reason": f"Source explicitly refutes or debunks this assertion ('{phrase}').",
                "conflict_type": "EXPLICIT_REFUTATION",
                "evidence_text": evidence_snippet,
            }

    # 3. Wire Disclaimer / Unverified Gate (Section 13)
    for phrase in NON_CONFIRMATION_PHRASES:
        if phrase in text_lower:
            return {
                "relationship": "NEUTRAL",
                "verdict": "INSUFFICIENT",
                "entailment_state": "UNKNOWN",
                "directness": "INDIRECT",
                "confidence": 0.35,
                "reason": f"Source notes it could not independently verify the claim ('{phrase}').",
                "conflict_type": None,
                "evidence_text": evidence_snippet,
            }

    # 4. Universal Action / Antonym Contradiction Detection (Section 9)
    # Check if the claim asserts an action and the evidence asserts an antonym or conflicting outcome
    for action_key, conflicting_set in ACTION_CONTRADICTION_MAP.items():
        if re.search(r"\b" + re.escape(action_key) + r"\b", claim_lower):
            for conflict_term in conflicting_set:
                if re.search(r"\b" + re.escape(conflict_term) + r"\b", text_lower):
                    return {
                        "relationship": "CONTRADICTS",
                        "verdict": "CONTRADICTS",
                        "entailment_state": "CONTRADICTS",
                        "directness": "DIRECT",
                        "confidence": 0.92,
                        "reason": f"Action conflict: claim asserts '{action_key}', but source reports '{conflict_term}'.",
                        "conflict_type": "ACTION_CONFLICT",
                        "evidence_text": evidence_snippet,
                    }

    # 5. Direct Negation Contradiction Detection (Section 9)
    # Example: claim asserts 'landed on moon', evidence asserts 'did not land' or 'will not land'
    negation_patterns = [
        r"\b(?:did\s+not|does\s+not|will\s+not|never|refused\s+to|failed\s+to|denied)\s+(\w+)\b"
    ]
    for pat in negation_patterns:
        neg_matches = re.findall(pat, text_lower)
        for neg_verb in neg_matches:
            if any(neg_verb in w for w in claim_words):
                return {
                    "relationship": "CONTRADICTS",
                    "verdict": "CONTRADICTS",
                    "entailment_state": "CONTRADICTS",
                    "directness": "DIRECT",
                    "confidence": 0.92,
                    "reason": f"Negation conflict: source explicitly states that the entity did not {neg_verb}.",
                    "conflict_type": "NEGATION_CONFLICT",
                    "evidence_text": evidence_snippet,
                }

    # 6. Structured Numerical Consistency Verification (Section 10)
    claim_metrics = extract_numbers_and_metrics(claim_clean)
    text_metrics = extract_numbers_and_metrics(text_clean)

    if claim_metrics and text_metrics:
        for c_val, c_unit in claim_metrics:
            for t_val, t_unit in text_metrics:
                if c_unit == t_unit and c_unit not in ("", "$", "€", "£"):
                    # Differing count/percentage on the same unit
                    if c_val != t_val:
                        return {
                            "relationship": "CONTRADICTS",
                            "verdict": "CONTRADICTS",
                            "entailment_state": "CONTRADICTS",
                            "directness": "DIRECT",
                            "confidence": 0.94,
                            "reason": f"Quantitative discrepancy: claim specifies {c_val:g} {c_unit}, but source reports {t_val:g} {t_unit}.",
                            "conflict_type": "NUMERIC_CONFLICT",
                            "evidence_text": evidence_snippet,
                        }
                elif c_unit in ("$", "€", "£") and t_unit == c_unit:
                    # Currency discrepancy ($5 billion vs $5 million)
                    if abs(c_val - t_val) / max(c_val, t_val, 1) > 0.1:
                        return {
                            "relationship": "CONTRADICTS",
                            "verdict": "CONTRADICTS",
                            "entailment_state": "CONTRADICTS",
                            "directness": "DIRECT",
                            "confidence": 0.94,
                            "reason": f"Financial discrepancy: claim specifies {c_unit}{c_val:g}, but source reports {t_unit}{t_val:g}.",
                            "conflict_type": "NUMERIC_CONFLICT",
                            "evidence_text": evidence_snippet,
                        }

    # 7. Structured Date & Temporal Consistency Verification (Section 11)
    claim_dates = extract_dates_and_years(claim_clean)
    text_dates = extract_dates_and_years(text_clean)

    # Historical year conflict (e.g. 2026 vs 2023)
    if claim_dates["years"] and text_dates["years"]:
        if not (claim_dates["years"] & text_dates["years"]):
            # Describing a different historical year
            return {
                "relationship": "UNRELATED",
                "verdict": "UNRELATED",
                "entailment_state": "UNKNOWN",
                "directness": "NONE",
                "confidence": 0.85,
                "reason": f"Historical event year mismatch: claim specifies {list(claim_dates['years'])[0]}, but source reports on {list(text_dates['years'])[0]}.",
                "conflict_type": "DATE_CONFLICT",
                "evidence_text": evidence_snippet,
            }

    # Specific day/month conflict within the same event (October 4 vs October 5)
    if claim_dates["full_dates"] and text_dates["full_dates"]:
        if not (claim_dates["full_dates"] & text_dates["full_dates"]):
            c_date = list(claim_dates["full_dates"])[0]
            t_date = list(text_dates["full_dates"])[0]
            return {
                "relationship": "CONTRADICTS",
                "verdict": "CONTRADICTS",
                "entailment_state": "CONTRADICTS",
                "directness": "DIRECT",
                "confidence": 0.90,
                "reason": f"Date mismatch: claim states event happened on {c_date}, but source reports {t_date}.",
                "conflict_type": "DATE_CONFLICT",
                "evidence_text": evidence_snippet,
            }

    # 8. Structured Location Verification
    claim_locs = extract_locations(claim_clean)
    text_locs = extract_locations(text_clean)
    if claim_locs and text_locs:
        if not (claim_locs & text_locs):
            # Known major city conflicts (Delhi vs Mumbai, London vs Paris, etc.)
            major_cities = {"delhi", "mumbai", "bengaluru", "kolkata", "chennai", "london", "paris", "new york", "washington", "tokyo", "beijing", "sriharikota"}
            if any(l in major_cities for l in claim_locs) and any(l in major_cities for l in text_locs):
                c_loc = list(claim_locs)[0].title()
                t_loc = list(text_locs)[0].title()
                return {
                    "relationship": "CONTRADICTS",
                    "verdict": "CONTRADICTS",
                    "entailment_state": "CONTRADICTS",
                    "directness": "DIRECT",
                    "confidence": 0.88,
                    "reason": f"Location conflict: claim states event happened in {c_loc}, but source reports {t_loc}.",
                    "conflict_type": "LOCATION_CONFLICT",
                    "evidence_text": evidence_snippet,
                }

    # 9. Topical Relevance vs. Exact Action Entailment (Section 7 & 8 - THE CORE FIX)
    # Check if the source is merely discussing, planning, or mentioning background without affirming the action
    for hyp in HYPOTHETICAL_MARKERS:
        if hyp in text_lower:
            return {
                "relationship": "NEUTRAL",
                "verdict": "INSUFFICIENT",
                "entailment_state": "UNKNOWN",
                "directness": "INDIRECT",
                "confidence": 0.40,
                "reason": f"Topical mention only: source indicates the topic was discussed or hypothetical ('{hyp}'), but does not confirm that the event actually occurred.",
                "conflict_type": None,
                "evidence_text": evidence_snippet,
            }

    # Find the main action verb in the claim
    claim_actions = []
    for word in claim_words:
        for group in UNIVERSAL_SYNONYMS:
            if word in group:
                claim_actions.append((word, group))
                break

    # If the claim specifies an action (e.g. 'launched', 'landed', 'resigned', 'won', 'signed', 'approved')
    if claim_actions:
        action_word, synonym_group = claim_actions[0]
        # Does the evidence contain the action or its direct synonyms?
        action_found = any(re.search(r"\b" + re.escape(syn) + r"\b", text_lower) for syn in synonym_group)
        if not action_found:
            # The source mentions the company/subject, but DOES NOT affirm the action!
            # Example: "Company X develops Product Y" for "Company X launched Product Y" -> NEUTRAL!
            return {
                "relationship": "NEUTRAL",
                "verdict": "INSUFFICIENT",
                "entailment_state": "UNKNOWN",
                "directness": "INDIRECT",
                "confidence": 0.45,
                "reason": f"Topical background only: source mentions the subject/topic, but does not confirm that it {action_word}.",
                "conflict_type": None,
                "evidence_text": evidence_snippet,
            }

    # 10. Affirmative Entailment Verification (SUPPORTS)
    # Both subject and action confirmed, with semantic word overlap and consistent modifiers
    matched_words = 0
    for w in claim_words:
        if w in text_lower:
            matched_words += 1
            continue
        for group in UNIVERSAL_SYNONYMS:
            if w in group and any(syn in text_lower for syn in group):
                matched_words += 1
                break

    overlap_ratio = matched_words / len(claim_words) if claim_words else 0

    if overlap_ratio >= 0.55 or (claim_actions and overlap_ratio >= 0.45):
        directness = "DIRECT" if overlap_ratio >= 0.65 else "INDIRECT"
        return {
            "relationship": "SUPPORTS",
            "verdict": "SUPPORTS",
            "entailment_state": "ENTAILS",
            "directness": directness,
            "confidence": 0.88 if directness == "DIRECT" else 0.76,
            "reason": f"Source directly corroborates and affirms the factual assertion ({matched_words}/{len(claim_words)} key terms and core predicate confirmed).",
            "conflict_type": None,
            "evidence_text": evidence_snippet,
        }
    elif overlap_ratio >= 0.25:
        return {
            "relationship": "NEUTRAL",
            "verdict": "INSUFFICIENT",
            "entailment_state": "UNKNOWN",
            "directness": "INDIRECT",
            "confidence": 0.35,
            "reason": "Source mentions related background context but does not contain enough specific detail to definitively confirm the assertion.",
            "conflict_type": None,
            "evidence_text": evidence_snippet,
        }
    else:
        return {
            "relationship": "UNRELATED",
            "verdict": "UNRELATED",
            "entailment_state": "UNKNOWN",
            "directness": "NONE",
            "confidence": 0.25,
            "reason": "Source text does not sufficiently address or establish the specific claim.",
            "conflict_type": None,
            "evidence_text": evidence_snippet,
        }
