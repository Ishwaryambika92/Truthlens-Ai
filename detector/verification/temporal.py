"""
Temporal context detection for TruthLens AI.

Distinguishes between:
  - "current": present / recent news (e.g. 2023-2026, breaking news, ongoing events)
  - "historical": past news (e.g. 2015-2020, matching the era of the training corpus)
  - "unspecified": articles where no reliable temporal anchor was detected

Why this matters:
  The core ML model was trained on historical news articles (circa 2015-2017).
  For historical news, ML style signals are informative alongside available evidence.
  For current/present news, ML style classification alone cannot verify new events
  it has never seen; current news must rely primarily on external evidence,
  fact-checking, and multi-source corroboration, and report UNCERTAIN if no
  corroborating evidence can be found.
"""
import re
from datetime import datetime

CURRENT_YEAR_MIN = 2023
HISTORICAL_YEAR_MAX = 2020

RECENT_YEAR_PATTERN = re.compile(r"\b(202[3-9]|203[0-9])\b")
HISTORICAL_YEAR_PATTERN = re.compile(r"\b(201[0-9]|202[0-1])\b")

CURRENT_PHRASES = [
    "breaking news", "this morning", "yesterday", "this week", "earlier today",
    "announced today", "announced this week", "just released", "in recent days",
    "earlier this month", "current administration", "ongoing investigation",
    "at a press briefing today", "press release today", "spokesperson said today",
    "eastern frontline", "frontline command post", "zelenskyy", "volodymyr zelenskyy",
    "zelensky", "ukraine war", "russian invasion", "covid-19", "coronavirus",
    "chatgpt", "openai", "generative ai", "rishi sunak", "keir starmer", "king charles",
    "chandrayaan", "artemis program", "deeptech fund",
]


HISTORICAL_PHRASES = [
    "back in 2015", "back in 2016", "2016 presidential election", "2016 campaign",
    "during the 2016", "during the 2015", "during the 2017", "in 2016,", "in 2015,",
    "historic 2016", "the 2016 race",
]


def detect_temporal_context(text: str, metadata: dict | None = None, explicit_choice: str | None = None) -> dict:
    """Detects whether an article is current/recent or historical news.

    Returns:
        {
            "context": "current" | "historical" | "unspecified",
            "label": str,
            "detected_year": int | None,
            "confidence": "high" | "medium" | "low",
            "reasons": [str, ...]
        }
    """
    metadata = metadata or {}
    text = text or ""
    lower = text.lower()
    reasons = []

    # 1. Explicit user selection (overrides automatic detection if set)
    choice = (explicit_choice or metadata.get("temporal_preference") or "").strip().lower()
    if choice in ("current", "historical"):
        label = "Current / Present News" if choice == "current" else "Historical / Past News"
        return {
            "context": choice,
            "label": label,
            "detected_year": None,
            "confidence": "high",
            "reasons": [f"User explicitly specified this article as {label.lower()}."],
        }

    detected_year = None

    # 2. Check metadata published_date
    pub_date_str = str(metadata.get("published_date") or "").strip()
    if pub_date_str:
        # Search for 4-digit year in date string
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", pub_date_str)
        if year_match:
            try:
                year = int(year_match.group(1))
                detected_year = year
                if year >= CURRENT_YEAR_MIN:
                    reasons.append(f"Publication date indicates year {year} (current era >= {CURRENT_YEAR_MIN}).")
                    return {
                        "context": "current",
                        "label": "Current / Present News",
                        "detected_year": year,
                        "confidence": "high",
                        "reasons": reasons,
                    }
                elif year <= HISTORICAL_YEAR_MAX:
                    reasons.append(f"Publication date indicates year {year} (historical era <= {HISTORICAL_YEAR_MAX}).")
                    return {
                        "context": "historical",
                        "label": "Historical / Past News",
                        "detected_year": year,
                        "confidence": "high",
                        "reasons": reasons,
                    }
            except ValueError:
                pass

    # 3. Analyze text for explicit year references
    recent_years = [int(y) for y in RECENT_YEAR_PATTERN.findall(text)]
    hist_years = [int(y) for y in HISTORICAL_YEAR_PATTERN.findall(text)]

    recent_hits = [p for p in CURRENT_PHRASES if p in lower]
    hist_hits = [p for p in HISTORICAL_PHRASES if p in lower]

    # Evaluate indicators
    if recent_years and not hist_years:
        detected_year = max(recent_years)
        reasons.append(f"Article text references recent year(s): {sorted(set(recent_years))}.")
        if recent_hits:
            reasons.append(f"Recent event phrasing detected: {', '.join(recent_hits[:3])}.")
        return {
            "context": "current",
            "label": "Current / Present News",
            "detected_year": detected_year,
            "confidence": "high" if len(recent_years) >= 2 or recent_hits else "medium",
            "reasons": reasons,
        }

    if hist_years and not recent_years:
        detected_year = min(hist_years)
        reasons.append(f"Article text references historical year(s): {sorted(set(hist_years))}.")
        if hist_hits:
            reasons.append(f"Historical context phrasing detected: {', '.join(hist_hits[:2])}.")
        return {
            "context": "historical",
            "label": "Historical / Past News",
            "detected_year": detected_year,
            "confidence": "high" if len(hist_years) >= 2 or hist_hits else "medium",
            "reasons": reasons,
        }

    if recent_years and hist_years:
        # Both present - count frequency and look at recent hits
        if len(recent_years) > len(hist_years) or recent_hits:
            detected_year = max(recent_years)
            reasons.append(f"Article primarily references recent year {detected_year} with current temporal markers.")
            return {
                "context": "current",
                "label": "Current / Present News",
                "detected_year": detected_year,
                "confidence": "medium",
                "reasons": reasons,
            }
        else:
            detected_year = min(hist_years)
            reasons.append(f"Article primarily references historical era {detected_year}.")
            return {
                "context": "historical",
                "label": "Historical / Past News",
                "detected_year": detected_year,
                "confidence": "medium",
                "reasons": reasons,
            }

    if recent_hits:
        reasons.append(f"Recent news phrasing detected: {', '.join(recent_hits[:3])}.")
        return {
            "context": "current",
            "label": "Current / Present News",
            "detected_year": None,
            "confidence": "medium",
            "reasons": reasons,
        }

    if hist_hits:
        reasons.append(f"Historical news phrasing detected: {', '.join(hist_hits[:2])}.")
        return {
            "context": "historical",
            "label": "Historical / Past News",
            "detected_year": None,
            "confidence": "medium",
            "reasons": reasons,
        }

    # 4. Fallback: unspecified date
    reasons.append("No explicit publication date or clear temporal anchors were found in the text.")
    return {
        "context": "unspecified",
        "label": "Unspecified Date",
        "detected_year": None,
        "confidence": "low",
        "reasons": reasons,
    }
