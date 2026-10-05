"""
Layer 2 - Article quality / validity check.

Runs BEFORE any ML/verification work. If the article isn't usable, we
return that plainly (resulting in UNCERTAIN) instead of forcing a REAL/FAKE
call on content that was never enough to judge in the first place.
"""
import re

MIN_RELIABLE_CHARS = 150   # below this, ML/claim signals are unreliable
MIN_RELIABLE_WORDS = 25
MAX_REPEATED_LINE_RATIO = 0.4  # if >40% of lines are exact repeats, likely spam/junk


def check_article_quality(text: str, metadata: dict | None = None) -> dict:
    """Returns {"usable": bool, "reasons": [...], "warnings": [...]}.

    `usable=False` means the pipeline should stop and report
    "Insufficient content for reliable verification" (assessment: UNCERTAIN)
    rather than proceeding.
    `warnings` are non-blocking (e.g. missing author/date) that later layers
    can factor in without refusing to analyze.
    """
    text = (text or "").strip()
    reasons = []
    warnings = []

    if not text:
        return {"usable": False, "reasons": ["No article text was provided or extracted."], "warnings": []}

    char_count = len(text)
    word_count = len(text.split())

    if char_count < MIN_RELIABLE_CHARS or word_count < MIN_RELIABLE_WORDS:
        reasons.append(
            f"The article is too short for reliable verification "
            f"({word_count} words, {char_count} characters; minimum required is {MIN_RELIABLE_WORDS} words)."
        )

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 5:
        unique_ratio = len(set(lines)) / len(lines)
        if unique_ratio < (1 - MAX_REPEATED_LINE_RATIO):
            reasons.append("The content contains excessive repeated text and does not look like a normal article.")

    non_alpha_ratio = sum(1 for c in text if not (c.isalnum() or c.isspace())) / max(len(text), 1)
    if non_alpha_ratio > 0.35:
        reasons.append("The content is dominated by symbols/formatting rather than readable prose.")

    if metadata:
        if not metadata.get("title"):
            warnings.append("No article title could be detected.")
        if not metadata.get("author"):
            warnings.append("No author information could be detected.")
        if not metadata.get("published_date"):
            warnings.append("No publication date could be detected.")
        if metadata.get("is_https") is False:
            warnings.append("The source site does not use HTTPS.")

    opinion_phrases = [
        "it is time we", "it's time we", "i believe", "in my opinion", "in my view",
        "i think", "we must realize", "honest conversation about how",
    ]
    lower_text = text.lower()
    is_opinion = sum(1 for p in opinion_phrases if p in lower_text) >= 2
    if is_opinion:
        warnings.append("Content appears to be an opinion piece or editorial commentary.")

    return {"usable": len(reasons) == 0, "reasons": reasons, "warnings": warnings, "is_opinion": is_opinion}
