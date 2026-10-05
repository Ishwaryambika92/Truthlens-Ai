"""
Indic Language Translation Service for TruthLens AI.

Powered by Sarvam AI (api.sarvam.ai).
Enables automated translation of regional Indian news (Hindi, Bengali, Tamil,
Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia) to English
so that the downstream verification pipeline and ML models can analyze them.

Gated behind SARVAM_API_KEY. Gracefully skips translation if not configured
or if text is already English / Roman script.
"""
import re
from typing import Any, Dict
from . import ai_services

# Unicode ranges for Indic scripts:
# Devanagari: \u0900-\u097F
# Bengali:    \u0980-\u09FF
# Gurmukhi:   \u0A00-\u0A7F
# Gujarati:   \u0A80-\u0AFF
# Oriya:      \u0B00-\u0B7F
# Tamil:      \u0B80-\u0BFF
# Telugu:     \u0C00-\u0C7F
# Kannada:    \u0C80-\u0CFF
# Malayalam:  \u0D00-\u0D7F
INDIC_UNICODE_PATTERN = re.compile(r"[\u0900-\u0D7F]")


def detect_indic_script(text: str) -> bool:
    """Checks whether the text contains a notable proportion of Indic script characters."""
    if not text:
        return False
    indic_chars = len(INDIC_UNICODE_PATTERN.findall(text))
    return (indic_chars / max(len(text), 1)) > 0.05


def translate_if_indic(text: str) -> Dict[str, Any]:
    """Translates Indic language text to English if Sarvam AI is configured.

    Returns:
        {
            "translated": bool,
            "original_text": str,
            "processed_text": str,
            "message": Optional[str],
            "sarvam_available": bool,
        }
    """
    if not detect_indic_script(text):
        return {
            "translated": False,
            "original_text": text,
            "processed_text": text,
            "message": None,
            "sarvam_available": bool(ai_services.get_secret("SARVAM_API_KEY") or ai_services.get_secret("SARVAM_AI_API_KEY")),
        }

    key = ai_services.get_secret("SARVAM_API_KEY") or ai_services.get_secret("SARVAM_AI_API_KEY")
    if not key:
        return {
            "translated": False,
            "original_text": text,
            "processed_text": text,
            "message": "Indic script detected. To enable automated translation, configure SARVAM_API_KEY in .env.",
            "sarvam_available": False,
        }

    res = ai_services.translate_indic(text)
    if res.get("success") and res.get("translated_text"):
        return {
            "translated": True,
            "original_text": text,
            "processed_text": res["translated_text"],
            "message": "Article successfully translated from Indic script via Sarvam AI.",
            "sarvam_available": True,
        }

    return {
        "translated": False,
        "original_text": text,
        "processed_text": text,
        "message": f"Sarvam AI translation could not be completed: {res.get('message')}",
        "sarvam_available": True,
    }
