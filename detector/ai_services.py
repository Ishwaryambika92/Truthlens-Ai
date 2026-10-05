"""
TruthLens AI - Unified Secure External API Adapter.

Integrates and securely configures:
  1. Google Gemini API (GEMINI_API_KEY) - Generative AI for summarization, claim structuring,
     grounded explanations, and content analysis.
  2. Google Fact Check Tools API (FACT_CHECK_API_KEY) - Accredited journalist fact-check search.
  3. Tavily Search API (TAVILY_API_KEY) - Real-time independent factual web search.
  4. Anthropic Claude API (ANTHROPIC_API_KEY) - Alternate/secondary LLM provider.
  5. Sarvam AI API (SARVAM_API_KEY) - Indic language translation & regional news support.

SECURITY & RESILIENCE RULES:
  - All keys are loaded from environment / .env via os.environ.
  - Never hard-code any API keys.
  - Never display, print, or log secret key values.
  - Missing or invalid keys degrade gracefully with clear user-safe messages.
  - Never crash the TruthLens application when external services are unreachable.
  - ML classification and external evidence verification remain strictly decoupled.
  - Never fabricate or guess evidence when an API is unavailable.
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

# Default model identifiers
GEMINI_MODELS = ["gemini-3.1-flash-lite-preview", "gemini-3.8-flash", "gemini-flash-latest"]
DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_TIMEOUT = 20


def get_secret(name: str) -> str:
    """Retrieve an environment variable key safely without whitespace."""
    val = os.environ.get(name, "").strip()
    if not val:
        if name == "GEMINI_API_KEY":
            val = os.getenv("GOOGLE_API_KEY", "").strip()
        elif name == "GOOGLE_API_KEY":
            val = os.getenv("GEMINI_API_KEY", "").strip()
    return val


# ---------------------------------------------------------------------------
# 1. Google Gemini API
# ---------------------------------------------------------------------------

def call_gemini(
    prompt: str,
    system_instruction: Optional[str] = None,
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Execute a text generation call via Google Gemini REST API.

    Returns:
        {"success": bool, "text": Optional[str], "error": Optional[str], "model": Optional[str]}
    """
    key = get_secret("GEMINI_API_KEY")
    if not key:
        return {
            "success": False,
            "text": None,
            "error": "Google Gemini API key is not configured (GEMINI_API_KEY missing).",
            "model": None,
        }

    last_error = None
    for model_name in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        contents = []
        if system_instruction:
            contents.append({
                "role": "user",
                "parts": [{"text": f"SYSTEM INSTRUCTION: {system_instruction}"}],
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I will strictly follow this instruction."}],
            })
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    extracted_text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
                    if extracted_text:
                        return {
                            "success": True,
                            "text": extracted_text,
                            "error": None,
                            "model": model_name,
                        }
            else:
                err_data = resp.json().get("error", {})
                err_msg = err_data.get("message", f"HTTP {resp.status_code}")
                last_error = f"Gemini ({model_name}): {err_msg}"
        except requests.exceptions.Timeout:
            last_error = f"Gemini ({model_name}) request timed out."
        except Exception as e:
            last_error = f"Gemini ({model_name}) error: {type(e).__name__}"

    return {
        "success": False,
        "text": None,
        "error": last_error or "Gemini API request failed.",
        "model": None,
    }


# ---------------------------------------------------------------------------
# 2. Anthropic Claude API
# ---------------------------------------------------------------------------

def call_anthropic(
    prompt: str,
    system_instruction: Optional[str] = None,
    max_tokens: int = 800,
) -> Dict[str, Any]:
    """Execute a text generation call via Anthropic Claude REST API.

    Returns:
        {"success": bool, "text": Optional[str], "error": Optional[str]}
    """
    key = get_secret("ANTHROPIC_API_KEY")
    if not key:
        return {
            "success": False,
            "text": None,
            "error": "Anthropic API key is not configured (ANTHROPIC_API_KEY missing).",
        }

    url = "https://api.anthropic.com/v1/messages"
    payload: Dict[str, Any] = {
        "model": DEFAULT_ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_instruction:
        payload["system"] = system_instruction

    try:
        resp = requests.post(
            url,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            text = "\n".join(blocks).strip()
            return {"success": True, "text": text, "error": None}
        else:
            err_msg = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
            return {"success": False, "text": None, "error": f"Anthropic error: {err_msg}"}
    except requests.exceptions.Timeout:
        return {"success": False, "text": None, "error": "Anthropic request timed out."}
    except Exception as e:
        return {"success": False, "text": None, "error": f"Anthropic connection error: {type(e).__name__}"}


# ---------------------------------------------------------------------------
# Unified LLM Provider: tries Gemini first (fast/active), then Anthropic
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    max_tokens: int = 800,
) -> Dict[str, Any]:
    """Call the active LLM provider. Priority: Gemini -> Anthropic -> Fallback."""
    # 1. Try Google Gemini if configured
    if get_secret("GEMINI_API_KEY"):
        res = call_gemini(prompt, system_instruction, max_tokens=max_tokens)
        if res["success"]:
            res["provider"] = "google_gemini"
            return res

    # 2. Try Anthropic if configured
    if get_secret("ANTHROPIC_API_KEY"):
        res = call_anthropic(prompt, system_instruction, max_tokens=max_tokens)
        if res["success"]:
            res["provider"] = "anthropic_claude"
            return res

    return {
        "success": False,
        "text": None,
        "provider": None,
        "error": "No LLM provider available (configure GEMINI_API_KEY or ANTHROPIC_API_KEY in .env).",
    }


# ---------------------------------------------------------------------------
# 3. Tavily Search API (Factual Web Search)
# ---------------------------------------------------------------------------

def search_tavily(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web for factual candidate sources using Tavily API.

    Returns:
        List of {"title": str, "url": str, "content": str}
    """
    key = get_secret("TAVILY_API_KEY")
    if not key or not query.strip():
        return []

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": key,
        "query": query.strip(),
        "search_depth": "basic",
        "max_results": max_results,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return [
                {
                    "title": r.get("title", "").strip(),
                    "url": r.get("url", "").strip(),
                    "content": r.get("content", "").strip(),
                }
                for r in data.get("results", [])
                if r.get("url")
            ]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# 4. Google Fact Check Tools API
# ---------------------------------------------------------------------------

def align_fact_checks_to_claims(article_claims: List[str], fact_checks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Determine whether each fact-check review SUPPORTS, CONTRADICTS, or is UNRELATED to the article's claims."""
    if not fact_checks or not article_claims:
        return fact_checks

    # 1. Try AI-powered alignment (Gemini/Claude)
    try:
        fc_block = "\n".join(
            f"{i+1}. Investigated Claim: \"{fc.get('claim', '')}\" | Rating: \"{fc.get('status', '')}\" (Publisher: {fc.get('source', '')})"
            for i, fc in enumerate(fact_checks)
        )
        claims_block = "\n".join(f"- {c}" for c in article_claims[:5])

        prompt = (
            "You are an objective fact-checking alignment analyst.\n\n"
            f"ARTICLE CLAIMS:\n{claims_block}\n\n"
            f"RETRIEVED FACT-CHECK REVIEWS:\n{fc_block}\n\n"
            "For EACH fact-check review above, determine its relationship to the ARTICLE CLAIMS:\n"
            "- CONTRADICTS: The fact-check proves that a claim made in the article is false, fabricated, or debunked.\n"
            "- SUPPORTS: The fact-check confirms the article's claims, OR debunks a counter-claim/hoax that questioned the article's factual event (e.g. debunking 'Apollo 11 was staged' supports the article stating Apollo 11 landed on the Moon).\n"
            "- UNRELATED: The fact-check is about an entirely different topic or secondary joke not asserted in the article.\n\n"
            "Respond ONLY with a JSON array of objects, one per fact-check in order:\n"
            "[{\"index\": 1, \"alignment\": \"CONTRADICTS\" | \"SUPPORTS\" | \"UNRELATED\", \"reason\": \"brief explanation\"}]"
        )
        res = call_llm(prompt, max_tokens=600)
        if res.get("success") and res.get("text"):
            raw = res["text"].strip()
            bracket_match = re.search(r"(\[.*\])", raw, re.DOTALL)
            if bracket_match:
                parsed = json.loads(bracket_match.group(1))
                if isinstance(parsed, list) and len(parsed) == len(fact_checks):
                    for fc, item in zip(fact_checks, parsed):
                        align = str(item.get("alignment", "UNRELATED")).upper()
                        if align in ("CONTRADICTS", "SUPPORTS", "UNRELATED"):
                            fc["alignment"] = align
                            fc["alignment_reason"] = item.get("reason", "")
                    return fact_checks
    except Exception:
        pass

    # 2. Heuristic fallback
    user_text_lower = " ".join(article_claims).lower()
    user_has_hoax_words = any(h in user_text_lower for h in ("fake", "hoax", "staged", "conspiracy", "animation", "fraud"))

    for fc in fact_checks:
        fc_claim_lower = (fc.get("claim") or "").lower()
        rating_lower = (fc.get("status") or "").lower()
        is_false_rating = any(k in rating_lower for k in ("false", "fake", "pants on fire", "incorrect", "hoax", "debunked"))
        is_true_rating = any(k in rating_lower for k in ("true", "correct", "accurate", "verified", "real"))
        is_hoax_counterclaim = any(h in fc_claim_lower for h in ("fake", "hoax", "staged", "conspiracy", "animation", "fraud", "flubbed"))

        if is_false_rating:
            if is_hoax_counterclaim and not user_has_hoax_words:
                fc["alignment"] = "SUPPORTS"
                fc["alignment_reason"] = "Fact-checker debunked a conspiracy theory/counterclaim questioning the real event."
            else:
                fc["alignment"] = "CONTRADICTS"
                fc["alignment_reason"] = "Fact-checker rated the claimed assertion as false or debunked."
        elif is_true_rating:
            if is_hoax_counterclaim and not user_has_hoax_words:
                fc["alignment"] = "CONTRADICTS"
                fc["alignment_reason"] = "Fact-checker confirmed a counterclaim contrary to the article."
            else:
                fc["alignment"] = "SUPPORTS"
                fc["alignment_reason"] = "Fact-checker confirmed the accuracy of this assertion."
        else:
            fc["alignment"] = "UNRELATED"
            fc["alignment_reason"] = "Fact-check was not a definitive match."

    return fact_checks


def verify_google_factcheck(query: str, claims_list: Optional[List[str]] = None) -> Dict[str, Any]:
    """Query Google Fact Check Tools API for accredited fact-check reviews.
    Accepts either an article text/query or a list of extracted central claims."""
    key = get_secret("FACT_CHECK_API_KEY")
    if not key:
        return {
            "available": False,
            "claims": [],
            "message": "Fact verification unavailable - external verification service is not configured (FACT_CHECK_API_KEY missing).",
        }

    queries_to_try = []
    effective_claims = claims_list or ([query] if query else [])
    if claims_list:
        for c in claims_list[:3]:
            c_clean = " ".join(c.split()[:12]).strip()
            if c_clean and c_clean not in queries_to_try:
                queries_to_try.append(c_clean)

    if not queries_to_try and query:
        # Extract first sentence or up to 10 words
        first_sentence = re.split(r"(?<=[.!?])\s+", query.strip())[0] if query.strip() else ""
        clean_query = " ".join((first_sentence or query).split()[:10]).strip()
        if clean_query:
            queries_to_try.append(clean_query)

    if not queries_to_try:
        return {"available": True, "claims": [], "message": "No specific claims provided to verify against fact-check database."}

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    collected_claims = []
    seen_urls = set()

    for q in queries_to_try:
        try:
            resp = requests.get(url, params={"query": q, "key": key}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for c in data.get("claims", [])[:4]:
                    review = (c.get("claimReview") or [{}])[0]
                    rev_url = review.get("url", "")
                    if rev_url and rev_url in seen_urls:
                        continue
                    if rev_url:
                        seen_urls.add(rev_url)
                    collected_claims.append({
                        "claim": c.get("text", ""),
                        "status": review.get("textualRating", "UNVERIFIED"),
                        "source": review.get("publisher", {}).get("name", "Unknown source"),
                        "url": rev_url,
                    })
        except Exception:
            continue

    if not collected_claims:
        return {
            "available": True,
            "claims": [],
            "message": "No matching fact-checks were found for this article's claims.",
        }

    aligned = align_fact_checks_to_claims(effective_claims, collected_claims[:6])
    return {"available": True, "claims": aligned, "message": None}


# ---------------------------------------------------------------------------
# 5. Sarvam AI API (Indic Language Translation)
# ---------------------------------------------------------------------------

def translate_indic(
    text: str,
    source_language_code: str = "hi-IN",
    target_language_code: str = "en-IN",
) -> Dict[str, Any]:
    """Translate Indic language text to English using Sarvam AI.

    Supported source codes include:
      hi-IN (Hindi), bn-IN (Bengali), ta-IN (Tamil), te-IN (Telugu),
      mr-IN (Marathi), gu-IN (Gujarati), kn-IN (Kannada), ml-IN (Malayalam),
      pa-IN (Punjabi), od-IN (Odia).
    """
    key = get_secret("SARVAM_API_KEY") or get_secret("SARVAM_AI_API_KEY")
    if not key:
        return {
            "success": False,
            "translated_text": text,
            "message": "Sarvam AI is not configured (set SARVAM_API_KEY in .env for Indic language support).",
        }

    url = "https://api.sarvam.ai/translate"
    payload = {
        "input": text[:4000],
        "source_language_code": source_language_code,
        "target_language_code": target_language_code,
        "mode": "formal",
        "model": "mayura:v1",
    }
    try:
        resp = requests.post(
            url,
            headers={"api-subscription-key": key, "content-type": "application/json"},
            json=payload,
            timeout=20,
        )
        if resp.status_code == 200:
            data = resp.json()
            translated = data.get("translated_text", "").strip()
            return {"success": True, "translated_text": translated or text, "message": None}
        else:
            return {
                "success": False,
                "translated_text": text,
                "message": f"Sarvam AI translation error (HTTP {resp.status_code}).",
            }
    except Exception as e:
        return {
            "success": False,
            "translated_text": text,
            "message": f"Sarvam AI connection failed: {type(e).__name__}.",
        }


# ---------------------------------------------------------------------------
# API Health and Connectivity Checker (No secrets displayed)
# ---------------------------------------------------------------------------

def check_all_api_status() -> Dict[str, Dict[str, Any]]:
    """Test and report connectivity for all configured external APIs.

    NEVER returns or logs secret keys.
    """
    status: Dict[str, Dict[str, Any]] = {}

    # 1. Google Fact Check Tools API
    fact_key = get_secret("FACT_CHECK_API_KEY")
    if not fact_key:
        status["Google Fact Check Tools API"] = {
            "configured": False,
            "connected": False,
            "purpose": "Authoritative claim verification against accredited fact-checkers",
            "notes": "FACT_CHECK_API_KEY is not set in .env",
        }
    else:
        try:
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            r = requests.get(url, params={"query": "health", "key": fact_key}, timeout=10)
            status["Google Fact Check Tools API"] = {
                "configured": True,
                "connected": r.status_code == 200,
                "purpose": "Authoritative claim verification against accredited fact-checkers",
                "notes": "Verified working (200 OK)" if r.status_code == 200 else f"HTTP {r.status_code}",
            }
        except Exception as e:
            status["Google Fact Check Tools API"] = {
                "configured": True,
                "connected": False,
                "purpose": "Authoritative claim verification against accredited fact-checkers",
                "notes": f"Connection error: {type(e).__name__}",
            }

    # 2. Google Gemini API
    gemini_key = get_secret("GEMINI_API_KEY")
    if not gemini_key:
        status["Google Gemini API"] = {
            "configured": False,
            "connected": False,
            "purpose": "Generative AI for article summarization, claim structuring, and assistant Q&A",
            "notes": "GEMINI_API_KEY is not set in .env",
        }
    else:
        res = call_gemini("Ping", max_tokens=5)
        status["Google Gemini API"] = {
            "configured": True,
            "connected": res["success"],
            "purpose": "Generative AI for article summarization, claim structuring, and assistant Q&A",
            "notes": f"Connected ({res.get('model')})" if res["success"] else res.get("error", "Failed"),
        }

    # 3. Tavily Search API
    tavily_key = get_secret("TAVILY_API_KEY")
    if not tavily_key:
        status["Tavily Search API"] = {
            "configured": False,
            "connected": False,
            "purpose": "Independent live web evidence retrieval for extracted claims",
            "notes": "TAVILY_API_KEY is not set in .env",
        }
    else:
        results = search_tavily("NASA news", max_results=1)
        status["Tavily Search API"] = {
            "configured": True,
            "connected": len(results) > 0,
            "purpose": "Independent live web evidence retrieval for extracted claims",
            "notes": "Verified working (live web search active)" if len(results) > 0 else "No search results returned",
        }

    # 4. Anthropic Claude API
    anthropic_key = get_secret("ANTHROPIC_API_KEY")
    if not anthropic_key:
        status["Anthropic Claude API"] = {
            "configured": False,
            "connected": False,
            "purpose": "Secondary/alternate LLM provider for summarization and Q&A",
            "notes": "ANTHROPIC_API_KEY is not set in .env",
        }
    else:
        res = call_anthropic("Ping", max_tokens=5)
        status["Anthropic Claude API"] = {
            "configured": True,
            "connected": res["success"],
            "purpose": "Secondary/alternate LLM provider for summarization and Q&A",
            "notes": "Connected" if res["success"] else res.get("error", "Failed"),
        }

    # 5. Sarvam AI API
    sarvam_key = get_secret("SARVAM_API_KEY") or get_secret("SARVAM_AI_API_KEY")
    if not sarvam_key:
        status["Sarvam AI API"] = {
            "configured": False,
            "connected": False,
            "purpose": "Indic regional language detection & translation (Hindi, Bengali, Tamil, etc.)",
            "notes": "SARVAM_API_KEY is not set in .env (optional for Indic languages)",
        }
    else:
        res = translate_indic("नमस्ते", source_language_code="hi-IN", target_language_code="en-IN")
        status["Sarvam AI API"] = {
            "configured": True,
            "connected": res["success"],
            "purpose": "Indic regional language detection & translation (Hindi, Bengali, Tamil, etc.)",
            "notes": "Connected" if res["success"] else res.get("message", "Failed"),
        }

    return status
