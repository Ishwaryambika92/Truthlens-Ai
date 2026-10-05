import os
import sys
import django

# Setup Django
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "truthlens.settings")
django.setup()

from detector.verification import aggregation, source_analysis, web_evidence

print("=" * 80)
print("TRUTHLENS AI - EVIDENCE VERIFICATION PIPELINE AUDIT")
print("=" * 80)

def print_result(label, res):
    print(f"\n[{label}]")
    print(f"Final Assessment: {res['final_assessment']}")
    print(f"Verification Confidence: {res['verification_confidence']} ({res['verification_confidence_score']}%)")
    print(f"ML Prediction: {res.get('ml_prediction')} (Confidence: {res.get('ml_confidence')}%)")
    ev = res.get("evidence_found", {})
    print(f"Evidence Found: Supported={ev.get('supported_count')}, Contradicted={ev.get('contradicted_count')}, Insufficient={ev.get('insufficient_count')}")
    print(f"Total Strength: Support={ev.get('total_support_strength')}, Contra={ev.get('total_contra_strength')}")
    print(f"Strong Sources: Support={ev.get('strong_support_count')}, Contra={ev.get('strong_contra_count')}, Primary Support={ev.get('primary_support_count')}")
    print("Reasoning:")
    for r in res.get("reasoning", []):
        print(f"  • {r}")

# ---------------------------------------------------------
# Test A: Known genuine current news
# ---------------------------------------------------------
test_a_ev = [{
    "claim": "World Health Organization released global air quality guidelines.",
    "status": "SUPPORTED",
    "available": True,
    "sources": [
        {
            "url": "https://who.int/news/item/air-quality-guidelines",
            "domain": "who.int",
            "publisher": "World Health Organization",
            "retrieved": True,
            "is_primary_source": True,
            "source_reliability": "VERY_HIGH",
            "source_reliability_weight": 2.0,
            "content_verdict": "SUPPORTS",
            "verdict": "SUPPORTS",
            "directness": "DIRECT",
            "strength_score": 2.0,
            "strength_label": "STRONG",
            "reason": "Official WHO announcement confirms the release.",
        },
        {
            "url": "https://bbc.com/news/health-guidelines",
            "domain": "bbc.com",
            "publisher": "BBC News",
            "retrieved": True,
            "is_primary_source": False,
            "source_reliability": "HIGH",
            "source_reliability_weight": 1.2,
            "content_verdict": "SUPPORTS",
            "verdict": "SUPPORTS",
            "directness": "DIRECT",
            "strength_score": 1.2,
            "strength_label": "STRONG",
            "reason": "BBC independently reports on the new safety thresholds.",
        }
    ],
    "summary": "Corroborated by official WHO portal and BBC News."
}]

res_a = aggregation.aggregate_signals(
    quality_result={"usable": True, "reasons": [], "warnings": []},
    prediction_result={"prediction": "REAL", "raw_model_label": "REAL", "confidence": 85.0, "confidence_display": "85.0%", "error": None},
    clickbait_result={"score": 12, "classification": "LOW", "signals": []},
    ai_text_result={"label": "Human", "signals": {}},
    source_result={"available": True, "domain": "who.int", "reference_list_match": "established_news_organization", "signals": []},
    fact_check_result={"available": False, "claims": []},
    evidence_result={"claim_evidence": test_a_ev},
    temporal_info={"context": "current", "label": "October 2026", "reasons": []}
)
print_result("TEST A: Known Genuine Current News", res_a)

# ---------------------------------------------------------
# Test B: Known fake news
# ---------------------------------------------------------
test_b_ev = [{
    "claim": "Government issued mandatory microchip implantation decree.",
    "status": "CONTRADICTED",
    "available": True,
    "sources": [
        {
            "url": "https://snopes.com/fact-check/mandatory-microchips",
            "domain": "snopes.com",
            "publisher": "Snopes Fact Check",
            "retrieved": True,
            "is_primary_source": False,
            "source_reliability": "HIGH",
            "source_reliability_weight": 1.2,
            "content_verdict": "CONTRADICTS",
            "verdict": "CONTRADICTS",
            "directness": "DIRECT",
            "strength_score": 1.2,
            "strength_label": "STRONG",
            "reason": "Snopes investigation confirms the claim is a fabricated rumor.",
        }
    ],
    "summary": "Directly debunked by Snopes."
}]

res_b = aggregation.aggregate_signals(
    quality_result={"usable": True, "reasons": [], "warnings": []},
    prediction_result={"prediction": "REAL", "raw_model_label": "REAL", "confidence": 62.0, "confidence_display": "62.0%", "error": None},
    clickbait_result={"score": 85, "classification": "HIGH", "signals": []},
    ai_text_result={"label": "Human", "signals": {}},
    source_result={"available": False, "signals": []},
    fact_check_result={
        "available": True,
        "claims": [{
            "claim": "Government issued mandatory microchip decree.",
            "status": "False",
            "source": "Snopes",
            "url": "https://snopes.com/fact-check/mandatory-microchips",
            "alignment": "CONTRADICTS"
        }]
    },
    evidence_result={"claim_evidence": test_b_ev},
    temporal_info={"context": "current", "label": "Viral Message", "reasons": []}
)
print_result("TEST B: Known Fake News", res_b)

# ---------------------------------------------------------
# Test C: Ambiguous news
# ---------------------------------------------------------
test_c_ev = [{
    "claim": "Local community center proposes rooftop telescope installation.",
    "status": "UNVERIFIED",
    "available": True,
    "sources": [],
    "summary": "No verified external reporting found."
}]

res_c = aggregation.aggregate_signals(
    quality_result={"usable": True, "reasons": [], "warnings": []},
    prediction_result={"prediction": "UNCERTAIN", "raw_model_label": "FAKE", "confidence": 53.0, "confidence_display": "53.0%", "error": None},
    clickbait_result={"score": 10, "classification": "LOW", "signals": []},
    ai_text_result={"label": "Human", "signals": {}},
    source_result={"available": False, "signals": []},
    fact_check_result={"available": False, "claims": []},
    evidence_result={"claim_evidence": test_c_ev},
    temporal_info={"context": "current", "label": "Recent Proposal", "reasons": []}
)
print_result("TEST C: Ambiguous News", res_c)

# ---------------------------------------------------------
# Test D: ML style and external evidence disagree
# 4 strong supporting sources vs 3 weak contradictory blogs
# ML predicts FAKE with 98.9% confidence
# ---------------------------------------------------------
test_d_supporting = [
    {
        "url": "https://delhi.gov.in/press-release/water-maintenance",
        "domain": "delhi.gov.in",
        "publisher": "Delhi Jal Board (Govt of Delhi)",
        "retrieved": True,
        "is_primary_source": True,
        "source_reliability": "VERY_HIGH",
        "source_reliability_weight": 2.0,
        "content_verdict": "SUPPORTS",
        "verdict": "SUPPORTS",
        "directness": "DIRECT",
        "strength_score": 2.0,
        "strength_label": "STRONG",
        "reason": "Official municipal gazette confirms 36-hour planned supply shutdown for pipeline interconnection.",
    },
    {
        "url": "https://thehindu.com/news/cities/Delhi/water-supply-maintenance",
        "domain": "thehindu.com",
        "publisher": "The Hindu",
        "retrieved": True,
        "is_primary_source": False,
        "source_reliability": "MEDIUM",
        "source_reliability_weight": 0.75,
        "content_verdict": "SUPPORTS",
        "verdict": "SUPPORTS",
        "directness": "DIRECT",
        "strength_score": 0.75,
        "strength_label": "MODERATE",
        "reason": "The Hindu reports water shutdown schedule across South Delhi colonies.",
    },
    {
        "url": "https://timesofindia.indiatimes.com/city/delhi/water-cut-schedule",
        "domain": "timesofindia.com",
        "publisher": "Times of India",
        "retrieved": True,
        "is_primary_source": False,
        "source_reliability": "MEDIUM",
        "source_reliability_weight": 0.75,
        "content_verdict": "SUPPORTS",
        "verdict": "SUPPORTS",
        "directness": "DIRECT",
        "strength_score": 0.75,
        "strength_label": "MODERATE",
        "reason": "Times of India details emergency tanker helpline and affected sectors.",
    },
    {
        "url": "https://ndtv.com/delhi-news/water-supply-hit",
        "domain": "ndtv.com",
        "publisher": "NDTV",
        "retrieved": True,
        "is_primary_source": False,
        "source_reliability": "MEDIUM",
        "source_reliability_weight": 0.75,
        "content_verdict": "SUPPORTS",
        "verdict": "SUPPORTS",
        "directness": "DIRECT",
        "strength_score": 0.75,
        "strength_label": "MODERATE",
        "reason": "NDTV reports civic maintenance work underway.",
    },
]

test_d_contradicting = [
    {
        "url": "https://unverified-blogger-xyz.com/post/water-panic",
        "domain": "unverified-blogger-xyz.com",
        "publisher": "Personal Blog",
        "retrieved": True,
        "is_primary_source": False,
        "source_reliability": "LOW",
        "source_reliability_weight": 0.3,
        "content_verdict": "CONTRADICTS",
        "verdict": "CONTRADICTS",
        "directness": "INDIRECT",
        "strength_score": 0.2,
        "strength_label": "WEAK",
        "reason": "Blog questions whether notice is genuine.",
    },
    {
        "url": "https://social-forum-mirror.net/thread/water",
        "domain": "social-forum-mirror.net",
        "publisher": "Community Forum",
        "retrieved": True,
        "is_primary_source": False,
        "source_reliability": "LOW",
        "source_reliability_weight": 0.3,
        "content_verdict": "CONTRADICTS",
        "verdict": "CONTRADICTS",
        "directness": "INDIRECT",
        "strength_score": 0.2,
        "strength_label": "WEAK",
        "reason": "User comment claiming supply was running normally in their building.",
    },
    {
        "url": "https://news-aggregator-scrape.co/water",
        "domain": "news-aggregator-scrape.co",
        "publisher": "Content Aggregator",
        "retrieved": True,
        "is_primary_source": False,
        "source_reliability": "LOW",
        "source_reliability_weight": 0.3,
        "content_verdict": "CONTRADICTS",
        "verdict": "CONTRADICTS",
        "directness": "INDIRECT",
        "strength_score": 0.2,
        "strength_label": "WEAK",
        "reason": "Scraped site claiming no official bulletin found on their feed.",
    },
]

test_d_ev = [{
    "claim": "Delhi Jal Board announced a 36-hour water supply disruption for planned maintenance.",
    "status": "SUPPORTED",
    "available": True,
    "sources": test_d_supporting + test_d_contradicting,
    "summary": "Corroborated by official government portal and established regional press, superseding weak blog commentary."
}]

res_d = aggregation.aggregate_signals(
    quality_result={"usable": True, "reasons": [], "warnings": []},
    prediction_result={"prediction": "LIKELY FAKE", "raw_model_label": "FAKE", "confidence": 98.9, "confidence_display": "98.9%", "error": None},
    clickbait_result={"score": 15, "classification": "LOW", "signals": []},
    ai_text_result={"label": "Human", "signals": {}},
    source_result={"available": False, "signals": []},
    fact_check_result={"available": False, "claims": []},
    evidence_result={"claim_evidence": test_d_ev},
    temporal_info={"context": "current", "label": "October 2026", "reasons": []}
)
print_result("TEST D: ML vs Evidence Disagreement (4 Strong Support vs 3 Weak Contra)", res_d)

# ---------------------------------------------------------
# Test E: Sources genuinely conflict
# ---------------------------------------------------------
test_e_ev = [{
    "claim": "Nation A and Nation B finalized bilateral treaty today.",
    "status": "MIXED",
    "available": True,
    "sources": [
        {
            "url": "https://apnews.com/article/treaty-finalized",
            "domain": "apnews.com",
            "publisher": "Associated Press",
            "retrieved": True,
            "is_primary_source": False,
            "source_reliability": "HIGH",
            "source_reliability_weight": 1.2,
            "content_verdict": "SUPPORTS",
            "verdict": "SUPPORTS",
            "directness": "DIRECT",
            "strength_score": 1.2,
            "strength_label": "STRONG",
            "reason": "AP diplomatic reporter confirms documents signed.",
        },
        {
            "url": "https://reuters.com/world/treaty-denied",
            "domain": "reuters.com",
            "publisher": "Reuters",
            "retrieved": True,
            "is_primary_source": False,
            "source_reliability": "HIGH",
            "source_reliability_weight": 1.2,
            "content_verdict": "CONTRADICTS",
            "verdict": "CONTRADICTS",
            "directness": "DIRECT",
            "strength_score": 1.2,
            "strength_label": "STRONG",
            "reason": "Reuters quotes Prime Minister's office denying any treaty was completed.",
        }
    ],
    "summary": "Genuine contradiction between AP and Reuters."
}]

res_e = aggregation.aggregate_signals(
    quality_result={"usable": True, "reasons": [], "warnings": []},
    prediction_result={"prediction": "REAL", "raw_model_label": "REAL", "confidence": 65.0, "confidence_display": "65.0%", "error": None},
    clickbait_result={"score": 20, "classification": "LOW", "signals": []},
    ai_text_result={"label": "Human", "signals": {}},
    source_result={"available": False, "signals": []},
    fact_check_result={"available": False, "claims": []},
    evidence_result={"claim_evidence": test_e_ev},
    temporal_info={"context": "current", "label": "Breaking Report", "reasons": []}
)
print_result("TEST E: Sources Genuinely Conflict", res_e)

print("\n" + "=" * 80)
print("AUDIT VALIDATION COMPLETE")
print("=" * 80)
