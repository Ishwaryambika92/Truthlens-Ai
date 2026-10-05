import json

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.http import require_POST

from detector.models import Analysis
from detector import services


def _analysis_context(analysis: Analysis) -> dict:
    from detector.verification.claims import build_authoritative_claims_structure
    claims_struct = build_authoritative_claims_structure(
        analysis=analysis,
        claims_list=analysis.claims,
        evidence_data=analysis.evidence,
        fact_checks=analysis.fact_checks.all(),
    )
    return {
        "id": analysis.id,
        "title": analysis.title,
        "input_type": analysis.get_input_type_display(),
        "final_assessment": analysis.final_assessment or claims_struct["final_assessment"]["verdict"] or "Not evaluated",
        "verification_confidence": analysis.verification_confidence or claims_struct["final_assessment"]["confidence_label"] or "N/A",
        "verification_confidence_score": (
            f"{analysis.verification_confidence_score:.0f}%"
            if analysis.verification_confidence_score is not None else
            (f"{claims_struct['final_assessment']['confidence_score']:.0f}%"
             if claims_struct.get("final_assessment", {}).get("confidence_score") is not None else "N/A")
        ),
        "verification_reasoning": analysis.verification_reasoning or [],
        "temporal_context": analysis.temporal_context or "unspecified",
        "ml_prediction": analysis.prediction or "N/A",
        "prediction": analysis.prediction,  # backward compatibility
        "confidence_display": f"{analysis.confidence}%" if analysis.confidence is not None else "unavailable",
        "confidence": analysis.confidence,
        "confidence_status": analysis.confidence_status or "N/A",
        "explanation": analysis.explanation or [],
        "top_terms": analysis.top_terms or [],
        "clickbait": {
            "score": analysis.clickbait_score,
            "classification": analysis.clickbait_classification,
            "signals": analysis.clickbait_signals or [],
        },
        "ai_text_label": analysis.ai_text_label,
        "summary": analysis.summary,
        "claims": analysis.claims or [],
        "evidence": analysis.evidence or {},
        "claim_counts": claims_struct["claim_counts"],
        "source_counts": claims_struct["source_counts"],
        "processed_claims": [
            {
                "claim": c["claim"],
                "status": c["status"],
                "summary": c["summary"],
                "sources_count": c["sources_count"],
            }
            for c in claims_struct["claims"]
        ],
        "fact_checks": [
            {"claim": fc.claim, "status": fc.status, "source": fc.source, "url": fc.url}
            for fc in analysis.fact_checks.all()
        ],
        "fact_check": {
            "available": analysis.fact_checks.exists() or bool(services.GOOGLE_FACTCHECK_API_KEY),
            "claims": [
                {"claim": fc.claim, "status": fc.status, "source": fc.source}
                for fc in analysis.fact_checks.all()
            ],
            "message": None if analysis.fact_checks.exists() else (
                "Fact verification unavailable - external verification service is not configured."
                if not services.GOOGLE_FACTCHECK_API_KEY else
                "No matching fact-checks were found for this article's claims."
            ),
        },
        "extracted_text_excerpt": analysis.extracted_text[:1500] if analysis.extracted_text else "",
    }


@login_required
def chatbot_view(request):
    analyses = Analysis.objects.filter(user=request.user)
    selected_id = request.GET.get("analysis")
    selected = None
    if selected_id:
        selected = get_object_or_404(Analysis, id=selected_id, user=request.user)
    elif analyses.exists():
        selected = analyses.first()

    return render(request, "chatbot.html", {
        "analyses": analyses,
        "selected": selected,
    })


@require_POST
@login_required
def ask_assistant(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid request format.", "success": False}, status=400)

    question = (payload.get("question") or "").strip()
    analysis_id = payload.get("analysis_id")
    history = payload.get("history") or []

    if not question:
        return JsonResponse({"error": "Please enter a question.", "success": False}, status=400)

    context = None
    if analysis_id:
        try:
            analysis = Analysis.objects.filter(id=analysis_id, user=request.user).first()
            if analysis:
                context = _analysis_context(analysis)
        except Exception:
            context = None

    try:
        answer = services.answer_assistant_question(question, context, history=history)
        return JsonResponse({
            "success": True,
            "answer": answer,
            "analysis_id": analysis_id if context else None,
        })
    except Exception as e:
        return JsonResponse({
            "success": False,
            "answer": "I'm having trouble connecting right now. Please try again in a moment.",
            "error": "Temporary assistant error",
        }, status=200)

