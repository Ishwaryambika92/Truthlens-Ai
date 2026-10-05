from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST

from detector.models import Analysis, SavedArticle
from .generate_pdf import build_analysis_pdf


@login_required
def reports_view(request):
    saved = SavedArticle.objects.filter(user=request.user).select_related("analysis")
    return render(request, "reports.html", {"saved": saved})


@login_required
@require_POST
def delete_saved_article(request, saved_id):
    saved_item = get_object_or_404(SavedArticle, id=saved_id, user=request.user)
    saved_item.delete()

    is_ajax = (
        request.headers.get("x-requested-with") == "XMLHttpRequest"
        or "application/json" in request.headers.get("accept", "")
    )
    if is_ajax:
        return JsonResponse({
            "success": True,
            "message": "Article removed from Saved.",
            "saved_id": saved_id,
        })

    messages.success(request, "Article removed from Saved.")
    return redirect("reports")


@login_required
def download_report(request, analysis_id):
    analysis = get_object_or_404(Analysis, id=analysis_id, user=request.user)
    fact_checks = analysis.fact_checks.all()
    try:
        pdf_bytes = build_analysis_pdf(analysis, fact_checks)
    except Exception:
        messages.error(request, "The PDF report could not be generated. Please try again.")
        return redirect("analysis_result", analysis_id=analysis.id)

    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="truthlens_report_{analysis.id}.pdf"'
    return response
