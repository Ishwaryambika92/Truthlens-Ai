from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404

from django.utils import timezone
from . import extract_text, services, translation
from .verification.pipeline import run_verification
from .forms import NewsInputForm
from .models import Analysis, FactCheckResult, Feedback, SavedArticle


@login_required
def detector_view(request):
    if request.method != "POST":
        return render(request, "detector.html", {"form": NewsInputForm()})

    form = NewsInputForm(request.POST, request.FILES)
    if not form.is_valid():
        for error in form.non_field_errors():
            messages.error(request, error)
        return render(request, "detector.html", {"form": form})

    input_type = form.cleaned_data["input_type"]
    text, filename, source_url, error, metadata = None, "", "", None, {}

    if input_type == "text":
        text, error = extract_text.validate_pasted_text(form.cleaned_data["news_text"])
    elif input_type in ("pdf", "txt"):
        f = form.cleaned_data["news_file"]
        filename = f.name
        if filename.lower().endswith(".txt"):
            text, error = extract_text.extract_from_txt(f)
        else:
            text, error = extract_text.extract_from_pdf(f)
    elif input_type == "url":
        source_url = form.cleaned_data["news_url"]
        text, error, metadata = extract_text.extract_from_url(source_url)

    if error:
        messages.error(request, error)
        return render(request, "detector.html", {"form": form})

    # Indic language translation check (Sarvam AI)
    indic_info = translation.translate_if_indic(text)
    if indic_info.get("translated"):
        messages.info(request, indic_info["message"])
        text = indic_info["processed_text"]
    elif indic_info.get("message"):
        messages.warning(request, indic_info["message"])

    # Deduplication: if user submitted the exact same text within the last 5 minutes, reuse result
    recent_match = Analysis.objects.filter(
        user=request.user,
        extracted_text=text,
        created_at__gte=timezone.now() - timezone.timedelta(minutes=5),
    ).order_by("-id").first()
    if recent_match:
        return redirect("analysis_result", analysis_id=recent_match.id)

    metadata = metadata or {}
    metadata["temporal_preference"] = form.cleaned_data.get("temporal_preference", "auto")

    analysis = _run_full_analysis(
        user=request.user,
        text=text,
        input_type=input_type,
        filename=filename,
        source_url=source_url,
        metadata=metadata,
    )

    return redirect("analysis_result", analysis_id=analysis.id)


def _run_full_analysis(user, text, input_type, filename="", source_url="", metadata=None):
    """Runs the full multi-layer verification pipeline and persists it.
    See detector/verification/pipeline.py for the layer-by-layer flow and
    detector/verification/aggregation.py for how the final assessment and
    verification confidence are computed."""
    result = run_verification(text, metadata=metadata)

    prediction_result = result["prediction"]
    clickbait_result = result["clickbait"]
    ai_result = result["ai_text"]
    source_result = result["source"]
    fact_check_result = result["fact_check"]
    evidence_result = result["evidence"]
    final = result["final"]

    explanation = services.build_explanation(prediction_result, clickbait_result)
    summary_result = services.summarize_text(text, length="medium") if result["quality"]["usable"] else \
        {"summary": "", "source": "not_run", "error": "Not run - article quality check failed."}

    temporal_info = final.get("temporal_info") or {}

    analysis = Analysis.objects.create(
        user=user,
        input_type=input_type,
        source_url=source_url or None,
        original_filename=filename,
        extracted_text=text,
        prediction=prediction_result.get("prediction"),
        confidence_status=prediction_result.get("confidence_status") or "",
        confidence=prediction_result.get("confidence"),
        explanation=explanation,
        top_terms=prediction_result.get("top_terms") or [],
        clickbait_score=clickbait_result["score"],
        clickbait_classification=clickbait_result["classification"],
        clickbait_signals=clickbait_result["signals"],
        ai_text_label=ai_result["label"],
        ai_text_signals=ai_result.get("signals", {}),
        summary=summary_result["summary"],
        summary_source=summary_result["source"],
        prediction_error=prediction_result.get("error") or "",
        source_domain=source_result.get("domain") or "",
        source_analysis=source_result,
        claims=result.get("claims") or [],
        evidence=evidence_result,
        final_assessment=final["final_assessment"],
        verification_confidence=final["verification_confidence"],
        verification_confidence_score=final.get("verification_confidence_score"),
        verification_reasoning=final["reasoning"],
        temporal_context=final.get("temporal_context") or "unspecified",
        temporal_detected_year=temporal_info.get("detected_year"),
    )

    for claim in fact_check_result.get("claims", []):
        FactCheckResult.objects.create(
            analysis=analysis,
            claim=claim["claim"],
            status=claim["status"],
            source=claim["source"],
            url=claim.get("url", ""),
        )

    analysis._fact_check_message = fact_check_result.get("message")
    return analysis


def _build_result_context_helpers(analysis, fact_checks=None):
    from .verification.canonical import build_canonical_verification_result

    # ONE canonical verification result object
    canonical = build_canonical_verification_result(
        analysis=analysis,
        fact_checks=fact_checks,
    )

    # Harmonize in-memory analysis verdict with canonical decision
    analysis.final_assessment = canonical["final_assessment"]["verdict"]
    analysis.verification_confidence = canonical["final_assessment"]["confidence_label"]
    analysis.verification_confidence_score = canonical["final_assessment"]["confidence_score"]

    raw_text = analysis.extracted_text or ""
    clean_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if clean_lines:
        headline = clean_lines[0]
        if len(headline) > 130:
            headline = headline[:127] + "..."
    else:
        headline = "Analyzed Article Content"

    temp_ctx = analysis.temporal_context or "unspecified"
    if temp_ctx == "current":
        temporal_label = "Current / Present News"
    elif temp_ctx == "historical":
        temporal_label = f"Historical Context ({analysis.temporal_detected_year or 'Past'})"
    else:
        temporal_label = "General News"

    article_meta = {
        "headline": headline,
        "word_count": len(raw_text.split()),
        "language": "English",
        "input_type": analysis.get_input_type_display(),
        "temporal_label": temporal_label,
    }

    if len(raw_text) > 300:
        article_preview_short = raw_text[:280] + "..."
        article_preview_rest = raw_text[280:]
        has_more_text = True
    else:
        article_preview_short = raw_text
        article_preview_rest = ""
        has_more_text = False

    return {
        "canonical_result": canonical,
        "verification_result": canonical,
        "claim_stats": canonical["claim_counts"],
        "claim_counts": canonical["claim_counts"],
        "source_counts": canonical["source_counts"],
        "evidence_counts": canonical["evidence_counts"],
        "evidence_summary": canonical["evidence_summary"],
        "processed_claims": canonical["claims"],
        "all_sources": canonical["sources"],
        "has_conflicting_evidence": (canonical["final_assessment"]["verdict"] == "MIXED / MISLEADING"),
        "has_verified_factcheck": canonical["has_verified_factcheck"],
        "primary_explanation": canonical["final_assessment"]["subtitle"],
        "why_points": canonical["final_assessment"]["why_points"],
        "key_takeaways": canonical["final_assessment"]["key_takeaways"],
        "ml_signal": canonical["ml_signal"],
        "timeline": canonical["timeline"],
        "article_meta": article_meta,
        "article_preview_short": article_preview_short,
        "article_preview_rest": article_preview_rest,
        "has_more_text": has_more_text,
    }



@login_required
def analysis_result_view(request, analysis_id):
    analysis = get_object_or_404(Analysis, id=analysis_id, user=request.user)
    fact_checks = analysis.fact_checks.all()
    user_feedback = Feedback.objects.filter(analysis=analysis, user=request.user).first()
    is_saved = SavedArticle.objects.filter(analysis=analysis, user=request.user).exists()

    fact_check_message = None
    if not fact_checks.exists():
        if not services.GOOGLE_FACTCHECK_API_KEY:
            fact_check_message = "Fact verification unavailable - external verification service is not configured."
        else:
            fact_check_message = "No matching fact-checks were found for this article's claims."

    helpers = _build_result_context_helpers(analysis, fact_checks)

    context = {
        "analysis": analysis,
        "fact_checks": fact_checks,
        "fact_check_message": fact_check_message,
        "user_feedback": user_feedback,
        "is_saved": is_saved,
        **helpers,
    }
    return render(request, "result.html", context)


@login_required
def submit_feedback(request, analysis_id):
    analysis = get_object_or_404(Analysis, id=analysis_id, user=request.user)
    if request.method == "POST":
        helpful = request.POST.get("helpful") == "yes"
        comment = request.POST.get("comment", "").strip()
        Feedback.objects.update_or_create(
            analysis=analysis, user=request.user,
            defaults={"helpful": helpful, "comment": comment},
        )
        if request.headers.get("x-requested-with") == "XMLHttpRequest" or request.content_type == "application/json":
            return JsonResponse({"status": "success", "message": "Thanks for your feedback!", "helpful": helpful})
        messages.success(request, "Thanks for your feedback!")
    return redirect("analysis_result", analysis_id=analysis.id)


@login_required
def toggle_save_article(request, analysis_id):
    analysis = get_object_or_404(Analysis, id=analysis_id, user=request.user)
    saved, created = SavedArticle.objects.get_or_create(analysis=analysis, user=request.user)
    if not created:
        saved.delete()
        is_saved = False
        msg = "Removed from saved articles."
    else:
        is_saved = True
        msg = "Article saved."

    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JsonResponse({"status": "success", "is_saved": is_saved, "message": msg})

    if is_saved:
        messages.success(request, msg)
    else:
        messages.info(request, msg)
    return redirect("analysis_result", analysis_id=analysis.id)

