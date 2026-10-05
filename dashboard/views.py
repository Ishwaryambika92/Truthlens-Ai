from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Q
from django.shortcuts import render
from django.utils import timezone

from detector.models import Analysis


@login_required(login_url="login")
def dashboard_view(request):
    analyses = Analysis.objects.filter(user=request.user)
    total = analyses.count()

    # Ground truth metrics using Final Assessment (with fallback to ML prediction for legacy records)
    real_count = analyses.filter(
        Q(final_assessment__iexact="LIKELY REAL") |
        Q(final_assessment__iexact="REAL") |
        (Q(final_assessment="") & Q(prediction__iexact="REAL")) |
        (Q(final_assessment__isnull=True) & Q(prediction__iexact="REAL"))
    ).count()

    fake_count = analyses.filter(
        Q(final_assessment__iexact="LIKELY FAKE") |
        Q(final_assessment__iexact="FAKE") |
        (Q(final_assessment="") & Q(prediction__iexact="FAKE")) |
        (Q(final_assessment__isnull=True) & Q(prediction__iexact="FAKE"))
    ).count()

    # Ensure total = real + fake + uncertain consistency
    uncertain_count = max(0, total - (real_count + fake_count))

    # Calculate average confidence: prioritize verification_confidence_score, fallback to ML confidence
    avg_conf = analyses.exclude(verification_confidence_score=None).aggregate(avg=Avg("verification_confidence_score"))["avg"]
    if avg_conf is None:
        avg_conf = analyses.exclude(confidence=None).aggregate(avg=Avg("confidence"))["avg"]

    today = timezone.now().date()
    today_count = analyses.filter(created_at__date=today).count()

    most_frequent = None
    if real_count or fake_count or uncertain_count:
        if real_count >= fake_count and real_count >= uncertain_count:
            most_frequent = "REAL"
        elif fake_count >= real_count and fake_count >= uncertain_count:
            most_frequent = "FAKE"
        else:
            most_frequent = "UNCERTAIN"

    # Deduplicate recent analyses so duplicate consecutive submissions of identical news are collapsed
    unique_recent = []
    seen_texts = set()
    for a in analyses[:30]:
        text_sig = (a.extracted_text or "").strip()[:100].lower()
        if text_sig not in seen_texts:
            seen_texts.add(text_sig)
            unique_recent.append(a)
            if len(unique_recent) >= 8:
                break
    recent = unique_recent

    # 30-Day Timeline with Real, Fake, and Uncertain series
    timeline_30 = []
    for i in range(29, -1, -1):
        day = today - timezone.timedelta(days=i)
        day_qs = analyses.filter(created_at__date=day)
        day_real = day_qs.filter(
            Q(final_assessment__iexact="LIKELY REAL") |
            Q(final_assessment__iexact="REAL") |
            (Q(final_assessment="") & Q(prediction__iexact="REAL")) |
            (Q(final_assessment__isnull=True) & Q(prediction__iexact="REAL"))
        ).count()
        day_fake = day_qs.filter(
            Q(final_assessment__iexact="LIKELY FAKE") |
            Q(final_assessment__iexact="FAKE") |
            (Q(final_assessment="") & Q(prediction__iexact="FAKE")) |
            (Q(final_assessment__isnull=True) & Q(prediction__iexact="FAKE"))
        ).count()
        day_total = day_qs.count()
        day_unc = max(0, day_total - (day_real + day_fake))

        timeline_30.append({
            "date": day.strftime("%b %d"),
            "real": day_real,
            "fake": day_fake,
            "uncertain": day_unc,
            "total": day_total,
        })

    # Generate insights ONLY when enough real data exists (>= 3 analyses)
    insight = None
    if total >= 3:
        if uncertain_count > (total / 2):
            insight = "Your recent verification activity is mostly uncertain because available supporting evidence across web sources is limited."
        elif fake_count > real_count:
            pct_fake = round((fake_count / total) * 100)
            insight = f"{pct_fake}% of your analyzed articles contain fabricated or unverified claims refuted by independent evidence."
        elif real_count >= fake_count:
            pct_real = round((real_count / total) * 100)
            insight = f"{pct_real}% of analyzed stories have been corroborated as factual by independent news archives."
        else:
            insight = "Verification records indicate a balanced distribution of authentic and refuted claims."

    context = {
        "username": request.user.username,
        "email": request.user.email,
        "total": total,
        "real_count": real_count,
        "fake_count": fake_count,
        "uncertain_count": uncertain_count,
        "avg_confidence": round(avg_conf, 1) if avg_conf is not None else None,
        "today_count": today_count,
        "most_frequent": most_frequent,
        "recent": recent,
        "timeline_30": timeline_30,
        "timeline_14": timeline_30[-14:],
        "timeline_7": timeline_30[-7:],
        "timeline": timeline_30[-14:],  # default 14-day
        "insight": insight,
    }

    return render(request, "dashboard.html", context)


def about_view(request):
    return render(request, "about.html")

