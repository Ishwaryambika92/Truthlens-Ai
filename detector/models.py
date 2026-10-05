# pyrefly: ignore [missing-import]
from django.conf import settings
# pyrefly: ignore [missing-import]
from django.db import models


class Analysis(models.Model):
    INPUT_TYPES = [
        ("text", "Pasted Text"),
        ("pdf", "PDF Upload"),
        ("txt", "TXT Upload"),
        ("url", "URL"),
    ]
    PREDICTIONS = [
        ("REAL", "Likely Real"),
        ("FAKE", "Likely Fake"),
        ("UNCERTAIN", "Uncertain"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="analyses")

    input_type = models.CharField(max_length=10, choices=INPUT_TYPES)
    source_url = models.URLField(blank=True, null=True)
    original_filename = models.CharField(max_length=255, blank=True, null=True)

    extracted_text = models.TextField()

    prediction = models.CharField(max_length=10, choices=PREDICTIONS, null=True, blank=True)
    confidence_status = models.CharField(max_length=10, blank=True)  # HIGH / MEDIUM / LOW
    confidence = models.FloatField(null=True, blank=True)
    explanation = models.JSONField(default=list, blank=True)
    top_terms = models.JSONField(default=list, blank=True)

    clickbait_score = models.IntegerField(null=True, blank=True)
    clickbait_classification = models.CharField(max_length=10, blank=True)
    clickbait_signals = models.JSONField(default=list, blank=True)

    ai_text_label = models.CharField(max_length=40, blank=True)
    ai_text_signals = models.JSONField(default=dict, blank=True)

    summary = models.TextField(blank=True)
    summary_source = models.CharField(max_length=30, blank=True)

    prediction_error = models.CharField(max_length=255, blank=True)

    # --- New verification-architecture fields (additive; existing fields
    # above are preserved as-is for backward compatibility) ---
    source_domain = models.CharField(max_length=255, blank=True)
    source_analysis = models.JSONField(default=dict, blank=True)
    claims = models.JSONField(default=list, blank=True)
    evidence = models.JSONField(default=dict, blank=True)
    final_assessment = models.CharField(max_length=20, blank=True)  # "LIKELY REAL" / "LIKELY FAKE" / "UNCERTAIN"
    verification_confidence = models.CharField(max_length=10, blank=True)  # HIGH / MEDIUM / LOW
    verification_confidence_score = models.FloatField(null=True, blank=True)
    verification_reasoning = models.JSONField(default=list, blank=True)
    temporal_context = models.CharField(max_length=20, blank=True, default="unspecified")
    temporal_detected_year = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Analysis #{self.pk} ({self.prediction or 'N/A'}) - {self.user}"

    @property
    def title(self):
        text = self.extracted_text.strip()
        return (text[:80] + "...") if len(text) > 80 else (text or "Untitled analysis")


class FactCheckResult(models.Model):
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="fact_checks")
    claim = models.TextField()
    status = models.CharField(max_length=50)
    source = models.CharField(max_length=255, blank=True)
    url = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.status}: {self.claim[:50]}"


class Feedback(models.Model):
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="feedback_items")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    helpful = models.BooleanField()
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("analysis", "user")

    def __str__(self):
        return f"Feedback on #{self.analysis_id}: {'helpful' if self.helpful else 'not helpful'}"


class SavedArticle(models.Model):
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="saved_by")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="saved_articles")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("analysis", "user")

    def __str__(self):
        return f"Saved #{self.analysis_id} by {self.user}"
