# pyrefly: ignore [missing-import]
from django import forms


class NewsInputForm(forms.Form):
    INPUT_CHOICES = [
        ("text", "Paste Text"),
        ("pdf", "Upload PDF"),
        ("txt", "Upload TXT"),
        ("url", "Article URL"),
    ]

    TEMPORAL_CHOICES = [
        ("auto", "Auto-detect (Recommended)"),
        ("current", "Current / Present News"),
        ("historical", "Historical / Past News"),
    ]

    input_type = forms.ChoiceField(choices=INPUT_CHOICES, required=True)
    temporal_preference = forms.ChoiceField(choices=TEMPORAL_CHOICES, required=False, initial="auto")
    news_text = forms.CharField(required=False, widget=forms.Textarea)
    news_file = forms.FileField(required=False)
    news_url = forms.URLField(required=False)

    def clean(self):
        cleaned = super().clean()
        input_type = cleaned.get("input_type")

        if input_type == "text" and not (cleaned.get("news_text") or "").strip():
            raise forms.ValidationError("Please paste some article text.")
        if input_type in ("pdf", "txt") and not cleaned.get("news_file"):
            raise forms.ValidationError("Please choose a file to upload.")
        if input_type == "url" and not (cleaned.get("news_url") or "").strip():
            raise forms.ValidationError("Please enter an article URL.")

        return cleaned
