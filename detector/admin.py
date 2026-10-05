from django.contrib import admin
from .models import Analysis, FactCheckResult, Feedback, SavedArticle

admin.site.register(Analysis)
admin.site.register(FactCheckResult)
admin.site.register(Feedback)
admin.site.register(SavedArticle)
