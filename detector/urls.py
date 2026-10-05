from django.urls import path
from . import views

urlpatterns = [
    path("detector/", views.detector_view, name="detector"),
    path("analysis/<int:analysis_id>/", views.analysis_result_view, name="analysis_result"),
    path("analysis/<int:analysis_id>/feedback/", views.submit_feedback, name="submit_feedback"),
    path("analysis/<int:analysis_id>/save/", views.toggle_save_article, name="toggle_save_article"),
]
