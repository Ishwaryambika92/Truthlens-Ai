from django.urls import path
from . import views

urlpatterns = [
    path("reports/", views.reports_view, name="reports"),
    path("saved/<int:saved_id>/delete/", views.delete_saved_article, name="delete_saved_article"),
    path("analysis/<int:analysis_id>/report/", views.download_report, name="download_report"),
]
