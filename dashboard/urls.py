from django.urls import path
from .views import dashboard_view, about_view

urlpatterns = [
    path("dashboard/", dashboard_view, name="dashboard"),
    path("about/", about_view, name="about"),
]