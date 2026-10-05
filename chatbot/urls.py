from django.urls import path
from . import views

urlpatterns = [
    path("chatbot/", views.chatbot_view, name="chatbot"),
    path("chatbot/ask/", views.ask_assistant, name="ask_assistant"),
]
