from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from detector.models import Analysis


@login_required
def history_view(request):
    analyses = Analysis.objects.filter(user=request.user)
    return render(request, "history.html", {"analyses": analyses})
