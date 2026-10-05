from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect

from .forms import RegisterForm
from .models import UserProfile
from detector.models import Analysis


# ===========================
# User Registration
# ===========================
def register_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            UserProfile.objects.create(user=user)
            messages.success(request, "Account created successfully. Please login.")
            return redirect("login")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = RegisterForm()

    return render(request, "register.html", {"form": form})


# ===========================
# User Login
# ===========================
def login_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome {user.username}!")
            next_url = request.GET.get("next")
            if next_url:
                return redirect(next_url)
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, "login.html")


# ===========================
# User Logout
# ===========================
def logout_view(request):
    logout(request)
    messages.success(request, "Logged out successfully.")
    return redirect("login")


from django.contrib.auth.models import User
# ===========================
# Profile
# ===========================
@login_required
def profile_view(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        first_name = request.POST.get("first_name", "").strip()
        last_name = request.POST.get("last_name", "").strip()
        bio = request.POST.get("bio", "").strip()

        errors = []
        if not username:
            errors.append("Username cannot be empty.")
        elif User.objects.filter(username__iexact=username).exclude(id=request.user.id).exists():
            errors.append("This username is already taken. Please choose another.")

        if email and User.objects.filter(email__iexact=email).exclude(id=request.user.id).exists():
            errors.append("This email address is already in use by another account.")

        if errors:
            for err in errors:
                messages.error(request, err)
        else:
            request.user.username = username
            request.user.email = email
            request.user.first_name = first_name
            request.user.last_name = last_name
            request.user.save()

            profile.bio = bio
            if request.FILES.get("profile_image"):
                profile.profile_image = request.FILES["profile_image"]
            profile.save()

            messages.success(request, "Your profile has been updated successfully!")
            return redirect("profile")

    total_analyses = Analysis.objects.filter(user=request.user).count()

    return render(request, "profile.html", {
        "profile": profile,
        "total_analyses": total_analyses,
    })

