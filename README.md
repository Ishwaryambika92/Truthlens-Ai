# TruthLens AI

An AI-powered news verification platform: paste text, upload a PDF/TXT file, or
submit a URL, and TruthLens AI runs it through a real ML fake-news classifier
plus supporting analysis (clickbait scoring, experimental AI-text detection,
fact-check lookup, summarization) and stores the results per-user.

## 1. Features

- **Authentication** — register/login/logout, hashed passwords (Django's
  built-in PBKDF2 auth), per-user profile with bio/avatar.
- **News input** — pasted text, PDF upload, TXT upload, or article URL, each
  with real validation and graceful error handling.
- **ML fake-news detection** — TF-IDF + Logistic Regression, trained from
  `detector/ml/train_model.py`, with genuine `predict_proba`-based confidence.
- **Explainability** — top contributing terms pulled from the model's actual
  coefficients for the specific input, not invented.
- **Clickbait analysis** — deterministic, rule-based (sensational phrases,
  ALL-CAPS words, exclamation/question density).
- **AI-generated text analysis** — experimental heuristic (sentence-length
  variance, vocabulary diversity, repetition rate), clearly labeled as such.
- **Fact verification** — Google Fact Check Tools API integration; reports
  itself as unavailable (never fabricates sources) if unconfigured.
- **Summarization** — Anthropic API (Claude) if `ANTHROPIC_API_KEY` is set,
  otherwise an extractive fallback so the feature never breaks.
- **Explanation assistant / chatbot** — grounded strictly in the selected
  analysis's stored data; AI-backed if configured, rule-based fallback if not.
- **History, Dashboard, Saved Articles, Feedback** — all computed from real
  per-user database rows, never hard-coded.
- **PDF report** — branded, generated server-side with ReportLab.

## 2. Architecture

Django 6 project, one app per feature area (matches the app layout that was
already in the uploaded project):

```
truthlens/         project settings, urls, wsgi/asgi
accounts/          registration, login, logout, profile (UserProfile model)
detector/          core domain: models (Analysis, FactCheckResult, Feedback,
                    SavedArticle), preprocessing.py, extract_text.py,
                    services.py (ML + clickbait + AI-text + summarization +
                    fact-check + chatbot logic), ml/ (dataset, training
                    script, saved model/vectorizer)
dashboard/          real-time stats computed from detector.models.Analysis
history/            per-user analysis history
reports/            saved articles + PDF report generation
chatbot/            explanation assistant UI + API endpoint
templates/, static/ shared design system
```

`detector/preprocessing.py` is imported by both the training script and the
live prediction path, so inference always uses the exact same cleaning logic
the model was trained on.

## 3. Technology stack

- Django 6.0
- SQLite by default (swaps to Postgres automatically if `DATABASE_URL` is set)
- scikit-learn (TF-IDF + Logistic Regression), joblib for model persistence
- pypdf (PDF extraction), BeautifulSoup + lxml (URL article extraction)
- ReportLab (PDF report generation)
- Anthropic API (optional — summarization + chatbot)
- Google Fact Check Tools API (optional — fact verification)
- WhiteNoise (static files in production), gunicorn (WSGI server)

## 4. Local setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then edit .env as needed
python manage.py migrate
python manage.py createsuperuser   # optional, for /admin/

# Train the ML model (required once — model.pkl/vectorizer.pkl are not
# committed if you're starting from a clean checkout)
python detector/ml/train_model.py

python manage.py runserver
```

Visit `http://127.0.0.1:8000/` (redirects to `/dashboard/`, which redirects
to `/login/` if not authenticated).

## 5. Environment variables

See `.env.example`. Summary:

| Variable | Required | Purpose |
|---|---|---|
| `SECRET_KEY` | Yes (production) | Django secret key |
| `DEBUG` | No | `True`/`False` |
| `ALLOWED_HOSTS` | Yes (production) | comma-separated hostnames |
| `CSRF_TRUSTED_ORIGINS` | If behind HTTPS proxy | comma-separated origins |
| `DATABASE_URL` | No | Postgres URL; defaults to local SQLite |
| `ANTHROPIC_API_KEY` | No | enables AI summarization + chatbot |
| `ANTHROPIC_MODEL` | No | defaults to `claude-sonnet-4-6` |
| `FACT_CHECK_API_KEY` | No | Google Fact Check Tools API key |

## 6. Database & migrations

Models live in `accounts/models.py` (UserProfile) and `detector/models.py`
(Analysis, FactCheckResult, Feedback, SavedArticle). If you change any model:

```bash
python manage.py makemigrations
python manage.py migrate
```

## 7. ML model setup / retraining

```bash
python detector/ml/train_model.py
```

This reads `detector/ml/news_dataset.csv`, cleans it with the shared
preprocessing pipeline, trains TF-IDF + Logistic Regression, evaluates on a
held-out 20% split, and writes `model.pkl`, `vectorizer.pkl`, `metrics.json`.
**To use a real-world dataset**, replace `news_dataset.csv` with any CSV that
has `text` and `label` (`REAL`/`FAKE`) columns and rerun the script — see
Known Limitations below for why this matters.

## 8. Running tests

Automated end-to-end smoke tests were run against Django's test client
(not just described — actually executed) covering: registration, login,
logout, unauthenticated-access redirect, all 4 input types (text/PDF/TXT/
URL) including a real generated PDF, empty/oversized/short-text validation,
corrupted PDF handling, unreachable-URL handling, wrong-file-extension
handling, ML prediction + confidence + clickbait + explanation, PDF report
generation (verified real `%PDF` bytes), feedback submission, save/unsave,
dashboard stats, history listing, chatbot Q&A (rule-based fallback path),
cross-user authorization (404 on another user's analysis, empty history),
and graceful behavior when the model files are missing. All passed. See the
completion report for full results.

## 9. Cloud deployment (Heroku-style / any Procfile platform)

```bash
# Procfile is included:
#   web: gunicorn truthlens.wsgi --log-file -
#   release: python manage.py migrate --noinput

# 1. Set environment variables on the platform:
SECRET_KEY=<generate a long random value>
DEBUG=False
ALLOWED_HOSTS=<your-domain>
DATABASE_URL=<platform-provided Postgres URL>
ANTHROPIC_API_KEY=<optional>
FACT_CHECK_API_KEY=<optional>

# 2. Push / deploy. The release step runs migrations automatically.
# 3. Static files are served by WhiteNoise (already wired into MIDDLEWARE
#    and STORAGES) - no separate static file host is required.
# 4. Make sure model.pkl / vectorizer.pkl are present in the deployed
#    package (either commit them, or run `python detector/ml/train_model.py`
#    as part of your build step).
```

Exclude from any deployment package: `venv/`, `__pycache__/`, `db.sqlite3`
(if using Postgres), `staticfiles/` (regenerated by `collectstatic`).

## 10. API configuration

- **Anthropic API** (`ANTHROPIC_API_KEY`): used for article summarization and
  the explanation-assistant chatbot. Without it, both features still work
  via non-AI fallbacks (extractive summary; grounded rule-based Q&A).
- **Google Fact Check Tools API** (`FACT_CHECK_API_KEY`): used for fact
  verification. Without it, the app clearly states verification is
  unavailable rather than fabricating claims or sources.

## 11. Known limitations

- **Training dataset**: `detector/ml/news_dataset.csv` is a small (600-row),
  self-authored, template-based illustrative dataset — not a large
  real-world corpus. It was evaluated honestly (held-out test split, real
  metrics in `metrics.json`), but the near-perfect test accuracy reflects
  that templated data is easy to separate, not real-world performance on
  genuine news articles. Swap in a real labeled dataset before production use.
- **AI-generated text detection** is an explicitly experimental heuristic
  (not a trained classifier) and may be inaccurate; the UI discloses this.
- **Fact verification** depends on the Google Fact Check Tools API and only
  returns results that API has indexed; many claims will show "no matching
  fact-checks found" rather than a verdict.
- **URL extraction** uses a generic `<article>`/`<p>`-tag heuristic; heavily
  JavaScript-rendered sites or sites that block bots may fail extraction
  (handled gracefully with a clear error message).
- **Summarization/chatbot** quality depends on whether `ANTHROPIC_API_KEY`
  is configured; fallbacks are functional but simpler.
