# Raw dataset provenance

**File:** `fake_or_real_news.csv`
**Rows:** 6,335 (id, title, text, label)
**Labels:** `REAL` (3,171) / `FAKE` (3,164) — near-balanced

**Origin:** Commonly known as the "Getting Real about Fake News" / George
McIntire fake-vs-real news dataset. Real-world political news articles and
headlines gathered primarily around the 2016 U.S. election cycle. `REAL`
articles are drawn from mainstream news outlets; `FAKE` articles are drawn
from sites flagged as unreliable/hyperpartisan (originally categorized via
the OpenSources.co / BS Detector project methodology). It has been widely
used as a teaching/research dataset for fake-news text classification,
including the well-known KDnuggets article
"How to Build a Fake News Classification Model" (kdnuggets.com/2017/04) and
the associated DataCamp tutorial "Detecting Fake News with Scikit-Learn".

**Downloaded from (this session):**
https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv
(a public mirror of the original dataset; repository: github.com/lutzhamel/fake-news)
Retrieved: 2026-08-11.

**Label definition (as inherited from the source dataset):**
Labels reflect the *publishing source's* known reliability classification
at time of labeling, NOT a per-claim fact-check of each individual article.
This is an important, honest caveat: the model learns to recognize the
*writing style and content patterns* typical of real vs. hyperpartisan/fake
news sources circa 2015-2016 — it does not verify individual factual claims.

**Known limitations of this dataset (documented honestly, not hidden):**
- Topic coverage skews heavily toward U.S. politics/2016 election content;
  diversity across topics like sports, technology, or international news
  unrelated to politics is limited.
- No article `date` or `source domain` field is present, so a time-aware
  split and explicit source-based leakage removal are not possible with
  this file alone; near-duplicate detection is used instead (see
  `prepare_dataset.py` / `dataset_report.json`).
- License: no explicit license is attached to the GitHub mirror. The
  dataset has circulated widely for research/education use since ~2017.
  If TruthLens AI will be used commercially, verify redistribution/usage
  terms independently before relying on this exact file — this is
  flagged here rather than silently assumed.

**Recommended future improvement:** merging in a larger, source-diverse
corpus (e.g. the ISOT Fake News Dataset, Ahmed/Traore/Saad, University of
Victoria, ~44,898 articles, CC0-licensed release on Kaggle) would improve
topic diversity and generalization further.

---

## UPDATE (this session): ISOT dataset added

The ISOT dataset described above has now been added and merged in by
`prepare_dataset.py`. Two more files live in this `raw/` folder:

- `isot_true.csv` (21,417 rows) — REAL, heavily Reuters/worldnews sourced
- `isot_fake.csv` (23,481 rows) — FAKE, various flagged sites

**Downloaded from (this session):**
https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/master/data/True.csv
https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/master/data/Fake.csv
(a public mirror of the ISOT dataset; repo: github.com/laxmimerit/fake-real-news-dataset)
Retrieved: 2026-08-19.

**IMPORTANT - severe leakage found and fixed:** 21,247 / 21,417 (99.2%) of
ISOT's True/REAL articles contain the literal substring `"(Reuters)"`,
versus 9 / 23,481 (0.04%) of Fake articles. A model could reach ~99%
"accuracy" on ISOT alone just by detecting that one token — this teaches
nothing about real-vs-fake writing style and is a well-documented critique
of this dataset in the literature. `prepare_dataset.py` strips:
  - wire-service datelines (`"WASHINGTON (Reuters) - "` style prefixes)
  - bare mentions of publisher/wire-service names (Reuters, Breitbart,
    InfoWars, 21st Century Wire, etc.) anywhere in the text
  - photo-credit / social-embed scraper chrome (`"Featured image via..."`,
    `"Getty Images"`, `pic.twitter.com/...` links, raw URLs) — found at
    similarly extreme skew (e.g. "Featured image" in 40.8% of FAKE text
    vs. 0% of REAL) and traced to leftover HTML/media-caption artifacts
    from the original scrape, not genuine article content.

A follow-up audit (documented in `dataset_report.json` ->
`diagnosed_leakage`) confirmed these interventions reduced the skew on
every flagged term from double-digit-percentage-point gaps down to low
single digits or zero. See `dataset_report.json` for the exact before/
after numbers from this project's own runs.

**Reproduction commands** (from the project root, venv active):

```powershell
# download the raw sources into detector/ml/raw/ (if not already present)
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv" -OutFile "detector\ml\raw\fake_or_real_news.csv"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/master/data/True.csv" -OutFile "detector\ml\raw\isot_true.csv"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/master/data/Fake.csv" -OutFile "detector\ml\raw\isot_fake.csv"

python detector\ml\prepare_dataset.py
python detector\ml\train_model.py
python detector\ml\test_model.py
```
