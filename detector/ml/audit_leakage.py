import pandas as pd
import re
from collections import Counter

df = pd.read_csv("detector/ml/news_dataset.csv")
TEXT_COL = "text"
LABEL_COL = "label"

n_total = len(df)
n_real = (df[LABEL_COL] == "REAL").sum()
n_fake = (df[LABEL_COL] == "FAKE").sum()

def pct_by_label(mask):
    real_hits = (mask & (df[LABEL_COL] == "REAL")).sum()
    fake_hits = (mask & (df[LABEL_COL] == "FAKE")).sum()
    real_pct = 100 * real_hits / max(n_real, 1)
    fake_pct = 100 * fake_hits / max(n_fake, 1)
    return real_hits, real_pct, fake_hits, fake_pct

patterns = {
    "Reuters mention":            r"\bReuters\b",
    "(Reuters) dateline tag":     r"\(Reuters\)",
    "AP mention":                 r"\bAssociated Press\b|\bAP\b",
    "URL present":                r"https?://|www\.[a-z]",
    "Email address present":      r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "Twitter handle (@name)":     r"(?<!\w)@\w{3,}",
    "SUBSCRIBE / MEMBER CTA":     r"\bSUBSCRIBE\b|\bBECOME A MEMBER\b",
    "21WIRE site mention":        r"21WIRE|21wire",
    "Hashtag present":            r"(?<!\w)#\w{3,}",
    "'Featured image' caption":   r"Featured image|Image via|Image credit",
    "'Read more at' boilerplate": r"Read [Mm]ore (at|AT)",
    "All-caps SEO stuffing (3+ repeats of a phrase)": None,  # handled separately below
}

print(f"Total rows: {n_total} | REAL: {n_real} | FAKE: {n_fake}\n")
print(f"{'Pattern':38s} {'REAL hits':>10s} {'REAL %':>8s} {'FAKE hits':>10s} {'FAKE %':>8s}")
print("-" * 80)

for name, pat in patterns.items():
    if pat is None:
        continue
    mask = df[TEXT_COL].astype(str).str.contains(pat, regex=True, na=False)
    real_hits, real_pct, fake_hits, fake_pct = pct_by_label(mask)
    print(f"{name:38s} {real_hits:10d} {real_pct:7.1f}% {fake_hits:10d} {fake_pct:7.1f}%")

# Detect repeated-phrase spam (same 3+ word phrase repeated many times in one row)
def has_repetitive_spam(text, min_repeats=6):
    words = str(text).lower().split()
    if len(words) < 20:
        return False
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    counts = Counter(trigrams)
    return counts.most_common(1)[0][1] >= min_repeats if counts else False

spam_mask = df[TEXT_COL].apply(has_repetitive_spam)
real_hits, real_pct, fake_hits, fake_pct = pct_by_label(spam_mask)
print(f"{'Repetitive trigram spam (6+ repeats)':38s} {real_hits:10d} {real_pct:7.1f}% {fake_hits:10d} {fake_pct:7.1f}%")

# Top 15 most common capitalized multi-word phrases that could be source names/brands
def extract_brandlike(text):
    return re.findall(r"\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){1,2}\b", str(text))

print("\n--- Top 20 recurring capitalized phrases (possible source/brand names) in REAL ---")
real_phrases = Counter()
for t in df.loc[df[LABEL_COL]=="REAL", TEXT_COL].astype(str):
    real_phrases.update(extract_brandlike(t))
for phrase, cnt in real_phrases.most_common(20):
    print(f"{cnt:6d}  {phrase}")

print("\n--- Top 20 recurring capitalized phrases (possible source/brand names) in FAKE ---")
fake_phrases = Counter()
for t in df.loc[df[LABEL_COL]=="FAKE", TEXT_COL].astype(str):
    fake_phrases.update(extract_brandlike(t))
for phrase, cnt in fake_phrases.most_common(20):
    print(f"{cnt:6d}  {phrase}")