import pandas as pd

df = pd.read_csv("detector/ml/news_dataset.csv")

# Adjust column names below if yours differ
TEXT_COL = "text"
LABEL_COL = "label"

df["char_len"] = df[TEXT_COL].astype(str).str.len()
df["word_len"] = df[TEXT_COL].astype(str).str.split().str.len()

top10 = df.sort_values("char_len", ascending=False).head(10)

print("=== TOP 10 LONGEST ARTICLES ===\n")
for idx, row in top10.iterrows():
    text = str(row[TEXT_COL])
    print(f"--- Row index: {idx} ---")
    print(f"Label: {row[LABEL_COL]}")
    print(f"Char count: {row['char_len']}")
    print(f"Word count: {row['word_len']}")
    print(f"First 200 chars: {text[:200]!r}")
    print(f"Last 200 chars:  {text[-200:]!r}")
    print()

print("=== OVERALL LENGTH STATS ===")
print(df["char_len"].describe())