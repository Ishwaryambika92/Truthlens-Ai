"""
Centralized text preprocessing.

CRITICAL: this module is imported by BOTH the model training script
(detector/ml/train_model.py) and the live prediction service
(detector/services.py). Do not change the cleaning logic here without
retraining the model, or predictions will be made on text that does not
match what the model was trained on.
"""
import re
import string

# A small, dependency-free English stopword list (avoids requiring NLTK
# downloads at runtime/deploy time, which can fail in restricted network
# environments).
STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
    "be","because","been","before","being","below","between","both","but","by","can","did","do","does",
    "doing","don","down","during","each","few","for","from","further","had","has","have","having","he",
    "her","here","hers","herself","him","himself","his","how","i","if","in","into","is","it","its","itself",
    "just","me","more","most","my","myself","no","nor","not","now","of","off","on","once","only","or",
    "other","our","ours","ourselves","out","over","own","s","same","she","should","so","some","such","t",
    "than","that","the","their","theirs","them","themselves","then","there","these","they","this","those",
    "through","to","too","under","until","up","very","was","we","were","what","when","where","which",
    "while","who","whom","why","will","with","you","your","yours","yourself","yourselves",
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """Basic normalization: strip control chars, collapse whitespace."""
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_stem(word: str) -> str:
    """Very lightweight suffix-stripping stemmer (Porter-lite).

    A full Porter stemmer/lemmatizer is avoided so the pipeline has no
    heavyweight NLP dependency (e.g. NLTK corpora) that could be missing in
    a deployment environment. This keeps behaviour identical between
    training and inference, which matters more than stemming sophistication.
    """
    for suffix in ("ational", "tional", "izer", "ation", "ator", "ing", "edly", "edness",
                   "fulness", "iveness", "ness", "ment", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def clean_text(text: str, remove_stopwords: bool = True, stem: bool = True) -> str:
    """Full preprocessing pipeline used for BOTH training and inference.

    Raw Text -> normalize -> lowercase -> strip symbols/punctuation ->
    tokenize -> stopword removal -> lightweight stemming -> clean text
    """
    text = normalize_text(text)
    text = text.lower()

    # Remove URLs and email addresses (common noise in news text/HTML dumps)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove punctuation/symbols, keep letters/numbers/spaces
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split(" ")
    tokens = [t for t in tokens if t]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    if stem:
        tokens = [simple_stem(t) for t in tokens]

    return " ".join(tokens)


def word_count(text: str) -> int:
    return len(normalize_text(text).split())
