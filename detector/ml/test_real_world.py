"""
Interactive real-world testing entry point (thin wrapper).

Usage:
    python detector/ml/test_real_world.py

Paste any completely new news article (something you copied from a live
news site today, not from news_dataset.csv) and see the model's raw
prediction/confidence - the exact same logic detector/services.py uses in
the Django app. See test_model.py for the shared implementation and a set
of curated example categories.
"""
from test_model import load_pipeline, run_interactive

if __name__ == "__main__":
    pipeline = load_pipeline()
    run_interactive(pipeline)
