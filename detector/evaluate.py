"""
Standalone entry point for TruthLens AI system evaluation.

Usage:
  python detector/evaluate.py
  python detector/evaluate.py --dataset path/to/dataset.json
  python detector/evaluate.py --limit 10
"""
import os
import sys
from pathlib import Path

# Setup Django environment
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "truthlens.settings")

import django
django.setup()

from django.core.management import call_command

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate TruthLens AI verification pipeline on a labelled dataset.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSON labelled dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of articles")
    args = parser.parse_args()

    cmd_kwargs = {}
    if args.dataset:
        cmd_kwargs["dataset"] = args.dataset
    if args.limit:
        cmd_kwargs["limit"] = args.limit

    call_command("evaluate_system", **cmd_kwargs)
