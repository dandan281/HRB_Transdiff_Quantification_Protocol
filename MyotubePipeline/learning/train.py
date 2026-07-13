"""Fit one interpretable model per case type from the accumulated feedback, save them, and write
a human-readable summary (learning/model_summary.json) with the learned rule for each type.

Run after log_feedback.py (the orchestrator does this on --resume). Safe to run anytime.
"""
from __future__ import annotations
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FEATURE_KEYS, fit, DATA_DIR, MODELS_DIR, SUMMARY  # noqa: E402


def load_records(ctype):
    path = os.path.join(DATA_DIR, f"{ctype}.csv")
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                vec = [float(row[k]) for k in FEATURE_KEYS[ctype]]
                recs.append((vec, int(row["label"])))
            except (KeyError, ValueError):
                continue
    return recs


def main():
    import joblib
    os.makedirs(MODELS_DIR, exist_ok=True)
    summary = {}
    for ctype in FEATURE_KEYS:
        recs = load_records(ctype)
        pipe, info = fit(recs, ctype)
        mpath = os.path.join(MODELS_DIR, f"{ctype}.joblib")
        if pipe is not None:
            joblib.dump(pipe, mpath)
            print(f"[{ctype}] fitted on n={info['n']} (pos={info['n_pos']}) "
                  f"acc={info['train_accuracy']} | {info['rule']}")
        else:
            if os.path.exists(mpath):
                os.remove(mpath)              # not enough data yet -> no model influences defaults
            print(f"[{ctype}] {info['status']} (n={info['n']}) -> conservative defaults kept")
        summary[ctype] = info
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"summary -> {SUMMARY}")


if __name__ == "__main__":
    main()
