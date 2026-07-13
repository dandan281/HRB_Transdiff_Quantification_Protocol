"""Learning core: turn ambiguous-case features + your decisions into a small, interpretable model.

Classical ML only (scikit-learn LogisticRegression on a handful of hand-crafted features per case
type). No deep learning, no pytorch -- the data is small (tens of cases per well) and you want to
SEE the learned rule. One model per case type: split / merge / occluded.
"""
from __future__ import annotations
import os
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")           # accumulated feedback CSVs
MODELS_DIR = os.path.join(HERE, "models")        # fitted sklearn pipelines (joblib)
SUMMARY = os.path.join(HERE, "model_summary.json")

# canonical feature order per case type (must match the `features` dict flag.py attaches)
FEATURE_KEYS = {
    "split": ["length_px", "brightness", "longest_dark_px", "max_bend_deg", "n_gap", "n_kink"],
    "merge": ["gap_px", "angle_deg", "continuity", "len_min", "len_max", "bright_min", "bright_max"],
    "occluded": ["length_px", "brightness"],
}

# how a saved decision becomes a 0/1 label (the "yes, do the edit" class).
# NOTE: the model learns "split-or-not", not the split MODE -- both point-split and split_n count as
# positive, and the served default for a positive is point-`split` (the common case). N is never
# learned; the reviewer can still pick split_n in the page.
POSITIVE_ACTIONS = {
    "split": {"split", "split_n"},      # vs keep/reject
    "merge": {"merge"},                 # vs separate
    "occluded": {"restore", "keep"},    # vs drop
}

# explicit 0/1 mapping per known action. A raw 'redraw' action is interpreted in log_feedback.py
# by comparing the pipeline trace count with the reviewer-drawn trace count, then recorded here as
# a synthetic action such as 'redraw_split' or 'redraw_keep'.
KNOWN_LABELS = {
    "split": {"split": 1, "split_n": 1, "keep": 0, "reject": 0},
    "merge": {"merge": 1, "separate": 0},
    "occluded": {"restore": 1, "keep": 1, "drop": 0},
}

# how a positive/negative prediction maps back to a default review action
DEFAULT_FOR = {
    "split": ("split", "keep"),
    "merge": ("merge", "separate"),
    "occluded": ("restore", "drop"),
}

MIN_SAMPLES = 12        # below this (or single-class), the model does not influence defaults


def features_vector(features: dict, ctype: str):
    return [float(features.get(k, 0.0)) for k in FEATURE_KEYS[ctype]]


def label_for(ctype: str, action: str):
    """0/1 binary label, or None for actions that are not a binary decision (e.g. 'redraw')."""
    return KNOWN_LABELS.get(ctype, {}).get(action)


def fit(records, ctype: str):
    """records: list of (feature_vector, label). Returns (pipeline_or_None, info_dict)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X = np.array([r[0] for r in records], dtype=float)
    y = np.array([r[1] for r in records], dtype=int)
    info = {"type": ctype, "n": int(len(y)), "n_pos": int(y.sum()), "features": FEATURE_KEYS[ctype]}
    if len(y) < MIN_SAMPLES or y.min() == y.max():
        info["status"] = "insufficient" if len(y) < MIN_SAMPLES else "single_class"
        info["majority"] = int(round(y.mean())) if len(y) else 0
        return None, info

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ])
    pipe.fit(X, y)
    info["status"] = "fitted"
    info["train_accuracy"] = round(float(pipe.score(X, y)), 3)
    # interpretable: standardized coefficients ranked by |weight|
    coefs = pipe.named_steps["clf"].coef_[0]
    ranked = sorted(zip(FEATURE_KEYS[ctype], coefs), key=lambda kv: -abs(kv[1]))
    info["coef_std"] = {k: round(float(c), 3) for k, c in ranked}
    info["rule"] = _describe(ctype, ranked, y)
    return pipe, info


def _describe(ctype, ranked, y):
    pos = POSITIVE_ACTIONS[ctype]
    top = ranked[0]
    direction = "higher" if top[1] > 0 else "lower"
    verb = "/".join(sorted(pos))
    return (f"you tend to '{verb}' a {ctype} case when {top[0]} is {direction} "
            f"(top driver, std-coef {top[1]:+.2f}); base rate {y.mean():.0%}")


def predict_default(ctype: str, features: dict):
    """Return (action, proba) using the saved model for ctype, or (None, None) if no usable model.

    Self-contained + non-fatal: refuses to serve unless model_summary reports this type 'fitted'
    (so a stale/under-data joblib never influences defaults), and any load/predict/shape error
    degrades to (None, None) instead of raising. `proba` is the confidence in the SHOWN action.
    """
    path = os.path.join(MODELS_DIR, f"{ctype}.joblib")
    if not os.path.exists(path):
        return None, None
    summ = load_summary().get(ctype, {})
    if summ.get("status") != "fitted":          # under-data / single-class / stale -> don't serve
        return None, None
    try:
        import joblib
        pipe = joblib.load(path)
        x = np.array([features_vector(features, ctype)], dtype=float)
        if getattr(pipe, "n_features_in_", x.shape[1]) != x.shape[1]:
            return None, None                   # feature schema drifted since training
        p = float(pipe.predict_proba(x)[0, 1])  # P(positive class)
    except Exception:
        return None, None
    pos_action, neg_action = DEFAULT_FOR[ctype]
    if p >= 0.5:
        return pos_action, round(p, 3)
    return neg_action, round(1.0 - p, 3)        # report confidence in the chosen (negative) action


def load_summary() -> dict:
    if os.path.exists(SUMMARY):
        with open(SUMMARY, encoding="utf-8") as fh:
            return json.load(fh)
    return {}
