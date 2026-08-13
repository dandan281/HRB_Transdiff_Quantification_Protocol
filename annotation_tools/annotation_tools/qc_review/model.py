"""Learning core: turn proposal features + your accept/reject clicks into a small,
interpretable model that pre-decides the next batch.

Classical ML only (scikit-learn LogisticRegression on hand-crafted features). No
deep learning -- the data is small and you want to SEE the learned rule. One
binary model: is this proposal a real, complete myotube (accept) or not (reject)?
'ambiguous' decisions are kept for the annotator but never used as training labels.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(HERE, "models", "accept.joblib")
SUMMARY = os.path.join(HERE, "models", "model_summary.json")

# Features the model reads (subset of pipeline.proposal_features), fixed order.
FEATURE_KEYS = ["length_um", "width_um", "area_um2", "aspect", "solidity",
                "fiber_mean", "territory_overlap", "touches_border"]

# decision -> binary label (accept=1, reject=0). 'ambiguous' is not a label.
LABEL_FOR = {"accept": 1, "reject": 0}

MIN_SAMPLES = 12        # below this (or single-class) the model does not set defaults


def features_vector(features: dict) -> list[float]:
    return [float(features.get(k, 0.0)) for k in FEATURE_KEYS]


def fit(records):
    """records: list of (feature_vector, label 0/1). Returns (pipeline_or_None, info)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X = np.array([r[0] for r in records], dtype=float)
    y = np.array([r[1] for r in records], dtype=int)
    info = {"n": int(len(y)), "n_accept": int(y.sum()), "features": FEATURE_KEYS}
    if len(y) < MIN_SAMPLES or y.min() == y.max():
        info["status"] = "insufficient" if len(y) < MIN_SAMPLES else "single_class"
        return None, info

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ])
    pipe.fit(X, y)
    info["status"] = "fitted"
    info["train_accuracy"] = round(float(pipe.score(X, y)), 3)
    coefs = pipe.named_steps["clf"].coef_[0]
    ranked = sorted(zip(FEATURE_KEYS, coefs), key=lambda kv: -abs(kv[1]))
    info["coef_std"] = {k: round(float(c), 3) for k, c in ranked}
    top = ranked[0]
    direction = "higher" if top[1] > 0 else "lower"
    info["rule"] = (f"you tend to ACCEPT a proposal when {top[0]} is {direction} "
                    f"(top driver, std-coef {top[1]:+.2f}); accept rate {y.mean():.0%}")
    return pipe, info


def predict_default(features: dict):
    """Return (action, proba) from the saved model, or (None, None) if none usable.

    Non-fatal: refuses unless model_summary reports 'fitted'; any error degrades to
    (None, None). ``proba`` is confidence in the SHOWN action.
    """
    if not os.path.exists(MODEL_PATH):
        return None, None
    if load_summary().get("status") != "fitted":
        return None, None
    try:
        import joblib
        pipe = joblib.load(MODEL_PATH)
        x = np.array([features_vector(features)], dtype=float)
        if getattr(pipe, "n_features_in_", x.shape[1]) != x.shape[1]:
            return None, None
        p = float(pipe.predict_proba(x)[0, 1])
    except Exception:
        return None, None
    if p >= 0.5:
        return "accept", round(p, 3)
    return "reject", round(1.0 - p, 3)


def load_summary() -> dict:
    if os.path.exists(SUMMARY):
        with open(SUMMARY, encoding="utf-8") as fh:
            return json.load(fh)
    return {}
