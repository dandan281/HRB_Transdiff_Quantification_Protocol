"""Build the dataset, then train + evaluate the join/no-join fragment-linker.

Evaluation respects generalization: leave-one-well-out CV, plus an explicit train-PLATE_32 /
test-PLATE_23 split (the honest held-out test). Reports metrics on the JOIN (positive) class, since
that is the decision that fixes fragmentation.
"""
from __future__ import annotations

import csv
import json

import numpy as np

from . import config as LC
from . import dataset as D


# ---- dataset ---------------------------------------------------------------------------------
def build(write=True, verbose=True):
    all_rows, stats = [], []
    if verbose:
        print("building linker dataset:")
    for w in LC.train_wells():
        rows, st = D.build_well(w, verbose=verbose)
        all_rows += rows
        stats.append(st)
    if write:
        LC.DATA.mkdir(parents=True, exist_ok=True)
        cols = ["stem", "plate", "case_id", "label"] + D.FEATURE_KEYS
        with open(LC.DATA / "pairs.csv", "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=cols)
            wtr.writeheader()
            wtr.writerows(all_rows)
        (LC.DATA / "build_stats.json").write_text(json.dumps(stats, indent=2, default=float))
        print(f"\nwrote {len(all_rows)} pairs -> {LC.DATA/'pairs.csv'}")
    tot = len(all_rows); pos = sum(r["label"] for r in all_rows)
    print(f"total: {tot} pairs, {pos} join ({pos/tot:.1%}), {tot-pos} no-join across {len(stats)} wells")
    return all_rows


def _xy(rows):
    X = np.array([[r[k] for k in D.FEATURE_KEYS] for r in rows], dtype=float)
    y = np.array([int(r["label"]) for r in rows], dtype=int)
    groups = np.array([r["stem"] for r in rows])
    plates = np.array([r["plate"] for r in rows])
    return X, y, groups, plates


def _pipe(kind):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logreg":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])
    return Pipeline([("clf", GradientBoostingClassifier(random_state=0))])


def _scores(y, yhat, proba):
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    return dict(
        precision=float(precision_score(y, yhat, zero_division=0)),
        recall=float(recall_score(y, yhat, zero_division=0)),
        f1=float(f1_score(y, yhat, zero_division=0)),
        auc=float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        n=int(len(y)), n_pos=int(y.sum()),
    )


def evaluate(rows=None):
    if rows is None:
        rows = build(write=False, verbose=False)
    X, y, groups, plates = _xy(rows)

    print(f"\n=== dataset: {len(y)} pairs, {y.sum()} join ({y.mean():.1%}), {len(set(groups))} wells ===")

    for kind in ("logreg", "gbm"):
        print(f"\n--- {kind} : leave-one-well-out CV (join-class metrics) ---")
        f1s = []
        for held in sorted(set(groups)):
            tr, te = groups != held, groups == held
            if len(np.unique(y[tr])) < 2:
                continue
            pipe = _pipe(kind).fit(X[tr], y[tr])
            proba = pipe.predict_proba(X[te])[:, 1]
            yhat = (proba >= 0.5).astype(int)
            s = _scores(y[te], yhat, proba)
            f1s.append(s["f1"])
            print(f"  hold {held:22s} P={s['precision']:.3f} R={s['recall']:.3f} "
                  f"F1={s['f1']:.3f} AUC={s['auc']:.3f} (n={s['n']}, pos={s['n_pos']})")
        if f1s:
            print(f"  mean LOWO join-F1 = {np.mean(f1s):.3f}")

        # honest held-out: train PLATE_32, test PLATE_23
        tr, te = plates == "PLATE_32", plates == "PLATE_23"
        if te.sum() and len(np.unique(y[tr])) > 1:
            pipe = _pipe(kind).fit(X[tr], y[tr])
            proba = pipe.predict_proba(X[te])[:, 1]
            s = _scores(y[te], (proba >= 0.5).astype(int), proba)
            print(f"  train PLATE_32 -> test PLATE_23: P={s['precision']:.3f} R={s['recall']:.3f} "
                  f"F1={s['f1']:.3f} AUC={s['auc']:.3f} (n={s['n']}, pos={s['n_pos']})")

    _report_logreg_coefs(X, y)


def _report_logreg_coefs(X, y):
    pipe = _pipe("logreg").fit(X, y)
    coefs = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(coefs))
    print("\n--- logreg standardized coefficients (join direction) ---")
    for k in order:
        print(f"  {D.FEATURE_KEYS[k]:16s} {coefs[k]:+.3f}")


def train_final(kind="gbm"):
    """Fit the final model (GBM by default — it wins decisively) on ALL train wells and persist it."""
    import joblib
    rows = build(write=True, verbose=True)
    X, y, _, _ = _xy(rows)
    pipe = _pipe(kind).fit(X, y)
    LC.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipe": pipe, "features": D.FEATURE_KEYS, "kind": kind}, LC.MODELS / "link.joblib")
    print(f"\nsaved final {kind} model -> {LC.MODELS/'link.joblib'}")
