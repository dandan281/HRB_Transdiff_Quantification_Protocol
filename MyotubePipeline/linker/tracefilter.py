"""Learned keep/drop confidence filter — applied AFTER the linker chains fragments.

Drops chained traces that are noise (don't lie on a real fibre) while keeping real fibres. Auto-
labeled from GT: a chained trace is KEEP=1 if it maps to a GT fibre, DROP=0 otherwise. Because it
runs post-chaining, a real fibre merged to full length is easy to separate from an isolated short
noise piece. Composes with the linker to cut the over-count and the residual %<300 bias.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

from benchmark import config as BC
from benchmark import geometry as g
from benchmark import io_load as io
from benchmark import matching as M
from . import config as LC
from . import dataset as D
from . import chain as CH

sys.path.insert(0, os.path.join(BC.PIPE, "common"))

FEATURE_KEYS = ["length_px", "signal_mean", "signal_frac", "straightness", "signal_p25"]
FIBER_T = 15   # 8-bit signal >= T => fibre present (matches common/signalmap)


def trace_features(trace, signal):
    from signalmap import trace_signal_profile
    length = g.polylen(trace)
    prof = np.asarray(trace_signal_profile(signal, trace, step=2.0), dtype=float)
    end = np.hypot(*(np.asarray(trace[-1], float) - np.asarray(trace[0], float)))
    return {
        "length_px": length,
        "signal_mean": float(prof.mean()) if prof.size else 0.0,
        "signal_frac": float(np.mean(prof >= FIBER_T)) if prof.size else 0.0,
        "straightness": float(end / length) if length > 0 else 0.0,
        "signal_p25": float(np.percentile(prof, 25)) if prof.size else 0.0,
    }


# ---- dataset (chain training wells, label each chained trace by GT overlap) ------------------
def build(write=True, verbose=True):
    from signalmap import load_signal
    pipe, feats = CH._load_model(LC.MODELS / "link.joblib")
    rows = []
    for w in LC.train_wells():
        frags = D.load_fragments(w["run_stem"])
        chained = CH.chain_fragments(frags, pipe, feats, threshold=0.5)
        gt = io.read_imagej_zip(BC.QPLATES / w["plate"] / w["roi_zip"])
        trace_gt = D.fragment_to_gt(chained, gt)      # chained trace -> GT idx or -1
        signal = load_signal(str(BC.RUNS / w["run_stem"] / "stage1_threshold" / "signal.png"))
        n_keep = 0
        for i, tr in enumerate(chained):
            label = 1 if trace_gt[i] >= 0 else 0
            n_keep += label
            rows.append(dict(stem=w["well_id"], plate=w["plate"], label=label, **trace_features(tr, signal)))
        if verbose:
            print(f"  {w['well_id']:22s} chained={len(chained):4d} keep={n_keep} drop={len(chained)-n_keep}")
    if write:
        LC.DATA.mkdir(parents=True, exist_ok=True)
        with open(LC.DATA / "traces.csv", "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=["stem", "plate", "label"] + FEATURE_KEYS)
            wtr.writeheader(); wtr.writerows(rows)
    pos = sum(r["label"] for r in rows)
    print(f"total: {len(rows)} chained traces, {pos} keep ({pos/len(rows):.1%})")
    return rows


def _xy(rows):
    X = np.array([[r[k] for k in FEATURE_KEYS] for r in rows], float)
    y = np.array([int(r["label"]) for r in rows], int)
    return X, y, np.array([r["stem"] for r in rows]), np.array([r["plate"] for r in rows])


def evaluate_cv(rows=None):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    if rows is None:
        rows = build(write=False, verbose=False)
    X, y, groups, _ = _xy(rows)
    print(f"\n=== keep/drop filter: {len(y)} traces, {y.sum()} keep ({y.mean():.1%}) ===")
    print("--- leave-one-well-out (keep-class) ---")
    f1s = []
    for held in sorted(set(groups)):
        tr, te = groups != held, groups == held
        if len(np.unique(y[tr])) < 2:
            continue
        clf = GradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
        pr = clf.predict_proba(X[te])[:, 1]; yh = (pr >= 0.5).astype(int)
        print(f"  hold {held:22s} P={precision_score(y[te],yh,zero_division=0):.3f} "
              f"R={recall_score(y[te],yh,zero_division=0):.3f} F1={f1_score(y[te],yh,zero_division=0):.3f} "
              f"AUC={roc_auc_score(y[te],pr):.3f}")
        f1s.append(f1_score(y[te], yh, zero_division=0))
    print(f"  mean LOWO keep-F1 = {np.mean(f1s):.3f}")
    clf = GradientBoostingClassifier(random_state=0).fit(X, y)
    imp = clf.feature_importances_
    print("--- feature importances ---")
    for k in np.argsort(-imp):
        print(f"  {FEATURE_KEYS[k]:14s} {imp[k]:.3f}")


def train_final():
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    rows = build(write=True, verbose=True)
    X, y, _, _ = _xy(rows)
    clf = GradientBoostingClassifier(random_state=0).fit(X, y)
    LC.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "features": FEATURE_KEYS}, LC.MODELS / "trace_filter.joblib")
    print(f"\nsaved keep/drop filter -> {LC.MODELS/'trace_filter.joblib'}")


# ---- end-to-end eval: chain -> filter, on held-out wells -------------------------------------
def evaluate_end_to_end(wids=None, link_thr=0.5, keep_thr=0.5):
    import joblib
    from signalmap import load_signal
    link_pipe, link_feats = CH._load_model(LC.MODELS / "link.joblib")
    fm = joblib.load(LC.MODELS / "trace_filter.joblib"); clf, ff = fm["clf"], fm["features"]
    wids = wids or ["P26_B02_Ctrl", "P26_C08_BR223_IGF1R", "P26_B06_ACT104_TrkA"]
    print(f"\nChain(bright+dim, thr={link_thr}) -> keep/drop filter (thr={keep_thr}), held-out\n"
          f"{'well':22s} {'method':20s} {'n':>5s} {'cDelta':>6s} {'P':>5s} {'R':>5s} {'F1':>5s} "
          f"{'%<300':>6s} {'GT%<300':>7s}")
    for wid in wids:
        w = BC.well_by_id(wid)
        frags = D.load_fragments(w["run_stem"])
        signal = load_signal(str(BC.RUNS / w["run_stem"] / "stage1_threshold" / "signal.png"))
        gt = io.read_imagej_zip(BC.QPLATES / w["plate"] / w["roi_zip"])
        gt_um = io.extract_lengths(BC.QPLATES / w["plate"] / w["results_csv"])
        gt_below = 100 * np.mean(gt_um < BC.THRESH_UM)
        final = io.read_traces(BC.pred_paths(w["run_stem"])[0])

        chained = CH.chain_fragments(frags, link_pipe, link_feats, link_thr)
        Xf = np.array([[trace_features(t, signal)[k] for k in ff] for t in chained], float)
        keep = clf.predict_proba(Xf)[:, 1] >= keep_thr
        filtered = [t for t, k in zip(chained, keep) if k]

        for name, polys in {"pipeline-final": final, "chain only": chained,
                            "chain+filter": filtered}.items():
            s = CH._score(polys, gt, gt_um)
            print(f"{wid if name=='pipeline-final' else '':22s} {name:20s} {s['n']:5d} "
                  f"{s['count_delta']:+6d} {s['P']:.3f} {s['R']:.3f} {s['f1']:.3f} "
                  f"{s['below']:6.1f} {gt_below:7.1f}")
        print()
