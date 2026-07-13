"""Endpoint-extend: grow under-traced fibres outward along continuing signal.

Applied last (after chain -> filter). From each trace endpoint, greedily step in a forward cone
toward the direction of maximum 8-bit fibre signal, stopping when the signal drops below a threshold
(a true fibre end) or a max distance is reached. Self-limiting: at a real end the signal is already
gone, so nothing is added. Targets the residual under-tracing (matched fibres ~70-83% of GT length).
"""
from __future__ import annotations

import os
import sys
from math import cos, radians, sin

import numpy as np

from benchmark import config as BC
from benchmark import io_load as io
from . import config as LC
from . import chain as CH
from . import dataset as D

sys.path.insert(0, os.path.join(BC.PIPE, "common"))


def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 0 else np.array([0.0, 0.0])


def _rot(v, a):
    c, s = cos(a), sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _extend_end(pts, signal, end, step, cone_deg, n_dirs, thr, max_dist, smooth):
    from signalmap import sample_max
    H, W = signal.shape
    k = min(6, len(pts) - 1)
    if end == 0:
        p = np.asarray(pts[0], float); h = _unit(pts[0] - pts[k])
    else:
        p = np.asarray(pts[-1], float); h = _unit(pts[-1] - pts[-1 - k])
    added, dist = [], 0.0
    while dist < max_dist:
        best, best_s = None, -1.0
        for a in np.linspace(-cone_deg, cone_deg, n_dirs):
            d = _rot(h, radians(a))
            q = p + step * d
            s = sample_max(signal, q[0], q[1])
            if s > best_s:
                best_s, best = s, (q, d)
        if best_s < thr:
            break
        q, d = best
        if not (0 <= q[0] < W and 0 <= q[1] < H):
            break
        added.append(q)
        h = _unit((1 - smooth) * h + smooth * d)
        p = q; dist += step
    return added


def extend_trace(trace, signal, step=3.0, cone_deg=40.0, n_dirs=9, thr=50.0,
                 max_dist=160.0, smooth=0.5):
    pts = [np.asarray(v, float) for v in trace]
    if len(pts) < 2:
        return np.asarray(trace)
    head = _extend_end(pts, signal, 0, step, cone_deg, n_dirs, thr, max_dist, smooth)
    tail = _extend_end(pts, signal, 1, step, cone_deg, n_dirs, thr, max_dist, smooth)
    out = [np.asarray(v) for v in reversed(head)] + list(np.asarray(trace)) + tail
    return np.asarray(out)


def extend_all(traces, signal, **kw):
    return [extend_trace(t, signal, **kw) for t in traces]


# ---- full-pipeline evaluation: chain -> filter -> extend, on held-out wells -------------------
def evaluate_full(wids=None, link_thr=0.5, keep_thr=0.5, extend_thr=50.0):
    import joblib
    from signalmap import load_signal
    link_pipe, link_feats = CH._load_model(LC.MODELS / "link.joblib")
    fm = joblib.load(LC.MODELS / "trace_filter.joblib"); clf, ff = fm["clf"], fm["features"]
    from . import tracefilter as TF
    wids = wids or ["P26_B02_Ctrl", "P26_C08_BR223_IGF1R", "P26_B06_ACT104_TrkA"]
    print(f"chain(thr={link_thr}) -> filter(thr={keep_thr}) -> extend(sig>={extend_thr}), held-out\n"
          f"{'well':22s} {'method':22s} {'n':>5s} {'cDelta':>6s} {'P':>5s} {'R':>5s} {'F1':>5s} "
          f"{'lenRatio':>8s} {'%<300':>6s} {'GT%<300':>7s}")
    for wid in wids:
        w = BC.well_by_id(wid)
        frags = D.load_fragments(w["run_stem"])
        signal = load_signal(str(BC.RUNS / w["run_stem"] / "stage1_threshold" / "signal.png"))
        gt = io.read_imagej_zip(BC.QPLATES / w["plate"] / w["roi_zip"])
        gt_um = io.extract_lengths(BC.QPLATES / w["plate"] / w["results_csv"])
        gt_below = 100 * np.mean(gt_um < BC.THRESH_UM)
        final = io.read_traces(BC.pred_paths(w["run_stem"])[0])

        chained = CH.chain_fragments(frags, link_pipe, link_feats, link_thr)
        Xf = np.array([[TF.trace_features(t, signal)[k] for k in ff] for t in chained], float)
        filt = [t for t, keep in zip(chained, clf.predict_proba(Xf)[:, 1] >= keep_thr) if keep]
        extended = extend_all(filt, signal, thr=extend_thr)

        for name, polys in {"pipeline-final": final, "chain+filter": filt,
                            "chain+filter+extend": extended}.items():
            s = CH._score(polys, gt, gt_um)
            print(f"{wid if name=='pipeline-final' else '':22s} {name:22s} {s['n']:5d} "
                  f"{s['count_delta']:+6d} {s['P']:.3f} {s['R']:.3f} {s['f1']:.3f} "
                  f"{s['len_ratio']:8.3f} {s['below']:6.1f} {gt_below:7.1f}")
        print()
