"""Phase 3: assemble fragments into full-length traces with the learned linker, and benchmark it
head-to-head against the geometric merge on the SAME raw fragments.

Greedy chaining: score every candidate endpoint pair with the trained model, join highest-proba
first (each fragment end used once, no cycles), then reconstruct each chain into one polyline.
Length is Euclidean vertex-sum * pixel_um (verified == Fiji).
"""
from __future__ import annotations

import os
import sys

import numpy as np

from benchmark import config as BC
from benchmark import geometry as g
from benchmark import io_load as io
from benchmark import matching as M
from . import config as LC
from . import dataset as D

sys.path.insert(0, os.path.join(BC.PIPE, "common"))


class _UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def _load_model(path):
    import joblib
    m = joblib.load(path)
    return m["pipe"], m["features"]


def score_candidates(frags, pipe, feats):
    lens = [g.polylen(p) for p in frags]
    cands = D.candidate_pairs(frags)
    if not cands:
        return []
    X = np.array([[D.pair_features(frags, lens, fi, ei, fj, ej)[k] for k in feats]
                  for (fi, ei, fj, ej) in cands], dtype=float)
    proba = pipe.predict_proba(X)[:, 1]
    return sorted(zip(proba, cands), key=lambda t: -t[0])


def chain_fragments(frags, pipe, feats, threshold=0.5):
    """Greedy join of candidate endpoint pairs -> reconstructed full-length polylines."""
    conn = {}                       # (frag,end) -> (frag,end)
    used = set()
    uf = _UF(len(frags))
    for proba, (fi, ei, fj, ej) in score_candidates(frags, pipe, feats):
        if proba < threshold:
            break
        if (fi, ei) in used or (fj, ej) in used or uf.find(fi) == uf.find(fj):
            continue
        conn[(fi, ei)] = (fj, ej); conn[(fj, ej)] = (fi, ei)
        used.add((fi, ei)); used.add((fj, ej)); uf.union(fi, fj)

    # reconstruct chains by walking from free ends
    visited, out = set(), []
    def free_ends(f): return [e for e in (0, 1) if (f, e) not in conn]
    for s in range(len(frags)):
        if s in visited or not free_ends(s):
            continue
        chain, f, entry = [], s, free_ends(s)[0]
        while f not in visited:
            visited.add(f); chain.append((f, entry))
            nxt = conn.get((f, 1 - entry))
            if nxt is None:
                break
            f, entry = nxt
        pts = []
        for (fr, en) in chain:
            seg = frags[fr] if en == 0 else frags[fr][::-1]
            pts.extend(np.asarray(seg))
        out.append(np.asarray(pts))
    for f in range(len(frags)):        # isolated fragments (both ends free, never started)
        if f not in visited:
            out.append(np.asarray(frags[f])); visited.add(f)
    return out


# ---- evaluation ------------------------------------------------------------------------------
def _score(pred_polys, gt_polys, gt_um):
    W = BC.IMAGE_SHAPE[1]
    gm = [g.rasterize(p, BC.DILATE_RADIUS_PX, W) for p in gt_polys]
    pm = [g.rasterize(p, BC.DILATE_RADIUS_PX, W) for p in pred_polys]
    ov = M.build_overlap(gm, pm)
    matches = M.greedy_match(ov, BC.IOU_THRESH)
    det = M.detection_metrics(matches, len(gm), len(pm))
    lens_um = np.array([g.polylen(p) * BC.PIXEL_UM for p in pred_polys])
    below = 100 * np.mean(lens_um < BC.THRESH_UM) if len(lens_um) else float("nan")
    gl = [g.polylen(gt_polys[i]) for i, _ in matches]
    pl = [g.polylen(pred_polys[j]) for _, j in matches]
    ratios = [p / q for p, q in zip(pl, gl) if q > 0]
    return dict(n=len(pred_polys), P=det["precision"], R=det["recall"], f1=det["f1"],
                below=below, count_delta=len(pred_polys) - len(gt_um),
                len_ratio=(float(np.median(ratios)) if ratios else float("nan")))


load_fragments = D.load_fragments


def evaluate(wids=None, threshold=0.5, model_path=None):
    from merge import merge
    from signalmap import load_signal

    pipe, feats = _load_model(model_path or (LC.MODELS / "link.joblib"))
    wids = wids or ["P26_B02_Ctrl", "P26_C08_BR223_IGF1R", "P26_B06_ACT104_TrkA"]
    print(f"Phase-3 bright vs bright+dim chaining (threshold={threshold}, model={('%s' % (model_path or 'link.joblib')).split(chr(92))[-1]})\n"
          f"{'well':22s} {'method':16s} {'n':>5s} {'cDelta':>6s} {'P':>5s} {'R':>5s} {'F1':>5s} "
          f"{'%<300':>6s} {'GT%<300':>7s}")
    for wid in wids:
        w = BC.well_by_id(wid)
        bright = load_fragments(w["run_stem"], include_dim=False)
        pooled = load_fragments(w["run_stem"], include_dim=True)
        signal = load_signal(str(BC.RUNS / w["run_stem"] / "stage1_threshold" / "signal.png"))
        gt = io.read_imagej_zip(BC.QPLATES / w["plate"] / w["roi_zip"])
        gt_um = io.extract_lengths(BC.QPLATES / w["plate"] / w["results_csv"])
        gt_below = 100 * np.mean(gt_um < BC.THRESH_UM)
        final = io.read_traces(BC.pred_paths(w["run_stem"])[0])   # the pipeline's actual output

        variants = {
            "pipeline-final": final,
            "geo(bright)": merge(bright, signal),
            "learn(bright)": chain_fragments(bright, pipe, feats, threshold),
            "learn(bright+dim)": chain_fragments(pooled, pipe, feats, threshold),
        }
        first = True
        for name, polys in variants.items():
            s = _score(polys, gt, gt_um)
            print(f"{wid if first else '':22s} {name:16s} {s['n']:5d} "
                  f"{s['count_delta']:+6d} {s['P']:.3f} {s['R']:.3f} {s['f1']:.3f} "
                  f"{s['below']:6.1f} {gt_below:7.1f}")
            first = False
        print()
