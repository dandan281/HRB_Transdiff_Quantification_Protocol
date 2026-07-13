"""Stage 4 -- flag over/under-segmentation for review.

Combines Stage 2 (bright) + Stage 3 (dim) + Stage 3's bright-mask-excluded traces, then proposes:

  * SPLIT (under-segmentation): a trace crossing an internal DARK gap (relative to the fibre's own
    brightness) OR a sharp KINK (two touching fibres seen as one bent ridge). Dark-gap points and
    kink points are tracked separately; only dark-gap points are eligible for auto-split.
  * MERGE (over-segmentation): two traces with close, collinear endpoints AND CONTINUOUS fiber
    signal between them -- probably one fibre wrongly broken apart.
  * OCCLUDED: a dim fibre Stage 3 hid behind the bright mask -- surfaced so the human can RESTORE
    a genuinely separate dim fibre (default: stay dropped).

Each case gets a zoomed crop and a confidence. Only ultra-confident merges are auto-applied;
everything else goes to the human review page. Writes `combined_traces.txt` (ordered union =
bright+dim+excluded, the index space cases refer to) and `flags.json`.

Usage:
  python flag.py --stage1 <dir> --stage2 <dir> --stage3 <dir> --out <stage4 dir> --stem <stem>
"""
from __future__ import annotations
import os
import sys
import json
import argparse

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from iohelpers import read_traces, write_traces, write_json  # noqa: E402
from geometry import (polylen, midpoint, end_direction, angle_between, unit,  # noqa: E402
                      spatial_order, sharp_turns, arclen_of_point)
from signalmap import load_signal, all_dark_gaps_px, brightness_mean, FIBER_T  # noqa: E402
import math  # noqa: E402

# learning layer (optional): pre-set each case default to your likely choice once a model exists
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "learning"))
try:
    from model import predict_default as _predict_default
except Exception:  # learning module/sklearn unavailable -> conservative defaults only
    _predict_default = None

PAD = 90  # crop padding (px)


def signal_between(signal, p, q, thr=FIBER_T):
    """Fraction of points on segment p->q with fiber signal present (windowed-max)."""
    H, W = signal.shape
    d = math.hypot(q[0] - p[0], q[1] - p[1])
    n = max(3, int(d / 2))
    ok = 0
    for s in range(1, n):
        t = s / n
        x = int(round(p[0] + (q[0] - p[0]) * t)); y = int(round(p[1] + (q[1] - p[1]) * t))
        w = signal[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        if w.size and w.max() >= thr:
            ok += 1
    return ok / max(1, n - 1)


DARK_REL = 0.45          # a dark gap must be < max(FIBER_T, 0.45*p75) of the fibre's own profile


def _dedupe_points(trace, points, min_spacing=14.0, end_margin=14.0, exclude=()):
    """Keep only points that are >= min_spacing apart (by arc-length), away from the endpoints,
    and not within min_spacing of an already-kept `exclude` point. Returns [[x,y],...]."""
    total = polylen(trace)
    kept_s = [arclen_of_point(trace, p) for p in exclude]
    kept = []
    for p in points:
        s = arclen_of_point(trace, p)
        if s <= end_margin or s >= total - end_margin:
            continue
        if any(abs(s - ks) < min_spacing for ks in kept_s):
            continue
        kept_s.append(s)
        kept.append([round(p[0], 1), round(p[1], 1)])
    return kept


def find_splits(traces, indices, signal, split_min_gap, hard_split, auto_split, kink_angle):
    """Two kinds of under-segmentation: a DARK internal gap (merge bridged separate fibres) and a
    sharp KINK (two touching fibres detected as one bent ridge, no dark gap).

    Gap-derived cut points and kink-derived cut points are stored separately: only the gap points
    are eligible for auto-split, and near-duplicate proposals are deduped so the per-point UI
    reflects cuts reconcile can actually realise."""
    cases = []
    for i in indices:
        t = traces[i]
        gaps = all_dark_gaps_px(signal, t, min_gap_px=split_min_gap, rel=DARK_REL)
        kinks = sharp_turns(t, angle_thr=kink_angle)
        gap_pts = _dedupe_points(t, [c for _, c in gaps])
        kink_pts = _dedupe_points(t, [p for _, p in kinks], exclude=[tuple(p) for p in gap_pts])
        if not gap_pts and not kink_pts:
            continue
        proposed = gap_pts + kink_pts
        longest = max((g[0] for g in gaps), default=0.0)
        max_bend = max((b for b, _ in kinks), default=0.0)
        reasons = []
        if gap_pts:
            reasons.append(f"dark gap(s) longest {longest:.0f}px")
        if kink_pts:
            reasons.append(f"sharp bend(s) max {max_bend:.0f}deg (review only)")
        # dark gaps are strong evidence; kinks are weak (curved real fibres also kink) -> capped low
        gap_conf = min(1.0, longest / 120.0) if gap_pts else 0.0
        kink_conf = min(0.55, (max_bend - kink_angle) / 120.0 + 0.2) if kink_pts else 0.0
        cases.append(dict(
            id=f"split_{i:04d}", type="split", trace_indices=[i],
            reason="; ".join(reasons),
            longest_dark_px=round(longest, 1), max_bend_deg=round(max_bend, 1),
            proposed_splits=proposed,
            gap_splits=gap_pts,                 # auto-split uses ONLY these
            confidence=round(float(max(gap_conf, kink_conf)), 3),
            auto=bool(auto_split and gap_pts and longest >= hard_split),   # only dark-gap auto-splits
            features=dict(length_px=round(polylen(t), 1),
                          brightness=round(brightness_mean(signal, t), 1),
                          longest_dark_px=round(longest, 1), max_bend_deg=round(max_bend, 1),
                          n_gap=len(gap_pts), n_kink=len(kink_pts)),
        ))
    return cases


def find_occluded(traces, occ_indices, signal):
    """One review case per dim fibre Stage 3 hid behind the bright mask: default DROP (it is
    usually a bright fragment), but the human can RESTORE a genuinely separate dim fibre."""
    out = []
    for k in occ_indices:
        t = traces[k]
        out.append(dict(
            id=f"occlude_{k:04d}", type="occluded", trace_indices=[k],
            reason="excluded by bright-mask (>60% over a bright fibre); restore if a separate dim fibre",
            proposed_splits=[], confidence=0.2, auto=False,
            features=dict(length_px=round(polylen(t), 1),
                          brightness=round(brightness_mean(signal, t), 1)),
        ))
    return out


def find_merges(traces, indices, signal, merge_gap, merge_angle, merge_continuity,
                auto_merge_gap, auto_merge_angle):
    # endpoint list with outward direction (active traces only)
    eps = []
    for i in indices:
        t = traces[i]
        eps.append((i, "head", t[0], end_direction(t, "head")))
        eps.append((i, "tail", t[-1], end_direction(t, "tail")))
    cell = max(merge_gap, 1.0)
    grid = {}
    for k, (i, end, pt, d) in enumerate(eps):
        grid.setdefault((int(pt[0] // cell), int(pt[1] // cell)), []).append(k)

    seen = set()
    cases = []
    for k, (i, ei, pi, di) in enumerate(eps):
        cx, cy = int(pi[0] // cell), int(pi[1] // cell)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for m in grid.get((gx, gy), ()):
                    j, ej, pj, dj = eps[m][0], eps[m][1], eps[m][2], eps[m][3]
                    if j <= i:
                        continue
                    gap = math.hypot(pj[0] - pi[0], pj[1] - pi[1])
                    if gap < 1e-6 or gap > merge_gap:
                        continue
                    conn = unit(pj[0] - pi[0], pj[1] - pi[1])
                    a1 = angle_between(di, conn)
                    a2 = angle_between(dj, (-conn[0], -conn[1]))
                    if a1 > merge_angle or a2 > merge_angle:
                        continue
                    cont = signal_between(signal, pi, pj, merge_continuity_thr())
                    if cont < merge_continuity:
                        continue
                    pair = (i, j)
                    score = (1 - gap / merge_gap) * (1 - (a1 + a2) / (2 * merge_angle)) * cont
                    if pair in seen:
                        continue
                    seen.add(pair)
                    auto = (gap <= auto_merge_gap and max(a1, a2) <= auto_merge_angle and cont >= 0.8)
                    li, lj = polylen(traces[i]), polylen(traces[j])
                    bi, bj = brightness_mean(signal, traces[i]), brightness_mean(signal, traces[j])
                    cases.append(dict(
                        id=f"merge_{i:04d}_{j:04d}", type="merge", trace_indices=[i, j],
                        reason=f"collinear, gap {gap:.0f}px, angle {max(a1,a2):.0f}deg, signal {cont:.2f}",
                        gap_px=round(gap, 1), angle_deg=round(max(a1, a2), 1),
                        continuity=round(cont, 3), confidence=round(float(min(1.0, score)), 3),
                        auto=bool(auto),
                        features=dict(gap_px=round(gap, 1), angle_deg=round(max(a1, a2), 1),
                                      continuity=round(cont, 3),
                                      len_min=round(min(li, lj), 1), len_max=round(max(li, lj), 1),
                                      bright_min=round(min(bi, bj), 1), bright_max=round(max(bi, bj), 1)),
                    ))
    return cases


def merge_continuity_thr():
    return FIBER_T


def bbox_of(traces, idxs, W, H):
    xs, ys = [], []
    for i in idxs:
        for (x, y) in traces[i]:
            xs.append(x); ys.append(y)
    x0 = max(0, int(min(xs)) - PAD); y0 = max(0, int(min(ys)) - PAD)
    x1 = min(W, int(max(xs)) + PAD); y1 = min(H, int(max(ys)) + PAD)
    return [x0, y0, x1, y1]


def make_raw_crop(base_img, traces, case, out_raw):
    """Save the CLEAN raw crop (the page draws my proposed overlay as canvas vectors, so no baked
    overlay PNG is needed). Returns (bbox, scale) so the page can map a click to full-image coords:
        full_x = bbox[0] + crop_natural_x / scale ;  full_y = bbox[1] + crop_natural_y / scale."""
    W, H = base_img.size
    bbox = bbox_of(traces, case["trace_indices"], W, H)
    crop = base_img.crop(bbox).convert("RGB")
    sc = 2 if (bbox[2] - bbox[0]) < 500 else 1   # zoom small crops 2x for visibility
    if sc != 1:
        crop = crop.resize(((bbox[2] - bbox[0]) * sc, (bbox[3] - bbox[1]) * sc), Image.BILINEAR)
    crop.save(out_raw)
    return bbox, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--stage3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--split-min-gap", type=float, default=35.0)
    ap.add_argument("--hard-split", type=float, default=90.0)
    ap.add_argument("--kink-angle", type=float, default=70.0)   # higher -> fewer false kink flags
    ap.add_argument("--auto-split", action="store_true")
    # Wider net to PROPOSE more fragment-join candidates for review (curvy fibres meet at >22deg
    # over bigger gaps). The signal-continuity gate (0.6) still keeps proposals to pairs with real
    # fibre between them, not distinct fibres across a dark gap. AUTO thresholds stay strict, so
    # nothing merges without a human click.
    ap.add_argument("--merge-gap", type=float, default=90.0)
    ap.add_argument("--merge-angle", type=float, default=32.0)
    ap.add_argument("--merge-continuity", type=float, default=0.6)
    ap.add_argument("--auto-merge-gap", type=float, default=16.0)
    ap.add_argument("--auto-merge-angle", type=float, default=10.0)
    a = ap.parse_args()

    signal = load_signal(os.path.join(a.stage1, "signal.png"))
    H, W = signal.shape

    bright = read_traces(os.path.join(a.stage2, "bright_traces.txt"))
    dim = read_traces(os.path.join(a.stage3, "dim_traces.txt"))
    excluded = read_traces(os.path.join(a.stage3, "excluded_by_brightmask.txt"))
    combined = bright + dim + excluded
    origin = (["bright"] * len(bright)) + (["dim"] * len(dim)) + (["excluded"] * len(excluded))
    order = spatial_order(combined)
    combined = [combined[i] for i in order]
    origin = [origin[i] for i in order]
    write_traces(os.path.join(a.out, "combined_traces.txt"), combined)

    active = [i for i, o in enumerate(origin) if o != "excluded"]   # bright + dim get detection
    occ_idx = [i for i, o in enumerate(origin) if o == "excluded"]  # excluded get a restore case
    splits = find_splits(combined, active, signal, a.split_min_gap, a.hard_split,
                         a.auto_split, a.kink_angle)
    merges = find_merges(combined, active, signal, a.merge_gap, a.merge_angle, a.merge_continuity,
                         a.auto_merge_gap, a.auto_merge_angle)
    occluded = find_occluded(combined, occ_idx, signal)
    cases = splits + merges + occluded

    # learned defaults: if a model for this case type exists, pre-set the case to your likely choice
    n_learned = 0
    if _predict_default is not None:
        for c in cases:
            try:                                   # learning is best-effort: NEVER crash detection
                act, proba = _predict_default(c["type"], c.get("features", {}))
            except Exception:
                act, proba = None, None            # bad/incompatible model -> conservative defaults
            if act is None:
                continue
            # a learned 'split' on a kink-only case (no dark-gap points) isn't auto-applicable;
            # leave it conservative so the badge, the shown radio, and the emitted action all agree.
            if c["type"] == "split" and act == "split" and not c.get("gap_splits"):
                continue
            c["learned_default"], c["learned_proba"] = act, proba
            n_learned += 1

    # crop base: prefer the composite (shows DAPI + overlap), else the green adjusted primary
    comp = os.path.join(a.out, f"{a.stem}_composite.png")
    meta = json.load(open(os.path.join(a.stage1, "metadata.json"), encoding="utf-8"))
    p = meta["channels"]["primary"]
    adj = os.path.join(a.stage1, f"ch{p}_adjusted8.tif")
    sigp = os.path.join(a.stage1, "signal.png")
    if os.path.exists(comp):
        base = Image.open(comp).convert("RGB")
    elif os.path.exists(adj):
        base = Image.open(adj).convert("RGB")
    else:                                            # last resort: the fiber signal map
        base = Image.open(sigp).convert("RGB")
    crops_dir = os.path.join(a.out, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    for c in cases:
        c["crop_raw"] = os.path.join("crops", c["id"] + ".png").replace("\\", "/")
        bbox, sc = make_raw_crop(base, combined, c, os.path.join(a.out, "crops", c["id"] + ".png"))
        c["bbox"] = bbox
        c["crop_scale"] = sc          # full_x = bbox[0] + crop_natural_x / scale (for click->coord)
        # the involved trace polylines (full-image coords) so the page can draw them as INTERACTIVE
        # vectors over the raw image (not a baked PNG) -- the user can highlight/reject/redraw them.
        c["trace_polys"] = [[[round(x, 1), round(y, 1)] for (x, y) in combined[i]]
                            for i in c["trace_indices"]]

    cases.sort(key=lambda c: -c["confidence"])
    flags = dict(
        stem=a.stem, n_bright=len(bright), n_dim=len(dim), n_excluded=len(excluded),
        n_combined=len(combined),
        params=dict(split_min_gap=a.split_min_gap, merge_gap=a.merge_gap, merge_angle=a.merge_angle,
                    merge_continuity=a.merge_continuity, kink_angle=a.kink_angle),
        origin=origin,
        n_auto=sum(1 for c in cases if c["auto"]),
        n_review=sum(1 for c in cases if not c["auto"]),
        n_learned=n_learned,
        cases=cases,
    )
    write_json(os.path.join(a.out, "flags.json"), flags)
    print(f"combined={len(combined)} (bright={len(bright)} dim={len(dim)} excluded={len(excluded)}) "
          f"splits={len(splits)} merges={len(merges)} occluded={len(occluded)} "
          f"auto={flags['n_auto']} review={flags['n_review']} learned_defaults={n_learned}")


if __name__ == "__main__":
    main()
