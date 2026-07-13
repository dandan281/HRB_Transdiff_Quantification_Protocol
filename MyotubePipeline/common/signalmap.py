"""Sample the 8-bit signal map (bg-subtracted primary @ display max) along traces.

Used by Stage 2/3 selection (brightness proxy) and Stage 4 flagging (dark-gap detection),
so the "is there fiber here?" question is answered identically everywhere.
"""
from __future__ import annotations
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from geometry import resample, polylen

WIN = 3            # half-window for windowed-max sampling
FIBER_T = 15       # 8-bit signal >= T  => fiber present (matches audit_b06.py)


def load_signal(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def sample_max(signal: np.ndarray, x: float, y: float, win: int = WIN) -> int:
    H, W = signal.shape
    xi, yi = int(round(x)), int(round(y))
    w = signal[max(0, yi - win):min(H, yi + win + 1), max(0, xi - win):min(W, xi + win + 1)]
    return int(w.max()) if w.size else 0


def trace_signal_profile(signal: np.ndarray, pts, step: float = 2.0):
    """Windowed-max signal value sampled at ~step px along the polyline."""
    return [sample_max(signal, x, y) for (x, y) in resample(pts, step)]


def brightness_mean(signal: np.ndarray, pts, step: float = 2.0) -> float:
    prof = trace_signal_profile(signal, pts, step)
    return float(np.mean(prof)) if prof else 0.0


def _effective_dark_thr(vals, thr: int, rel: float) -> float:
    """A point is 'dark' if below this. The relative term guards against carving up a uniformly
    DIM fibre: a real internal gap is dark relative to the fibre's own bright stretches, not just
    below an absolute floor. thr = hard absolute floor; rel*p75(profile) = relative component."""
    if not vals or rel <= 0:
        return float(thr)
    return max(float(thr), rel * float(np.percentile(vals, 75)))


def longest_dark_run_px(signal: np.ndarray, pts, step: float = 2.0, thr: int = FIBER_T,
                        rel: float = 0.0):
    """Length (px) of the longest contiguous dark run along the trace, and its mid-point.

    Returns (dark_px, total_px, gap_center_xy or None). The center is the midpoint of the
    longest dark stretch -- Stage 4 proposes splitting there.
    """
    samples = list(resample(pts, step))
    vals = [sample_max(signal, x, y) for (x, y) in samples]
    t = _effective_dark_thr(vals, thr, rel)
    best = cur = 0
    best_start = cur_start = 0
    for i, v in enumerate(vals):
        if v < t:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best:
                best = cur
                best_start = cur_start
        else:
            cur = 0
    if best == 0:
        return 0.0, len(vals) * step, None
    mid = samples[best_start + best // 2]
    return best * step, len(vals) * step, (float(mid[0]), float(mid[1]))


def rasterize_traces(traces, shape, dilate: int = 0, step: float = 2.0) -> np.ndarray:
    """Boolean mask of pixels covered by the given polylines, optionally dilated by `dilate` px."""
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    for pts in traces:
        for (x, y) in resample(pts, step):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < H and 0 <= xi < W:
                mask[yi, xi] = True
    if dilate > 0:
        mask = ndi.binary_dilation(mask, iterations=int(dilate))
    return mask


def overlap_fraction(mask: np.ndarray, pts, step: float = 2.0) -> float:
    """Fraction of a trace's sampled points that fall inside `mask`."""
    H, W = mask.shape
    pts_s = list(resample(pts, step))
    if not pts_s:
        return 0.0
    inside = sum(1 for (x, y) in pts_s
                 if 0 <= int(round(y)) < H and 0 <= int(round(x)) < W and mask[int(round(y)), int(round(x))])
    return inside / len(pts_s)


def all_dark_gaps_px(signal: np.ndarray, pts, step: float = 2.0, thr: int = FIBER_T,
                     min_gap_px: float = 30.0, rel: float = 0.0):
    """Return every internal dark gap >= min_gap_px as (length_px, center_xy). For split proposals.

    With rel>0 the 'dark' threshold is max(thr, rel*p75(profile)): an internal hole in a bright
    fibre is flagged, but a uniformly faint fibre (low p75) is NOT carved up."""
    samples = list(resample(pts, step))
    vals = [sample_max(signal, x, y) for (x, y) in samples]
    t = _effective_dark_thr(vals, thr, rel)
    gaps = []
    cur = 0; start = 0
    for i, v in enumerate(vals + [10**9]):   # sentinel high value to flush a trailing run
        if v < t:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur * step >= min_gap_px and start > 0 and i < len(vals):  # internal only
                mid = samples[start + cur // 2]
                gaps.append((cur * step, (float(mid[0]), float(mid[1]))))
            cur = 0
    return gaps
