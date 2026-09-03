"""The standard length-class summary — the operator's metric of record.

The operator reads the PROPORTION of myotubes per length class, not total
length ("when you over-count both short and long, it basically cancels
out"). Every per-well quantification should therefore report these shares
beside counts and totals. One definition, used everywhere:

    50-150 / 150-300 / 300-500 / 500-800 / >800  (um, >= 50 um counted)

Also home to the freeline measurement convention (2026-08-27 §7d): raw
point-to-point arc of a freehand trace inflates 10-15% with drawing
jitter; Fiji smooths before measuring. `smooth_polyline` is the 5-point
moving average that reproduces Fiji's freeline Length to ~1.3% — apply it
to any HUMAN freehand trace before measuring; tracer walk paths are
already smooth (raw/smoothed = 1.007).
"""
from __future__ import annotations

import numpy as np

BINS_UM = (50.0, 150.0, 300.0, 500.0, 800.0, np.inf)
LABELS = ("50-150", "150-300", "300-500", "500-800", ">800")


def smooth_polyline(p: np.ndarray, w: int = 5) -> np.ndarray:
    """Fiji-style smoothing for freehand traces.

    A moving average whose window stays SYMMETRIC about each point and
    shrinks near the ends (radius = min(w//2, i, n-1-i)). Symmetry is the
    property that matters: it reproduces a straight line exactly and leaves
    the endpoints untouched by construction. The obvious implementation --
    edge-padding plus a fixed window -- fails both: padding replicates the
    first point, so on a straight line the second point is pulled inward
    (1 -> 1.2 at w=5), putting a small kink at each end of every trace.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n < w + 2:
        return p
    idx = np.arange(n)
    r = np.minimum(np.minimum(idx, n - 1 - idx), w // 2)
    cs = np.vstack([np.zeros((1, p.shape[1])), np.cumsum(p, axis=0)])
    lo, hi = idx - r, idx + r + 1
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


def arc_um(p, pixel_um: float, *, smoothed: bool = False) -> float:
    p = np.asarray(p, dtype=float)
    if smoothed:
        p = smooth_polyline(p)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) * pixel_um


def class_shares(lengths_um) -> dict:
    """{label: share} over fibres >= 50 um, plus n. Shares sum to 1."""
    lens = np.asarray([v for v in lengths_um if v >= BINS_UM[0]], dtype=float)
    h, _ = np.histogram(lens, bins=BINS_UM)
    n = int(h.sum())
    out = {lbl: (round(float(c) / n, 4) if n else 0.0)
           for lbl, c in zip(LABELS, h)}
    out["n"] = n
    return out


def format_shares(shares: dict) -> str:
    return "  ".join(f"{lbl} {shares[lbl] * 100:4.1f}%" for lbl in LABELS)
