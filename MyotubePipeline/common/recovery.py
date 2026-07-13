"""Recover missed fiber centerlines from residual signal.

Ridge Detection can miss broad, low-contrast, or texture-like Desmin-positive myotubes. This module
creates conservative extra candidates by skeletonizing signal that is not already close to an
existing trace, then keeping long/bright skeleton paths.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

from geometry import polylen
from geometry import end_direction, angle_between, unit
from signalmap import brightness_mean, overlap_fraction, rasterize_traces


def _signal_fraction(signal: np.ndarray, a, b, *, threshold: int, win: int = 3) -> float:
    h, w = signal.shape
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    n_steps = max(3, int(dist / 3))
    ok = total = 0
    for s in range(1, n_steps):
        t = s / n_steps
        x = int(round(a[0] + (b[0] - a[0]) * t))
        y = int(round(a[1] + (b[1] - a[1]) * t))
        patch = signal[max(0, y - win):min(h, y + win + 1), max(0, x - win):min(w, x + win + 1)]
        if patch.size and patch.max() >= threshold:
            ok += 1
        total += 1
    return ok / max(1, total)


def stitch_continuous_traces(
    traces,
    signal: np.ndarray,
    *,
    gap_max: float = 300.0,
    angle_max: float = 30.0,
    signal_threshold: int = 15,
    min_signal_fraction: float = 0.75,
):
    """Greedily stitch fragmented recall traces across bright, aligned signal bridges.

    This is intentionally stricter than a graph union: each chain endpoint accepts only its best
    matching neighbor. That avoids fusing whole dense neighborhoods while still fixing cases where
    one intact myotube was recovered as several collinear pieces.
    """
    remaining = [(list(t), [i]) for i, t in enumerate(traces) if len(t) >= 2]
    stitched = []
    groups = []

    while remaining:
        start = max(range(len(remaining)), key=lambda k: polylen(remaining[k][0]))
        chain, ids = remaining.pop(start)
        while True:
            changed = False
            for which in ("tail", "head"):
                pt = chain[-1] if which == "tail" else chain[0]
                d_chain = end_direction(chain, which)
                best = None
                for j, (other, other_ids) in enumerate(remaining):
                    for end, opt in (("head", other[0]), ("tail", other[-1])):
                        gap = math.hypot(opt[0] - pt[0], opt[1] - pt[1])
                        if gap < 1e-6 or gap > gap_max:
                            continue
                        conn = unit(opt[0] - pt[0], opt[1] - pt[1])
                        d_other = end_direction(other, end)
                        a1 = angle_between(d_chain, conn)
                        a2 = angle_between(d_other, (-conn[0], -conn[1]))
                        if max(a1, a2) > angle_max:
                            continue
                        frac = _signal_fraction(signal, pt, opt, threshold=signal_threshold)
                        if frac < min_signal_fraction:
                            continue
                        score = max(a1, a2) * 3.0 + gap * 0.05 - frac * 20.0
                        if best is None or score < best[0]:
                            best = (score, j, end)
                if best is None:
                    continue
                _, j, end = best
                other, other_ids = remaining.pop(j)
                if which == "tail":
                    if end == "head":
                        chain.extend(other)
                        ids.extend(other_ids)
                    else:
                        chain.extend(other[::-1])
                        ids.extend(other_ids[::-1])
                else:
                    if end == "tail":
                        chain[:0] = other
                        ids[:0] = other_ids
                    else:
                        chain[:0] = other[::-1]
                        ids[:0] = other_ids[::-1]
                changed = True
                break
            if not changed:
                break
        stitched.append(chain)
        groups.append(ids)
    return stitched, groups


def _neighbors(pt, pixels):
    r, c = pt
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            q = (r + dr, c + dc)
            if q in pixels:
                yield q


def _farthest(start, pixels):
    dist = {start: 0.0}
    prev = {start: None}
    pq = [(0.0, start)]
    while pq:
        d, p = heapq.heappop(pq)
        if d != dist[p]:
            continue
        for q in _neighbors(p, pixels):
            w = math.hypot(q[0] - p[0], q[1] - p[1])
            nd = d + w
            if nd < dist.get(q, float("inf")):
                dist[q] = nd
                prev[q] = p
                heapq.heappush(pq, (nd, q))
    far = max(dist, key=dist.get)
    return far, dist[far], prev


def _longest_skeleton_path(coords):
    """Return a longest-path approximation through one skeleton component as [(x,y),...]."""
    pixels = {tuple(map(int, p)) for p in coords}  # skimage gives (row, col)
    if len(pixels) < 2:
        return []
    endpoints = [p for p in pixels if sum(1 for _ in _neighbors(p, pixels)) == 1]
    seed = endpoints[0] if endpoints else next(iter(pixels))
    a, _, _ = _farthest(seed, pixels)
    b, _, prev = _farthest(a, pixels)
    path = []
    p = b
    while p is not None:
        path.append((float(p[1]), float(p[0])))
        p = prev[p]
    path.reverse()
    return path


def recover_signal_traces(
    signal: np.ndarray,
    existing_traces,
    *,
    threshold: int = 25,
    existing_dilate: int = 8,
    dedupe_dilate: int = 10,
    min_object_px: int = 80,
    min_skeleton_px: int = 30,
    min_len_px: float = 80.0,
    min_brightness: float = 18.0,
    max_overlap: float = 0.35,
    decimate: int = 6,
):
    """Return extra trace candidates from signal not already covered by `existing_traces`.

    These are candidates, not final biological truth. They should be reviewed, or when used to
    patch a finalized well, inspected as a separate recall output first.
    """
    base_mask = signal >= threshold
    base_mask = morphology.remove_small_objects(base_mask, min_size=min_object_px)
    base_mask = ndi.binary_opening(base_mask, structure=np.ones((2, 2), dtype=bool))

    existing_mask = rasterize_traces(existing_traces, signal.shape, dilate=existing_dilate)
    residual = base_mask & ~existing_mask
    residual = ndi.binary_closing(residual, structure=np.ones((3, 3), dtype=bool))
    residual = morphology.remove_small_objects(residual, min_size=max(1, min_object_px // 2))

    skeleton = morphology.skeletonize(residual)
    labels, n_labels = ndi.label(skeleton, structure=np.ones((3, 3), dtype=int))

    accepted = []
    accepted_mask = rasterize_traces(existing_traces, signal.shape, dilate=dedupe_dilate)
    for label_id in range(1, n_labels + 1):
        coords = np.argwhere(labels == label_id)
        if len(coords) < min_skeleton_px:
            continue
        path = _longest_skeleton_path(coords)
        if len(path) < 2:
            continue
        if polylen(path) < min_len_px:
            continue
        if brightness_mean(signal, path) < min_brightness:
            continue
        if overlap_fraction(accepted_mask, path) >= max_overlap:
            continue
        compact = path[::max(1, int(decimate))]
        if compact[-1] != path[-1]:
            compact.append(path[-1])
        if polylen(compact) < min_len_px:
            continue
        accepted.append(compact)
        accepted_mask |= rasterize_traces([compact], signal.shape, dilate=dedupe_dilate)
    return accepted
