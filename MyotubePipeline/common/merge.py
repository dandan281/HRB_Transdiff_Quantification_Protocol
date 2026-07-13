"""Collinear segment stitching with a signal-continuity gate (refactored from merge_gen.py).

`merge(segs, signal)` stitches ridge fragments into traces but never bridges a gap that lacks
fiber signal in `signal` (the 8-bit bg-subtracted primary @ display max). Reused by Stage 2 and
Stage 3. Parameters are exposed so the bright/dim regimes can tune the gate.
"""
from __future__ import annotations
import math

import numpy as np

# defaults from the validated merge_gen.py
GAP_MAX, GAP_MIN, ANGLE, DIRPTS, CELL = 170.0, 45.0, 33.0, 6, 170.0
SIG_T, SIG_FRAC, GAP_CHECK_MIN = 18, 0.55, 18


def _unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 0 else (0.0, 0.0)


def _out_dir(pts, end, dirpts):
    k = min(dirpts, len(pts) - 1)
    if end == "tail":
        ax, ay = pts[-1]; bx, by = pts[-1 - k]
    else:
        ax, ay = pts[0]; bx, by = pts[k]
    return _unit(ax - bx, ay - by)


def _ang(u, v):
    return math.degrees(math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))))


def merge(segs, signal: np.ndarray, *, gap_max=GAP_MAX, gap_min=GAP_MIN, angle=ANGLE,
          dirpts=DIRPTS, cell=CELL, sig_t=SIG_T, sig_frac=SIG_FRAC, gap_check_min=GAP_CHECK_MIN):
    """segs: list[list[(x,y)]]; signal: HxW uint8. Returns merged list[list[(x,y)]]."""
    H, W = signal.shape

    def signal_ok(ax, ay, bx, by):
        dist = math.hypot(bx - ax, by - ay)
        if dist < gap_check_min:
            return True
        nsteps = max(3, int(dist / 3))
        ok = tot = 0
        for s in range(1, nsteps):
            t = s / nsteps
            xi = int(round(ax + (bx - ax) * t)); yi = int(round(ay + (by - ay) * t))
            win = signal[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4]
            if win.size and win.max() >= sig_t:
                ok += 1
            tot += 1
        return tot == 0 or ok / tot >= sig_frac

    def neighbors(grid, x, y):
        cx, cy = int(x // cell), int(y // cell)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                yield from grid.get((gx, gy), ())

    def best_match(used, pt, d_chain, grid):
        best = None
        for j, end, px, py in neighbors(grid, pt[0], pt[1]):
            if used[j]:
                continue
            gx, gy = px - pt[0], py - pt[1]
            dist = math.hypot(gx, gy)
            if dist < 1e-6 or dist > gap_max:
                continue
            ghat = (gx / dist, gy / dist)
            d_out = _out_dir(segs[j], end, dirpts)
            a1 = _ang(d_chain, ghat); a2 = _ang(d_chain, (-d_out[0], -d_out[1]))
            if a1 > angle or a2 > angle:
                continue
            if dist > gap_max - (gap_max - gap_min) * ((a1 + a2) / (2 * angle)):
                continue
            if not signal_ok(pt[0], pt[1], px, py):
                continue
            score = a1 + a2 + dist * 0.05
            if best is None or score < best[0]:
                best = (score, j, end)
        return best

    used = [False] * len(segs)
    grid = {}
    for j, pts in enumerate(segs):
        for end, (px, py) in (("head", pts[0]), ("tail", pts[-1])):
            grid.setdefault((int(px // cell), int(py // cell)), []).append((j, end, px, py))
    out = []
    for s in sorted(range(len(segs)), key=lambda j: -len(segs[j])):
        if used[s]:
            continue
        chain = list(segs[s]); used[s] = True
        for which in ("tail", "head"):
            while True:
                pt = chain[-1] if which == "tail" else chain[0]
                m = best_match(used, pt, _out_dir(chain, which, dirpts), grid)
                if m is None:
                    break
                _, j, end = m
                if which == "tail":
                    chain.extend(segs[j] if end == "head" else segs[j][::-1])
                else:
                    chain[:0] = segs[j] if end == "tail" else segs[j][::-1]
                used[j] = True
        out.append(chain)
    return out
