"""Geometry helpers shared by selection, merge, flagging, and numbering.

A 'trace' is a polyline: list[(x, y)]. Pure stdlib + numpy.
"""
from __future__ import annotations
import math


def polylen(pts) -> float:
    return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
               for i in range(1, len(pts)))


def midpoint(pts) -> tuple[float, float]:
    """The vertex nearest the arc-length midpoint (matches the old gen_render m=floor(np/2))."""
    m = len(pts) // 2
    return pts[m]


def endpoints(pts):
    return pts[0], pts[-1]


def resample(pts, step: float = 2.0):
    """Yield (x, y) points along the polyline at ~step px spacing (incl. last vertex)."""
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / step))
        for s in range(n):
            t = s / n
            yield (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    yield pts[-1]


def unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 0 else (0.0, 0.0)


def end_direction(pts, end: str, k: int = 6):
    """Outward unit direction at an endpoint ('head' or 'tail')."""
    kk = min(k, len(pts) - 1)
    if end == "tail":
        ax, ay = pts[-1]
        bx, by = pts[-1 - kk]
    else:
        ax, ay = pts[0]
        bx, by = pts[kk]
    return unit(ax - bx, ay - by)


def angle_between(u, v) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))))


def sharp_turns(pts, win: float = 22.0, angle_thr: float = 50.0):
    """Find sharp kinks: points where the trace bends by > angle_thr over a ~win-px arc.

    Two fibres that physically touch are often detected as one bent ridge (under-segmentation
    with NO dark gap). Real myotubes are fairly straight or gently curved, so a localised sharp
    bend is a split candidate. Returns [(bend_deg, (x, y)), ...], one per local maximum.
    """
    pl = list(resample(pts, win))
    if len(pl) < 5:
        return []
    bends = []
    for i in range(1, len(pl) - 1):
        a = unit(pl[i][0] - pl[i - 1][0], pl[i][1] - pl[i - 1][1])
        b = unit(pl[i + 1][0] - pl[i][0], pl[i + 1][1] - pl[i][1])
        bends.append(angle_between(a, b))
    out = []
    for i in range(1, len(bends) - 1):
        if bends[i] >= angle_thr and bends[i] >= bends[i - 1] and bends[i] >= bends[i + 1]:
            out.append((round(bends[i], 1), pl[i + 1]))   # pl index offset (+1 from bends)
    return out


def _cumlen(pts):
    cl = [0.0]
    for i in range(1, len(pts)):
        cl.append(cl[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    return cl


def arclen_of_point(pts, pt):
    """Arc-length position along `pts` of the closest point to `pt`."""
    best_d = float("inf")
    best_s = 0.0
    cl = _cumlen(pts)
    for i in range(1, len(pts)):
        ax, ay = pts[i - 1]
        bx, by = pts[i]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 == 0:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / seg2))
        px, py = ax + t * dx, ay + t * dy
        d = math.hypot(pt[0] - px, pt[1] - py)
        if d < best_d:
            best_d = d
            best_s = cl[i - 1] + t * math.hypot(dx, dy)
    return best_s


def point_at_arclen(pts, s):
    """Coordinate at arc-length `s` along the polyline."""
    cl = _cumlen(pts)
    if s <= 0:
        return pts[0]
    if s >= cl[-1]:
        return pts[-1]
    for i in range(1, len(pts)):
        if cl[i] >= s:
            seg = cl[i] - cl[i - 1]
            t = 0.0 if seg == 0 else (s - cl[i - 1]) / seg
            return (pts[i - 1][0] + t * (pts[i][0] - pts[i - 1][0]),
                    pts[i - 1][1] + t * (pts[i][1] - pts[i - 1][1]))
    return pts[-1]


def equal_split_points(pts, n):
    """n-1 cut points dividing the polyline into n equal-length pieces."""
    total = _cumlen(pts)[-1]
    return [point_at_arclen(pts, k * total / n) for k in range(1, n)]


def split_at_points(pts, cut_pts):
    """Split polyline `pts` at each point in `cut_pts`. Returns list of sub-polylines.

    The cut coordinate is INSERTED as a shared vertex at each boundary (so a cut falling between
    two far-apart vertices does not lose a segment). Sub-polylines shorter than 2 vertices are
    dropped. If no cut lands strictly inside the polyline, the trace is returned unchanged.
    """
    if not cut_pts:
        return [pts]
    cl = _cumlen(pts)
    total = cl[-1]
    cuts = sorted(s for s in (arclen_of_point(pts, c) for c in cut_pts) if 1.0 < s < total - 1.0)
    if not cuts:
        return [pts]

    def push(piece, p):
        if not piece or (abs(piece[-1][0] - p[0]) > 1e-6 or abs(piece[-1][1] - p[1]) > 1e-6):
            piece.append(p)

    pieces = []
    cur = [pts[0]]
    ci = 0
    for k in range(1, len(pts)):
        while ci < len(cuts) and cuts[ci] <= cl[k] + 1e-9:
            cp = point_at_arclen(pts, cuts[ci])
            push(cur, cp)
            if len(cur) >= 2:
                pieces.append(cur)
            cur = [cp]
            ci += 1
        push(cur, pts[k])
    if len(cur) >= 2:
        pieces.append(cur)
    return pieces if pieces else [pts]


def chain_merge(traces):
    """Greedily concatenate polylines into one by repeatedly joining the nearest endpoints."""
    remaining = [list(t) for t in traces if len(t) >= 2]
    if not remaining:
        return []
    chain = remaining.pop(0)
    while remaining:
        # find the remaining trace + orientation whose endpoint is closest to either chain end
        best = None
        for j, t in enumerate(remaining):
            for ce, cpt in (("head", chain[0]), ("tail", chain[-1])):
                for te, tpt in (("head", t[0]), ("tail", t[-1])):
                    d = math.hypot(cpt[0] - tpt[0], cpt[1] - tpt[1])
                    if best is None or d < best[0]:
                        best = (d, j, ce, te)
        _, j, ce, te = best
        t = remaining.pop(j)
        if te == "tail":
            t = t[::-1]               # so t starts at the joining end
        if ce == "tail":
            chain = chain + t
        else:
            chain = t[::-1] + chain
    return chain


def spatial_order(items, key=lambda t: t):
    """Return indices of `items` sorted top->bottom, left->right by their midpoint.

    `key(item)` -> polyline. Matches old pipeline: sort by mid_y*4000 + mid_x.
    """
    def sort_key(i):
        mx, my = midpoint(key(items[i]))
        return my * 4000.0 + mx
    return sorted(range(len(items)), key=sort_key)
