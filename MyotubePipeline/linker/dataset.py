"""Build the fragment-linker training dataset: candidate endpoint-pairs -> features + GT label."""
from __future__ import annotations

import numpy as np

from benchmark import config as BC
from benchmark import geometry as g
from benchmark import io_load as io
from . import config as LC

FEATURE_KEYS = ["gap_px", "align_deg", "a1_deg", "a2_deg", "len_min", "len_max", "len_ratio",
                "end_dist_ratio"]


def load_fragments(run_stem, include_dim=None):
    """Raw bright fragments, optionally pooled with stage-3 dim pieces (dim_traces.txt) so the
    linker can bridge dim gaps between bright segments of the same fibre."""
    include_dim = LC.INCLUDE_DIM if include_dim is None else include_dim
    frags = io.read_traces(LC.fragments_path(run_stem))
    if include_dim:
        dim = BC.RUNS / run_stem / "stage3_dim" / "dim_traces.txt"
        if dim.exists():
            frags = frags + io.read_traces(dim)
    return frags


# ---- fragment endpoints + headings ----------------------------------------------------------
def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    return (v / n) if n > 0 else np.array([0.0, 0.0])


def endpoint(pts, end):
    return np.asarray(pts[0] if end == 0 else pts[-1], dtype=float)


def outward_heading(pts, end, k=LC.DIRPTS):
    """Unit vector pointing OUT of the fragment at the given end (0=head, 1=tail)."""
    pts = np.asarray(pts, dtype=float)
    k = min(k, len(pts) - 1)
    if end == 0:
        return _unit(pts[0] - pts[k])
    return _unit(pts[-1] - pts[-1 - k])


def _angle(u, v):
    """Angle (deg) between two unit vectors, in [0,180]."""
    d = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


# ---- candidate enumeration -------------------------------------------------------------------
def candidate_pairs(frags):
    """One candidate per DIFFERENT-fragment pair whose nearest endpoints are within GAP_MAX_PX,
    taken at those CLOSEST endpoints (the natural break point a linker would bridge). Emitting only
    the closest-endpoint combination — not all four — keeps the heading/angle features meaningful.
    """
    from scipy.spatial import cKDTree

    eps, meta = [], []
    for fi, p in enumerate(frags):
        eps.append(endpoint(p, 0)); meta.append((fi, 0))
        eps.append(endpoint(p, 1)); meta.append((fi, 1))
    if len(eps) < 2:
        return []
    eps = np.array(eps)
    tree = cKDTree(eps)
    best = {}                       # (fi<fj) -> (dist, ei, ej)
    for a, b in tree.query_pairs(r=LC.GAP_MAX_PX):
        fi, ei = meta[a]
        fj, ej = meta[b]
        if fi == fj:
            continue
        d = float(np.hypot(*(eps[a] - eps[b])))
        key = (fi, fj) if fi < fj else (fj, fi)
        cei, cej = (ei, ej) if fi < fj else (ej, ei)
        if key not in best or d < best[key][0]:
            best[key] = (d, cei, cej)
    return [(k[0], v[1], k[1], v[2]) for k, v in best.items()]


def pair_features(frags, lens, fi, ei, fj, ej):
    pa, pb = frags[fi], frags[fj]
    ea, eb = endpoint(pa, ei), endpoint(pb, ej)
    gap = float(np.hypot(*(eb - ea)))
    ha, hb = outward_heading(pa, ei), outward_heading(pb, ej)
    gv = _unit(eb - ea)                       # a -> b
    a1 = _angle(ha, gv)                        # does a head toward b?
    a2 = _angle(hb, -gv)                       # does b's end continue back toward a?
    la, lb = lens[fi], lens[fj]
    lmin, lmax = (la, lb) if la <= lb else (lb, la)
    # distance between the two fragments' NEAR endpoints vs their far endpoints (context)
    far_a = endpoint(pa, 1 - ei); far_b = endpoint(pb, 1 - ej)
    far = float(np.hypot(*(far_b - far_a)))
    return {
        "gap_px": gap, "align_deg": a1 + a2, "a1_deg": a1, "a2_deg": a2,
        "len_min": lmin, "len_max": lmax, "len_ratio": (lmin / lmax if lmax else 0.0),
        "end_dist_ratio": (gap / far if far > 0 else 1.0),
    }


# ---- auto-labeling from GT -------------------------------------------------------------------
def fragment_to_gt(frags, gt_polys, frag_radius=LC.FRAG_RADIUS_PX,
                   gt_radius=LC.GT_ASSOC_RADIUS_PX, cov=LC.FRAG_TO_GT_COV):
    """Map each fragment -> index of the GT fibre it lies on (>= cov of the fragment covered), else -1.

    GT is dilated to a wider band (fibre half-width) so a fragment anywhere on the fibre body maps,
    even when the Ridge centerline is laterally offset from the hand-trace.
    """
    W = BC.IMAGE_SHAPE[1]
    fmasks = [g.rasterize(p, frag_radius, W) for p in frags]
    gmasks = [g.rasterize(p, gt_radius, W) for p in gt_polys]
    cand = g.bbox_overlap_pairs(fmasks, gmasks)
    best = {i: (0.0, -1) for i in range(len(frags))}
    for i, j in cand:
        area = g.intersection_area(fmasks[i], gmasks[j])
        if area <= 0:
            continue
        cov_frag = area / fmasks[i].area if fmasks[i].area else 0.0
        if cov_frag > best[i][0]:
            best[i] = (cov_frag, int(j))
    return {i: (j if c >= cov else -1) for i, (c, j) in best.items()}


# ---- per-well dataset ------------------------------------------------------------------------
def build_well(w, verbose=True):
    """Return (rows, stats) for one well. rows: list of dict(stem, case_id, label, **features)."""
    frags = load_fragments(w["run_stem"])
    gt = io.read_imagej_zip(BC.QPLATES / w["plate"] / w["roi_zip"])
    lens = [g.polylen(p) for p in frags]
    frag_gt = fragment_to_gt(frags, gt)
    mapped = sum(1 for v in frag_gt.values() if v >= 0)

    rows = []
    n_skipped = 0
    for fi, ei, fj, ej in candidate_pairs(frags):
        gi, gj = frag_gt[fi], frag_gt[fj]
        if LC.BOTH_MAPPED_ONLY and (gi < 0 or gj < 0):
            n_skipped += 1          # >=1 fragment is off-GT (noise/untraced) -> can't label cleanly
            continue
        label = 1 if (gi >= 0 and gi == gj) else 0
        feats = pair_features(frags, lens, fi, ei, fj, ej)
        rows.append(dict(stem=w["well_id"], plate=w["plate"],
                         case_id=f"link_{fi}_{ei}_{fj}_{ej}", label=label, **feats))
    n_pos = sum(r["label"] for r in rows)
    stats = dict(well=w["well_id"], plate=w["plate"], n_frag=len(frags), n_gt=len(gt),
                 frag_mapped=mapped, frag_mapped_frac=(mapped / len(frags) if frags else 0.0),
                 n_pairs=len(rows), n_pos=n_pos, n_skipped_unmapped=n_skipped,
                 pos_frac=(n_pos / len(rows) if rows else 0.0))
    if verbose:
        print(f"  {w['well_id']:22s} frags={len(frags):4d} gt={len(gt):4d} "
              f"mapped={stats['frag_mapped_frac']:.2f} | pairs={len(rows):5d} pos={n_pos:5d} "
              f"({stats['pos_frac']:.2f})  skipped_unmapped={n_skipped}")
    return rows, stats
