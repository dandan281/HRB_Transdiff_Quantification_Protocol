"""Decompose-and-retrace: turn one dense field into K sparse ones, re-trace.

The operator's proposal (sketch, 2026-08-28), grounded in two measured
facts: density is the difficulty axis (CV loss tracks well density), and
cuts concentrate at crossings (break-point attribution: 2-3x enrichment).

Pipeline per well:

1. **First pass** â€” the frozen walk + weld on the full field.
2. **Conflict graph** â€” objects whose paths come within `contact_px` at a
   TRANSVERSE angle (> `conflict_deg` axial) conflict; co-linear contacts
   do NOT (possible pieces of one fibre must stay together so the re-trace
   can rejoin them). Greedy coloring -> K groups, mutually non-crossing
   within each group.
3. **Sub-images** â€” for group g, keep the normalized image under the
   group's corridors (path discs of `mask_r_px`, endpoints extended
   `extend_px` along the end tangent so dim middles between fragments are
   INSIDE the sparse image); elsewhere the image's background level.
4. **Re-trace** each sub-image: re-predict fields (crossings that lost
   their transverse partner become plain fibre â€” the orientation head sees
   ONE structure now), same frozen walk + weld.
5. **Combine**: union of all groups' objects; the dense field's
   quantification is the sum of the sub-fields'.

v1 knobs are fixed, not swept (contact 6 px, conflict 30 deg, mask 10 px,
extend 90 px, background = 30th percentile); if the mechanism earns a
sweep it happens on the tune wells per the standing rule. Missed-fibre
recall cannot improve by construction (a fibre absent from pass 1 is in no
sub-image) â€” this targets crossing cuts and identity, judged on
false_split_count / identity / the length-class mix.

    python model_labs/tracer_lab/decompose_retrace.py --wells C05 --render
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "annotation_tools",
           ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CV = ROOT / "model_labs/tracer_lab/_runs/net_cv"
SWEEP_CACHE = ROOT / "model_labs/tracer_lab/_runs/sweep_cache"
OUT_DIR = ROOT / "model_labs/tracer_lab/_runs/decompose_v1"

WALK = dict(seed_thresh=0.4, support_thresh=0.3, claim_radius_px=3.5,
            rescue_window_steps=1)
WELD = dict(weld_dist_px=14.0, weld_deg=12.5, crossing_gate_px=12.0)
CONTACT_PX = 6.0
CONFLICT_DEG = 30.0
MASK_R_PX = 10.0
# Two DIFFERENT extension lengths, deliberately. DETECT reaches far so a
# fibre cut at a crossing still registers its transverse partner as a
# conflict. MASK is short: each piece of a cut pair extends toward the
# other (40 + 40 spans the median 90 px inter-piece gap), but a corridor
# paved 90 px past a TRUE fibre end walks the tracer into the next
# co-linear fibre â€” measured on C05: mdape 0.32 -> 1.07, merges 59 -> 131,
# the 2026-07 fragments-joined class rebuilt by the mask.
DETECT_EXT_PX = 90.0
MASK_EXT_PX = 20.0
BG_PCT = 30.0
WITNESS_PX = 40.0   # min px of a re-trace along EACH member to testify a merge


def dense_points(res):
    """(points Nx2, object_id N, theta N) for every path, ~3 px spacing."""
    from tracer_lab.centreline_targets import resample_polyline
    pts, oids, thetas = [], [], []
    for pid, path in enumerate(res["paths"], start=1):
        p = np.asarray(path, float)
        if p.ndim != 2 or len(p) < 2:
            continue
        d = resample_polyline(p, 3.0)
        if len(d) < 2:
            continue
        t = np.diff(d, axis=0)
        th = np.arctan2(t[:, 0], t[:, 1])
        th = np.append(th, th[-1])
        pts.append(d)
        oids.append(np.full(len(d), res["object_of"][pid]))
        thetas.append(th)
    return np.concatenate(pts), np.concatenate(oids), np.concatenate(thetas)


def conflict_groups(res, min_groups: int = 3):
    """Color the transverse-conflict graph; co-linear contact never splits.

    Conflicts are computed on the EXTENDED spines (paths + endpoint
    extensions): a fibre that was CUT at a crossing stops short of it, so
    its traced path never touches the crosser â€” the very conflicts the
    decomposition exists for are invisible to path-contact alone (measured
    on C05: 45 edges, group 0 = essentially the whole image). The
    extension reaches across the cut and registers the transverse partner.

    Non-conflicting objects are spread round-robin so every sub-image is
    genuinely sparse, not one dense group plus empties.
    """
    import math
    from scipy.spatial import cKDTree
    from tracer_lab.centreline_targets import resample_polyline

    P0, O0, TH0 = dense_points(res)
    ext_pts, ext_oid, ext_th = [], [], []
    for pid, path in enumerate(res["paths"], start=1):
        p = np.asarray(path, float)
        if p.ndim != 2 or len(p) < 2:
            continue
        d = resample_polyline(p, 3.0)
        if len(d) < 2:
            continue
        oid = res["object_of"][pid]
        for sign in (0, -1):
            k = min(4, len(d) - 1)
            t = (d[-1] - d[-1 - k]) if sign == -1 else (d[0] - d[k])
            n = float(np.linalg.norm(t))
            if n < 1e-6:
                continue
            t = t / n
            steps = np.arange(3.0, DETECT_EXT_PX + 1.0, 3.0)
            q = d[sign][None, :] + steps[:, None] * t[None, :]
            ext_pts.append(q)
            ext_oid.append(np.full(len(q), oid))
            ext_th.append(np.full(len(q), math.atan2(t[0], t[1])))
    P = np.concatenate([P0] + ext_pts) if ext_pts else P0
    O = np.concatenate([O0] + ext_oid) if ext_pts else O0
    TH = np.concatenate([TH0] + ext_th) if ext_pts else TH0
    # origin flag: True where the sample is a real PATH point (not an
    # endpoint extension). Pieces of ONE cut fibre touch only via
    # extensions (there is a gap between their paths); PARALLEL neighbours
    # touch path-to-path along a sustained stretch. The flag separates the
    # two co-linear cases so rejoinable pieces stay together while
    # side-by-side fibres are forced into different groups (same-group
    # parallel corridors fuse and merge — measured: merges 59 -> 96+).
    IS_PATH = np.concatenate(
        [np.ones(len(P0), bool)]
        + [np.zeros(len(q), bool) for q in ext_pts]) if ext_pts \
        else np.ones(len(P0), bool)

    tree = cKDTree(P)
    pair_angles: dict[tuple[int, int], list[float]] = {}
    pair_pp_colinear: dict[tuple[int, int], int] = {}
    for i, j in tree.query_pairs(CONTACT_PX):
        a, b = int(O[i]), int(O[j])
        if a == b:
            continue
        d = (TH[i] - TH[j]) % math.pi
        d = min(d, math.pi - d)
        key = (min(a, b), max(a, b))
        pair_angles.setdefault(key, []).append(d)
        if IS_PATH[i] and IS_PATH[j] \
                and d <= math.radians(CONFLICT_DEG):
            pair_pp_colinear[key] = pair_pp_colinear.get(key, 0) + 1

    edges: dict[int, set[int]] = {}
    for (a, b), angles in pair_angles.items():
        transverse = np.median(angles) > math.radians(CONFLICT_DEG)
        parallel_run = pair_pp_colinear.get((a, b), 0) >= 8  # ~24 px side-by-side
        if transverse or parallel_run:
            edges.setdefault(a, set()).add(b)
            edges.setdefault(b, set()).add(a)

    objs = sorted(set(int(v) for v in O))
    conflicted = sorted((o for o in objs if edges.get(o)),
                        key=lambda o: -len(edges[o]))
    color: dict[int, int] = {}
    for o in conflicted:
        used = {color[n] for n in edges.get(o, ()) if n in color}
        c = 0
        while c in used:
            c += 1
        color[o] = c
    k = max(max(color.values()) + 1 if color else 1, min_groups)
    load = [sum(1 for c in color.values() if c == g) for g in range(k)]
    for o in objs:
        if o not in color:
            g = int(np.argmin(load))
            color[o] = g
            load[g] += 1
    return color, k, len(pair_angles), sum(len(v) for v in edges.values()) // 2


def group_mask(res, members: set[int], shape):
    """Corridors of one group: path discs + endpoint extensions."""
    from scipy import ndimage
    from tracer_lab.centreline_targets import resample_polyline

    H, W = shape
    spine = np.zeros(shape, dtype=bool)
    for pid, path in enumerate(res["paths"], start=1):
        if res["object_of"][pid] not in members:
            continue
        p = np.asarray(path, float)
        if p.ndim != 2 or len(p) < 2:
            continue
        d = resample_polyline(p, 1.0)
        segs = [d]
        for sign in (0, -1):
            k = min(4, len(d) - 1)
            t = (d[-1] - d[-1 - k]) if sign == -1 else (d[0] - d[k])
            n = float(np.linalg.norm(t))
            if n > 1e-6:
                t = t / n
                steps = np.arange(1.0, MASK_EXT_PX + 1.0)
                segs.append(d[sign][None, :] + steps[:, None] * t[None, :])
        for s in segs:
            r = np.clip(np.round(s[:, 0]).astype(int), 0, H - 1)
            c = np.clip(np.round(s[:, 1]).astype(int), 0, W - 1)
            spine[r, c] = True
    rr = int(np.ceil(MASK_R_PX))
    yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
    disc = (yy ** 2 + xx ** 2) <= MASK_R_PX ** 2
    return ndimage.binary_dilation(spine, disc)


def member_spine_tree(res, members: set[int]):
    """KDTree over the group's member spines + endpoint extensions."""
    from scipy.spatial import cKDTree
    from tracer_lab.centreline_targets import resample_polyline

    pts = []
    for pid, path in enumerate(res["paths"], start=1):
        if res["object_of"][pid] not in members:
            continue
        p = np.asarray(path, float)
        if p.ndim != 2 or len(p) < 2:
            continue
        d = resample_polyline(p, 2.0)
        pts.append(d)
        for sign in (0, -1):
            k = min(4, len(d) - 1)
            t = (d[-1] - d[-1 - k]) if sign == -1 else (d[0] - d[k])
            n = float(np.linalg.norm(t))
            if n > 1e-6:
                steps = np.arange(2.0, MASK_EXT_PX + 1.0, 2.0)
                pts.append(d[sign][None, :]
                           + steps[:, None] * (t / n)[None, :])
    return cKDTree(np.concatenate(pts)) if pts else None


def responsibility_filter(res_g, tree, keep_frac=0.5, r=6.0):
    """Keep only re-traced objects that lie along THIS group's members.

    A sub-image also contains fragments of other groups' fibres (crossing
    overlaps; endpoint extensions sweeping through them). Reporting those
    double-counts across sub-images and manufactures splits â€” each
    sub-image answers only for its own myotubes (the sum-of-submatrices
    semantics of the operator's sketch). Measured on C05 v1 without this
    filter: splits 105 -> 128; the filter exists to remove that artefact.
    """
    from tracer_lab.centreline_targets import resample_polyline

    if tree is None:
        return {**res_g, "paths": [], "object_of": {}}
    obj_pts: dict[int, list] = {}
    for pid, path in enumerate(res_g["paths"], start=1):
        p = np.asarray(path, float)
        if p.ndim != 2 or len(p) < 2:
            continue
        obj_pts.setdefault(res_g["object_of"][pid], []).append(
            resample_polyline(p, 2.0))
    keep_obj = set()
    for oid, chunks in obj_pts.items():
        P = np.concatenate(chunks)
        d, _ = tree.query(P, distance_upper_bound=r)
        if np.isfinite(d).mean() >= keep_frac:
            keep_obj.add(oid)
    paths, object_of = [], {}
    for pid, path in enumerate(res_g["paths"], start=1):
        if res_g["object_of"][pid] in keep_obj:
            paths.append(path)
            object_of[len(paths)] = res_g["object_of"][pid]
    return {**res_g, "paths": paths, "object_of": object_of}


def group_member_tree(res, members: set[int]):
    """(KDTree, member-id per point) over the group's member spines
    (+ short endpoint extensions), for witness attribution."""
    from scipy.spatial import cKDTree
    from tracer_lab.centreline_targets import resample_polyline

    pts, ids = [], []
    for pid, path in enumerate(res["paths"], start=1):
        oid = res["object_of"][pid]
        if oid not in members:
            continue
        p = np.asarray(path, float)
        if p.ndim != 2 or len(p) < 2:
            continue
        d = resample_polyline(p, 2.0)
        pts.append(d)
        ids.append(np.full(len(d), oid))
        for sign in (0, -1):
            k = min(4, len(d) - 1)
            t = (d[-1] - d[-1 - k]) if sign == -1 else (d[0] - d[k])
            n = float(np.linalg.norm(t))
            if n > 1e-6:
                steps = np.arange(2.0, MASK_EXT_PX + 1.0, 2.0)
                q = d[sign][None, :] + steps[:, None] * (t / n)[None, :]
                pts.append(q)
                ids.append(np.full(len(q), oid))
    if not pts:
        return None, None
    return cKDTree(np.concatenate(pts)), np.concatenate(ids)


def identity_repair(base, sub_results, color):
    """Option 2: sub-image re-traces are WITNESSES only, geometry stays base.

    A re-traced object in group g that substantially touches >= 2 of the
    group's first-pass members (>= 15 px along each) testifies that those
    members are one myotube; their identities are unioned. No re-traced
    path enters the output, so material the first pass never saw — which
    the sub-images absorb into whatever corridor covers it (the measured
    v3-v6 failure) — cannot contaminate any measurement. Pure identity
    repair: recall and lengths are pass-1's; only the grouping changes.
    """
    from tracer_lab.centreline_targets import resample_polyline

    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_witness = 0
    k = max(color.values()) + 1
    for g in range(k):
        members = {o for o, c in color.items() if c == g}
        tree, labels = group_member_tree(base, members)
        if tree is None:
            continue
        res_g = sub_results[g]
        obj_pts: dict[int, list] = {}
        for pid, path in enumerate(res_g["paths"], start=1):
            p = np.asarray(path, float)
            if p.ndim != 2 or len(p) < 2:
                continue
            obj_pts.setdefault(res_g["object_of"][pid], []).append(
                resample_polyline(p, 2.0))
        for oid, chunks in obj_pts.items():
            P = np.concatenate(chunks)
            d, idx = tree.query(P, distance_upper_bound=8.0)
            ok = np.isfinite(d)
            if not ok.any():
                continue
            touched, counts = np.unique(labels[idx[ok]], return_counts=True)
            witness = [int(t) for t, n in zip(touched, counts)
                       if n * 2.0 >= WITNESS_PX]
            if len(witness) >= 2:
                n_witness += 1
                r0 = find(witness[0])
                for w in witness[1:]:
                    rw = find(w)
                    if rw != r0:
                        parent[rw] = r0
    object_of = {pid: find(oid) for pid, oid in base["object_of"].items()}
    n_before = len(set(base["object_of"].values()))
    n_after = len(set(object_of.values()))
    return {**base, "object_of": object_of}, \
        {"witness_merges": n_witness, "objects": (n_before, n_after)}


def combine(results):
    """Union sub-field results into one result dict with disjoint ids."""
    paths, object_of = [], {}
    offset = 0
    max_obj = 0
    for res in results:
        for pid, path in enumerate(res["paths"], start=1):
            paths.append(path)
            object_of[offset + pid] = max_obj + res["object_of"][pid]
        offset += len(res["paths"])
        if res["object_of"]:
            max_obj += max(res["object_of"].values())
    return {"paths": paths, "object_of": object_of, "merge_events": [],
            "stop_reasons": {}}


def run_well(well: str, render: bool = False, mode: str = "full") -> dict:
    from tracer_lab.infer_trace import predict_fields, fields_for_walk
    from tracer_lab.oracle_trace import (
        TraceParams, score_against_gt, trace_field, weld_objects)
    from tracer_lab.train_tracer import load_well

    t0 = time.time()
    image, gt, _ = load_well(well)
    npz = SWEEP_CACHE / f"{well}.npz"
    if npz.exists():
        z = np.load(npz)
        pred = {k: z[k] for k in ("centre", "orient", "crossing")}
    else:
        pred = predict_fields(image, CV / well / "best.pt")

    def walk(p):
        wf = fields_for_walk(p, crossing_thresh=0.4, valid_thresh=0.2,
                             prep="nms")
        res = trace_field(wf, TraceParams(**WALK))
        return weld_objects(res, wf, **WELD), wf

    base, wf_full = walk(pred)
    wf_full["instance"] = gt["instance"]
    wf_full["traces"] = gt["traces"]
    sc_base = score_against_gt(base, wf_full)

    color, k, n_contact, n_conflict = conflict_groups(base)
    bg = float(np.percentile(image, BG_PCT))
    sub_results = []
    sub_imgs = []
    for g in range(k):
        members = {o for o, c in color.items() if c == g}
        m = group_mask(base, members, image.shape)
        sub = np.where(m, image, bg).astype(np.float32)
        p_g = predict_fields(sub, CV / well / "best.pt")
        res_g, _ = walk(p_g)
        res_g = responsibility_filter(
            res_g, member_spine_tree(base, members))
        sub_results.append(res_g)
        if render:
            sub_imgs.append((sub, res_g))

    if mode == "repair":
        final, info = identity_repair(base, sub_results, color)
        print(f"  repair: {info['witness_merges']} witness merges, "
              f"objects {info['objects'][0]} -> {info['objects'][1]}",
              flush=True)
    else:
        final = combine(sub_results)
    sc_dec = score_against_gt(final, wf_full)
    # duplication diagnostic: if union-length mdape is much lower than
    # sum-length mdape, the inflation is duplicate arcs, not bridges
    u_dec = score_against_gt(final, wf_full, length_mode="union")
    u_base = score_against_gt(base, wf_full, length_mode="union")
    print(f"  mdape sum vs union: base {sc_base['length_mdape']:.3f}/"
          f"{u_base['length_mdape']:.3f}  decomposed "
          f"{sc_dec['length_mdape']:.3f}/{u_dec['length_mdape']:.3f}",
          flush=True)

    out = {"well": well, "k_groups": k,
           "contact_pairs": n_contact, "conflict_edges": n_conflict,
           "base": {m: sc_base[m] for m in
                    ("false_split_count", "false_merge_count",
                     "recall_traces", "identity_through_crossing",
                     "length_mdape", "n_objects")},
           "decomposed": {m: sc_dec[m] for m in
                          ("false_split_count", "false_merge_count",
                           "recall_traces", "identity_through_crossing",
                           "length_mdape", "n_objects")},
           "seconds": round(time.time() - t0)}
    b, d = out["base"], out["decomposed"]
    print(f"{well}: K={k} ({n_conflict} conflicts) | "
          f"splits {b['false_split_count']}->{d['false_split_count']}  "
          f"merges {b['false_merge_count']}->{d['false_merge_count']}  "
          f"recall {b['recall_traces']:.2f}->{d['recall_traces']:.2f}  "
          f"idx {b['identity_through_crossing']:.2f}->"
          f"{d['identity_through_crossing']:.2f}  "
          f"mdape {b['length_mdape']:.3f}->{d['length_mdape']:.3f}  "
          f"({out['seconds']}s)", flush=True)

    if render:
        import imageio.v2 as imageio
        sz = 1100
        r0 = c0 = (image.shape[0] - sz) // 2
        panels = [np.clip(image[r0:r0 + sz, c0:c0 + sz] * 0.9, 0, 1)]
        for sub, _res in sub_imgs[:4]:
            panels.append(np.clip(sub[r0:r0 + sz, c0:c0 + sz] * 0.9, 0, 1))
        gap = np.ones((sz, 10)) * 0.5
        row = panels[0]
        for p in panels[1:]:
            row = np.concatenate([row, gap, p], axis=1)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        f = OUT_DIR / f"decompose_demo_{well}.png"
        imageio.imwrite(f, (row[0::2, 0::2] * 255).astype(np.uint8))
        print(f"  demo -> {f}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wells", nargs="+", default=["C05"])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--mode", default="full", choices=("full", "repair"))
    a = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [run_well(w, render=a.render, mode=a.mode) for w in a.wells]
    if len(rows) > 1:
        for tag in ("base", "decomposed"):
            print(f"POOLED {tag}: splits "
                  f"{sum(r[tag]['false_split_count'] for r in rows)}  "
                  f"merges {sum(r[tag]['false_merge_count'] for r in rows)}  "
                  f"recall {np.mean([r[tag]['recall_traces'] for r in rows]):.3f}  "
                  f"idx {np.mean([r[tag]['identity_through_crossing'] for r in rows]):.3f}  "
                  f"mdape {np.mean([r[tag]['length_mdape'] for r in rows]):.3f}")
    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"-> {OUT_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
