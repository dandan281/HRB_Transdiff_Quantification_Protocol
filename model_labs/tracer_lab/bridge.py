"""Bridge missing fibre middles by walking the ORIENTATION field.

Diagnosis (report §11): a split fibre's pieces are a median of 90 px apart --
the walk lost the centre ridge for tens of pixels (dim stretch, tangle, NMS
hole) and the middle of the fibre is simply untraced. Endpoint stitching is
refuted at that range, and blind bridging is the 2026-07 linker failure.

The bridge uses the one head that stays reliable where the centre head goes
dim: orientation (7 deg median axial error on-ridge). From each free fragment
end, a PROBE walks outward following the predicted orientation field --

* sign resolved by continuity, the same 25 deg follow-gate as the main walk
  (a field direction that disagrees with the probe's heading is another
  fibre's, so the probe dead-reckons past it);
* requiring only weak RAW centre support (``bridge_floor``, swept) -- the
  pre-NMS field, because NMS zeroed exactly the holes being crossed;
* for at most ``max_len_px`` (swept; the split-gap p75 is 176 px).

A probe succeeds ONLY by landing on another object's painted claim
co-linearly (axial angle < 25 deg). Transverse contact, running out of
support, or running out of length all discard the probe silently. A
successful probe unions the two objects and its path joins the object, so
recovered length is REAL traced length along the image, not an invented
straight segment.

Tuning protocol identical to the stitcher: sweep on TUNE wells, freeze,
claim on TEST wells; false merges guarded as hard as mdape.

    python model_labs/tracer_lab/bridge.py --sweep
    python model_labs/tracer_lab/bridge.py --apply
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CORPUS = ROOT / "PrecisionMyotube/annotation_work/plate32_dense_v1"
CV = ROOT / "model_labs/tracer_lab/_runs/net_cv"
TUNE_WELLS = ("C02", "C03", "C05", "C11", "D02")
TEST_WELLS = ("B02", "D04", "D08", "D09", "D11")

FROZEN = {"bridge_floor": 0.15, "max_len_px": 120.0}


def _free_ends(obj_paths, min_sep=6.0):
    """Endpoints of an object's paths that are not interior joints."""
    from tracer_lab.centreline_targets import resample_polyline

    dense = [resample_polyline(np.asarray(p), 2.0) for p in obj_paths
             if len(p) >= 4]
    ends = []
    for i, p in enumerate(dense):
        h = min(5, len(p) - 1)
        for pt, inner in ((p[0], p[h]), (p[-1], p[-1 - h])):
            t = pt - inner                      # outward direction
            n = np.linalg.norm(t)
            if n < 1e-9:
                continue
            near_other = any(
                np.min(np.linalg.norm(q - pt[None, :], axis=1)) < min_sep
                for j, q in enumerate(dense) if j != i)
            if not near_other:
                ends.append((pt.astype(float), t / n))
    return ends


def bridge(result, pred, *, bridge_floor, max_len_px, step_px=3.0,
           gate_deg=25.0, land_deg=25.0, land_end_px=12.0,
           require_mutual=True):
    """-> (new result with unions + bridge paths, n_bridges).

    v2 guards, added after the v1 sweep repaired ~100 splits per 5 wells at
    the cost of ~+160 false merges at EVERY config (the floor axis is dead:
    the halo keeps raw centre above 0.25 in nearly all corridors):

    * ``land_end_px`` -- a probe must land within this distance of a FREE END
      of the target object. A co-linear landing mid-body is a parallel bundle
      being captured, not a missing middle being repaired.
    * ``require_mutual`` -- a bridge a->b counts only if some probe from b
      also lands on a. A true missing middle is approachable from both rims;
      a bundle capture rarely is.
    """
    from tracer_lab.centreline_targets import resample_polyline
    from tracer_lab.oracle_trace import _axial_diff

    centre = pred["centre"]                    # RAW field, pre-NMS
    orient = pred["orient"]
    H, W = centre.shape

    # paint every object's claim at 1 px density
    claimed_id = np.zeros((H, W), np.int32)
    claimed_th = np.zeros((H, W), np.float32)
    groups: dict[int, list] = {}
    parent = dict(result["object_of"])

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pid, path in enumerate(result["paths"], start=1):
        groups.setdefault(find(pid), []).append(np.asarray(path))
    for oid, paths in groups.items():
        for path in paths:
            d = resample_polyline(path, 1.0)
            for k in range(len(d)):
                a, b = d[max(k - 2, 0)], d[min(k + 2, len(d) - 1)]
                th = math.atan2(float(b[0] - a[0]), float(b[1] - a[1]))
                r, c = int(round(d[k, 0])), int(round(d[k, 1]))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W \
                                and claimed_id[rr, cc] == 0:
                            claimed_id[rr, cc] = oid
                            claimed_th[rr, cc] = th

    new_paths = list(result["paths"])
    n_bridges = 0
    max_steps = int(max_len_px / step_px)

    free_ends = {oid: _free_ends(paths) for oid, paths in groups.items()}
    candidates = {}                      # (src_root, dst_root) -> probe path

    for oid, paths in groups.items():
        for p0, v0 in free_ends[oid]:
            p = p0.copy()
            v = v0.copy()
            probe = [p.copy()]
            landed = 0
            for _ in range(max_steps):
                ri, ci = int(round(p[0])), int(round(p[1]))
                if not (0 <= ri < H and 0 <= ci < W):
                    break
                th = 0.5 * math.atan2(float(orient[1, ri, ci]),
                                      float(orient[0, ri, ci]))
                u = np.array([math.sin(th), math.cos(th)])
                if float(u @ v) < 0:
                    u = -u
                if _axial_diff(th, math.atan2(v[0], v[1])) \
                        > math.radians(gate_deg):
                    u = v                       # foreign direction: hold course
                q = p + step_px * u
                rq, cq = int(round(q[0])), int(round(q[1]))
                if not (0 <= rq < H and 0 <= cq < W):
                    break
                if float(centre[rq, cq]) < bridge_floor:
                    break
                cid = int(claimed_id[rq, cq])
                if cid and find_static(parent, cid) != find_static(parent, oid):
                    if _axial_diff(math.atan2(u[0], u[1]),
                                   float(claimed_th[rq, cq])) \
                            < math.radians(land_deg):
                        landed = cid
                    break                       # transverse landing: discard
                probe.append(q.copy())
                v = u
                p = q
            if landed and len(probe) >= 2:
                land_pt = probe[-1]
                near_end = any(
                    float(np.linalg.norm(land_pt - ep)) <= land_end_px
                    for ep, _ in free_ends.get(find(landed), []))
                if not near_end:
                    continue
                ra, rb = find(oid), find(landed)
                if ra != rb and (ra, rb) not in candidates:
                    candidates[(ra, rb)] = np.asarray(probe)

    for (ra, rb), probe in list(candidates.items()):
        if find(ra) == find(rb):
            continue                     # already united via another pair
        if require_mutual and (rb, ra) not in candidates:
            continue
        pid_new = len(new_paths) + 1
        new_paths.append(probe)
        parent[pid_new] = find(rb)
        parent[find(ra)] = find(rb)
        n_bridges += 1

    object_of = {pid: find(pid) for pid in range(1, len(new_paths) + 1)}
    return ({**result, "paths": new_paths, "object_of": object_of},
            n_bridges)


def find_static(parent, x):
    while parent[x] != x:
        x = parent[x]
    return x


def _run_well(well, params):
    from tracer_lab.train_tracer import load_well
    from tracer_lab.infer_trace import predict_fields, fields_for_walk
    from tracer_lab.oracle_trace import (
        TraceParams, score_against_gt, trace_field)

    image, gt, _ = load_well(well)
    pred = predict_fields(image, CV / well / "best.pt")
    wf = fields_for_walk(pred, crossing_thresh=0.4, valid_thresh=0.2,
                         prep="nms")
    wf["instance"] = gt["instance"]
    wf["traces"] = gt["traces"]
    res = trace_field(wf, TraceParams(
        seed_thresh=0.4, support_thresh=0.3, claim_radius_px=3.5,
        rescue_window_steps=1))
    before = score_against_gt(res, wf)
    bridged, nb = bridge(res, pred, **params)
    after = score_against_gt(bridged, wf)
    return before, after, nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)

    if a.sweep:
        # v2 grid: the floor axis was measured dead (827-835 bridges at
        # 0.10/0.15/0.25 alike); fixed at 0.15. Axes that can still matter:
        # probe length, and the two new guards (one diagnostic row each).
        grid_rows = [
            {"bridge_floor": 0.15, "max_len_px": 60.0},
            {"bridge_floor": 0.15, "max_len_px": 120.0},
            {"bridge_floor": 0.15, "max_len_px": 200.0},
            {"bridge_floor": 0.15, "max_len_px": 120.0,
             "require_mutual": False},
            {"bridge_floor": 0.15, "max_len_px": 120.0,
             "land_end_px": 1e9},
        ]
        print(f"{'guards':>7}{'len':>6} | {'mdape':>14}{'ident':>14}"
              f"{'merge':>12}{'split':>12}{'bridges':>9}")
        out = []
        for params in grid_rows:
            bs, as_, nbs = [], [], 0
            for w in TUNE_WELLS:
                b, af, nb = _run_well(w, params)
                bs.append(b)
                as_.append(af)
                nbs += nb
            mb = np.median([x["length_mdape"] for x in bs])
            ma = np.median([x["length_mdape"] for x in as_])
            ib = np.mean([x["identity_through_crossing"] for x in bs])
            ia = np.mean([x["identity_through_crossing"] for x in as_])
            gb = sum(x["false_merge_count"] for x in bs)
            ga = sum(x["false_merge_count"] for x in as_)
            sb = sum(x["false_split_count"] for x in bs)
            sa = sum(x["false_split_count"] for x in as_)
            tag = ("both" if params.get("require_mutual", True)
                   and params.get("land_end_px", 12.0) < 1e8 else
                   ("end" if not params.get("require_mutual", True)
                    else "mutual"))
            print(f"{tag:>7}{params['max_len_px']:>6.0f}"
                  f" |{mb:>7.3f}->{ma:<6.3f}{ib:>7.3f}->{ia:<6.3f}"
                  f"{gb:>5}->{ga:<6}{sb:>5}->{sa:<6}{nbs:>9}", flush=True)
            out.append({**params, "mdape_after": float(ma),
                        "ident_after": float(ia), "merge_after": int(ga),
                        "split_after": int(sa), "bridges": int(nbs)})
        (ROOT / "model_labs/tracer_lab/_runs/bridge_sweep.json").write_text(
            json.dumps(out, indent=2))
        print("\nwritten: _runs/bridge_sweep.json")
        return 0

    if a.apply:
        print(f"frozen: {FROZEN}  (tuned on {TUNE_WELLS})")
        keys = ("length_mdape", "identity_through_crossing", "recall_traces",
                "false_split_count", "false_merge_count", "n_objects")
        rows = []
        for w in TEST_WELLS:
            b, af, nb = _run_well(w, FROZEN)
            rows.append({"well": w, "bridges": nb,
                         "before": {k: b[k] for k in keys},
                         "after": {k: af[k] for k in keys}})
            print(f"{w}: mdape {b['length_mdape']:.3f}->"
                  f"{af['length_mdape']:.3f}  ident "
                  f"{b['identity_through_crossing']:.3f}->"
                  f"{af['identity_through_crossing']:.3f}  merges "
                  f"{b['false_merge_count']}->{af['false_merge_count']}  "
                  f"splits {b['false_split_count']}->"
                  f"{af['false_split_count']}  bridges {nb}", flush=True)
        (ROOT / "model_labs/tracer_lab/_runs/bridge_test.json").write_text(
            json.dumps({"frozen": FROZEN, "rows": rows}, indent=2))
        print("\nwritten: _runs/bridge_test.json")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
