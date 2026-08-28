"""Stitch co-linear fragments after the walk: the per-fibre-length attack.

Cross-validated state (report §10): the nms arm's well-level totals already
track the operator (0.95x, rank rho +0.90) but matched per-fibre length sits
at 0.32 against the operator's own repeatability of 0.096 -- and the overlays
show why: single fibres come back as chains of separately-coloured fragments.
The walk's in-flight merge machinery joins walks that TOUCH; it cannot join
fragments separated by a support gap (an NMS crest hole, a dim patch).

This joins them afterwards, under three conditions ALL required:

* **end-to-end geometry** -- an endpoint of one object within ``gap_px`` of an
  endpoint of another. Endpoint-to-middle contact is NOT a stitch candidate:
  that is a crossing or a spur, and bridging it is how the 2026-07 linker
  manufactured ~65% wrong merges.
* **co-linearity** -- the end tangents of both fragments AND the gap vector
  agree within ``angle_deg`` (axial). A fibre continues straight through a
  small gap; anything that turns is somebody else.
* **image support** -- the predicted centre field along the bridged segment
  averages at least ``bridge_support``. A real fibre leaves signal in the gap;
  background does not. This is the condition the old linker never had.

Endpoints are matched greedily by score, each endpoint used at most once, and
stitching iterates so chains of three or more fragments can reunite.

Tuning protocol: knobs are swept on the TUNE wells only (C02 C03 C05 C11
D02), frozen at the plateau, and the claim is made on the TEST wells
(B02 D04 D08 D09 D11) -- the CV fold outputs are never-seen w.r.t. network
training, and this split keeps them never-seen w.r.t. the stitcher too.

    python model_labs/tracer_lab/stitcher.py --sweep    # tune wells only
    python model_labs/tracer_lab/stitcher.py --apply    # frozen, test wells
"""
from __future__ import annotations

import argparse
import itertools
import json
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

FROZEN = {"gap_px": 12.0, "angle_deg": 25.0, "bridge_support": 0.10}
# ^ overwritten by --sweep's plateau before --apply is trusted; the JSON this
#   file writes records what was actually used.


def _endpoints(obj_paths):
    """-> list of (point, inward unit tangent) for the 2 ends of each path.

    The tangent points INTO the fragment, estimated over ~10 px so hand-drawn
    jitter does not decide co-linearity (the 15 px direction-window lesson).
    """
    out = []
    for path in obj_paths:
        p = np.asarray(path)
        if len(p) < 6:
            continue
        h = min(10, len(p) - 1)
        t0 = p[0] - p[h]
        t1 = p[-1] - p[-1 - h]
        for pt, t in ((p[0], t0), (p[-1], t1)):
            n = np.linalg.norm(t)
            if n > 1e-9:
                out.append((pt.astype(float), t / n))
    return out


def stitch(result: dict, centre: np.ndarray, *, gap_px: float,
           angle_deg: float, bridge_support: float, max_rounds: int = 4):
    """-> new object_of mapping with co-linear end-to-end fragments joined."""
    paths = result["paths"]
    parent = dict(result["object_of"])

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    H, W = centre.shape
    cos_gate = np.cos(np.radians(angle_deg))

    for _ in range(max_rounds):
        groups: dict[int, list] = {}
        for pid, path in enumerate(paths, start=1):
            groups.setdefault(find(pid), []).append(path)

        cand = []
        keys = list(groups.keys())
        ends = {k: _endpoints(groups[k]) for k in keys}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                best = None
                for pa, ta in ends[a]:
                    for pb, tb in ends[b]:
                        gap = pb - pa
                        d = float(np.linalg.norm(gap))
                        if d > gap_px or d < 1e-6:
                            continue
                        g = gap / d
                        # fragment A ends heading OUT along -ta; the gap must
                        # continue that heading, and B must continue the gap
                        if float(-ta @ g) < cos_gate:
                            continue
                        if float(tb @ g) < cos_gate:
                            continue
                        n_s = max(int(d), 2)
                        seg = pa[None, :] + np.linspace(0, 1, n_s)[:, None] * gap[None, :]
                        r = np.clip(np.rint(seg[:, 0]).astype(int), 0, H - 1)
                        c = np.clip(np.rint(seg[:, 1]).astype(int), 0, W - 1)
                        sup = float(centre[r, c].mean())
                        if sup < bridge_support:
                            continue
                        score = sup / (1.0 + d)
                        if best is None or score > best[0]:
                            best = (score, a, b)
                if best:
                    cand.append(best)

        if not cand:
            break
        cand.sort(reverse=True)
        used = set()
        joined = 0
        for score, a, b in cand:
            ra, rb = find(a), find(b)
            if ra == rb or ra in used or rb in used:
                continue
            parent[ra] = rb
            used.add(ra)
            used.add(rb)
            joined += 1
        if joined == 0:
            break

    return {**result, "object_of": {pid: find(pid) for pid in parent}}


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
    # bridge support is judged on the PRE-nms centre map: the crest holes the
    # stitch must cross are exactly where NMS zeroed the field
    stitched = stitch(res, pred["centre"], **params)
    after = score_against_gt(stitched, wf)
    return before, after


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", action="store_true", help="tune wells only")
    ap.add_argument("--apply", action="store_true",
                    help="frozen params, test wells")
    a = ap.parse_args(argv)

    keys = ("length_mdape", "identity_through_crossing", "recall_traces",
            "false_split_count", "false_merge_count", "n_objects")

    if a.sweep:
        grid = {"gap_px": [8.0, 12.0, 18.0], "angle_deg": [15.0, 25.0, 35.0],
                "bridge_support": [0.05, 0.10, 0.20]}
        print(f"{'gap':>5}{'ang':>5}{'sup':>6} | {'mdape':>14}{'ident':>14}"
              f"{'merge':>12}{'split':>12}")
        out = []
        base = {w: None for w in TUNE_WELLS}
        for combo in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), combo))
            agg_b, agg_a = [], []
            for w in TUNE_WELLS:
                b, af = _run_well(w, params)
                agg_b.append(b)
                agg_a.append(af)
            mb = np.median([x["length_mdape"] for x in agg_b])
            ma = np.median([x["length_mdape"] for x in agg_a])
            ib = np.mean([x["identity_through_crossing"] for x in agg_b])
            ia = np.mean([x["identity_through_crossing"] for x in agg_a])
            gb = sum(x["false_merge_count"] for x in agg_b)
            ga = sum(x["false_merge_count"] for x in agg_a)
            sb = sum(x["false_split_count"] for x in agg_b)
            sa = sum(x["false_split_count"] for x in agg_a)
            print(f"{params['gap_px']:>5.0f}{params['angle_deg']:>5.0f}"
                  f"{params['bridge_support']:>6.2f} |"
                  f"{mb:>7.3f}->{ma:<6.3f}{ib:>7.3f}->{ia:<6.3f}"
                  f"{gb:>5}->{ga:<6}{sb:>5}->{sa:<6}", flush=True)
            out.append({**params, "mdape_before": float(mb),
                        "mdape_after": float(ma), "ident_after": float(ia),
                        "merge_before": int(gb), "merge_after": int(ga),
                        "split_after": int(sa)})
        (ROOT / "model_labs/tracer_lab/_runs/stitch_sweep.json").write_text(
            json.dumps(out, indent=2))
        print("\nwritten: _runs/stitch_sweep.json -- freeze the plateau into "
              "FROZEN before --apply")
        return 0

    if a.apply:
        print(f"frozen params: {FROZEN}   (tuned on {TUNE_WELLS})")
        rows = []
        for w in TEST_WELLS:
            b, af = _run_well(w, FROZEN)
            rows.append({"well": w,
                         "before": {k: b[k] for k in keys},
                         "after": {k: af[k] for k in keys}})
            print(f"{w}: mdape {b['length_mdape']:.3f}->"
                  f"{af['length_mdape']:.3f}  ident "
                  f"{b['identity_through_crossing']:.3f}->"
                  f"{af['identity_through_crossing']:.3f}  merges "
                  f"{b['false_merge_count']}->{af['false_merge_count']}  "
                  f"splits {b['false_split_count']}->"
                  f"{af['false_split_count']}  nobj {b['n_objects']}->"
                  f"{af['n_objects']}", flush=True)
        (ROOT / "model_labs/tracer_lab/_runs/stitch_test.json").write_text(
            json.dumps({"frozen": FROZEN, "tune_wells": TUNE_WELLS,
                        "rows": rows}, indent=2))
        print("\nwritten: _runs/stitch_test.json")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
