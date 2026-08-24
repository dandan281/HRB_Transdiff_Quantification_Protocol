"""Score every trained checkpoint end-to-end, identically, on one table.

Field statistics (ridge FWHM, mass ratio) are proxies. This is the thing they
are proxies FOR: raw image in, fields predicted, the same frozen walk, scored
against the operator's polylines. Every version gets the same walk parameters
and the same well, so the column that changes is the network.

The oracle row is the ceiling (perfect fields, same walk); the classical floor
0.3169 is context from a different GT and is printed, not compared.

Runs on a TRAINING well by default. B02 is held out and is refused, because a
version chosen by its held-out score is no longer held out.

    python model_labs/tracer_lab/benchmark_versions.py --well D04
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RUNS = ROOT / "model_labs/tracer_lab/_runs"


def main(argv=None) -> int:
    from tracer_lab.train_tracer import load_well, CORPUS
    from tracer_lab.infer_trace import predict_fields, fields_for_walk
    from tracer_lab.oracle_trace import (
        TraceParams, score_against_gt, trace_field)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--well", default="D04")
    ap.add_argument("--held-out", default="B02")
    ap.add_argument("--preps", default="raw,nms")
    ap.add_argument("--versions", default="")
    a = ap.parse_args(argv)

    if a.well == a.held_out:
        print(f"refusing: {a.well} is held out", file=sys.stderr)
        return 1

    versions = ([v for v in a.versions.split(",") if v] or
                sorted(p.name for p in RUNS.glob("net_v*")
                       if (p / "best.pt").exists()))
    preps = [p for p in a.preps.split(",") if p]

    image, gt, _ = load_well(a.well)
    um = json.loads((CORPUS / a.well / "well_manifest.json")
                    .read_text())["pixel_um"]
    # one walk configuration for every row -- the plateau of the net_v3 sweep
    walk = dict(seed_thresh=0.4, support_thresh=0.3, claim_radius_px=3.5,
                rescue_window_steps=1)

    def arc(p):
        return float(np.linalg.norm(np.diff(np.asarray(p), axis=0),
                                    axis=1).sum())

    gt_total_mm = sum(arc(t) * um for t in gt["traces"]) / 1000.0
    print(f"well {a.well} (TRAINING well) -- {len(gt['traces'])} operator "
          f"traces, {gt_total_mm:.1f} mm traced")
    print(f"walk: {walk}\n")
    hdr = (f"{'version':<9}{'prep':<6}{'id_x':>7}{'mdape':>9}{'split':>7}"
           f"{'merge':>7}{'recall':>8}{'nobj':>7}{'mm':>8}{'sec':>6}")
    print(hdr)
    print("-" * len(hdr))

    rows = []

    # the ceiling, same walk, perfect fields
    t0 = time.time()
    res = trace_field(gt, TraceParams(**walk))
    sc = score_against_gt(res, gt)
    mm = sum(arc(p) for p in res["paths"]) * um / 1000.0
    print(f"{'ORACLE':<9}{'-':<6}{sc['identity_through_crossing']:>7.3f}"
          f"{sc['length_mdape']:>9.3f}{sc['false_split_count']:>7}"
          f"{sc['false_merge_count']:>7}{sc['recall_traces']:>8.3f}"
          f"{sc['n_objects']:>7}{mm:>8.1f}{time.time() - t0:>6.0f}")
    rows.append({"version": "ORACLE", "prep": "-", "mm": mm, **sc})

    for v in versions:
        ck = RUNS / v / "best.pt"
        pred = predict_fields(image, ck)
        for prep in preps:
            t0 = time.time()
            if prep == "raw":
                crossing = pred["crossing"] >= 0.4
                wf = {"centre": pred["centre"], "orient": pred["orient"],
                      "crossing": crossing,
                      "orient_valid": (pred["centre"] >= 0.2) & ~crossing}
            else:
                try:
                    wf = fields_for_walk(pred, crossing_thresh=0.4,
                                         valid_thresh=0.2, prep=prep)
                except KeyError:
                    print(f"{v:<9}{prep:<6}  (no offset head in this "
                          f"checkpoint -- skipped)")
                    continue
            wf["instance"] = gt["instance"]
            wf["traces"] = gt["traces"]
            res = trace_field(wf, TraceParams(**walk))
            sc = score_against_gt(res, wf)
            mm = sum(arc(p) for p in res["paths"]) * um / 1000.0
            print(f"{v:<9}{prep:<6}{sc['identity_through_crossing']:>7.3f}"
                  f"{sc['length_mdape']:>9.3f}{sc['false_split_count']:>7}"
                  f"{sc['false_merge_count']:>7}{sc['recall_traces']:>8.3f}"
                  f"{sc['n_objects']:>7}{mm:>8.1f}"
                  f"{time.time() - t0:>6.0f}", flush=True)
            rows.append({"version": v, "prep": prep, "mm": mm, **sc})

    out = RUNS / f"benchmark_{a.well}.json"
    out.write_text(json.dumps(
        {"well": a.well, "walk": walk, "gt_traces": len(gt["traces"]),
         "gt_total_mm": gt_total_mm, "rows": rows}, indent=2))
    print(f"\nwritten: {out}")
    print("operator total length "
          f"{gt_total_mm:.1f} mm -- the `mm` column should approach it")
    print("classical floor for context: length_mdape 0.3169 (different GT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
