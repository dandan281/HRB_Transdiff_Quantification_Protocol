"""Sweep the predicted-field thresholds -- on TRAINING wells only.

The oracle's walk parameters were swept on perfect fields and are frozen.
Predicted fields introduce three new knobs that perfect fields never needed:
the crossing probability cut, the validity/support cut, and the seed cut
(predicted centre is dimmer than a perfect Gaussian, so the oracle's 0.9 seed
threshold may find nothing). Per the project rule those are swept full-range
on wells the net TRAINED on, frozen, and only then applied to held-out B02.
Sweeping them on B02 would be threshold search on the test set.

Fields are predicted once per well and cached; each config re-runs only the
walk (~5 s), so the grid is cheap.

    python model_labs/tracer_lab/sweep_infer.py --well D04 \
        --ckpt model_labs/tracer_lab/_runs/net_v1/best.pt
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


def main(argv=None) -> int:
    from tracer_lab.train_tracer import load_well
    from tracer_lab.infer_trace import predict_fields, fields_for_walk
    from tracer_lab.oracle_trace import (
        TraceParams, score_against_gt, trace_field)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--well", default="D04",
                    help="a TRAINING well; never the held-out well")
    ap.add_argument("--ckpt",
                    default="model_labs/tracer_lab/_runs/net_v1/best.pt")
    ap.add_argument("--held-out", default="B02")
    ap.add_argument("--prep", default="nms",
                    choices=("nms", "steer", "offset"),
                    help="how the predicted centre map is turned into a "
                         "walkable ridge")
    ap.add_argument("--out", default="model_labs/tracer_lab/_runs/net_v1")
    a = ap.parse_args(argv)

    if a.well == a.held_out:
        print(f"refusing: {a.well} is the held-out well; sweeping thresholds "
              "on it would be threshold search on the test set",
              file=sys.stderr)
        return 1

    image, gt_fields, _ = load_well(a.well)
    pred = predict_fields(image, ROOT / a.ckpt)
    print(f"well {a.well}: fields predicted; sweeping\n")

    # The first sweep (net_v1) started at seed 0.5 and found NOTHING at any
    # config: the predicted ridge peaked near 0.35 and every threshold in the
    # grid sat above the entire field. The walk's thresholds are in the
    # field's own brightness units -- so the grid must reach down to where
    # the predicted ridge actually lives, and `support_thresh` (which gates
    # both stopping and rescue) has to be swept with it, not inherited from
    # the oracle's perfect-brightness setting.
    # Axes chosen from the measured net_v2 failures: `claim` because the
    # 1.5 px oracle-tuned claim radius lets duplicate walks re-trace a wide
    # fuzzy band (the "marker-pen" objects); `rescue` because gated rescue
    # cannot bridge NMS crest gaps (the fragmentation); support/valid axes
    # went inert in both v2 sweeps and are pinned at their plateau.
    grid = {
        "seed": [0.3, 0.5],
        "crossing": [0.4, 0.6],
        "claim": [1.5, 2.5, 3.5],
        "rescue": [1, 10 ** 9],       # gated to crossings vs always-on
    }
    print(f"{'seed':>5}{'cross':>7}{'claim':>7}{'rescue':>8} | {'id_x':>7}"
          f"{'mdape':>9}{'split':>6}{'merge':>6}{'recall':>7}{'nobj':>6}")
    rows = []
    for s, x, cl, rw in itertools.product(*grid.values()):
        wf = fields_for_walk(pred, crossing_thresh=x, valid_thresh=0.2,
                             prep=a.prep)
        wf["instance"] = gt_fields["instance"]
        wf["traces"] = gt_fields["traces"]
        res = trace_field(wf, TraceParams(
            seed_thresh=s, support_thresh=0.3, claim_radius_px=cl,
            rescue_window_steps=rw))
        sc = score_against_gt(res, wf)
        rows.append({"seed": s, "crossing": x, "claim": cl, "rescue": rw,
                     **{k: sc[k] for k in
                        ("identity_through_crossing", "length_mdape",
                         "false_split_count", "false_merge_count",
                         "recall_traces", "n_objects")}})
        print(f"{s:>5.2f}{x:>7.2f}{cl:>7.1f}{'always' if rw > 99 else 'xing':>8}"
              f" | {sc['identity_through_crossing']:>7.3f}"
              f"{sc['length_mdape']:>9.4f}{sc['false_split_count']:>6}"
              f"{sc['false_merge_count']:>6}{sc['recall_traces']:>7.3f}"
              f"{sc['n_objects']:>6}", flush=True)

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rec = out / f"threshold_sweep_{a.well}.json"
    rec.write_text(json.dumps({"well": a.well, "ckpt": str(a.ckpt),
                               "rows": rows}, indent=2))
    print(f"\nwritten: {rec}")
    print("pick the plateau (metric priority: length_mdape, then "
          "false_split_count, then recall), freeze it, and only then run "
          "infer_trace on the held-out well.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
