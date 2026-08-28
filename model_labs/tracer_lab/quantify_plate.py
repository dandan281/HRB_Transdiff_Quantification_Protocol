"""Quantify a whole plate with the tracer and compare to the operator.

The question this answers is the biological one, not the pixel one: across
every well, how many fibres does the automated pipeline find and how much
total fibre length does it measure, against the human annotation?

Two configurations are run because the candidate does not have one good
operating point, it has two bad ones that fail in opposite directions
(measured on D04): `raw` fields give near-complete coverage with lengths
~2.2x too long, `nms` fields give near-floor length error with half the
fibres missing. Reporting one of them alone would misrepresent the state.

The held-out well is run and labelled but MUST NOT be used to choose between
configurations -- the configurations were frozen on training wells before this
ran, and picking a winner by its held-out score would spend the one clean
comparison the lane has left.

    python model_labs/tracer_lab/quantify_plate.py --ckpt <run>/best.pt
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

CORPUS = ROOT / "PrecisionMyotube/annotation_work/plate32_dense_v1"


def _arc(p):
    return float(np.linalg.norm(np.diff(np.asarray(p), axis=0), axis=1).sum())


def main(argv=None) -> int:
    from tracer_lab.train_tracer import load_well
    from tracer_lab.infer_trace import predict_fields, fields_for_walk
    from tracer_lab.oracle_trace import (
        TraceParams, score_against_gt, trace_field)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt",
                    default="model_labs/tracer_lab/_runs/net_v2/best.pt")
    ap.add_argument("--ckpt-nms",
                    default="model_labs/tracer_lab/_runs/net_v6/best.pt",
                    help="checkpoint for the clean/low-recall arm")
    ap.add_argument("--held-out", default="B02")
    ap.add_argument("--min-um", type=float, default=50.0,
                    help="fibres shorter than this are not counted, matching "
                         "the operator's practice")
    ap.add_argument("--out", default="model_labs/tracer_lab/_runs")
    a = ap.parse_args(argv)

    wells = sorted(p.name for p in CORPUS.iterdir() if p.is_dir())
    walk = dict(seed_thresh=0.4, support_thresh=0.3, claim_radius_px=3.5,
                rescue_window_steps=1)

    hdr = (f"{'well':<6}{'human n':>9}{'human mm':>10}"
           f"{'raw n':>8}{'raw mm':>9}{'nms n':>8}{'nms mm':>9}"
           f"{'raw rec':>9}{'nms rec':>9}")
    print(f"checkpoints: raw={a.ckpt}  nms={a.ckpt_nms}")
    print(f"walk: {walk}   min length {a.min_um} um\n")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for well in wells:
        t0 = time.time()
        image, gt, _ = load_well(well)
        um = json.loads((CORPUS / well / "well_manifest.json")
                        .read_text())["pixel_um"]
        h_len = np.array([_arc(t) * um for t in gt["traces"]])
        h_len = h_len[h_len >= a.min_um]

        rec = {"well": well, "held_out": well == a.held_out,
               "human_lengths_um": [round(float(v), 2) for v in h_len],
               "human_n": int(len(h_len)),
               "human_mm": float(h_len.sum() / 1000.0),
               "human_median_um": float(np.median(h_len)) if len(h_len) else 0.0}

        for tag, ckpt, prep in (("raw", a.ckpt, "raw"),
                                ("nms", a.ckpt_nms, "nms")):
            pred = predict_fields(image, ROOT / ckpt)
            if prep == "raw":
                xg = pred["crossing"] >= 0.4
                wf = {"centre": pred["centre"], "orient": pred["orient"],
                      "crossing": xg,
                      "orient_valid": (pred["centre"] >= 0.2) & ~xg}
            else:
                wf = fields_for_walk(pred, crossing_thresh=0.4,
                                     valid_thresh=0.2, prep="nms")
            wf["instance"] = gt["instance"]
            wf["traces"] = gt["traces"]
            res = trace_field(wf, TraceParams(**walk))
            sc = score_against_gt(res, wf)

            obj_len: dict[int, float] = {}
            for pid, path in enumerate(res["paths"], start=1):
                oid = res["object_of"][pid]
                obj_len[oid] = obj_len.get(oid, 0.0) + _arc(path) * um
            lens = np.array([v for v in obj_len.values() if v >= a.min_um])
            rec[f"{tag}_n"] = int(len(lens))
            rec[f"{tag}_mm"] = float(lens.sum() / 1000.0)
            rec[f"{tag}_median_um"] = (float(np.median(lens))
                                       if len(lens) else 0.0)
            # per-fibre lengths, needed for distribution figures; summary
            # statistics cannot be un-summarised later
            rec[f"{tag}_lengths_um"] = [round(float(v), 2) for v in lens]
            rec[f"{tag}_recall"] = float(sc["recall_traces"])
            rec[f"{tag}_identity"] = float(sc["identity_through_crossing"])
            rec[f"{tag}_mdape"] = float(sc["length_mdape"])

        rows.append(rec)
        flag = " (HELD OUT)" if rec["held_out"] else ""
        print(f"{well:<6}{rec['human_n']:>9}{rec['human_mm']:>10.1f}"
              f"{rec['raw_n']:>8}{rec['raw_mm']:>9.1f}"
              f"{rec['nms_n']:>8}{rec['nms_mm']:>9.1f}"
              f"{rec['raw_recall']:>9.2f}{rec['nms_recall']:>9.2f}{flag}",
              flush=True)
        print(f"       ({time.time() - t0:.0f} s)", flush=True)

    hn = sum(r["human_n"] for r in rows)
    hm = sum(r["human_mm"] for r in rows)
    print("-" * len(hdr))
    print(f"{'PLATE':<6}{hn:>9}{hm:>10.1f}"
          f"{sum(r['raw_n'] for r in rows):>8}"
          f"{sum(r['raw_mm'] for r in rows):>9.1f}"
          f"{sum(r['nms_n'] for r in rows):>8}"
          f"{sum(r['nms_mm'] for r in rows):>9.1f}")

    out = ROOT / a.out / "plate32_quantification.json"
    out.write_text(json.dumps({"walk": walk, "min_um": a.min_um,
                               "ckpt_raw": a.ckpt, "ckpt_nms": a.ckpt_nms,
                               "rows": rows}, indent=2))
    print(f"\nwritten: {out}")
    print("NOTE: the held-out well is reported, not used to choose a "
          "configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
