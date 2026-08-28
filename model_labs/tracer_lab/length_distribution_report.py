"""Length-class proportions (operator vs tracer) + the convention restate.

Two questions, both across every annotated well the project owns:

1. **Restate totals under the smoothed convention** (Fiji smooths freehand
   lines before measuring; raw point-to-point arc inflates human traces
   10-15% — session report 2026-08-27 §7d). Old raw-arc numbers are shown
   beside the smoothed ones; nothing is silently replaced.
2. **The distribution question the operator actually cares about**: the
   PROPORTION of myotubes per length class, operator vs tracer — short vs
   long mix, where total-length errors cancel. Output feeds the bar chart.

Wells: PLATE_32 (10, dense corpus annotation, fields cached, each well its
never-seen fold) and PLATE_26/28 (8, never-seen plates, fold-B02, operator
lengths from their own Fiji CSVs). Tracer = frozen nms walk; object
identity = with the frozen weld (the current version); the un-welded
grouping is what the old CV table used and is restated for comparability.

    python model_labs/tracer_lab/length_distribution_report.py
"""
from __future__ import annotations

import csv
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

QP = ROOT / "Q_PLATES/Q_Plates"
CV = ROOT / "model_labs/tracer_lab/_runs/net_cv"
SWEEP_CACHE = ROOT / "model_labs/tracer_lab/_runs/sweep_cache"
UNSEEN_CACHE = ROOT / "model_labs/tracer_lab/_runs/unseen_cache"
OUT = ROOT / "model_labs/tracer_lab/_runs/length_distribution_report.json"

MIN_UM = 50.0
BINS_UM = [50, 150, 300, 500, 800, np.inf]
BIN_LABELS = ["50-150", "150-300", "300-500", "500-800", ">800"]
WALK = dict(seed_thresh=0.4, support_thresh=0.3, claim_radius_px=3.5,
            rescue_window_steps=1)
WELD = dict(weld_dist_px=14.0, weld_deg=12.5, crossing_gate_px=12.0)
UM32 = 0.650017
UM_UNSEEN = {"PLATE_26": 0.649269, "PLATE_28": 0.649341}

UNSEEN = [("PLATE_26", "B02", "B02_Ctrl"),
          ("PLATE_26", "B06", "B06_ACT104_TrkA"),
          ("PLATE_26", "C08", "C08_BR223_IGF1R"),
          ("PLATE_28", "B02", "B02_Ctrl"),
          ("PLATE_28", "B04", "B04_BR223_EGFR"),
          ("PLATE_28", "B08", "B08_BMPR2_HER2"),
          ("PLATE_28", "E08", "E08_BR223_EGFR"),
          ("PLATE_28", "E10", "E10_BR223_IGF1R")]


def arc(p):
    return float(np.linalg.norm(np.diff(np.asarray(p, float), axis=0),
                                axis=1).sum())


def smooth(p, w=5):
    p = np.asarray(p, float)
    if len(p) < w + 2:
        return p
    k = np.ones(w) / w
    q = p.copy()
    for d in range(2):
        q[:, d] = np.convolve(np.pad(p[:, d], (w // 2, w // 2), mode="edge"),
                              k, mode="valid")
    q[0], q[-1] = p[0], p[-1]
    return q


def object_lengths(res, object_of, um, *, smoothed):
    from tracer_lab.centreline_targets import resample_polyline
    out: dict[int, float] = {}
    for pid, path in enumerate(res["paths"], start=1):
        oid = object_of[pid]
        d = resample_polyline(np.asarray(path), 1.0)
        if smoothed:
            d = smooth(d)
        out[oid] = out.get(oid, 0.0) + arc(d) * um
    return np.array([v for v in out.values() if v >= MIN_UM])


def main() -> int:
    from tracer_lab.infer_trace import predict_fields, fields_for_walk
    from tracer_lab.oracle_trace import (
        TraceParams, trace_field, weld_objects)
    from tracer_lab.train_tracer import load_well

    report = {"bins_um": BIN_LABELS, "min_um": MIN_UM,
              "wells_p32": [], "wells_unseen": []}
    human32, tracer32 = [], []
    human_un, tracer_un = [], []

    print("=== PLATE_32 (10 wells, never-seen folds) — convention restate ===")
    hdr = (f"{'well':<6}{'hum raw':>9}{'hum sm':>8}{'trc raw':>9}"
           f"{'trc sm':>8}{'ratio old':>10}{'ratio new':>10}")
    print(hdr)
    wells32 = sorted(p.name for p in
                     (ROOT / "PrecisionMyotube/annotation_work/"
                      "plate32_dense_v1").iterdir() if p.is_dir())
    for well in wells32:
        t0 = time.time()
        image, gt, _ = load_well(well)
        z = np.load(SWEEP_CACHE / f"{well}.npz")
        pred = {k: z[k] for k in ("centre", "orient", "crossing")}
        wf = fields_for_walk(pred, crossing_thresh=0.4, valid_thresh=0.2,
                             prep="nms")
        res = trace_field(wf, TraceParams(**WALK))
        welded = weld_objects(res, wf, **WELD)

        h_raw = np.array([arc(t) * UM32 for t in gt["traces"]])
        h_sm = np.array([arc(smooth(t)) * UM32 for t in gt["traces"]])
        h_raw, h_sm = h_raw[h_raw >= MIN_UM], h_sm[h_sm >= MIN_UM]
        t_raw = object_lengths(res, res["object_of"], UM32, smoothed=False)
        t_sm = object_lengths(welded, welded["object_of"], UM32,
                              smoothed=True)
        rec = {"well": well,
               "human_mm_raw": round(h_raw.sum() / 1000, 2),
               "human_mm_smooth": round(h_sm.sum() / 1000, 2),
               "tracer_mm_raw_noweld": round(t_raw.sum() / 1000, 2),
               "tracer_mm_smooth_weld": round(t_sm.sum() / 1000, 2),
               "human_n": len(h_sm), "tracer_n": len(t_sm)}
        report["wells_p32"].append(rec)
        human32.append(h_sm)
        tracer32.append(t_sm)
        print(f"{well:<6}{rec['human_mm_raw']:>9.1f}"
              f"{rec['human_mm_smooth']:>8.1f}"
              f"{rec['tracer_mm_raw_noweld']:>9.1f}"
              f"{rec['tracer_mm_smooth_weld']:>8.1f}"
              f"{rec['tracer_mm_raw_noweld'] / rec['human_mm_raw']:>10.2f}"
              f"{rec['tracer_mm_smooth_weld'] / rec['human_mm_smooth']:>10.2f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

    print("\n=== PLATE_26/28 (8 never-seen wells) — vs the operator's own "
          "CSVs ===")
    hdr = (f"{'well':<14}{'CSV mm':>8}{'trc sm mm':>10}{'ratio':>7}"
           f"{'old(inflated)':>14}")
    print(hdr)
    for plate, well, stem in UNSEEN:
        t0 = time.time()
        um = UM_UNSEEN[plate]
        rows = list(csv.DictReader((QP / plate / f"{stem}_Results.csv")
                                   .open()))
        csv_len = np.array([float(r["Length"]) for r in rows])
        csv_len = csv_len[csv_len >= MIN_UM]

        img = np.load(UNSEEN_CACHE / f"{plate}_{well}_fiber.npy") \
            .astype(np.float32)
        lo, hi = np.percentile(img, [1.0, 99.9])
        norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) \
            .astype(np.float32)
        pred = predict_fields(norm, CV / "B02" / "best.pt")
        wf = fields_for_walk(pred, crossing_thresh=0.4, valid_thresh=0.2,
                             prep="nms")
        res = trace_field(wf, TraceParams(**WALK))
        welded = weld_objects(res, wf, **WELD)
        t_sm = object_lengths(welded, welded["object_of"], um, smoothed=True)
        t_raw = object_lengths(welded, welded["object_of"], um,
                               smoothed=False)
        # the old table divided tracer raw by the INFLATED raw re-measure of
        # the ROIs; shown for the record
        from tracer_lab.centreline_targets import targets_from_roi_zip
        gt = targets_from_roi_zip(QP / plate / f"{stem}_ROIs.zip", norm.shape)
        h_raw = np.array([arc(t) * um for t in gt["traces"]])
        h_raw = h_raw[h_raw >= MIN_UM]

        rec = {"plate": plate, "well": well,
               "csv_mm": round(csv_len.sum() / 1000, 2),
               "tracer_mm_smooth": round(t_sm.sum() / 1000, 2),
               "ratio_vs_csv": round(t_sm.sum() / csv_len.sum(), 2),
               "old_inflated_ratio": round(t_raw.sum() / h_raw.sum(), 2),
               "human_n": len(csv_len), "tracer_n": len(t_sm)}
        report["wells_unseen"].append(rec)
        human_un.append(csv_len)
        tracer_un.append(t_sm)
        print(f"{plate[-2:]}/{well:<11}{rec['csv_mm']:>8.1f}"
              f"{rec['tracer_mm_smooth']:>10.1f}{rec['ratio_vs_csv']:>7.2f}"
              f"{rec['old_inflated_ratio']:>14.2f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # length-class proportions, pooled
    def props(lens):
        h, _ = np.histogram(lens, bins=BINS_UM)
        return (h / max(h.sum(), 1)).tolist(), int(h.sum())

    from scipy.stats import ks_2samp
    pools = {"PLATE_32": (np.concatenate(human32), np.concatenate(tracer32)),
             "PLATE_26/28": (np.concatenate(human_un),
                             np.concatenate(tracer_un))}
    report["proportions"] = {}
    for name, (h, t) in pools.items():
        ph, nh = props(h)
        pt, nt = props(t)
        ks = ks_2samp(h, t)
        report["proportions"][name] = {
            "human": ph, "human_n": nh, "tracer": pt, "tracer_n": nt,
            "ks_D": round(float(ks.statistic), 3),
            "ks_p": float(ks.pvalue),
            "human_median_um": round(float(np.median(h)), 1),
            "tracer_median_um": round(float(np.median(t)), 1)}
        print(f"\n{name}: human n={nh} median {np.median(h):.0f}um | "
              f"tracer n={nt} median {np.median(t):.0f}um | "
              f"KS D={ks.statistic:.3f} p={ks.pvalue:.2g}")
        for lbl, a, b in zip(BIN_LABELS, ph, pt):
            print(f"  {lbl:>8}: you {a * 100:5.1f}%   tracer {b * 100:5.1f}%")

    OUT.write_text(json.dumps(report, indent=2))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
