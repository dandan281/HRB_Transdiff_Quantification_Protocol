"""Cross-validated plate report: every well scored by the fold that never saw it.

The single-split plate quantification (`quantify_plate.py`) scored 9 of 10
wells with a network trained ON them; only B02 was clean, and B02 is
anomalous (+3.5 SD above the density trend). Here each well is predicted by
its OWN fold's checkpoint from the leave-one-well-out run, so every number is
a never-seen-this-well number and the plate-level correlations are honest.

Both walk preparations are reported (raw = coverage arm, nms = clean arm),
same frozen walk parameters as every earlier table, and the human ceiling
from the blind re-trace is printed beside the tracer rows -- metrics are
judged against measured human repeatability, not against an implied 1.0.

    python model_labs/tracer_lab/cv_report.py
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
CV = ROOT / "model_labs/tracer_lab/_runs/net_cv"

HUMAN_CEILING = {"recall": 0.71, "count_ratio": 1.72, "mdape": 0.096,
                 "identity": 0.27}


def _arc(p):
    return float(np.linalg.norm(np.diff(np.asarray(p), axis=0), axis=1).sum())


def main(argv=None) -> int:
    from tracer_lab.train_tracer import load_well
    from tracer_lab.infer_trace import predict_fields, fields_for_walk
    from tracer_lab.oracle_trace import (
        TraceParams, score_against_gt, trace_field)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min-um", type=float, default=50.0)
    ap.add_argument("--out", default="model_labs/tracer_lab/_runs")
    a = ap.parse_args(argv)

    wells = sorted(p.name for p in CORPUS.iterdir() if p.is_dir())
    missing = [w for w in wells if not (CV / w / "manifest.json").exists()]
    if missing:
        print(f"folds not finished yet: {missing}", file=sys.stderr)
        return 1

    walk = dict(seed_thresh=0.4, support_thresh=0.3, claim_radius_px=3.5,
                rescue_window_steps=1)
    hdr = (f"{'well':<6}{'human n':>9}{'human mm':>10}"
           f"{'raw n':>8}{'raw mm':>9}{'nms n':>8}{'nms mm':>9}"
           f"{'raw rec':>9}{'nms rec':>9}{'nms mdape':>11}")
    print("every well scored by the fold that NEVER saw it")
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
        rec = {"well": well,
               "human_n": int(len(h_len)),
               "human_mm": float(h_len.sum() / 1000.0),
               "human_lengths_um": [round(float(v), 2) for v in h_len]}

        pred = predict_fields(image, CV / well / "best.pt")
        for tag in ("raw", "nms"):
            if tag == "raw":
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
            rec[f"{tag}_lengths_um"] = [round(float(v), 2) for v in lens]
            rec[f"{tag}_recall"] = float(sc["recall_traces"])
            rec[f"{tag}_identity"] = float(sc["identity_through_crossing"])
            rec[f"{tag}_mdape"] = float(sc["length_mdape"])
        rows.append(rec)
        print(f"{well:<6}{rec['human_n']:>9}{rec['human_mm']:>10.1f}"
              f"{rec['raw_n']:>8}{rec['raw_mm']:>9.1f}"
              f"{rec['nms_n']:>8}{rec['nms_mm']:>9.1f}"
              f"{rec['raw_recall']:>9.2f}{rec['nms_recall']:>9.2f}"
              f"{rec['nms_mdape']:>11.3f}  ({time.time() - t0:.0f}s)",
              flush=True)

    print("-" * len(hdr))
    hn = sum(r["human_n"] for r in rows)
    hm = sum(r["human_mm"] for r in rows)
    print(f"{'PLATE':<6}{hn:>9}{hm:>10.1f}"
          f"{sum(r['raw_n'] for r in rows):>8}"
          f"{sum(r['raw_mm'] for r in rows):>9.1f}"
          f"{sum(r['nms_n'] for r in rows):>8}"
          f"{sum(r['nms_mm'] for r in rows):>9.1f}")

    from scipy.stats import spearmanr
    hmv = np.array([r["human_mm"] for r in rows])
    hnv = np.array([r["human_n"] for r in rows])
    summary = {}
    for tag in ("raw", "nms"):
        am = np.array([r[f"{tag}_mm"] for r in rows])
        an = np.array([r[f"{tag}_n"] for r in rows])
        summary[tag] = {
            "length_ratio_mean": float(np.mean(am / hmv)),
            "length_pearson_r": float(np.corrcoef(hmv, am)[0, 1]),
            "length_spearman": float(spearmanr(hmv, am).correlation),
            "count_ratio_mean": float(np.mean(an / hnv)),
            "count_pearson_r": float(np.corrcoef(hnv, an)[0, 1]),
            "count_spearman": float(spearmanr(hnv, an).correlation),
            "recall_mean": float(np.mean([r[f"{tag}_recall"] for r in rows])),
            "mdape_median": float(np.median([r[f"{tag}_mdape"]
                                             for r in rows])),
        }
        s = summary[tag]
        print(f"\n== {tag} (all wells NEVER-SEEN)")
        print(f"  total length ratio {s['length_ratio_mean']:.2f}x"
              f"   r={s['length_pearson_r']:+.3f}"
              f"   rank rho={s['length_spearman']:+.3f}")
        print(f"  count ratio {s['count_ratio_mean']:.2f}x"
              f"   r={s['count_pearson_r']:+.3f}"
              f"   rank rho={s['count_spearman']:+.3f}")
        print(f"  mean recall {s['recall_mean']:.3f}"
              f"   median per-fibre mdape {s['mdape_median']:.3f}")
    print(f"\nhuman ceiling (blind re-trace, one window): "
          f"recall {HUMAN_CEILING['recall']}, count {HUMAN_CEILING['count_ratio']}x, "
          f"per-fibre length {HUMAN_CEILING['mdape']}, "
          f"identity {HUMAN_CEILING['identity']}")

    out = ROOT / a.out / "plate32_cv_report.json"
    out.write_text(json.dumps({"walk": walk, "min_um": a.min_um,
                               "human_ceiling": HUMAN_CEILING,
                               "summary": summary, "rows": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
