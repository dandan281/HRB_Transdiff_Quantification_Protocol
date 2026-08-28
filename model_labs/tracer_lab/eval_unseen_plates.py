"""Trace PLATE_26 / PLATE_28 — data the tracer has NEVER seen — vs annotation.

These two plates took no part in anything: training and threshold tuning
used PLATE_32, the sealed T03 benchmark used PLATE_23. Every nd2 here is
hash-distinct from every training/benchmark image (the recurring
``23_B02_ctrl.nd2`` name is a per-plate acquisition index, not the same
field — verified). So this is the purest evaluation the project owns:
never trained, never tuned, never previously scored.

Predeclared, fixed before any of these images were opened:

* checkpoint ``_runs/net_cv/B02/best.pt`` (the cross-plate model, same as
  the sealed T03 run);
* frozen nms walk (seed 0.4 / support 0.3 / claim 3.5 / rescue 1,
  crossing 0.4 / valid 0.2);
* PRIMARY = the frozen junction weld (dist 14 / deg 12.5 / gate 12);
  the un-welded baseline is reported alongside for attribution;
* comparison per `cv_report.py` conventions: well totals at >= 50 um,
  polyline metrics from `score_against_gt`, everything judged against the
  measured human ceiling, never an implied 1.0.

Two stages because the GPU env has no nd2 reader:

    python model_labs/tracer_lab/eval_unseen_plates.py --extract   # pm-annotate
    python model_labs/tracer_lab/eval_unseen_plates.py             # pm-omnipose
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

QP = ROOT / "Q_PLATES/Q_Plates"
CACHE = ROOT / "model_labs/tracer_lab/_runs/unseen_cache"
OUT = ROOT / "model_labs/tracer_lab/_runs/eval_unseen_plates.json"
CKPT = ROOT / "model_labs/tracer_lab/_runs/net_cv/B02/best.pt"
PIXEL_UM_EXPECTED = 0.650017
MIN_UM = 50.0
WALK = dict(seed_thresh=0.4, support_thresh=0.3, claim_radius_px=3.5,
            rescue_window_steps=1)
WELD = dict(weld_dist_px=14.0, weld_deg=12.5, crossing_gate_px=12.0)

HUMAN_CEILING = {"recall": 0.71, "count_ratio": 1.72, "mdape": 0.096,
                 "identity": 0.27}


def well_pairs(plate: str) -> list[tuple[str, Path, Path]]:
    """(well, nd2, roi_zip) pairs, matched on the well token (B02, E08...)."""
    pdir = QP / plate
    zips = {z.name.split("_")[0].upper(): z for z in pdir.glob("*_ROIs.zip")}
    out = []
    for nd in sorted(pdir.glob("*.nd2")):
        well = nd.stem.split("_")[1].upper()
        if well in zips:
            out.append((well, nd, zips[well]))
    if len(out) != len(zips):
        raise SystemExit(f"{plate}: {len(zips)} ROI zips but only "
                         f"{len(out)} matched to an nd2")
    return out


def extract() -> int:
    from precision_myotube.io import load_nd2

    CACHE.mkdir(parents=True, exist_ok=True)
    for plate in ("PLATE_26", "PLATE_28"):
        for well, nd, _zip in well_pairs(plate):
            dst = CACHE / f"{plate}_{well}_fiber.npy"
            if dst.exists():
                print(f"{plate} {well}: cached")
                continue
            channels, pixel_um, sizes = load_nd2(nd)
            if abs(pixel_um - PIXEL_UM_EXPECTED) > 1e-3:
                raise SystemExit(
                    f"{nd.name}: pixel_um {pixel_um} != expected "
                    f"{PIXEL_UM_EXPECTED} — different acquisition, "
                    f"do NOT trace with the PLATE_32 model unrescaled")
            fiber = channels[1]  # validated Q-plate layout: ch1 = desmin
            np.save(dst, fiber)
            print(f"{plate} {well}: {nd.name} ch1 -> {dst.name} "
                  f"{fiber.shape} px={pixel_um:.6f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args(argv)
    if a.extract:
        return extract()

    from tracer_lab.centreline_targets import targets_from_roi_zip
    from tracer_lab.infer_trace import predict_fields, fields_for_walk
    from tracer_lab.oracle_trace import (
        TraceParams, score_against_gt, trace_field, weld_objects)

    def _arc(p):
        return float(np.linalg.norm(np.diff(np.asarray(p), axis=0),
                                    axis=1).sum())

    rows = []
    for plate in ("PLATE_26", "PLATE_28"):
        for well, nd, zp in well_pairs(plate):
            t0 = time.time()
            src = CACHE / f"{plate}_{well}_fiber.npy"
            if not src.exists():
                raise SystemExit(f"missing {src} — run --extract first "
                                 f"(pm-annotate)")
            img = np.load(src).astype(np.float32)
            lo, hi = np.percentile(img, [1.0, 99.9])
            norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) \
                .astype(np.float32)
            gt = targets_from_roi_zip(zp, norm.shape)

            pred = predict_fields(norm, CKPT)
            wf = fields_for_walk(pred, crossing_thresh=0.4,
                                 valid_thresh=0.2, prep="nms")
            wf["instance"] = gt["instance"]
            wf["traces"] = gt["traces"]
            res = trace_field(wf, TraceParams(**WALK))
            welded = weld_objects(res, wf, **WELD)

            um = PIXEL_UM_EXPECTED
            h_len = np.array([_arc(t) * um for t in gt["traces"]])
            h_len = h_len[h_len >= MIN_UM]
            rec = {"plate": plate, "well": well, "roi_zip": zp.name,
                   "nd2": nd.name, "human_n": int(len(h_len)),
                   "human_mm": round(float(h_len.sum() / 1000.0), 2)}
            for tag, r in (("base", res), ("weld", welded)):
                sc = score_against_gt(r, wf)
                obj_len: dict[int, float] = {}
                for pid, path in enumerate(r["paths"], start=1):
                    oid = r["object_of"][pid]
                    obj_len[oid] = obj_len.get(oid, 0.0) + _arc(path) * um
                lens = np.array([v for v in obj_len.values() if v >= MIN_UM])
                rec[tag] = {
                    "n": int(len(lens)),
                    "mm": round(float(lens.sum() / 1000.0), 2),
                    "recall": round(sc["recall_traces"], 3),
                    "mdape": round(sc["length_mdape"], 3),
                    "splits": sc["false_split_count"],
                    "merges": sc["false_merge_count"],
                    "identity_x": round(sc["identity_through_crossing"], 3),
                    "n_welds": len(r.get("weld_events", []))}
            rows.append(rec)
            w = rec["weld"]
            print(f"{plate} {well}: human n {rec['human_n']:>3} "
                  f"mm {rec['human_mm']:>6.1f} | weld n {w['n']:>4} "
                  f"mm {w['mm']:>6.1f} recall {w['recall']:.2f} "
                  f"splits {w['splits']:>3} idx {w['identity_x']:.2f} "
                  f"mdape {w['mdape']:.3f}  ({time.time() - t0:.0f}s)",
                  flush=True)

    from scipy.stats import spearmanr
    hn = np.array([r["human_n"] for r in rows], float)
    hm = np.array([r["human_mm"] for r in rows], float)
    pooled = {}
    for tag in ("base", "weld"):
        an = np.array([r[tag]["n"] for r in rows], float)
        am = np.array([r[tag]["mm"] for r in rows], float)
        pooled[tag] = {
            "count_ratio_mean": round(float(np.mean(an / hn)), 3),
            "length_ratio_mean": round(float(np.mean(am / hm)), 3),
            "count_spearman": round(float(spearmanr(hn, an).correlation), 3),
            "length_spearman": round(float(spearmanr(hm, am).correlation), 3),
            "recall_mean": round(float(np.mean(
                [r[tag]["recall"] for r in rows])), 3),
            "mdape_median": round(float(np.median(
                [r[tag]["mdape"] for r in rows])), 3),
            "splits_total": int(sum(r[tag]["splits"] for r in rows)),
            "merges_total": int(sum(r[tag]["merges"] for r in rows)),
            "identity_x_mean": round(float(np.mean(
                [r[tag]["identity_x"] for r in rows])), 3)}

    print("\n=== pooled over 8 never-seen wells (2 plates) ===")
    for tag in ("base", "weld"):
        p = pooled[tag]
        print(f"{tag:>5}: len ratio {p['length_ratio_mean']:.2f}x "
              f"rho {p['length_spearman']:+.2f} | count "
              f"{p['count_ratio_mean']:.2f}x rho {p['count_spearman']:+.2f}"
              f" | recall {p['recall_mean']:.2f} | splits "
              f"{p['splits_total']} merges {p['merges_total']} | "
              f"idx {p['identity_x_mean']:.2f} | mdape {p['mdape_median']:.3f}")
    print(f"human ceiling: recall {HUMAN_CEILING['recall']}, count "
          f"{HUMAN_CEILING['count_ratio']}x, per-fibre length "
          f"{HUMAN_CEILING['mdape']}, identity {HUMAN_CEILING['identity']}")

    OUT.write_text(json.dumps({
        "checkpoint": str(CKPT), "walk": WALK, "weld": WELD,
        "min_um": MIN_UM, "human_ceiling": HUMAN_CEILING,
        "evidence": "never trained, never tuned, never previously scored",
        "pooled": pooled, "rows": rows}, indent=2))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
