"""Per-well runner + aggregation.

For each well with a completed prediction and usable GT geometry, computes Tier-1 detection metrics
and Tier-2 endpoint metrics, writes a per-well JSON, and rolls everything into two summary CSVs:
  - detection_summary.csv : per-well P/R/F1 + error-class counts
  - endpoint_summary.csv  : per-group %below/above 300 um (GT vs predicted), mirroring
                            Q_Plates/myotube_length_summary.csv
"""
from __future__ import annotations

import json

import numpy as np

from . import config as C
from . import geometry as g
from . import io_load as io
from . import matching as M
from . import scientific as S


def score_well(w, radius=None, verbose=True) -> dict:
    """Run Tier-1 + Tier-2 for one well. Returns a flat result dict."""
    radius = C.DILATE_RADIUS_PX if radius is None else radius
    ft, fr = C.pred_paths(w["run_stem"])
    gt_zip = C.QPLATES / w["plate"] / w["roi_zip"]
    gt_csv = C.QPLATES / w["plate"] / w["results_csv"]

    res = dict(well_id=w["well_id"], plate=w["plate"], group=w["group"], split=w["split"],
               run_stem=w["run_stem"], radius_px=radius,
               gt_geom_partial=w["gt_geom_partial"], tier1=None, tier2=None, notes=[])

    # ---- Tier 2 (endpoint) — always available from the two Results CSVs
    gt_um = io.extract_lengths(gt_csv)
    pred_um = io.read_pred_lengths(fr)
    res["tier2"] = S.endpoint_metrics(gt_um, pred_um, C.THRESH_UM)

    # ---- Tier 1 (detection geometry) — needs usable GT ROI geometry
    if not w["gt_roi_ok"] or not io.gt_zip_ok(gt_zip):
        res["notes"].append("GT ROI geometry unusable; Tier-1 skipped (Tier-2 only).")
        return res

    gt_polys = io.read_imagej_zip(gt_zip)
    pred_polys = io.read_traces(ft)
    if not gt_polys or not pred_polys:
        res["notes"].append("empty GT or prediction polylines; Tier-1 skipped.")
        return res

    # coordinate sanity: everything must be within the image (catches a (y,x) swap)
    H, W = C.IMAGE_SHAPE
    for tag, polys in (("GT", gt_polys), ("pred", pred_polys)):
        allpts = np.concatenate(polys, axis=0)
        if allpts[:, 0].max() >= W + 5 or allpts[:, 1].max() >= H + 5 or allpts.min() < -5:
            res["notes"].append(f"WARNING: {tag} coords out of image bounds — possible convention error.")

    gt_lens_px = np.array([g.polylen(p) for p in gt_polys])
    pred_lens_px = np.array([g.polylen(p) for p in pred_polys])
    gt_lens_um = gt_lens_px * C.PIXEL_UM      # per-fibre um aligned to mask index (not the CSV order)
    pred_lens_um = pred_lens_px * C.PIXEL_UM

    gt_masks = [g.rasterize(p, radius, W) for p in gt_polys]
    pred_masks = [g.rasterize(p, radius, W) for p in pred_polys]

    ov = M.build_overlap(gt_masks, pred_masks)
    matches = M.greedy_match(ov, C.IOU_THRESH)
    det = M.detection_metrics(matches, len(gt_polys), len(pred_polys))
    err = M.classify_errors(ov, matches, gt_lens_px, pred_lens_px,
                            C.MIN_OVERLAP_FRAC, C.TOO_SHORT_RATIO, C.TOO_LONG_RATIO, C.FRAG_COV_FRAC)

    tier1 = dict(
        n_gt_geom=len(gt_polys), n_pred=len(pred_polys),
        precision=det["precision"], recall=det["recall"], f1=det["f1"],
        tp=det["tp"], fp=det["fp"], fn=det["fn"],
        too_short_count=err["too_short_count"], too_short_rate=err["too_short_rate"],
        false_split_count=err["false_split_count"], false_split_rate=err["false_split_rate"],
        fragmented_count=err["fragmented_count"], fragmented_rate=err["fragmented_rate"],
        extra_fragments=err["extra_fragments"],
        over_merge_count=err["over_merge_count"], over_merge_rate=err["over_merge_rate"],
        too_long_count=err["too_long_count"], median_len_ratio=err["median_len_ratio"],
        boundary_flip_rate=S.boundary_flip_rate(matches, gt_lens_um, pred_lens_um, C.THRESH_UM),
        boundary_weighted_mae_um=S.boundary_weighted_mae(
            matches, gt_lens_um, pred_lens_um, C.THRESH_UM, C.BOUNDARY_SIGMA_UM),
    )
    res["tier1"] = tier1
    if w["gt_geom_partial"]:
        res["notes"].append("GT ROI zip is a partial re-trace vs Results.csv; Tier-1 recall understated.")
    if verbose:
        _print_well(res)
    return res


def _fmt(x, p=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def _print_well(res):
    t1, t2 = res["tier1"], res["tier2"]
    print(f"\n=== {res['well_id']}  ({res['group']}, {res['split']}, r={res['radius_px']}px) ===")
    if t1:
        print(f"  Tier1  P={_fmt(t1['precision'])} R={_fmt(t1['recall'])} F1={_fmt(t1['f1'])} "
              f"| GT_geom={t1['n_gt_geom']} pred={t1['n_pred']} tp={t1['tp']} fp={t1['fp']} fn={t1['fn']}")
        print(f"         too_short={t1['too_short_count']} ({_fmt(t1['too_short_rate'],2)})  "
              f"fragmented={t1['fragmented_count']} ({_fmt(t1['fragmented_rate'],2)}, +{t1['extra_fragments']} pieces)  "
              f"over_merge={t1['over_merge_count']}  med_len_ratio={_fmt(t1['median_len_ratio'],2)}")
        print(f"         boundary_flip={_fmt(t1['boundary_flip_rate'],2)}  "
              f"boundary_wMAE={_fmt(t1['boundary_weighted_mae_um'],1)}um")
    else:
        print("  Tier1  skipped")
    print(f"  Tier2  count GT={t2['n_gt']} pred={t2['n_pred']} (delta {t2['count_delta']:+d})  "
          f"%<300 GT={_fmt(t2['gt_pct_below'],1)} pred={_fmt(t2['pred_pct_below'],1)} "
          f"(delta {_fmt(t2['pct_below_delta'],1)})")
    for n in res["notes"]:
        print(f"  note: {n}")


# ---- aggregation -----------------------------------------------------------------------------
def write_outputs(results):
    C.OUT.mkdir(parents=True, exist_ok=True)
    for res in results:
        (C.OUT / f"{res['well_id']}.json").write_text(json.dumps(res, indent=2, default=float))

    # detection_summary.csv
    det_rows = ["well_id,plate,group,split,n_gt_geom,n_pred,precision,recall,f1,"
                "too_short,false_split,fragmented,extra_fragments,over_merge,too_long,median_len_ratio,"
                "boundary_flip_rate,boundary_wMAE_um,gt_geom_partial"]
    for r in results:
        t1 = r["tier1"]
        if not t1:
            det_rows.append(f"{r['well_id']},{r['plate']},{r['group']},{r['split']},,,,,,,,,,,,,,,{r['gt_geom_partial']}")
            continue
        det_rows.append(",".join(str(x) for x in [
            r["well_id"], r["plate"], r["group"], r["split"], t1["n_gt_geom"], t1["n_pred"],
            f"{t1['precision']:.4f}", f"{t1['recall']:.4f}", f"{t1['f1']:.4f}",
            t1["too_short_count"], t1["false_split_count"], t1["fragmented_count"],
            t1["extra_fragments"], t1["over_merge_count"], t1["too_long_count"],
            f"{t1['median_len_ratio']:.3f}", f"{t1['boundary_flip_rate']:.3f}",
            f"{t1['boundary_weighted_mae_um']:.2f}", r["gt_geom_partial"]]))
    (C.OUT / "detection_summary.csv").write_text("\n".join(det_rows) + "\n")

    # endpoint_summary.csv — per group, GT vs predicted %below300 (mirrors myotube_length_summary.csv)
    by_group = {}
    for r in results:
        by_group.setdefault(r["group"], []).append(r["tier2"])
    ep_rows = ["group,n_wells,gt_pct_below_300,gt_sem,pred_pct_below_300,pred_sem,mean_delta"]
    per_well_delta_rows = ["well_id,group,gt_pct_below_300,pred_pct_below_300,delta,count_delta"]
    for grp in C.GROUP_ORDER:
        ts = by_group.get(grp)
        if not ts:
            continue
        gtb = np.array([t["gt_pct_below"] for t in ts], dtype=float)
        prb = np.array([t["pred_pct_below"] for t in ts], dtype=float)
        n = len(ts)
        sem = lambda a: (float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0)
        ep_rows.append(",".join(str(x) for x in [
            grp, n, f"{np.mean(gtb):.1f}", f"{sem(gtb):.1f}",
            f"{np.mean(prb):.1f}", f"{sem(prb):.1f}", f"{np.mean(prb - gtb):.1f}"]))
    for r in results:
        t2 = r["tier2"]
        per_well_delta_rows.append(",".join(str(x) for x in [
            r["well_id"], r["group"], f"{t2['gt_pct_below']:.1f}", f"{t2['pred_pct_below']:.1f}",
            f"{t2['pct_below_delta']:.1f}", t2["count_delta"]]))
    (C.OUT / "endpoint_summary.csv").write_text("\n".join(ep_rows) + "\n")
    (C.OUT / "endpoint_per_well.csv").write_text("\n".join(per_well_delta_rows) + "\n")
    print(f"\nWrote per-well JSON + detection_summary.csv + endpoint_summary.csv -> {C.OUT}")
