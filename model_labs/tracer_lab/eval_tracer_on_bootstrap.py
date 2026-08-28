"""Score the tracer on the sealed PLATE_23 bootstrap — the T03 test set.

The trace->mask bridge: the tracer emits polylines; the sealed benchmark
scores instance masks. Each traced object's member paths are stamped at the
corpus ribbon convention (``width_px: 8.0``, band = distance <= width/2 from
the 1 px-dense spine, exactly `annotation_tools.relabel.raster.ribbon_mask`
but computed on a bounding-box crop), OR-ed per object, and exported through
the same `export_prediction` -> `benchmark_instances` path as every other
candidate. Overlapping objects keep their overlap — masks are overlap-safe,
which is the representation advantage this lane exists for.

Predeclarations, fixed before any PLATE_23 inference ran:

* **checkpoint**: ``_runs/net_cv/B02/best.pt`` — the fold trained on the nine
  non-B02 wells: the same nine-well training set as the lane's historical
  single-split models, and the fold that retains all five wells the walk
  thresholds were frozen on. No bootstrap image was seen by any fold
  (verified: the ``23_B02_ctrl`` / ``29_C05`` name collisions with the dense
  corpus are different fields — pixel correlation ~0.001 / 0.006). No
  ensembling: ensemble-mean fields would shift the centre statistics the
  frozen thresholds were calibrated on, and that shift cannot be validated
  on any PLATE_32 well without leakage.
* **walk + preps**: the frozen CV configuration verbatim
  (`cv_report.py`): walk seed 0.4 / support 0.3 / claim 3.5 px / rescue 1;
  ``nms`` prep crossing 0.4, valid 0.2 (the shipped arm), ``raw`` prep
  crossing 0.4, valid 0.2 (the coverage arm, reported alongside because the
  lane's standing rule is that either arm alone misrepresents the state).
* **length gates**: objects shorter than the gate are not exported.
  **Primary row: nms arm, 50 um** (the lane's frozen counting convention);
  0 um and 25 um (the classical winner's own gate) are computed in the same
  pass and all reported — disclosure, not selection.
* **metrics, in the predeclared order**: ``length_mdape`` vs floor 0.3169,
  ``false_split_count`` vs 52/375, pooled ``recall`` vs 0.928. Precision/F1
  are not interpretable against this sparse proposal-conditioned GT (see
  `eval_on_bootstrap.py`) and carry the same disclaimer here.

Numbers go to Codex; Codex rules. This script scores nothing into a ruling.

Run in the GPU env::

    python model_labs/tracer_lab/eval_tracer_on_bootstrap.py \
        --out model_labs/tracer_lab/_runs/eval_bootstrap_v1
"""
from __future__ import annotations

import argparse
import hashlib
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

BOOTSTRAP_REL = "PrecisionMyotube/annotation_work/bootstrap_v1"
PIXEL_UM = 0.650017          # same acquisition family as the dense corpus
WIDTH_PX = 8.0               # the corpus ribbon width — not a knob
FLOORS = {"length_mdape": 0.3169, "false_split_count": 52, "recall": 0.928}
GATES_UM = (0.0, 25.0, 50.0)
PRIMARY = ("nms", 50.0)

WALK = dict(seed_thresh=0.4, support_thresh=0.3, claim_radius_px=3.5,
            rescue_window_steps=1)
CROSSING_THRESH = 0.4
VALID_THRESH = 0.2


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _arc(p) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(p), axis=0), axis=1).sum())


def stamp_object(paths: list[np.ndarray], shape: tuple[int, int],
                 width_px: float = WIDTH_PX) -> np.ndarray:
    """One object's ribbon mask: `ribbon_mask` semantics on a bbox crop.

    The corpus convention is band = EDT(spine) <= width/2 with the spine
    rasterised at ~1 px density (`polyline_pixels`). A full-frame EDT per
    object is exact but ~1 s x thousands of objects; the EDT of a crop padded
    by width/2 + 2 px is identical inside the band (no spine pixel outside
    the pad can influence a distance <= width/2).
    """
    from scipy import ndimage as ndi

    from annotation_tools.relabel.raster import polyline_pixels

    H, W = shape
    rr_all, cc_all = [], []
    for p in paths:
        p = np.asarray(p, dtype=float)
        if p.ndim != 2 or p.shape[0] < 2:
            continue
        r, c = polyline_pixels(p)
        rr_all.append(r)
        cc_all.append(c)
    mask = np.zeros(shape, dtype=bool)
    if not rr_all:
        return mask
    rr = np.clip(np.round(np.concatenate(rr_all)).astype(int), 0, H - 1)
    cc = np.clip(np.round(np.concatenate(cc_all)).astype(int), 0, W - 1)
    pad = int(np.ceil(width_px / 2.0)) + 2
    r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad + 1, H)
    c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad + 1, W)
    spine = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    spine[rr - r0, cc - c0] = True
    dist = ndi.distance_transform_edt(~spine)
    mask[r0:r1, c0:c1] = dist <= max(width_px, 1.0) / 2.0
    return mask


def objects_from_walk(res: dict) -> dict[int, list[np.ndarray]]:
    """Group the walk's paths by final object id."""
    objects: dict[int, list[np.ndarray]] = {}
    for pid, path in enumerate(res["paths"], start=1):
        oid = res["object_of"][pid]
        objects.setdefault(oid, []).append(np.asarray(path, dtype=float))
    return objects


def main(argv=None) -> int:
    import tifffile

    from _shared.eval_gt import build_eval_gt
    from _shared.predict_export import ModelProvenance, export_prediction
    from precision_myotube.benchmark import benchmark_instances
    from tracer_lab.infer_trace import predict_fields, fields_for_walk
    from tracer_lab.oracle_trace import TraceParams, trace_field

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt",
                    default="model_labs/tracer_lab/_runs/net_cv/B02/best.pt")
    ap.add_argument("--bootstrap", default=str(ROOT / BOOTSTRAP_REL))
    ap.add_argument("--out",
                    default="model_labs/tracer_lab/_runs/eval_bootstrap_v1")
    ap.add_argument("--wells", nargs="+", default=None)
    args = ap.parse_args(argv)

    bootstrap = Path(args.bootstrap)
    manifest_path = bootstrap / "bootstrap_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wells = args.wells or sorted(manifest["per_well"])
    ckpt = ROOT / args.ckpt
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    prov = ModelProvenance(
        model="tracer", version="cv_foldB02_v1",
        architecture=("TracerNet 4-head U-Net 1.93M params (centre/orient/"
                      "crossing/offset), v5 recipe, trained on the 9 non-B02 "
                      "wells of the dense PLATE_32 corpus; deterministic walk "
                      "on predicted fields; polylines stamped as ribbons at "
                      f"the corpus width_px={WIDTH_PX}"),
        checkpoint_hash=sha256_file(ckpt),
        environment_hash="", data_hash="", seed=0,
        thresholds={"walk": WALK, "crossing_thresh": CROSSING_THRESH,
                    "valid_thresh": VALID_THRESH, "width_px": WIDTH_PX,
                    "gates_um": list(GATES_UM), "primary": list(PRIMARY)},
        channels="desmin_only", used_prompts=False)

    print(f"checkpoint : {ckpt}")
    print(f"test set   : {bootstrap}  ({len(wells)} wells)")
    print(f"walk {WALK}  crossing {CROSSING_THRESH}  valid {VALID_THRESH}"
          f"  width {WIDTH_PX}px  gates {GATES_UM}um  "
          f"PRIMARY {PRIMARY[0]}/{PRIMARY[1]:g}um\n")

    rows: dict[str, dict] = {}      # f"{arm}_min{gate:g}" -> accumulators
    per_well_all: dict[str, dict] = {}
    for well in wells:
        gt = build_eval_gt(manifest, well, out / "eval_gt")
        img = tifffile.imread(bootstrap / well / "image_fiber.tif") \
            .astype(np.float32)
        lo, hi = np.percentile(img, [1.0, 99.9])
        norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0) \
            .astype(np.float32)

        t0 = time.time()
        pred = predict_fields(norm, ckpt)
        t_pred = time.time() - t0

        for arm in ("nms", "raw"):
            if arm == "raw":
                xg = pred["crossing"] >= CROSSING_THRESH
                wf = {"centre": pred["centre"], "orient": pred["orient"],
                      "crossing": xg,
                      "orient_valid": (pred["centre"] >= VALID_THRESH) & ~xg}
            else:
                wf = fields_for_walk(pred, crossing_thresh=CROSSING_THRESH,
                                     valid_thresh=VALID_THRESH, prep="nms")
            t0 = time.time()
            res = trace_field(wf, TraceParams(**WALK))
            t_walk = time.time() - t0
            objects = objects_from_walk(res)
            length_um = {oid: sum(_arc(p) for p in paths) * PIXEL_UM
                         for oid, paths in objects.items()}

            for gate in GATES_UM:
                keep = [oid for oid, L in length_um.items() if L >= gate]
                tag = f"{arm}_min{gate:g}"
                t0 = time.time()
                exported = export_prediction(
                    out / "predictions" / tag, gt["image_id"],
                    gt["image_shape"], prov,
                    masks=(stamp_object(objects[oid], gt["image_shape"])
                           for oid in keep),
                    write_convenience_tiff=False, status="complete")
                m = benchmark_instances(gt["path"], exported["instances"])
                t_score = time.time() - t0

                acc = rows.setdefault(tag, {"n_gt": 0, "n_pred": 0, "tp": 0,
                                            "false_split_count": 0,
                                            "over_merge_count": 0,
                                            "mdapes": []})
                for k in ("n_gt", "n_pred", "tp", "false_split_count",
                          "over_merge_count"):
                    acc[k] += m[k]
                if m.get("length_mdape") is not None:
                    acc["mdapes"].append(m["length_mdape"])
                per_well_all.setdefault(tag, {})[well] = {
                    **{k: m[k] for k in (
                        "n_gt", "n_pred", "tp", "precision", "recall", "f1",
                        "mean_matched_iou", "false_split_count",
                        "false_split_rate", "over_merge_count",
                        "over_merge_rate", "length_mdape", "width_mdape",
                        "automatic_coverage")},
                    "seconds": {"predict": round(t_pred, 1),
                                "walk": round(t_walk, 1),
                                "stamp+score": round(t_score, 1)}}
                lm = m.get("length_mdape")
                print(f"  {well:<22}{tag:<12}n_gt {m['n_gt']:>4}  "
                      f"n_pred {m['n_pred']:>5}  tp {m['tp']:>4}  "
                      f"recall {m['recall']:.3f}  fsplit "
                      f"{m['false_split_count']:>3}  mdape "
                      f"{(f'{lm:.4f}' if lm is not None else 'n/a')}",
                      flush=True)

    print("\n=== pooled, per configuration ===")
    hdr = (f"{'config':<12}{'n_gt':>6}{'n_pred':>8}{'tp':>6}{'recall':>8}"
           f"{'fsplit':>8}{'omerge':>8}{'med mdape':>11}")
    print(hdr)
    print("-" * len(hdr))
    summary_rows = {}
    for tag, acc in rows.items():
        rec = acc["tp"] / max(acc["n_gt"], 1)
        med = float(np.median(acc["mdapes"])) if acc["mdapes"] else None
        summary_rows[tag] = {
            "n_gt": acc["n_gt"], "n_pred": acc["n_pred"], "tp": acc["tp"],
            "pooled_recall": round(rec, 4),
            "false_split_count": acc["false_split_count"],
            "over_merge_count": acc["over_merge_count"],
            "median_length_mdape": med}
        print(f"{tag:<12}{acc['n_gt']:>6}{acc['n_pred']:>8}{acc['tp']:>6}"
              f"{rec:>8.3f}{acc['false_split_count']:>8}"
              f"{acc['over_merge_count']:>8}"
              f"{(f'{med:.4f}' if med is not None else 'n/a'):>11}")

    ptag = f"{PRIMARY[0]}_min{PRIMARY[1]:g}"
    p = summary_rows[ptag]
    print(f"\n=== PRIMARY ({ptag}) against the classical floor ===")
    med = p["median_length_mdape"]
    print(f"  length_mdape       "
          f"{(f'{med:.4f}' if med is not None else 'n/a')}   "
          f"floor {FLOORS['length_mdape']}  "
          f"{'BETTER' if med is not None and med < FLOORS['length_mdape'] else 'WORSE'}")
    print(f"  false_split_count  {p['false_split_count']:<8} "
          f"floor {FLOORS['false_split_count']}     "
          f"{'BETTER' if p['false_split_count'] < FLOORS['false_split_count'] else 'WORSE'}")
    print(f"  recall (pooled)    {p['pooled_recall']:.3f}    "
          f"floor {FLOORS['recall']}  "
          f"{'BETTER' if p['pooled_recall'] > FLOORS['recall'] else 'WORSE'}")
    print("\nprecision/F1 deliberately not compared: reviewed `complete` is a "
          "sparse\ncertified subset, so unmatched predictions are not "
          "necessarily false objects.")

    (out / "eval_summary.json").write_text(json.dumps({
        "checkpoint": str(ckpt),
        "checkpoint_sha256": prov.checkpoint_hash,
        "test_set": str(bootstrap),
        "test_set_manifest_sha256": sha256_file(manifest_path),
        "n_wells": len(wells),
        "predeclared": {
            "primary": ptag, "walk": WALK,
            "crossing_thresh": CROSSING_THRESH, "valid_thresh": VALID_THRESH,
            "width_px": WIDTH_PX, "gates_um": list(GATES_UM),
            "metric_order": ["length_mdape", "false_split_count", "recall"],
            "classical_floor": FLOORS,
            "leakage_check": ("bootstrap 23_B02_ctrl / 29_C05 vs corpus "
                              "B02 / C05: pixel corr 0.0006 / 0.0061 — "
                              "different fields, no overlap with training")},
        "pooled": summary_rows,
        "interpretation_note": (
            "precision and F1 are not interpretable against this sparse "
            "proposal-conditioned GT; a densely-trained model finds far more "
            "fibres than were ever certified"),
        "evidence_class": "cross_plate_independent_test_single_operator",
        "per_well": per_well_all}, indent=2, default=str), encoding="utf-8")
    print(f"\n-> {out / 'eval_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
