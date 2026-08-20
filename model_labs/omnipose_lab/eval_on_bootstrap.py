"""Score a trained checkpoint on the sealed PLATE_23 bootstrap — the test set.

This is where the project's question gets answered. Training on the dense
PLATE_32 corpus and testing here is a genuinely independent evaluation: different
plate, different session, different annotation method (P23 was proposal-
conditioned triage, P32 was drawn directly), so leakage is structurally
impossible rather than merely audited.

Metrics, in the order that matters:

``length_mdape``       median absolute % error on matched instances. THE number.
                       Classical floor 0.3169 on the non-circular subset.
``false_split_count``  the predeclared T03 primary. Classical floor 52/375.
                       It is the MECHANISM by which length fails: one fibre found
                       as three reports three short myotubes.
``recall``             coverage. Classical floor 0.928.

**Precision and F1 are not interpretable here and are reported only for
completeness.** Reviewed `complete` is a small certified subset of the fibre-like
structure in each field, so an unmatched prediction is not necessarily a false
object -- and a model trained on dense annotation will find far more fibres than
were ever certified. A low precision here is the sparse-GT effect, not a
regression.

Run on a GPU node::

    python model_labs/omnipose_lab/eval_on_bootstrap.py \\
        --checkpoint <...>/v1-fold-B02-paint_out \\
        --out runs/eval_dense_v1
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
for _p in (ROOT / "PrecisionMyotube", ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BOOTSTRAP_REL = "PrecisionMyotube/annotation_work/bootstrap_v1"
PIXEL_UM = 0.650017

# Identical to the training-time inference path, so numbers are commensurable.
EVAL_KW = dict(net_avg=False, tile=True, bsize=224, tile_overlap=0.1,
               mask_threshold=0.0, flow_threshold=0.0, min_size=15,
               cluster=False, resample=True, compute_masks=True, verbose=False)


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None) -> int:
    import tifffile
    import torch
    from cellpose_omni import models

    from _shared.eval_gt import build_eval_gt
    from _shared.predict_export import ModelProvenance, export_prediction
    from omnipose_lab.data import normalize_field
    from omnipose_lab.infer_fold import iter_masks_from_labels
    from precision_myotube.benchmark import benchmark_instances

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bootstrap", default=str(ROOT / BOOTSTRAP_REL))
    ap.add_argument("--out", default="runs/eval_dense_v1")
    ap.add_argument("--wells", nargs="+", default=None)
    ap.add_argument("--nclasses", type=int, default=2)
    args = ap.parse_args(argv)

    bootstrap = Path(args.bootstrap)
    manifest_path = bootstrap / "bootstrap_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no bootstrap manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wells = args.wells or sorted(manifest["per_well"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.is_available()
    print(f"checkpoint : {args.checkpoint}")
    print(f"test set   : {bootstrap}  ({len(wells)} wells)  gpu={gpu}\n")

    model = models.CellposeModel(gpu=gpu, omni=True, dim=2, nchan=1,
                                 nclasses=args.nclasses, diam_mean=0.0,
                                 pretrained_model=str(args.checkpoint))

    prov = ModelProvenance(
        model="omnipose", version="dense_v1",
        architecture=("omnipose nchan=1 dim=2 nclasses=%d, rescale=False, "
                      "fine-tuned from bact_phase_affinity on the dense "
                      "operator-traced PLATE_32 corpus with links"
                      % args.nclasses),
        checkpoint_hash=sha256_file(args.checkpoint),
        environment_hash="", data_hash="", seed=0,
        thresholds=dict(EVAL_KW), channels="desmin_only", used_prompts=False)

    hdr = (f"{'well':<24}{'n_gt':>6}{'n_pred':>8}{'tp':>6}{'recall':>8}"
           f"{'fsplit':>8}{'omerge':>8}{'len_mdape':>11}{'sec':>7}")
    print(hdr); print("-" * len(hdr))

    per_well, totals = {}, {"n_gt": 0, "n_pred": 0, "tp": 0,
                            "false_split_count": 0, "over_merge_count": 0}
    mdapes = []
    for well in wells:
        gt = build_eval_gt(manifest, well, out / "eval_gt")
        img = tifffile.imread(bootstrap / well / "image_fiber.tif")
        norm, _ = normalize_field(img)

        if gpu:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        labels = np.asarray(model.eval(norm, channels=None, channel_axis=None,
                                       normalize=False, omni=True, rescale=None,
                                       diameter=None, **EVAL_KW)[0],
                            dtype=np.int32)
        secs = time.time() - t0

        exported = export_prediction(
            out / "predictions", gt["image_id"], gt["image_shape"], prov,
            masks=iter_masks_from_labels(labels),
            write_convenience_tiff=False, status="complete")
        m = benchmark_instances(gt["path"], exported["instances"])

        np.save(out / f"{well}__labels.npy", labels)
        per_well[well] = {k: m[k] for k in (
            "n_gt", "n_pred", "tp", "precision", "recall", "f1",
            "mean_matched_iou", "false_split_count", "false_split_rate",
            "over_merge_count", "over_merge_rate", "length_mdape",
            "width_mdape", "automatic_coverage")}
        per_well[well]["inference_seconds"] = round(secs, 1)
        for k in totals:
            totals[k] += m[k]
        if m.get("length_mdape") is not None:
            mdapes.append(m["length_mdape"])

        lm = m.get("length_mdape")
        print(f"{well:<24}{m['n_gt']:>6}{m['n_pred']:>8}{m['tp']:>6}"
              f"{m['recall']:>8.3f}{m['false_split_count']:>8}"
              f"{m['over_merge_count']:>8}"
              f"{(f'{lm:.4f}' if lm is not None else 'n/a'):>11}{secs:>7.0f}")

    print("-" * len(hdr))
    micro_recall = totals["tp"] / max(totals["n_gt"], 1)
    print(f"{'POOLED':<24}{totals['n_gt']:>6}{totals['n_pred']:>8}"
          f"{totals['tp']:>6}{micro_recall:>8.3f}"
          f"{totals['false_split_count']:>8}{totals['over_merge_count']:>8}")

    med_mdape = float(np.median(mdapes)) if mdapes else float("nan")
    print("\n=== against the classical floor ===")
    print(f"  length_mdape       {med_mdape:.4f}   floor 0.3169  "
          f"{'BETTER' if med_mdape < 0.3169 else 'WORSE'}")
    print(f"  false_split_count  {totals['false_split_count']:<8} floor 52     "
          f"{'BETTER' if totals['false_split_count'] < 52 else 'WORSE'}")
    print(f"  recall (pooled)    {micro_recall:.3f}    floor 0.928  "
          f"{'BETTER' if micro_recall > 0.928 else 'WORSE'}")
    print("\nprecision/F1 deliberately not compared: reviewed `complete` is a "
          "sparse\ncertified subset, so unmatched predictions are not "
          "necessarily false objects.")

    (out / "eval_summary.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": prov.checkpoint_hash,
        "test_set": str(bootstrap),
        "test_set_manifest_sha256": sha256_file(manifest_path),
        "n_wells": len(wells),
        "pooled": {**totals, "micro_recall": round(micro_recall, 4),
                   "median_length_mdape": med_mdape},
        "classical_floor": {"length_mdape": 0.3169,
                            "false_split_count": 52, "recall": 0.928},
        "interpretation_note": (
            "precision and F1 are not interpretable against this sparse "
            "proposal-conditioned GT; a densely-trained model finds far more "
            "fibres than were ever certified"),
        "evidence_class": "cross_plate_independent_test_single_operator",
        "per_well": per_well}, indent=2, default=str), encoding="utf-8")
    print(f"\n-> {out}/eval_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
