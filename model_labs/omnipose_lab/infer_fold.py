"""T02 candidate 2 - inference on a held-out well, and canonical sealed export.

Two things here are deliberate and easy to get wrong.

**Inference runs on the unpainted field.** `omnipose_lab.ignore_policy` paints
uncertified regions out of the *training* image only. A held-out well is scored
exactly as the microscope produced it, so the ambiguous fibres the model never
trained on are present at test time. That is the honest test of whether the paint
policy taught a real background suppressor or only a shortcut.

**Tiles are stitched in flow space, not mask space.** Omnipose's own `tile=True`
averages overlapping tile outputs across the flow/distance fields and then
reconstructs masks **once, over the whole field**. So an object crossing a tile
seam is never split and never needs a merge step -- which is what
`model_labs/omnipose/resource_note.md`'s completion check asks for, satisfied by
construction rather than by post-hoc reconciliation.

Export goes through the canonical adapter with ``status="complete"``. The
adapter's default is ``"ambiguous"``, and
`precision_myotube.benchmark.benchmark_instances` scores only ``"complete"``
predictions -- an export left at the default is silently scored as **zero
detections**. Masks are streamed one at a time: materialising one full-field
boolean per instance costs ~13 MB each and has already stalled this project once
at 15.7 GB.

Usage (from pm-omnipose)::

    python model_labs/omnipose_lab/infer_fold.py --held-out 23_B02_ctrl \\
        --checkpoint <path> --out model_labs/omnipose_lab/_runs/v1
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

MODEL_NAME = "omnipose"
MODEL_VERSION = "v1"
BOOTSTRAP = "PrecisionMyotube/annotation_work/bootstrap_v1/bootstrap_manifest.json"

DEFAULT_THRESHOLDS = {
    # Omnipose distance threshold; 0.0 is the library default for the distance field.
    "mask_threshold": 0.0,
    # Flow-error filtering is a Cellpose-era heuristic tuned on round cells and is
    # routinely disabled for Omnipose's elongated targets. 0.0 = off.
    "flow_threshold": 0.0,
    # Below the smallest reviewed instance, so the gate never removes a real target.
    "min_size": 15,
    "cluster": False,
    "resample": True,
    "bsize": 224,
    "tile_overlap": 0.1,
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iter_masks_from_labels(labels: np.ndarray):
    """Yield one boolean mask at a time from a label image.

    Never build the list. One 3636x3636 boolean is ~13 MB; a field with a thousand
    predicted instances would be ~13 GB held at once.
    """
    from scipy import ndimage

    boxes = ndimage.find_objects(labels)
    for value, box in enumerate(boxes, start=1):
        if box is None:
            continue
        mask = np.zeros(labels.shape, dtype=bool)
        mask[box] = labels[box] == value
        yield mask


def load_model(checkpoint: str | Path, nclasses: int = 2):
    from cellpose_omni import models

    return models.CellposeModel(
        gpu=True, omni=True, dim=2, nchan=1, nclasses=nclasses,
        diam_mean=0.0, pretrained_model=str(checkpoint))


def predict_field(model, image: np.ndarray, thresholds: dict) -> tuple[np.ndarray, dict]:
    """Full-field inference. Returns (label image, timing/debug)."""
    import torch

    from omnipose_lab.data import normalize_field

    # One intensity scale for the whole field, matching training. Omnipose's own
    # tiled path logs "Now normalizing each tile separately", which on a ~98%
    # background field stretches a fibre-free tile's noise across the full range
    # and hands the network manufactured signal.
    normalised, norm_stats = normalize_field(image)

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    masks, _flows, _styles = model.eval(
        normalised,
        channels=None, channel_axis=None, normalize=False,
        omni=True, rescale=None, diameter=None, net_avg=False,
        # tile=True stitches the *flow field* across tiles and reconstructs masks
        # once over the whole field, so seam-crossing objects are never split.
        tile=True, bsize=thresholds["bsize"], tile_overlap=thresholds["tile_overlap"],
        mask_threshold=thresholds["mask_threshold"],
        flow_threshold=thresholds["flow_threshold"],
        min_size=thresholds["min_size"], cluster=thresholds["cluster"],
        resample=thresholds["resample"], compute_masks=True, verbose=False)[:3]
    seconds = time.time() - started
    labels = np.asarray(masks, dtype=np.int32)
    return labels, {
        "inference_seconds": round(seconds, 1),
        "peak_gpu_gb": round(float(torch.cuda.max_memory_allocated()) / 1e9, 2),
        "n_predicted": int(labels.max()),
        "normalization": norm_stats,
    }


def infer_one_fold(held_out: str, checkpoint: Path, out_dir: Path, *,
                   policy: str, include_round2: bool, train_manifest: dict | None,
                   thresholds: dict, seed: int = 0) -> dict:
    import tifffile

    from _shared.eval_gt import build_eval_gt
    from _shared.predict_export import ModelProvenance, export_prediction
    from omnipose_lab.env import verify
    from precision_myotube.benchmark import benchmark_instances
    from precision_myotube.schema import InstanceSet

    # The training manifest is not optional: nclasses, data_hash and the held-out
    # leakage assertion all derive from it. A missing manifest used to fall back to
    # `{}`, which made the leakage guard vacuously true — a guard that silently
    # disarms is worse than none.
    if not train_manifest:
        raise ValueError(
            "train_manifest is required: refusing to score a checkpoint without "
            "training provenance (nclasses, dataset_sha256, train_wells)")
    train_wells = train_manifest["train_wells"]
    if not train_wells:
        raise ValueError("train_manifest lists no training wells")

    environment = verify()
    manifest_path = ROOT / BOOTSTRAP
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Same evaluation ground truth the classical floor was scored on -- reviewed
    # `complete` minus the two binding exclusions -- built with the *same* function
    # so T03 compares candidates on byte-identical GT rather than on two
    # independent re-derivations that happen to agree.
    gt = build_eval_gt(manifest, held_out, out_dir / "eval_gt")

    # Inference deliberately uses the raw field, never the painted training copy.
    image = tifffile.imread(
        ROOT / "PrecisionMyotube/annotation_work/bootstrap_v1" / held_out / "image_fiber.tif")
    model = load_model(checkpoint, nclasses=train_manifest["config"]["nclasses"])
    labels, debug = predict_field(model, image, thresholds)
    print(f"  {held_out}: {debug['n_predicted']} instances in "
          f"{debug['inference_seconds']}s, peak GPU {debug['peak_gpu_gb']} GB")

    tag = f"{MODEL_VERSION}-fold-{held_out}-{policy}" + ("-r2" if include_round2 else "")
    # Initialisation is part of what the candidate IS (training plan §5(f)); the
    # sealed manifest must say so. This string previously hardcoded "trained from
    # scratch" regardless of the actual initialisation.
    init_model = train_manifest.get("init_model")
    provenance = ModelProvenance(
        model=MODEL_NAME, version=tag,
        architecture=f"omnipose (cellpose_omni {environment['cellpose_omni']}) "
                     f"nchan=1 dim=2 nclasses={train_manifest['config']['nclasses']}, "
                     f"rescale=False, "
                     + (f"fine-tuned from {init_model}" if init_model
                        else "trained from scratch"),
        checkpoint_hash=sha256_file(checkpoint),
        environment_hash=environment["environment_hash"],
        data_hash=train_manifest["dataset_sha256"],
        seed=seed,
        thresholds={**thresholds, "ignore_policy": policy,
                    "include_round2": include_round2,
                    "trained_on_wells": train_wells,
                    "init_model": init_model,
                    "init_model_sha256": train_manifest.get("init_model_sha256")},
        channels="desmin_only", used_prompts=False)

    exported = export_prediction(
        out_dir / "predictions", gt["image_id"], gt["image_shape"], provenance,
        masks=iter_masks_from_labels(labels),
        # False so the exporter consumes the generator exactly once.
        write_convenience_tiff=False, status="complete")

    metrics = benchmark_instances(gt["path"], exported["instances"])

    # Leakage / authority guards, same as the classical candidate.
    exported_set = InstanceSet.load(exported["instances"])
    assert exported_set.image_id == gt["image_id"], "prediction image_id mismatch"
    assert all(not r.reviewed for r in exported_set.instances), "predictions must be unreviewed"
    assert held_out not in train_wells, "held-out well was in the training set"

    return {
        "held_out_well": held_out, "ignore_policy": policy,
        "include_round2": include_round2,
        "checkpoint": str(checkpoint), "checkpoint_sha256": provenance.checkpoint_hash,
        "thresholds": thresholds,
        "inference": debug,
        "metrics": {k: metrics[k] for k in (
            "n_gt", "n_pred", "tp", "precision", "recall", "f1",
            "precision_weighted_score", "mean_matched_iou", "false_split_count",
            "false_split_rate", "over_merge_count", "over_merge_rate",
            "length_mdape", "width_mdape", "automatic_coverage")},
        "prediction": {
            "instances": str(Path(exported["instances"]).relative_to(ROOT)),
            "instances_sha256": sha256_file(exported["instances"]),
            "manifest": str(Path(exported["manifest"]).relative_to(ROOT)),
        },
        "eval_gt": {"path": str(gt["path"].relative_to(ROOT)), "sha256": gt["sha256"],
                    "n_gt": gt["n_gt"], "excluded_ids": gt["excluded_ids"]},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--held-out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy", default="paint_out")
    parser.add_argument("--include-round2", action="store_true")
    parser.add_argument("--out", default="model_labs/omnipose_lab/_runs/v1")
    parser.add_argument("--mask-threshold", type=float,
                        default=DEFAULT_THRESHOLDS["mask_threshold"])
    parser.add_argument("--flow-threshold", type=float,
                        default=DEFAULT_THRESHOLDS["flow_threshold"])
    args = parser.parse_args(argv)

    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    checkpoint = Path(args.checkpoint)
    train_manifest_path = checkpoint.parent.parent / "train_manifest.json"
    if not train_manifest_path.is_file():
        raise SystemExit(
            f"FAIL: {train_manifest_path} not found. Scoring requires the training "
            "manifest written by train_fold.py next to the checkpoint; without it "
            "the leakage guard and data provenance cannot be established.")
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))

    thresholds = {**DEFAULT_THRESHOLDS, "mask_threshold": args.mask_threshold,
                  "flow_threshold": args.flow_threshold}
    record = infer_one_fold(args.held_out, checkpoint, out_dir, policy=args.policy,
                            include_round2=args.include_round2,
                            train_manifest=train_manifest, thresholds=thresholds)
    print(json.dumps(record["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
