"""Annotation-dataset audits and export of reviewed masks for model training."""
from __future__ import annotations

import json
from pathlib import Path
import csv

import numpy as np
from skimage.io import imsave

from .io import load_metadata, load_run_channel
from .schema import InstanceSet


def audit_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = {"complete": 0, "hard_cases": 0, "double_annotated": 0,
              "development_reviewed": 0, "development_double_annotated": 0}
    plates, splits = set(), set()
    errors = []
    for sample in manifest.get("samples", []):
        plates.add(sample["plate"]); splits.add(sample["split"])
        try:
            instances_path = Path(sample["instances"])
            if not instances_path.is_absolute():
                instances_path = manifest_path.parent / instances_path
            instance_set = InstanceSet.load(instances_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sample.get('image_id', '?')}: {exc}")
            continue
        counts["complete"] += sum(r.reviewed and r.status == "complete"
                                  for r in instance_set.instances)
        counts["hard_cases"] += sum(r.status in {"ambiguous", "occluded"}
                                    for r in instance_set.instances)
        reviewed_count = sum(r.reviewed for r in instance_set.instances)
        if sample.get("double_annotated"):
            counts["double_annotated"] += reviewed_count
        if sample.get("split") == "train":
            counts["development_reviewed"] += reviewed_count
            if sample.get("double_annotated"):
                counts["development_double_annotated"] += reviewed_count
    samples = manifest.get("samples", [])
    development_plates = plates - {"PLATE_26"}
    double_fraction = (counts["development_double_annotated"] /
                       counts["development_reviewed"]
                       if counts["development_reviewed"] else 0.0)
    requirements = {
        "complete_ge_1000": counts["complete"] >= 1000,
        "hard_cases_ge_250": counts["hard_cases"] >= 250,
        "development_plates_23_28_32": {"PLATE_23", "PLATE_28", "PLATE_32"}
                                         <= development_plates,
        "has_train_validation_test_splits": {"train", "validation", "test"} <= splits,
        "double_annotated_development_ge_20pct": double_fraction >= 0.20,
        "all_validation_test_double_annotated": all(
            s.get("double_annotated") for s in samples if s.get("split") in {"validation", "test"}
        ),
        "plate26_locked_test": (any(s.get("plate") == "PLATE_26" for s in samples) and
                                all(s.get("split") == "test" for s in samples
                                    if s.get("plate") == "PLATE_26")),
        "no_schema_errors": not errors,
    }
    return {"counts": counts, "plates": sorted(plates), "splits": sorted(splits),
            "development_double_annotation_fraction": double_fraction,
            "requirements": requirements, "ready_for_bakeoff": all(requirements.values()),
            "errors": errors}


def export_training_sample(run_dir: str | Path, instances_path: str | Path,
                           out_dir: str | Path, stem: str | None = None) -> dict:
    """Export one reviewed image/mask pair for Cellpose/Omnipose-style training.

    Ambiguous, occluded, unreviewed, and overlapping pixels are zeroed. This export is a
    convenience format; the authoritative source remains the overlap-safe instance JSON.
    """
    run, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    instance_set = InstanceSet.load(instances_path)
    image = np.asarray(load_run_channel(run, "fiber"))
    labels = np.zeros(instance_set.image_shape, dtype=np.uint16)
    overlap = np.zeros(instance_set.image_shape, dtype=bool)
    used = 0
    for record, bbox, mask in instance_set.cropped_masks():
        if not record.reviewed or record.status != "complete":
            continue
        used += 1
        r0, c0, r1, c1 = bbox
        label_crop = labels[r0:r1, c0:c1]
        overlap_crop = overlap[r0:r1, c0:c1]
        overlap_crop |= mask & (label_crop > 0)
        label_crop[mask & (label_crop == 0)] = used
    labels[overlap] = 0
    name = stem or instance_set.image_id
    imsave(out / f"{name}_img.tif", image, check_contrast=False)
    imsave(out / f"{name}_masks.tif", labels, check_contrast=False)
    imsave(out / f"{name}_ignore.tif", overlap.astype(np.uint8), check_contrast=False)
    result = {"image_id": instance_set.image_id, "instances_exported": used,
              "overlap_pixels_ignored": int(overlap.sum())}
    (out / f"{name}_export.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def export_annotation_package(run_dir: str | Path, out_dir: str | Path,
                              instances_path: str | Path | None = None) -> dict:
    """Export native-resolution TIFFs and optional starting labels for napari/µSAM.

    The returned label TIFF cannot encode overlapping objects; the authoritative JSON can. Any
    starting-label overlap is therefore placed in ``overlap_ignore.tif`` for explicit review.
    """
    run, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(run)
    fiber = np.asarray(load_run_channel(run, "fiber"))
    dapi = np.asarray(load_run_channel(run, "dapi"))
    territory_path = run / "myotube_territory.npy"
    territory = np.load(territory_path) if territory_path.exists() else np.zeros(fiber.shape, bool)
    imsave(out / "fiber_raw16.tif", fiber, check_contrast=False)
    imsave(out / "dapi_raw16.tif", dapi, check_contrast=False)
    imsave(out / "semantic_territory.tif", territory.astype(np.uint8), check_contrast=False)

    properties = []
    labels = np.zeros(fiber.shape, dtype=np.uint16)
    overlap = np.zeros(fiber.shape, dtype=bool)
    if instances_path:
        instance_set = InstanceSet.load(instances_path)
        for index, (record, bbox, mask) in enumerate(instance_set.cropped_masks(), start=1):
            r0, c0, r1, c1 = bbox
            label_crop = labels[r0:r1, c0:c1]
            overlap_crop = overlap[r0:r1, c0:c1]
            overlap_crop |= mask & (label_crop > 0)
            label_crop[mask & (label_crop == 0)] = index
            properties.append({"label": index, "id": record.id, "status": record.status,
                               "reviewed": record.reviewed, "source": record.source,
                               "notes": record.notes})
    labels[overlap] = 0
    imsave(out / "starting_labels.tif", labels, check_contrast=False)
    imsave(out / "overlap_ignore.tif", overlap.astype(np.uint8), check_contrast=False)
    with (out / "instance_properties.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["label", "id", "status", "reviewed", "source", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(properties)
    instructions = {
        "image_id": metadata["image_id"],
        "pixel_um": metadata["pixel_um"],
        "workflow": [
            "Open fiber_raw16.tif and dapi_raw16.tif in napari or micro-sam.",
            "Load starting_labels.tif as editable labels; proposals are not ground truth.",
            "Correct full myotube bodies using ANNOTATION_PROTOCOL.md.",
            "Record status/review state in instance_properties.csv.",
            "Export the corrected label TIFF and run import-labels; represent visible overlaps "
            "as separate masks in the authoritative instance JSON.",
        ],
        "warning": "A 2-D label TIFF cannot encode overlapping instances; inspect overlap_ignore.tif.",
    }
    (out / "README.json").write_text(json.dumps(instructions, indent=2), encoding="utf-8")
    return {"out_dir": out.as_posix(), "starting_instances": len(properties),
            "overlap_pixels_ignored": int(overlap.sum())}
