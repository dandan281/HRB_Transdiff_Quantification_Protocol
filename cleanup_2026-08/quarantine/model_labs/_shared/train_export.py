"""Export reviewed-complete masks + channels as model training data.

Writes the standard instance-segmentation training format: an image stack plus a
mutually-exclusive instance label map (only reviewed-complete instances), and an
overlap-ignore mask so a flat-label model is not penalised where two crossing
myotubes legitimately share pixels. This is what Omnipose/Cellpose-SAM consume.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema_bridge import InstanceSet


def export_training(instances_path, fiber: np.ndarray, out_dir,
                    dapi: np.ndarray | None = None) -> dict:
    import tifffile
    inst = InstanceSet.load(instances_path)
    H, W = inst.image_shape
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = np.zeros((H, W), dtype=np.int32)
    ignore = np.zeros((H, W), dtype=bool)
    n = 0
    for record, bbox, crop in inst.cropped_masks():
        if not (record.reviewed and record.status == "complete"):
            continue                      # only reviewed-complete are trainable targets
        n += 1
        r0, c0, r1, c1 = bbox
        region = label_map[r0:r1, c0:c1]
        ignore[r0:r1, c0:c1] |= (region > 0) & crop   # overlap with an earlier instance
        region[crop] = n

    tifffile.imwrite(out_dir / "image_fiber.tif", np.asarray(fiber))
    if dapi is not None:
        tifffile.imwrite(out_dir / "image_dapi.tif", np.asarray(dapi))
    tifffile.imwrite(out_dir / "labels.tif", label_map)
    tifffile.imwrite(out_dir / "ignore.tif", ignore.astype(np.uint8))
    manifest = {"image_id": inst.image_id, "image_shape": [H, W],
                "n_trainable_instances": n, "n_overlap_pixels": int(ignore.sum()),
                "channels": ["fiber"] + (["dapi"] if dapi is not None else []),
                "note": "labels.tif = reviewed-complete instances only; ignore.tif marks overlaps."}
    (out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
