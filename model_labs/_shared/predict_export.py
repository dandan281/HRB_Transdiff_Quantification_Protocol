"""Normalize model-lab predictions to the canonical InstanceSet contract.

This is the handoff to Codex's C02 adapter / common benchmark. Every prediction:

* is written to a per-model, per-version, per-image path (P0.2.4 naming), so no
  model can overwrite another's output;
* is marked ``reviewed=False`` with ``source=<model>`` (C02.4) so an automated
  prediction can never become authoritative through import;
* carries full provenance -- architecture, checkpoint hash, environment hash,
  thresholds, confidence (C02.5) -- so any result is reproducible;
* is emitted overlap-safe (a list of masks may share pixels for crossings) *and*
  as a convenience label TIFF + properties CSV consumable by ``import-labels``.

The convenience TIFF is explicitly non-authoritative: crossings collapse in a
flat raster. The InstanceSet JSON is the authoritative artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import hashlib
import json
from pathlib import Path

import numpy as np

from .schema_bridge import InstanceRecord, InstanceSet, encode_sparse_positions


@dataclass
class ModelProvenance:
    model: str                         # e.g. "omnipose", "microsam"
    version: str                       # checkpoint/version label
    architecture: str = ""
    checkpoint_hash: str = ""
    environment_hash: str = ""
    data_hash: str = ""                # M01 shared-export hash
    seed: int | None = None
    thresholds: dict = field(default_factory=dict)
    channels: str = ""                 # e.g. "desmin_only" / "desmin_dapi"
    used_prompts: bool = False         # M05: official candidate must be False


def prediction_dir(out_root: str | Path, prov: ModelProvenance) -> Path:
    """Per-model/version prediction directory (unambiguous, non-colliding)."""
    return Path(out_root) / prov.model / prov.version


def _hash_bytes(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def _masks_to_records(masks: list[np.ndarray], image_shape: tuple[int, int],
                      prov: ModelProvenance, confidences,
                      status: str = "ambiguous") -> list[InstanceRecord]:
    records = []
    for i, mask in enumerate(masks, start=1):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != tuple(image_shape):
            raise ValueError("prediction mask shape does not match image_shape")
        rows, cols = np.nonzero(mask)
        if not rows.size:
            continue                    # skip empty predictions
        positions = rows.astype(np.int64) + cols.astype(np.int64) * image_shape[0]
        conf = None
        if confidences is not None:
            conf = float(confidences[i - 1])
        records.append(InstanceRecord(
            id=f"{prov.model}_{i:04d}", status=status, reviewed=False,
            source=prov.model, confidence=conf,
            rle=encode_sparse_positions(tuple(image_shape), positions),
        ))
    return records


def masks_from_label_image(labels: np.ndarray) -> list[np.ndarray]:
    """Split a mutually exclusive label image into per-instance boolean masks."""
    labels = np.asarray(labels)
    out = []
    for value in np.unique(labels):
        if value == 0:
            continue
        out.append(labels == value)
    return out


def export_prediction(out_root: str | Path, image_id: str,
                      image_shape: tuple[int, int], prov: ModelProvenance, *,
                      masks: list[np.ndarray] | None = None,
                      label_image: np.ndarray | None = None,
                      confidences=None,
                      write_convenience_tiff: bool = True,
                      status: str = "ambiguous") -> dict:
    """Write the authoritative InstanceSet JSON + provenance manifest for one field.

    Provide either ``masks`` (overlap-safe, may share pixels) or ``label_image``
    (mutually exclusive; converted to masks). Returns a paths dict.

    ``status`` is the QC status stamped on every emitted record. It defaults to
    ``"ambiguous"`` (the safest reading of a raw, unvetted prediction). A
    candidate that is asserting "this is one full-area, fully measurable object"
    must pass ``status="complete"``, because
    :func:`precision_myotube.benchmark.benchmark_instances` scores only
    predictions whose status is ``"complete"`` -- an exported candidate left at
    the default would otherwise be silently scored as zero detections. ``status``
    is orthogonal to authority: ``reviewed`` is forced ``False`` either way, so a
    prediction still never grants itself human authority.
    """
    if (masks is None) == (label_image is None):
        raise ValueError("provide exactly one of `masks` or `label_image`")
    if masks is None:
        masks = masks_from_label_image(label_image)

    records = _masks_to_records(masks, tuple(image_shape), prov, confidences, status)
    instance_set = InstanceSet(tuple(image_shape), image_id, records)
    instance_set.validate()
    # Enforce the unreviewed invariant regardless of how records were built.
    assert all(not r.reviewed for r in instance_set.instances), "predictions must be unreviewed"

    out_dir = prediction_dir(out_root, prov)
    out_dir.mkdir(parents=True, exist_ok=True)
    instances_path = out_dir / f"{image_id}.instances.json"
    instance_set.save(instances_path)

    manifest = {
        "image_id": image_id,
        "image_shape": list(image_shape),
        "n_instances": len(records),
        "provenance": asdict(prov),
        "authoritative_export": instances_path.name,
        "instances_sha256": _hash_bytes(instances_path.read_bytes()),
        "note": ("Predictions are unreviewed proposals. Connected-component or "
                 "instance count here is NOT an authoritative independent-myotube "
                 "count until expert review."),
    }
    manifest_path = out_dir / f"{image_id}.prediction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = {"instances": str(instances_path), "manifest": str(manifest_path),
              "n_instances": len(records)}

    if write_convenience_tiff and records:
        import tifffile
        raster = np.zeros(tuple(image_shape), dtype=np.int32)
        for value, mask in enumerate(masks, start=1):
            raster[np.asarray(mask, dtype=bool)] = value  # last-writer wins at overlaps
        tiff_path = out_dir / f"{image_id}.labels_convenience.tif"
        tifffile.imwrite(str(tiff_path), raster)
        props_path = out_dir / f"{image_id}.instance_properties.csv"
        lines = ["label,id,status,reviewed,source,notes"]
        for value, rec in enumerate(records, start=1):
            lines.append(f"{value},{rec.id},{rec.status},{rec.reviewed},{rec.source},")
        props_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result["convenience_tiff"] = str(tiff_path)
        result["properties_csv"] = str(props_path)
        result["tiff_warning"] = ("Flat TIFF is mutually exclusive; crossings "
                                  "collapse. Use the InstanceSet JSON as truth.")
    return result
