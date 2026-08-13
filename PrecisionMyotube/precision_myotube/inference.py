"""Framework-neutral conversion of model predictions to canonical InstanceSet records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from skimage.draw import polygon

from .io import sha256_file
from .schema import InstanceRecord, InstanceSet, encode_rle, from_label_image


def _load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    import tifffile
    return tifffile.imread(path)


def _provenance(*, architecture: str, checkpoint: str | Path | None,
                environment: str | Path | dict | None, thresholds: dict | None,
                input_format: str) -> dict:
    result: dict[str, Any] = {
        "kind": "model_prediction",
        "architecture": architecture,
        "input_format": input_format,
        "thresholds": thresholds or {},
        "review_policy": "unreviewed predictions are never authoritative",
    }
    if checkpoint:
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        result["checkpoint"] = str(checkpoint_path)
        result["checkpoint_sha256"] = sha256_file(checkpoint_path)
    if isinstance(environment, dict):
        result["environment"] = environment
    elif environment:
        environment_path = Path(environment)
        if environment_path.is_file():
            result["environment"] = json.loads(environment_path.read_text(encoding="utf-8"))
            result["environment_manifest_sha256"] = sha256_file(environment_path)
        else:
            result["environment"] = str(environment)
    return result


def adapt_label_image(labels_path: str | Path, *, image_id: str,
                      expected_shape: tuple[int, int] | None = None,
                      architecture: str, checkpoint: str | Path | None = None,
                      environment: str | Path | dict | None = None,
                      thresholds: dict | None = None,
                      confidence_path: str | Path | None = None) -> InstanceSet:
    """Convert mutually exclusive model labels without granting review authority."""
    labels = np.asarray(_load_array(labels_path))
    if labels.ndim != 2:
        raise ValueError("prediction label image must be 2-D")
    if expected_shape and tuple(labels.shape) != tuple(expected_shape):
        raise ValueError(f"prediction shape {labels.shape} does not match source {expected_shape}")
    result = from_label_image(labels, image_id, source=f"model:{architecture}",
                              reviewed=False, default_status="complete")
    if confidence_path:
        confidence = np.asarray(_load_array(confidence_path), dtype=float)
        if confidence.shape != labels.shape:
            raise ValueError("confidence map shape does not match prediction labels")
        for record, mask in result.masks():
            values = confidence[mask]
            record.confidence = float(np.mean(values)) if values.size else None
    result.provenance = _provenance(
        architecture=architecture, checkpoint=checkpoint, environment=environment,
        thresholds=thresholds, input_format="label_image")
    result.validate()
    return result


def _polygon_mask(shape: tuple[int, int], polygons: list) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for coordinates in polygons:
        values = np.asarray(coordinates, dtype=float)
        if values.ndim == 1:
            if values.size < 6 or values.size % 2:
                raise ValueError("polygon requires at least three x,y points")
            values = values.reshape(-1, 2)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
            raise ValueError("polygon must be an Nx2 array of x,y points")
        rows, cols = polygon(values[:, 1], values[:, 0], shape=shape)
        mask[rows, cols] = True
    return mask


def adapt_overlap_json(path: str | Path, *, image_id: str,
                       expected_shape: tuple[int, int] | None = None,
                       architecture: str, checkpoint: str | Path | None = None,
                       environment: str | Path | dict | None = None,
                       thresholds: dict | None = None) -> InstanceSet:
    """Convert polygon or uncompressed-COCO-RLE predictions while preserving overlaps."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    source_id = payload.get("image_id")
    if source_id != image_id:
        raise ValueError(f"prediction image_id {source_id!r} does not match {image_id!r}")
    shape = tuple(int(x) for x in payload.get("image_shape", ()))
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("prediction JSON requires positive image_shape [height,width]")
    if expected_shape and shape != tuple(expected_shape):
        raise ValueError(f"prediction shape {shape} does not match source {expected_shape}")
    records = []
    for index, item in enumerate(payload.get("instances", []), start=1):
        if "rle" in item:
            rle = item["rle"]
        elif "polygons" in item or "polygon" in item:
            polygons = item.get("polygons", [item["polygon"]])
            rle = encode_rle(_polygon_mask(shape, polygons))
        else:
            raise ValueError(f"instance {index}: expected rle, polygon, or polygons")
        records.append(InstanceRecord(
            id=str(item.get("id") or f"prediction_{index:04d}"),
            status="complete",
            rle=rle,
            source=f"model:{architecture}",
            confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
            reviewed=False,
            notes=str(item.get("notes", "")),
        ))
    result = InstanceSet(
        image_shape=shape, image_id=image_id, instances=records,
        provenance=_provenance(
            architecture=architecture, checkpoint=checkpoint, environment=environment,
            thresholds=thresholds, input_format="polygon_or_rle"))
    result.validate()
    return result


def adapt_prediction(*, input_path: str | Path, input_format: str, output_path: str | Path,
                     image_id: str, architecture: str,
                     expected_shape: tuple[int, int] | None = None,
                     checkpoint: str | Path | None = None,
                     environment: str | Path | dict | None = None,
                     thresholds: dict | None = None,
                     confidence_path: str | Path | None = None) -> InstanceSet:
    if input_format == "labels":
        result = adapt_label_image(
            input_path, image_id=image_id, expected_shape=expected_shape,
            architecture=architecture, checkpoint=checkpoint, environment=environment,
            thresholds=thresholds, confidence_path=confidence_path)
    elif input_format == "json":
        result = adapt_overlap_json(
            input_path, image_id=image_id, expected_shape=expected_shape,
            architecture=architecture, checkpoint=checkpoint, environment=environment,
            thresholds=thresholds)
    else:
        raise ValueError(f"unsupported prediction format: {input_format}")
    result.save(output_path)
    return result
