"""Restartable, manifest-driven field and plate execution."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import analyze
from .io import load_metadata, prepare_run, sha256_file
from .report import create_reports
from .segmentation import create_component_proposals, create_nuclei, create_territory


STAGE_ORDER = ("prepare", "territory", "nuclei", "proposals", "analysis", "report")
STAGE_ARTIFACTS = {
    "prepare": ("metadata.json",),
    "territory": ("desmin_semantic_mask.npy", "myotube_territory.npy",
                  "territory_metadata.json"),
    "nuclei": ("nuclei_masks.npy",),
    "proposals": ("instance_proposals.json",),
    "analysis": ("analysis_summary.json", "qc_flags.json", "myotubes.csv",
                 "nuclei.csv", "field_summary.csv"),
    "report": ("qc_overlay.png", "review.html"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_batch_manifest(path: str | Path) -> dict:
    """Load and validate the frozen C01 batch-manifest interface."""
    manifest_path = Path(path).resolve()
    raw = _read_json(manifest_path)
    fields = raw.get("fields")
    if raw.get("schema_version") != "1.0" or not isinstance(fields, list) or not fields:
        raise ValueError("batch manifest requires schema_version '1.0' and a non-empty fields list")
    seen: set[str] = set()
    normalized = []
    for index, item in enumerate(fields):
        if not isinstance(item, dict):
            raise ValueError(f"field {index}: expected an object")
        missing = [name for name in ("image_id", "nd2", "plate", "well", "field", "output_dir")
                   if item.get(name) in (None, "")]
        if missing:
            raise ValueError(f"field {index}: missing required values {missing}")
        image_id = str(item["image_id"])
        if image_id in seen:
            raise ValueError(f"duplicate image_id: {image_id}")
        seen.add(image_id)
        record = dict(item)
        for name in ("nd2", "output_dir", "nuclei_masks", "instances"):
            resolved = _resolve(manifest_path.parent, record.get(name))
            if resolved is not None:
                record[name] = str(resolved)
        nd2 = Path(record["nd2"])
        if not nd2.is_file():
            raise FileNotFoundError(f"{image_id}: ND2 not found: {nd2}")
        for name in ("nuclei_masks", "instances"):
            if record.get(name) and not Path(record[name]).is_file():
                raise FileNotFoundError(f"{image_id}: {name} not found: {record[name]}")
        normalized.append(record)
    return {**raw, "manifest_path": str(manifest_path), "fields": normalized}


def _hashes(run: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: sha256_file(run / name) for name in names if (run / name).is_file()}


def _verified(run: Path, checkpoint: dict, stage: str) -> bool:
    saved = checkpoint.get("stages", {}).get(stage)
    required = STAGE_ARTIFACTS[stage]
    if not saved or saved.get("status") != "complete":
        return False
    current = _hashes(run, required)
    return len(current) == len(required) and current == saved.get("artifacts")


def _invalidate_from(checkpoint: dict, stage: str) -> None:
    start = STAGE_ORDER.index(stage)
    for name in STAGE_ORDER[start:]:
        checkpoint.setdefault("stages", {}).pop(name, None)


def _validate_prepared(field: dict, run: Path) -> dict:
    metadata = load_metadata(run)
    source = Path(field["nd2"])
    if metadata.get("image_id") != field["image_id"]:
        raise ValueError(
            f"{field['image_id']}: manifest image_id does not match prepared source "
            f"{metadata.get('image_id')!r}"
        )
    if metadata.get("source_sha256") != sha256_file(source):
        raise ValueError(f"{field['image_id']}: source ND2 hash changed")
    expected_roles = {"fiber": field.get("fiber_ch"), "dapi": field.get("dapi_ch")}
    for role, expected in expected_roles.items():
        if expected is not None and metadata["channels"][role] != int(expected):
            raise ValueError(f"{field['image_id']}: {role} channel-role mismatch")
    if not float(metadata.get("pixel_um", 0)) > 0:
        raise ValueError(f"{field['image_id']}: invalid pixel size")
    return metadata


def _load_external_labels(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    import tifffile
    return tifffile.imread(path)


def _validate_mask_shape(field: dict, run: Path, path: Path, kind: str) -> np.ndarray:
    labels = np.asarray(_load_external_labels(path))
    expected = tuple(load_metadata(run)["image_shape"])
    if labels.shape != expected:
        raise ValueError(
            f"{field['image_id']}: {kind} shape {labels.shape} does not match source {expected}"
        )
    return labels


def _execute_stage(stage: str, field: dict, run: Path) -> str:
    if stage == "prepare":
        prepare_run(field["nd2"], run, force_fiber=field.get("fiber_ch"),
                    force_dapi=field.get("dapi_ch"))
        _validate_prepared(field, run)
    elif stage == "territory":
        create_territory(run)
    elif stage == "nuclei":
        if field.get("nuclei_masks"):
            masks = _validate_mask_shape(field, run, Path(field["nuclei_masks"]), "nucleus mask")
            np.save(run / "nuclei_masks.npy", masks.astype(np.int32, copy=False))
        else:
            create_nuclei(run, field.get("cellpose_model"))
    elif stage == "proposals":
        create_component_proposals(run)
    elif stage == "analysis":
        if not field.get("instances"):
            return "review_required"
        result = analyze(run, field["instances"])
        # analyze performs shape checks before writing its canonical CSV outputs.
        field["_analysis"] = result
    elif stage == "report":
        if not field.get("instances"):
            return "review_required"
        result = field.pop("_analysis", None)
        if result is None:
            result = analyze(run, field["instances"])
        create_reports(run, result)
    return "complete"


def run_field(field: dict, *, resume: bool = False) -> dict:
    run = Path(field["output_dir"])
    run.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run / "batch_checkpoint.json"
    checkpoint = _read_json(checkpoint_path) if checkpoint_path.exists() else {
        "schema_version": "1.0", "image_id": field["image_id"], "stages": {}}
    if checkpoint.get("image_id") != field["image_id"]:
        raise ValueError(f"{run}: checkpoint belongs to another image")

    first_invalid = next((stage for stage in STAGE_ORDER if not _verified(run, checkpoint, stage)), None)
    if resume and first_invalid:
        _invalidate_from(checkpoint, first_invalid)
    elif not resume and checkpoint.get("stages"):
        raise FileExistsError(f"{run}: batch state exists; pass --resume")

    try:
        for stage in STAGE_ORDER:
            if resume and _verified(run, checkpoint, stage):
                if stage == "prepare":
                    _validate_prepared(field, run)
                continue
            status = _execute_stage(stage, field, run)
            if status == "review_required":
                checkpoint["status"] = status
                checkpoint["updated_utc"] = _utc_now()
                _write_json(checkpoint_path, checkpoint)
                return _field_result(field, status, stage="analysis")
            artifacts = _hashes(run, STAGE_ARTIFACTS[stage])
            if len(artifacts) != len(STAGE_ARTIFACTS[stage]):
                raise RuntimeError(f"{field['image_id']}: stage {stage} did not create all artifacts")
            checkpoint.setdefault("stages", {})[stage] = {
                "status": "complete", "completed_utc": _utc_now(), "artifacts": artifacts}
            checkpoint["status"] = "running"
            checkpoint["updated_utc"] = _utc_now()
            _write_json(checkpoint_path, checkpoint)
        checkpoint["status"] = "success"
        checkpoint["updated_utc"] = _utc_now()
        _write_json(checkpoint_path, checkpoint)
        return _field_result(field, "success")
    except Exception as exc:
        checkpoint["status"] = "failed"
        checkpoint["failed_stage"] = next(
            (stage for stage in STAGE_ORDER if not _verified(run, checkpoint, stage)), "unknown")
        checkpoint["error"] = f"{type(exc).__name__}: {exc}"
        checkpoint["updated_utc"] = _utc_now()
        _write_json(checkpoint_path, checkpoint)
        return _field_result(field, "failed", checkpoint["failed_stage"], checkpoint["error"])


def _field_result(field: dict, status: str, stage: str = "", error: str = "") -> dict:
    release_mode = "authoritative_reviewed_instances" if status == "success" else (
        "field_metrics_pending_instance_review" if status == "review_required" else "none")
    return {"image_id": field["image_id"], "plate": field["plate"],
            "well": field["well"], "field": field["field"],
            "output_dir": field["output_dir"], "status": status,
            "stage": stage, "release_mode": release_mode, "error": error}


def run_batch(manifest_path: str | Path, *, resume: bool = False,
              summary_dir: str | Path | None = None) -> dict:
    manifest = load_batch_manifest(manifest_path)
    results = [run_field(dict(field), resume=resume) for field in manifest["fields"]]
    destination = Path(summary_dir).resolve() if summary_dir else Path(manifest["manifest_path"]).parent
    destination.mkdir(parents=True, exist_ok=True)
    counts = {status: sum(row["status"] == status for row in results)
              for status in ("success", "failed", "review_required")}
    summary = {"schema_version": "1.0", "created_utc": _utc_now(),
               "manifest": manifest["manifest_path"], "counts": counts, "fields": results}
    _write_json(destination / "batch_summary.json", summary)
    columns = ("image_id", "plate", "well", "field", "output_dir", "status",
               "stage", "release_mode", "error")
    with (destination / "batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    return summary
