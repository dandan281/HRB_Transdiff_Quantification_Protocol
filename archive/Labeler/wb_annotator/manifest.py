from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from wb_annotator.schema import CellLineBlock, ExperimentMetadata, FileAnnotation, LaneAnnotation, RenameRecord

METADATA_FILENAME = "wb_metadata.json"
RENAME_LOG_FILENAME = "wb_rename_log.csv"
LABEL_EXPORT_CSV_FILENAME = "wb_label_export.csv"
LABEL_EXPORT_JSON_FILENAME = "wb_label_export.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_metadata(
    folder: Path | str,
    metadata: ExperimentMetadata,
    files: list[FileAnnotation],
    lanes: list[LaneAnnotation],
    records: list[RenameRecord],
    experiment_sets: dict[str, ExperimentMetadata] | None = None,
    cell_line_blocks: list[CellLineBlock] | None = None,
) -> Path:
    destination = Path(folder) / METADATA_FILENAME
    payload = {
        "schema_version": "1.0",
        "generated_at": utc_now_iso(),
        "experiment": metadata.as_dict(),
        "experiment_sets": {
            key: value.as_dict() for key, value in (experiment_sets or {"E01": metadata}).items()
        },
        "cell_line_blocks": [item.as_dict() for item in (cell_line_blocks or [])],
        "files": [item.as_dict() for item in files],
        "lanes": [item.as_dict() for item in lanes],
        "rename_plan": [record.as_dict() for record in records],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def write_rename_log(folder: Path | str, records: list[RenameRecord]) -> Path:
    destination = Path(folder) / RENAME_LOG_FILENAME
    fieldnames = [
        "source_path",
        "target_path",
        "original_name",
        "new_name",
        "status",
        "message",
        "size_bytes",
        "sha256",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_dict())
    return destination


def write_label_export(
    folder: Path | str,
    metadata: ExperimentMetadata,
    files: list[FileAnnotation],
    lanes: list[LaneAnnotation],
    records: list[RenameRecord],
    experiment_sets: dict[str, ExperimentMetadata] | None = None,
    cell_line_blocks: list[CellLineBlock] | None = None,
) -> tuple[Path, Path]:
    root = Path(folder)
    csv_path = root / LABEL_EXPORT_CSV_FILENAME
    json_path = root / LABEL_EXPORT_JSON_FILENAME

    record_by_original = {record.original_name: record for record in records}
    fieldnames = [
        "current_image_file",
        "labeled_filename",
        "status",
        "message",
        "experiment_key",
        "blot_id",
        "file_kind",
        "protein_label",
        "protein_role",
        "note",
        "size_bytes",
        "sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in files:
            record = record_by_original.get(item.original_name)
            writer.writerow(
                {
                    "current_image_file": item.original_name,
                    "labeled_filename": record.new_name if record else "",
                    "status": record.status if record else "MISSING_PREVIEW",
                    "message": record.message if record else "No label preview generated for this file",
                    "experiment_key": item.experiment_key,
                    "blot_id": item.blot_id,
                    "file_kind": item.file_kind,
                    "protein_label": item.protein_label,
                    "protein_role": item.protein_role,
                    "note": item.note,
                    "size_bytes": record.size_bytes if record else None,
                    "sha256": record.sha256 if record else None,
                }
            )

    payload = {
        "schema_version": "1.0",
        "export_type": "label_map",
        "generated_at": utc_now_iso(),
        "description": "Maps current image filenames to the labeled filenames applied by LABEL IMAGES.",
        "experiment": metadata.as_dict(),
        "experiment_sets": {
            key: value.as_dict() for key, value in (experiment_sets or {"E01": metadata}).items()
        },
        "cell_line_blocks": [item.as_dict() for item in (cell_line_blocks or [])],
        "files": [item.as_dict() for item in files],
        "lanes": [item.as_dict() for item in lanes],
        "label_plan": [record.as_dict() for record in records],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path
