from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from wb_annotator.manifest import write_metadata, write_rename_log
from wb_annotator.naming import canonical_filename
from wb_annotator.schema import CellLineBlock, ExperimentMetadata, FileAnnotation, LaneAnnotation, RenameRecord
from wb_annotator.validators import (
    validate_experiment,
    validate_file_annotations,
    validate_lanes,
    validate_target_names,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rename_plan(
    folder: Path | str,
    metadata: ExperimentMetadata,
    files: list[FileAnnotation],
    lanes: list[LaneAnnotation] | None = None,
    experiment_sets: dict[str, ExperimentMetadata] | None = None,
) -> list[RenameRecord]:
    root = Path(folder)
    lanes = lanes or []
    experiment_sets = experiment_sets or {"E01": metadata}
    global_errors = (
        _validate_experiment_sets(experiment_sets)
        + validate_file_annotations(files)
        + validate_lanes(lanes)
    )

    proposed_names = [
        canonical_filename(_metadata_for_file(metadata, experiment_sets, item), item, index)
        for index, item in enumerate(files, start=1)
    ]
    target_name_errors = validate_target_names(proposed_names)
    duplicate_targets = {
        name for name, count in Counter(proposed_names).items() if count > 1
    }
    global_errors.extend(target_name_errors)

    records: list[RenameRecord] = []
    for index, item in enumerate(files, start=1):
        source_path = root / item.original_name
        new_name = proposed_names[index - 1]
        target_path = root / new_name
        row_errors = list(global_errors)
        if item.experiment_key not in experiment_sets:
            row_errors.append(f"{item.original_name}: unknown experiment set '{item.experiment_key}'")

        if not source_path.exists():
            row_errors.append(f"Source file does not exist: {item.original_name}")
        elif not source_path.is_file():
            row_errors.append(f"Source path is not a file: {item.original_name}")

        if new_name in duplicate_targets:
            row_errors.append(f"Proposed filename is duplicated: {new_name}")

        if target_path.exists() and source_path.resolve() != target_path.resolve():
            row_errors.append(f"Target already exists: {new_name}")

        status = "BLOCKED" if row_errors else "OK"
        message = "; ".join(dict.fromkeys(row_errors)) if row_errors else "Ready"
        size = source_path.stat().st_size if source_path.exists() and source_path.is_file() else None

        records.append(
            RenameRecord(
                source_path=str(source_path),
                target_path=str(target_path),
                original_name=item.original_name,
                new_name=new_name,
                status=status,
                message=message,
                size_bytes=size,
            )
        )
    return records


def apply_rename_plan(
    folder: Path | str,
    metadata: ExperimentMetadata,
    files: list[FileAnnotation],
    lanes: list[LaneAnnotation],
    records: list[RenameRecord],
    experiment_sets: dict[str, ExperimentMetadata] | None = None,
    cell_line_blocks: list[CellLineBlock] | None = None,
) -> list[RenameRecord]:
    applied: list[RenameRecord] = []
    for record in records:
        source_path = Path(record.source_path)
        target_path = Path(record.target_path)

        if record.status != "OK":
            applied.append(record)
            continue

        if not source_path.exists():
            applied.append(_with_status(record, "FAILED", "Source file disappeared before rename"))
            continue
        if target_path.exists() and source_path.resolve() != target_path.resolve():
            applied.append(_with_status(record, "FAILED", "Target file exists before rename"))
            continue

        digest = sha256_file(source_path)
        if source_path.resolve() == target_path.resolve():
            applied.append(_with_status(record, "UNCHANGED", "Source already has proposed name", digest))
            continue

        source_path.rename(target_path)
        applied.append(_with_status(record, "RENAMED", "Renamed successfully", digest))

    write_metadata(folder, metadata, files, lanes, applied, experiment_sets, cell_line_blocks)
    write_rename_log(folder, applied)
    return applied


def _metadata_for_file(
    default_metadata: ExperimentMetadata,
    experiment_sets: dict[str, ExperimentMetadata],
    file_annotation: FileAnnotation,
) -> ExperimentMetadata:
    return experiment_sets.get(file_annotation.experiment_key, default_metadata)


def _validate_experiment_sets(experiment_sets: dict[str, ExperimentMetadata]) -> list[str]:
    errors: list[str] = []
    if not experiment_sets:
        return ["At least one experiment set is required"]
    for key, metadata in experiment_sets.items():
        for error in validate_experiment(metadata):
            errors.append(f"Experiment {key}: {error}")
    return errors


def _with_status(
    record: RenameRecord,
    status: str,
    message: str,
    sha256: str | None = None,
) -> RenameRecord:
    return RenameRecord(
        source_path=record.source_path,
        target_path=record.target_path,
        original_name=record.original_name,
        new_name=record.new_name,
        status=status,
        message=message,
        size_bytes=record.size_bytes,
        sha256=sha256 or record.sha256,
    )
