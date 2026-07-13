from __future__ import annotations

from collections import Counter

from wb_annotator.naming import is_windows_safe_filename
from wb_annotator.schema import (
    FILE_KIND_DEFINITIONS,
    LANE_ROLE_DEFINITIONS,
    PROTEIN_ROLE_DEFINITIONS,
    REQUIRED_EXPERIMENT_FIELDS,
    ExperimentMetadata,
    FileAnnotation,
    LaneAnnotation,
)


def validate_experiment(metadata: ExperimentMetadata) -> list[str]:
    errors: list[str] = []
    values = metadata.as_dict()
    for field in REQUIRED_EXPERIMENT_FIELDS:
        if not str(values.get(field, "")).strip():
            errors.append(f"Missing required experiment field: {field}")
    return errors


def validate_file_annotations(files: list[FileAnnotation]) -> list[str]:
    errors: list[str] = []
    for index, item in enumerate(files, start=1):
        if not item.original_name.strip():
            errors.append(f"File row {index}: missing original filename")
        if not item.experiment_key.strip():
            errors.append(f"{item.original_name}: missing experiment set")
        if not item.blot_id.strip():
            errors.append(f"{item.original_name}: missing blot ID")
        kind = item.file_kind.strip().upper()
        if kind not in FILE_KIND_DEFINITIONS:
            allowed = ", ".join(FILE_KIND_DEFINITIONS)
            errors.append(f"{item.original_name}: invalid file kind '{item.file_kind}' (allowed: {allowed})")
        role = item.protein_role.strip().upper() or "UNKNOWN"
        if role not in PROTEIN_ROLE_DEFINITIONS:
            allowed = ", ".join(PROTEIN_ROLE_DEFINITIONS)
            errors.append(f"{item.original_name}: invalid protein role '{item.protein_role}' (allowed: {allowed})")
    return errors


def validate_lanes(lanes: list[LaneAnnotation]) -> list[str]:
    errors: list[str] = []
    lane_keys = [(lane.experiment_key, lane.lane_number) for lane in lanes]
    duplicates = [key for key, count in Counter(lane_keys).items() if count > 1]
    if duplicates:
        formatted = ", ".join(f"{experiment} lane {number}" for experiment, number in duplicates)
        errors.append(f"Duplicate lane numbers: {formatted}")

    for lane in lanes:
        if lane.lane_number <= 0:
            errors.append(f"Invalid lane number: {lane.lane_number}")
        if lane.role.strip().upper() not in LANE_ROLE_DEFINITIONS:
            allowed = ", ".join(LANE_ROLE_DEFINITIONS)
            errors.append(f"Lane {lane.lane_number}: invalid role '{lane.role}' (allowed: {allowed})")
    return errors


def validate_target_names(new_names: list[str]) -> list[str]:
    errors: list[str] = []
    counts = Counter(new_names)
    for name, count in counts.items():
        if count > 1:
            errors.append(f"Duplicate proposed filename: {name}")
        if not is_windows_safe_filename(name):
            errors.append(f"Proposed filename is not Windows-safe: {name}")
    return errors
