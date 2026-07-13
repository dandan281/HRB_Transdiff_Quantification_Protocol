from __future__ import annotations

import re
from pathlib import Path

from wb_annotator.protein_db import infer_protein_from_text
from wb_annotator.scanner import preserved_extension
from wb_annotator.schema import FileAnnotation


def natural_key(text: str) -> list[int | str]:
    parts = re.split(r"(\d+)", text.casefold())
    return [int(part) if part.isdigit() else part for part in parts]


def stem_without_supported_extension(filename: str) -> str:
    extension = preserved_extension(filename)
    if extension:
        return filename[: -len(extension)]
    return Path(filename).stem


def acquisition_label(filename: str) -> str:
    stem = stem_without_supported_extension(filename).strip()
    match = re.search(r"\(([^()]*)\)\s*$", stem)
    return match.group(1).strip() if match else ""


def blot_group_key(filename: str) -> str:
    stem = stem_without_supported_extension(filename).strip()
    stem = re.sub(r"\s*\([^()]*\)\s*$", "", stem).strip()
    return stem or stem_without_supported_extension(filename)


def experiment_group_key(filename: str) -> str:
    group = blot_group_key(filename)
    group = re.sub(r"[_\-\s]+[0-9]+[_\-\s]+[A-Za-z][A-Za-z0-9./-]*\s*$", "", group).strip()
    group = re.sub(r"[_\-\s]+[0-9]+\s*$", "", group).strip()
    return group or blot_group_key(filename)


def infer_file_kind(filename: str) -> str:
    label = acquisition_label(filename).casefold()
    lower_name = filename.casefold()
    combined = f"{label} {lower_name}"

    if "composite" in combined or "merged" in combined or "merge" in combined:
        return "MERGE"
    if "chemiluminescence" in combined or "chemi" in combined:
        return "CHEMI"
    if "colorimetric" in combined or "colourimetric" in combined or "color" in label:
        return "COLOR"
    if "ladder" in combined or "marker" in combined:
        return "LAD"
    if any(token in combined for token in ("gapdh", "actin", "tubulin", "loading", "lc")):
        return "LC"
    if any(token in combined for token in ("target", "protein", "p-", "phospho")):
        return "TGT"
    return "RAW"


def infer_note(filename: str) -> str:
    notes: list[str] = []
    label = acquisition_label(filename)
    if label:
        notes.append(label)

    lower_name = filename.casefold()
    if lower_name.endswith((".raw16.tif", ".raw16tif")):
        notes.append("raw16 intensity")
    elif lower_name.endswith((".jpg", ".jpeg")):
        notes.append("jpeg preview")
    elif lower_name.endswith((".tif", ".tiff")):
        notes.append("tif export")

    return "; ".join(notes)


def auto_annotate_files(paths: list[Path]) -> list[FileAnnotation]:
    sorted_paths = sorted(paths, key=lambda path: natural_key(path.name))
    experiment_ids: dict[str, str] = {}
    group_ids: dict[str, str] = {}
    annotations: list[FileAnnotation] = []

    for path in sorted_paths:
        experiment_key = experiment_group_key(path.name)
        if experiment_key not in experiment_ids:
            experiment_ids[experiment_key] = f"E{len(experiment_ids) + 1:02d}"
        group_key = blot_group_key(path.name)
        if group_key not in group_ids:
            group_ids[group_key] = f"B{len(group_ids) + 1:02d}"
        protein_hit = infer_protein_from_text(path.name)
        note = infer_note(path.name)
        if protein_hit:
            note = f"{note}; matched protein alias '{protein_hit.matched_alias}'" if note else (
                f"matched protein alias '{protein_hit.matched_alias}'"
            )
        annotations.append(
            FileAnnotation(
                original_name=path.name,
                experiment_key=experiment_ids[experiment_key],
                blot_id=group_ids[group_key],
                file_kind=infer_file_kind(path.name),
                protein_label=protein_hit.label if protein_hit else "",
                protein_role=protein_hit.role if protein_hit else "UNKNOWN",
                note=note,
            )
        )
    return annotations
