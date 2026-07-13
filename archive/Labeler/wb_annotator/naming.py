from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from wb_annotator.scanner import preserved_extension
from wb_annotator.schema import ExperimentMetadata, FileAnnotation

WINDOWS_INVALID_CHARS = '<>:"/\\|?*'
RESERVED_WINDOWS_NAMES = {
    "CON",
    "CONIN$",
    "CONOUT$",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(0, 10)),
    *(f"LPT{i}" for i in range(0, 10)),
}

GREEK_TRANSLITERATION = {
    "α": "alpha",
    "Α": "Alpha",
    "β": "beta",
    "Β": "Beta",
    "γ": "gamma",
    "Γ": "Gamma",
    "δ": "delta",
    "Δ": "Delta",
    "κ": "kappa",
    "Κ": "Kappa",
    "λ": "lambda",
    "Λ": "Lambda",
    "μ": "u",
    "µ": "u",
    "Μ": "Mu",
    "π": "pi",
    "Π": "Pi",
    "τ": "tau",
    "Τ": "Tau",
    "φ": "phi",
    "Φ": "Phi",
}


def normalize_date(value: str) -> str:
    text = value.strip()
    compact = re.sub(r"[-_./\s]", "", text)
    if re.fullmatch(r"\d{8}", compact):
        return compact
    return slugify(text, default="DATE")


def slugify(value: str, default: str = "NA", max_length: int = 60) -> str:
    text = str(value or "").strip()
    for source, replacement in GREEK_TRANSLITERATION.items():
        text = text.replace(source, replacement)

    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.translate(str.maketrans({char: "-" for char in WINDOWS_INVALID_CHARS}))
    ascii_text = re.sub(r"[\s_]+", "-", ascii_text)
    ascii_text = re.sub(r"[^A-Za-z0-9.-]+", "-", ascii_text)
    ascii_text = re.sub(r"-{2,}", "-", ascii_text)
    ascii_text = ascii_text.strip(" .-")

    if not ascii_text:
        ascii_text = default

    if ascii_text.upper() in RESERVED_WINDOWS_NAMES:
        ascii_text = f"{ascii_text}-value"

    if len(ascii_text) > max_length:
        ascii_text = ascii_text[:max_length].rstrip(" .-")

    return ascii_text or default


def treatment_set(metadata: ExperimentMetadata) -> str:
    parts = [metadata.treatment_name, metadata.dose_series, metadata.treatment_time]
    return slugify("-".join(part for part in parts if str(part).strip()), default="NT", max_length=80)


def canonical_filename(
    metadata: ExperimentMetadata,
    file_annotation: FileAnnotation,
    image_index: int,
) -> str:
    source_extension = preserved_extension(file_annotation.original_name)
    protein_component = protein_filename_component(metadata, file_annotation)
    components = [
        slugify(metadata.cell_line, default="CELL"),
        slugify(metadata.modification, default="MOD"),
        treatment_filename_component(metadata),
        concentration_filename_component(metadata),
        protein_component,
        repeat_number_component(file_annotation, image_index),
        normalize_date(metadata.date),
    ]
    stem = "_".join(components)
    if len(stem) > 180:
        stem = stem[:180].rstrip(" ._-")
    return f"{stem}{source_extension}"


def treatment_filename_component(metadata: ExperimentMetadata) -> str:
    treatment = metadata.treatment_name.strip()
    if not treatment or treatment.casefold() in {"nt", "none", "no treatment", "not treated", "untreated"}:
        return "NoTreatment"
    return slugify(treatment, default="NoTreatment", max_length=60)


def concentration_filename_component(metadata: ExperimentMetadata) -> str:
    concentration = metadata.dose_series.strip()
    if not concentration:
        return "0"
    return slugify(concentration, default="0", max_length=80)


def repeat_number_component(file_annotation: FileAnnotation, image_index: int) -> str:
    blot_id = file_annotation.blot_id.strip()
    match = re.fullmatch(r"[Bb](\d+)", blot_id)
    if match:
        return f"R{int(match.group(1)):02d}"
    if blot_id:
        return slugify(blot_id, default=f"R{image_index:02d}", max_length=20)
    return f"R{image_index:02d}"


def protein_filename_component(metadata: ExperimentMetadata, file_annotation: FileAnnotation) -> str:
    role = file_annotation.protein_role.strip().upper() or "UNKNOWN"
    label = file_annotation.protein_label.strip()
    if label:
        prefix = role if role in {"LC", "TGT"} else "PROT"
        return slugify(f"{prefix}-{label}", default="PROT-Unlabeled", max_length=60)

    if role == "LC" and metadata.loading_control.strip():
        return slugify(f"LC-{metadata.loading_control}", default="LC-Unlabeled", max_length=60)
    if role == "TGT" and metadata.target_protein.strip():
        return slugify(f"TGT-{metadata.target_protein}", default="TGT-Unlabeled", max_length=60)
    return "PROT-Unlabeled"


def is_windows_safe_filename(filename: str) -> bool:
    path = Path(filename)
    name = path.name
    stem = path.stem
    if not name or name in {".", ".."}:
        return False
    if any(char in name for char in WINDOWS_INVALID_CHARS):
        return False
    if name.endswith((" ", ".")):
        return False
    if stem.upper() in RESERVED_WINDOWS_NAMES:
        return False
    return True
