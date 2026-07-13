from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


FILE_KIND_DEFINITIONS = {
    "RAW": "Original unprocessed full blot image when target/loading-control identity is not separated",
    "CHEMI": "Chemiluminescence acquisition",
    "COLOR": "Colorimetric acquisition",
    "MERGE": "Merged/composite/multiplex image",
    "LAD": "Ladder/marker image",
    "TGT": "Legacy target protein image label",
    "LC": "Legacy loading-control image label",
    "OTHER": "User-defined image type",
}

PROTEIN_ROLE_DEFINITIONS = {
    "TGT": "Target/effector protein",
    "LC": "Loading control",
    "UNKNOWN": "Protein identity not detected",
}

LANE_DIRECTION_DEFINITIONS = {
    "LR": "Left to right",
    "TB": "Top to bottom",
}

LANE_ROLE_DEFINITIONS = {
    "NC": "Negative control",
    "PC": "Positive control",
    "VC": "Vehicle control",
    "SMP": "Experimental sample",
    "STD": "Standard/calibrator/dilution reference",
    "BLK": "Blank or empty lane",
    "LAD": "Ladder/marker lane",
}

REQUIRED_EXPERIMENT_FIELDS = (
    "date",
    "experiment_id",
    "cell_line",
    "modification",
    "treatment_name",
)


@dataclass(frozen=True)
class ExperimentMetadata:
    date: str
    experiment_id: str
    cell_line: str
    modification: str
    treatment_name: str
    dose_series: str
    treatment_time: str
    target_protein: str = ""
    loading_control: str = ""
    lane_direction: str = "LR"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CellLineBlock:
    experiment_key: str
    block_number: int
    cell_line: str
    modification: str
    lane_start: int | None = None
    lane_end: int | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FileAnnotation:
    original_name: str
    experiment_key: str = "E01"
    blot_id: str = "B01"
    file_kind: str = "RAW"
    protein_label: str = ""
    protein_role: str = "UNKNOWN"
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LaneAnnotation:
    lane_number: int
    role: str
    condition: str
    concentration: str = ""
    experiment_key: str = "E01"
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RenameRecord:
    source_path: str
    target_path: str
    original_name: str
    new_name: str
    status: str
    message: str
    size_bytes: int | None = None
    sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
