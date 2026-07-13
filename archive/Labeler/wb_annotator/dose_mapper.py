from __future__ import annotations

import re

from wb_annotator.schema import CellLineBlock, ExperimentMetadata, LaneAnnotation


def parse_dose_series(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    normalized = raw.replace("，", ",").replace("；", ";").replace("、", ",")
    if any(separator in normalized for separator in [",", ";", "|"]):
        parts = [part.strip() for part in re.split(r"[,;|]+", normalized) if part.strip()]
        return _apply_trailing_unit(parts)

    return _parse_compact_series(normalized)


def build_lane_annotations_from_experiments(
    experiment_sets: dict[str, ExperimentMetadata],
    cell_line_blocks: list[CellLineBlock] | None = None,
    lanes_per_experiment: int | None = None,
) -> list[LaneAnnotation]:
    lanes: list[LaneAnnotation] = []
    blocks_by_experiment: dict[str, list[CellLineBlock]] = {}
    for block in cell_line_blocks or []:
        blocks_by_experiment.setdefault(block.experiment_key, []).append(block)

    for experiment_key, metadata in experiment_sets.items():
        doses = parse_dose_series(metadata.dose_series)
        if not doses:
            continue

        blocks = blocks_by_experiment.get(experiment_key, [])
        if blocks:
            lanes.extend(_lanes_for_blocks(experiment_key, metadata, doses, blocks))
            continue

        lane_count = lanes_per_experiment or len(doses)
        for index in range(lane_count):
            concentration = doses[index] if index < len(doses) else ""
            note_parts = [f"auto dose order {metadata.lane_direction}"]
            if index >= len(doses):
                note_parts.append("no dose value supplied")
            lanes.append(
                LaneAnnotation(
                    experiment_key=experiment_key,
                    lane_number=index + 1,
                    role="SMP",
                    condition=metadata.treatment_name,
                    concentration=concentration,
                    note="; ".join(note_parts),
                )
            )
    return lanes


def _lanes_for_blocks(
    experiment_key: str,
    metadata: ExperimentMetadata,
    doses: list[str],
    blocks: list[CellLineBlock],
) -> list[LaneAnnotation]:
    lanes: list[LaneAnnotation] = []
    next_lane_number = 1
    dose_index = 0
    for block in sorted(blocks, key=_block_sort_key):
        if block.lane_start is not None and block.lane_end is not None and block.lane_end >= block.lane_start:
            lane_numbers = list(range(block.lane_start, block.lane_end + 1))
        else:
            lane_numbers = list(range(next_lane_number, next_lane_number + len(doses)))

        next_lane_number = max(lane_numbers, default=next_lane_number - 1) + 1
        for lane_number in lane_numbers:
            concentration = doses[dose_index] if dose_index < len(doses) else ""
            note_parts = [
                f"auto dose order {metadata.lane_direction}",
                f"block {block.block_number}",
                block.cell_line,
                block.modification,
            ]
            if dose_index >= len(doses):
                note_parts.append("no dose value supplied")
            lanes.append(
                LaneAnnotation(
                    experiment_key=experiment_key,
                    lane_number=lane_number,
                    role="SMP",
                    condition=metadata.treatment_name,
                    concentration=concentration,
                    note="; ".join(part for part in note_parts if part).strip("; "),
                )
            )
            dose_index += 1
    return lanes


def _block_sort_key(block: CellLineBlock) -> tuple[int, int]:
    if block.lane_start is not None:
        return block.lane_start, block.block_number
    return 1_000_000 + block.block_number, block.block_number


def _parse_compact_series(text: str) -> list[str]:
    matches = list(
        re.finditer(
            r"(?P<number>\d+(?:\.\d+)?)(?:\s*(?P<unit>[A-Za-zµuμ/%]+(?:/[A-Za-zµuμ%]+)?))?",
            text,
        )
    )
    if not matches:
        return [text]

    unit = ""
    for match in reversed(matches):
        candidate = match.group("unit")
        if candidate:
            unit = _normalize_unit(candidate)
            break

    doses: list[str] = []
    for match in matches:
        number = match.group("number")
        local_unit = _normalize_unit(match.group("unit") or unit)
        doses.append(f"{number} {local_unit}".strip())

    return doses


def _apply_trailing_unit(parts: list[str]) -> list[str]:
    trailing_unit = ""
    for part in reversed(parts):
        match = re.search(r"\d+(?:\.\d+)?\s*([A-Za-zµuμ/%]+(?:/[A-Za-zµuμ%]+)?)\s*$", part)
        if match:
            trailing_unit = _normalize_unit(match.group(1))
            break

    doses: list[str] = []
    for part in parts:
        if re.search(r"[A-Za-zµuμ/%]", part):
            doses.append(_normalize_dose_text(part))
        elif trailing_unit:
            doses.append(f"{part} {trailing_unit}".strip())
        else:
            doses.append(part)
    return doses


def _normalize_dose_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().replace("μ", "u").replace("µ", "u"))


def _normalize_unit(unit: str) -> str:
    return unit.strip().replace("μ", "u").replace("µ", "u")
