from __future__ import annotations

from dataclasses import dataclass
import re

from wb_annotator.schema import FILE_KIND_DEFINITIONS, PROTEIN_ROLE_DEFINITIONS, FileAnnotation


@dataclass(frozen=True)
class ParsedBatchCommands:
    file_rules: list[BatchRule]
    cell_line_block_rules: list[CellLineBlockRule]
    experiment_rules: list[ExperimentRule]


@dataclass(frozen=True)
class BatchRule:
    field: str
    value: str
    second_value: str = ""
    blot_start: int | None = None
    blot_end: int | None = None
    name_contains: str = ""
    all_rows: bool = False

    def matches(self, annotation: FileAnnotation) -> bool:
        if self.all_rows:
            return True
        if self.name_contains:
            return self.name_contains.casefold() in annotation.original_name.casefold()
        if self.blot_start is not None and self.blot_end is not None:
            blot_number = _parse_blot_number(annotation.blot_id)
            return blot_number is not None and self.blot_start <= blot_number <= self.blot_end
        return False


@dataclass(frozen=True)
class CellLineBlockRule:
    experiment_key: str
    block_number: int
    lane_start: int
    lane_end: int


@dataclass(frozen=True)
class ExperimentRule:
    experiment_key: str
    field: str
    value: str


def parse_mixed_batch_commands(text: str) -> ParsedBatchCommands:
    file_rules: list[BatchRule] = []
    cell_line_block_rules: list[CellLineBlockRule] = []
    experiment_rules: list[ExperimentRule] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _clean_natural_line(raw_line)
        if not line or line.startswith("#"):
            continue
        if _is_natural_heading(line):
            continue
        try:
            section, rule = parse_wbscript_rule(line)
            if section == "files":
                file_rules.append(rule)
            elif section == "blocks":
                cell_line_block_rules.append(rule)
            elif section == "exp":
                experiment_rules.append(rule)
            continue
        except ValueError:
            pass
        try:
            cell_line_block_rules.append(parse_cell_line_block_rule(line))
            continue
        except ValueError:
            pass

        try:
            file_rules.append(parse_batch_rule(line))
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: {exc}") from exc
    return ParsedBatchCommands(
        file_rules=file_rules,
        cell_line_block_rules=cell_line_block_rules,
        experiment_rules=experiment_rules,
    )


def parse_batch_rules(text: str) -> list[BatchRule]:
    rules: list[BatchRule] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rules.append(parse_batch_rule(line))
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: {exc}") from exc
    return rules


def parse_wbscript_rule(line: str) -> tuple[str, BatchRule | CellLineBlockRule | ExperimentRule]:
    normalized = _normalize_command(_clean_natural_line(line)).rstrip(".")
    match = re.fullmatch(
        r"(?P<section>files?|blocks?|cellblocks?|cell_line_blocks?|exps?|experiments?)"
        r"\[(?P<selector>.*?)\]\s*(?:\.|\$)\s*(?P<field>[A-Za-z_]+)\s*(?:<-|=)\s*(?P<value>.+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Not a WBScript command")

    section = match.group("section").casefold()
    selector = match.group("selector").strip()
    field = match.group("field").strip().casefold()
    value = _strip_quotes(match.group("value").strip())

    if section in {"file", "files"}:
        return "files", _parse_wbscript_file_rule(selector, field, value)
    if section in {"block", "blocks", "cellblock", "cellblocks", "cell_line_blocks"}:
        return "blocks", _parse_wbscript_block_rule(selector, field, value)
    if section in {"exp", "exps", "experiment", "experiments"}:
        return "exp", _parse_wbscript_experiment_rule(selector, field, value)
    raise ValueError(f"Unknown WBScript section '{section}'")


def _parse_wbscript_file_rule(selector: str, field: str, value: str) -> BatchRule:
    field_map = {
        "exp": "experiment_key",
        "experiment": "experiment_key",
        "experiment_key": "experiment_key",
        "kind": "file_kind",
        "file_kind": "file_kind",
        "data": "file_kind",
        "datatype": "file_kind",
        "data_type": "file_kind",
        "role": "protein_role",
        "protein_role": "protein_role",
        "protein": "protein",
    }
    if field not in field_map:
        raise ValueError(f"Unsupported files field '{field}'")

    canonical = field_map[field]
    if canonical == "experiment_key":
        assignment = _normalize_experiment_key(value)
    elif canonical == "file_kind":
        assignment = value.upper()
        if assignment not in FILE_KIND_DEFINITIONS:
            raise ValueError(f"Unknown file kind '{value}'")
    elif canonical == "protein_role":
        assignment = value.upper()
        if assignment not in PROTEIN_ROLE_DEFINITIONS:
            raise ValueError(f"Unknown protein role '{value}'")
    elif canonical == "protein":
        label, role = _split_protein_assignment(value)
        kwargs = _parse_wbscript_file_selector(selector)
        return BatchRule(field="protein", value=label, second_value=role, **kwargs)
    else:
        assignment = value

    kwargs = _parse_wbscript_file_selector(selector)
    return BatchRule(field=canonical, value=assignment, **kwargs)


def _parse_wbscript_file_selector(selector: str) -> dict[str, object]:
    cleaned = selector.strip()
    if cleaned.casefold() in {"all", "*"}:
        return {"all_rows": True}

    name_match = re.fullmatch(r"name\s*(?:~|contains)\s*(.+)", cleaned, flags=re.IGNORECASE)
    if name_match:
        return {"name_contains": _strip_quotes(name_match.group(1).strip())}

    return _parse_selector(cleaned)


def _parse_wbscript_block_rule(selector: str, field: str, value: str) -> CellLineBlockRule:
    if field not in {"lanes", "lines", "lane_range", "line_range"}:
        raise ValueError(f"Unsupported blocks field '{field}'")

    experiment_key, block_number = _parse_block_selector(selector)
    lane_start, lane_end = _parse_lane_range(value)
    return CellLineBlockRule(
        experiment_key=experiment_key,
        block_number=block_number,
        lane_start=lane_start,
        lane_end=lane_end,
    )


def _parse_wbscript_experiment_rule(selector: str, field: str, value: str) -> ExperimentRule:
    field_map = {
        "date": "date",
        "id": "experiment_id",
        "experiment_id": "experiment_id",
        "cell": "cell_line",
        "cell_line": "cell_line",
        "mod": "modification",
        "mods": "modification",
        "modification": "modification",
        "plasmid": "modification",
        "plasmids": "modification",
        "treatment": "treatment_name",
        "treatment_name": "treatment_name",
        "dose": "dose_series",
        "doses": "dose_series",
        "dose_series": "dose_series",
        "time": "treatment_time",
        "treatment_time": "treatment_time",
        "direction": "lane_direction",
        "lane_direction": "lane_direction",
    }
    if field not in field_map:
        raise ValueError(f"Unsupported exp field '{field}'")
    experiment_key = _normalize_experiment_key(selector)
    canonical = field_map[field]
    if canonical == "lane_direction":
        value = value.upper()
    return ExperimentRule(experiment_key=experiment_key, field=canonical, value=value)


def parse_cell_line_block_rule(line: str) -> CellLineBlockRule:
    normalized = _normalize_command(_clean_natural_line(line)).rstrip(".")

    full = re.fullmatch(
        r"(?P<exp>E\s*0*\d+)\s*"
        r"(?:(?:cell\s+line\s+)?block\s*(?P<block>\d+)\s*)?"
        r"(?:starts?|start)\s+(?:at\s+)?(?:lane|line)\s*(?P<start>\d+)\s*"
        r"(?:and\s+)?(?:ends?|end)\s+(?:at\s+)?(?:lane|line)\s*(?P<end>\d+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if full:
        return CellLineBlockRule(
            experiment_key=_normalize_experiment_key(full.group("exp")),
            block_number=int(full.group("block") or 1),
            lane_start=int(full.group("start")),
            lane_end=int(full.group("end")),
        )

    from_to = re.fullmatch(
        r"(?:for\s+)?(?P<exp>E\s*0*\d+)\s*,?\s*"
        r"(?:(?:cell\s+line\s+)?block\s*(?P<block>\d+)\s*)?"
        r"(?:from\s+)?(?:lane|line)\s*(?P<start>\d+)\s*"
        r"(?:-|to|through|thru)\s*(?:lane|line)?\s*(?P<end>\d+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if from_to:
        return CellLineBlockRule(
            experiment_key=_normalize_experiment_key(from_to.group("exp")),
            block_number=int(from_to.group("block") or 1),
            lane_start=int(from_to.group("start")),
            lane_end=int(from_to.group("end")),
        )

    compact = re.fullmatch(
        r"(?P<exp>E\s*0*\d+)\s*"
        r"(?:(?:cell\s+line\s+)?block\s*(?P<block>\d+)\s*)?"
        r"(?:lanes?|lines?)\s*(?P<start>\d+)\s*(?:-|to|through|thru)\s*(?P<end>\d+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if compact:
        return CellLineBlockRule(
            experiment_key=_normalize_experiment_key(compact.group("exp")),
            block_number=int(compact.group("block") or 1),
            lane_start=int(compact.group("start")),
            lane_end=int(compact.group("end")),
        )

    raise ValueError("Not a cell-line block command")


def parse_batch_rule(line: str) -> BatchRule:
    normalized = _normalize_command(line)

    strict = re.fullmatch(r"set\s+(.+?)\s+where\s+(.+)", normalized, flags=re.IGNORECASE)
    if strict:
        assignment = strict.group(1).strip()
        selector = strict.group(2).strip()
        return _build_rule(selector, assignment)

    arrow = re.fullmatch(r"(.+?)\s*(?:=>|=)\s*(.+)", normalized)
    if arrow:
        selector = arrow.group(1).strip()
        assignment = arrow.group(2).strip()
        return _build_rule(selector, assignment)

    natural = re.fullmatch(
        r"(?:block|blocks|blot|blots)\s+(.+?)\s+(?:are|is)\s+(?:all\s+)?(.+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if natural:
        selector = f"block {natural.group(1).strip()}"
        assignment = natural.group(2).strip()
        return _build_rule(selector, assignment)

    raise ValueError(
        "Unsupported command. Try 'B07-B12 => E02', "
        "'block 07 to block 12 are all E02', or 'set exp E02 where blot B07-B12'."
    )


def apply_batch_rules(annotations: list[FileAnnotation], rules: list[BatchRule]) -> tuple[list[FileAnnotation], int]:
    updated: list[FileAnnotation] = []
    change_count = 0
    for annotation in annotations:
        current = annotation
        for rule in rules:
            if not rule.matches(current):
                continue
            new_annotation = _apply_rule(current, rule)
            if new_annotation != current:
                change_count += 1
            current = new_annotation
        updated.append(current)
    return updated, change_count


def _build_rule(selector: str, assignment: str) -> BatchRule:
    field, value, second_value = _parse_assignment(assignment)
    kwargs = _parse_selector(selector)
    return BatchRule(field=field, value=value, second_value=second_value, **kwargs)


def _parse_assignment(text: str) -> tuple[str, str, str]:
    parts = text.strip().split()
    if not parts:
        raise ValueError("Missing assignment")

    first = parts[0].casefold()
    if _looks_like_experiment_key(parts[0]):
        return "experiment_key", _normalize_experiment_key(parts[0]), ""

    if first in {"exp", "experiment", "experiment_key", "set"}:
        if len(parts) < 2:
            raise ValueError("Experiment assignment needs a value such as E02")
        return "experiment_key", _normalize_experiment_key(parts[1]), ""

    if first in {"kind", "file_kind", "data", "datatype", "data_type"}:
        if len(parts) < 2:
            raise ValueError("Data type assignment needs a value such as CHEMI")
        value = parts[1].upper()
        if value not in FILE_KIND_DEFINITIONS:
            raise ValueError(f"Unknown file kind '{parts[1]}'")
        return "file_kind", value, ""

    if first in {"role", "protein_role"}:
        if len(parts) < 2:
            raise ValueError("Role assignment needs a value such as LC or TGT")
        value = parts[1].upper()
        if value not in PROTEIN_ROLE_DEFINITIONS:
            raise ValueError(f"Unknown protein role '{parts[1]}'")
        return "protein_role", value, ""

    if first == "protein":
        if len(parts) < 2:
            raise ValueError("Protein assignment needs a label such as H3")
        label = parts[1]
        role = parts[2].upper() if len(parts) >= 3 else ""
        if role and role not in PROTEIN_ROLE_DEFINITIONS:
            raise ValueError(f"Unknown protein role '{role}'")
        return "protein", label, role

    raise ValueError(f"Unsupported assignment '{text}'")


def _parse_selector(text: str) -> dict[str, object]:
    selector = text.strip()
    selector = re.sub(r"^(where\s+)?(blot|blots|block|blocks)\s+", "", selector, flags=re.IGNORECASE)
    selector = re.sub(r"\b(blot|blots|block|blocks)\s+", "", selector, flags=re.IGNORECASE)

    name_match = re.fullmatch(r"(?:name\s+)?contains\s+(.+)", selector, flags=re.IGNORECASE)
    if name_match:
        return {"name_contains": name_match.group(1).strip().strip("'\"")}

    if selector.casefold() in {"all", "everything", "files", "all files"}:
        return {"all_rows": True}

    range_match = re.fullmatch(
        r"B?\s*(\d+)\s*(?::|-|to|through|thru)\s*(?:B\s*)?(\d+)",
        selector,
        flags=re.IGNORECASE,
    )
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        if end < start:
            start, end = end, start
        return {"blot_start": start, "blot_end": end}

    single_match = re.fullmatch(r"B?\s*(\d+)", selector, flags=re.IGNORECASE)
    if single_match:
        number = int(single_match.group(1))
        return {"blot_start": number, "blot_end": number}

    raise ValueError(f"Unsupported selector '{text}'")


def _parse_block_selector(selector: str) -> tuple[str, int]:
    parts = [part.strip() for part in selector.split(",") if part.strip()]
    if not parts:
        raise ValueError("Block selector needs an experiment key, such as E01")
    experiment_key = _normalize_experiment_key(parts[0])
    block_number = 1
    if len(parts) >= 2:
        block_match = re.fullmatch(r"(?:block\s*)?(\d+)", parts[1], flags=re.IGNORECASE)
        if not block_match:
            raise ValueError(f"Invalid block selector '{selector}'")
        block_number = int(block_match.group(1))
    return experiment_key, block_number


def _parse_lane_range(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(?:lane|line)?\s*(\d+)\s*(?::|-|to|through|thru)\s*(?:lane|line)?\s*(\d+)", value, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid lane range '{value}'")
    start = int(match.group(1))
    end = int(match.group(2))
    if end < start:
        start, end = end, start
    return start, end


def _split_protein_assignment(value: str) -> tuple[str, str]:
    parts = [part.strip() for part in re.split(r"[, ]+", value) if part.strip()]
    if not parts:
        raise ValueError("Protein assignment needs a protein label")
    label = parts[0]
    role = parts[1].upper() if len(parts) >= 2 else ""
    if role and role not in PROTEIN_ROLE_DEFINITIONS:
        raise ValueError(f"Unknown protein role '{role}'")
    return label, role


def _apply_rule(annotation: FileAnnotation, rule: BatchRule) -> FileAnnotation:
    data = annotation.as_dict()
    if rule.field == "protein":
        data["protein_label"] = rule.value
        if rule.second_value:
            data["protein_role"] = rule.second_value
    else:
        data[rule.field] = rule.value
    return FileAnnotation(**data)


def _parse_blot_number(blot_id: str) -> int | None:
    match = re.fullmatch(r"B?\s*0*(\d+)", blot_id.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _looks_like_experiment_key(text: str) -> bool:
    return bool(re.fullmatch(r"E\s*0*\d+", text.strip(), flags=re.IGNORECASE))


def _normalize_experiment_key(text: str) -> str:
    match = re.fullmatch(r"E\s*0*(\d+)", text.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid experiment key '{text}'")
    number = int(match.group(1))
    if number == 0:
        number = 1
    return f"E{number:02d}"


def _normalize_command(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _strip_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def _clean_natural_line(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", cleaned)
    return cleaned.strip()


def _is_natural_heading(line: str) -> bool:
    lowered = line.strip().casefold()
    if lowered.endswith(":"):
        return True
    heading_phrases = {
        "for the cell line blocks",
        "cell line blocks",
        "cell-line blocks",
        "for detected image files",
        "detected image files",
        "for lane order metadata",
        "lane order metadata",
    }
    return lowered.rstrip(":") in heading_phrases
