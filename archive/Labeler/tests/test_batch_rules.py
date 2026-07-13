import pytest

from wb_annotator.batch_rules import (
    apply_batch_rules,
    parse_batch_rule,
    parse_batch_rules,
    parse_cell_line_block_rule,
    parse_mixed_batch_commands,
    parse_wbscript_rule,
)
from wb_annotator.schema import FileAnnotation


def annotations() -> list[FileAnnotation]:
    return [
        FileAnnotation(original_name="a.tif", experiment_key="E01", blot_id="B06", file_kind="CHEMI"),
        FileAnnotation(original_name="b.tif", experiment_key="E01", blot_id="B07", file_kind="CHEMI"),
        FileAnnotation(original_name="c_h3.tif", experiment_key="E01", blot_id="B12", file_kind="CHEMI"),
        FileAnnotation(original_name="d.tif", experiment_key="E01", blot_id="B13", file_kind="CHEMI"),
    ]


def test_parse_short_blot_range_to_experiment_assignment() -> None:
    rule = parse_batch_rule("B07-B12 => E02")

    assert rule.field == "experiment_key"
    assert rule.value == "E02"
    assert rule.blot_start == 7
    assert rule.blot_end == 12


def test_parse_natural_language_blot_range() -> None:
    rule = parse_batch_rule("block 07 to block 012 are all E02")

    assert rule.field == "experiment_key"
    assert rule.value == "E02"
    assert rule.blot_start == 7
    assert rule.blot_end == 12


def test_apply_experiment_range_rule() -> None:
    updated, count = apply_batch_rules(annotations(), parse_batch_rules("B07-B12 => E02"))

    assert count == 2
    assert [item.experiment_key for item in updated] == ["E01", "E02", "E02", "E01"]


def test_apply_name_contains_protein_rule() -> None:
    updated, count = apply_batch_rules(annotations(), parse_batch_rules("set protein H3 LC where name contains h3"))

    assert count == 1
    assert updated[2].protein_label == "H3"
    assert updated[2].protein_role == "LC"


def test_rejects_unknown_file_kind() -> None:
    with pytest.raises(ValueError, match="Unknown file kind"):
        parse_batch_rule("all => kind banana")


def test_parse_cell_line_block_natural_language() -> None:
    rule = parse_cell_line_block_rule("E01 starts at line 1 and ends at line 2.")

    assert rule.experiment_key == "E01"
    assert rule.block_number == 1
    assert rule.lane_start == 1
    assert rule.lane_end == 2


def test_parse_cell_line_block_with_block_number_and_e0_typo() -> None:
    rule = parse_cell_line_block_rule("E0 block 2 starts at line 3 and ends at line 13")

    assert rule.experiment_key == "E01"
    assert rule.block_number == 2
    assert rule.lane_start == 3
    assert rule.lane_end == 13


def test_parse_mixed_commands_routes_to_sections() -> None:
    parsed = parse_mixed_batch_commands(
        """
        B07-B12 => E02
        E01 starts at line 1 and ends at line 2
        """
    )

    assert len(parsed.file_rules) == 1
    assert len(parsed.cell_line_block_rules) == 1


def test_parse_heading_and_bulleted_cell_line_block_commands() -> None:
    parsed = parse_mixed_batch_commands(
        """
        For the cell line blocks:
        - E01 starts at line 1 and ends at line 2.
        - E0 block 2 starts at line 3 and ends at line 13
        """
    )

    assert len(parsed.file_rules) == 0
    assert [(rule.experiment_key, rule.block_number, rule.lane_start, rule.lane_end) for rule in parsed.cell_line_block_rules] == [
        ("E01", 1, 1, 2),
        ("E01", 2, 3, 13),
    ]


def test_parse_more_natural_from_to_sentence() -> None:
    rule = parse_cell_line_block_rule("For E02, block 3 from lane 14 to lane 20")

    assert rule.experiment_key == "E02"
    assert rule.block_number == 3
    assert rule.lane_start == 14
    assert rule.lane_end == 20


def test_parse_wbscript_file_exp_assignment() -> None:
    section, rule = parse_wbscript_rule("files[B07:B12].exp = E02")

    assert section == "files"
    assert rule.field == "experiment_key"
    assert rule.value == "E02"
    assert rule.blot_start == 7
    assert rule.blot_end == 12


def test_parse_wbscript_cell_block_lanes() -> None:
    section, rule = parse_wbscript_rule("blocks[E01,2].lanes = 3:13")

    assert section == "blocks"
    assert rule.experiment_key == "E01"
    assert rule.block_number == 2
    assert rule.lane_start == 3
    assert rule.lane_end == 13


def test_parse_wbscript_experiment_field() -> None:
    section, rule = parse_wbscript_rule('exp[E02].dose <- "0-10-100nM"')

    assert section == "exp"
    assert rule.experiment_key == "E02"
    assert rule.field == "dose_series"
    assert rule.value == "0-10-100nM"


def test_parse_mixed_wbscript_commands() -> None:
    parsed = parse_mixed_batch_commands(
        """
        files[B07:B12].exp = E02
        blocks[E02,1].lanes = 1:6
        exp[E02].direction = TB
        """
    )

    assert len(parsed.file_rules) == 1
    assert len(parsed.cell_line_block_rules) == 1
    assert len(parsed.experiment_rules) == 1
