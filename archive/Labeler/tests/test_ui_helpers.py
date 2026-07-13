import pytest

from wb_annotator.schema import FileAnnotation
from wb_annotator.ui import _image_label_summary, _parse_lane_remove_selector


def test_image_label_summary_uses_detected_annotation() -> None:
    annotation = FileAnnotation(
        original_name="b2 vs b2h2 pERK.raw16.tif",
        experiment_key="E02",
        blot_id="B07",
        file_kind="CHEMI",
        protein_label="pERK1/2",
        protein_role="TGT",
    )

    assert _image_label_summary(annotation) == "E02 / B07 / CHEMI / TGT-pERK1/2"


def test_parse_lane_remove_selector_lane_only() -> None:
    assert _parse_lane_remove_selector("3") == (None, 3)
    assert _parse_lane_remove_selector("lane 7") == (None, 7)


def test_parse_lane_remove_selector_experiment_and_lane() -> None:
    assert _parse_lane_remove_selector("E02:3") == ("E02", 3)
    assert _parse_lane_remove_selector("e02 lane 3") == ("E02", 3)


def test_parse_lane_remove_selector_rejects_unclear_input() -> None:
    with pytest.raises(ValueError):
        _parse_lane_remove_selector("third lane")
