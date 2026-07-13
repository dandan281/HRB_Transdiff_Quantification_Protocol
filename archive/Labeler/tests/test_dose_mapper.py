from wb_annotator.dose_mapper import build_lane_annotations_from_experiments, parse_dose_series
from wb_annotator.schema import CellLineBlock, ExperimentMetadata


def metadata(**overrides: str) -> ExperimentMetadata:
    values = {
        "date": "20260622",
        "experiment_id": "E01",
        "cell_line": "CHO",
        "modification": "human BMPR2 wild-type",
        "treatment_name": "BMP4",
        "dose_series": "0-10-100nM",
        "treatment_time": "30min",
        "lane_direction": "LR",
    }
    values.update(overrides)
    return ExperimentMetadata(**values)


def test_parse_dose_series_compact_shared_unit() -> None:
    assert parse_dose_series("0-10-100nM") == ["0 nM", "10 nM", "100 nM"]
    assert parse_dose_series("0, 10, 100 nM") == ["0 nM", "10 nM", "100 nM"]
    assert parse_dose_series("0; 1; 5 uM") == ["0 uM", "1 uM", "5 uM"]


def test_build_lane_annotations_from_experiment_doses() -> None:
    lanes = build_lane_annotations_from_experiments({"E01": metadata()})

    assert [lane.lane_number for lane in lanes] == [1, 2, 3]
    assert [lane.concentration for lane in lanes] == ["0 nM", "10 nM", "100 nM"]
    assert [lane.condition for lane in lanes] == ["BMP4", "BMP4", "BMP4"]
    assert all(lane.experiment_key == "E01" for lane in lanes)


def test_cell_line_blocks_consume_dose_series_left_to_right_across_blocks() -> None:
    blocks = [
        CellLineBlock(
            experiment_key="E01",
            block_number=1,
            cell_line="CHO",
            modification="human BMPR2 wild-type",
            lane_start=1,
            lane_end=3,
        ),
        CellLineBlock(
            experiment_key="E01",
            block_number=2,
            cell_line="HEK293T",
            modification="empty vector",
            lane_start=4,
            lane_end=6,
        ),
    ]

    lanes = build_lane_annotations_from_experiments(
        {"E01": metadata(dose_series="0, 10, 100, 0.05, 0.1, 1 nM")},
        blocks,
    )

    assert [lane.lane_number for lane in lanes] == [1, 2, 3, 4, 5, 6]
    assert [lane.concentration for lane in lanes] == ["0 nM", "10 nM", "100 nM", "0.05 nM", "0.1 nM", "1 nM"]
    assert "CHO" in lanes[0].note
    assert "HEK293T" in lanes[3].note


def test_cell_line_block_extra_lanes_do_not_cycle_dose_series() -> None:
    blocks = [
        CellLineBlock(
            experiment_key="E01",
            block_number=1,
            cell_line="CHO",
            modification="human BMPR2 wild-type",
            lane_start=1,
            lane_end=5,
        )
    ]

    lanes = build_lane_annotations_from_experiments({"E01": metadata()}, blocks)

    assert [lane.lane_number for lane in lanes] == [1, 2, 3, 4, 5]
    assert [lane.concentration for lane in lanes] == ["0 nM", "10 nM", "100 nM", "", ""]
    assert "no dose value supplied" in lanes[3].note


def test_user_dose_series_keeps_position_after_cell_line_block_boundary() -> None:
    blocks = [
        CellLineBlock(
            experiment_key="E02",
            block_number=1,
            cell_line="CHO",
            modification="human B2H2 WT",
            lane_start=1,
            lane_end=2,
        ),
        CellLineBlock(
            experiment_key="E02",
            block_number=2,
            cell_line="CHO",
            modification="human B2mutH2",
            lane_start=3,
            lane_end=13,
        ),
    ]
    experiment = metadata(
        experiment_id="B2H2-E02",
        treatment_name="B2H2 Novokine",
        dose_series="0, 100, 0, 0.05, 0.1, 1, 2.5, 5, 10, 50, 100, 250, 500",
    )

    lanes = build_lane_annotations_from_experiments({"E02": experiment}, blocks)

    assert [lane.lane_number for lane in lanes[:6]] == [1, 2, 3, 4, 5, 6]
    assert [lane.concentration for lane in lanes[:6]] == ["0", "100", "0", "0.05", "0.1", "1"]
    assert "block 1" in lanes[1].note
    assert "block 2" in lanes[2].note
