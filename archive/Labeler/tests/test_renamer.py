import csv
import json
from pathlib import Path

from wb_annotator.manifest import write_label_export
from wb_annotator.renamer import apply_rename_plan, build_rename_plan
from wb_annotator.schema import CellLineBlock, ExperimentMetadata, FileAnnotation, LaneAnnotation


def metadata(**overrides: str) -> ExperimentMetadata:
    values = {
        "date": "20260622",
        "experiment_id": "E001",
        "cell_line": "HEK293T",
        "modification": "WT",
        "treatment_name": "EGF",
        "dose_series": "0-50 ngmL",
        "treatment_time": "30 min",
        "target_protein": "ERK1/2",
        "loading_control": "ACTB",
    }
    values.update(overrides)
    return ExperimentMetadata(**values)


def test_missing_required_metadata_blocks_rename(tmp_path: Path) -> None:
    (tmp_path / "blot.tif").write_bytes(b"image")

    records = build_rename_plan(
        tmp_path,
        metadata(cell_line=""),
        [FileAnnotation(original_name="blot.tif", blot_id="B01", file_kind="RAW")],
        [],
    )

    assert records[0].status == "BLOCKED"
    assert "Missing required experiment field: cell_line" in records[0].message
    assert (tmp_path / "blot.tif").exists()


def test_existing_target_blocks_rename(tmp_path: Path) -> None:
    (tmp_path / "blot.tif").write_bytes(b"image")
    target = "HEK293T_WT_EGF_0-50-ngmL_PROT-Unlabeled_R01_20260622.tif"
    (tmp_path / target).write_bytes(b"existing")

    records = build_rename_plan(
        tmp_path,
        metadata(),
        [FileAnnotation(original_name="blot.tif", blot_id="B01", file_kind="RAW")],
        [],
    )

    assert records[0].status == "BLOCKED"
    assert "Target already exists" in records[0].message


def test_apply_renames_files_and_writes_manifest_and_log(tmp_path: Path) -> None:
    (tmp_path / "target.tif").write_bytes(b"target-image")
    (tmp_path / "gapdh.jpg").write_bytes(b"loading-control")
    files = [
        FileAnnotation(original_name="target.tif", blot_id="B01", file_kind="TGT", note="target exposure"),
        FileAnnotation(original_name="gapdh.jpg", blot_id="B01", file_kind="LC", note="loading control"),
    ]
    lanes = [
        LaneAnnotation(lane_number=1, role="NC", condition="Vehicle", concentration="0", note="negative"),
        LaneAnnotation(lane_number=2, role="PC", condition="EGF", concentration="50 ngmL", note="positive"),
        LaneAnnotation(lane_number=3, role="SMP", condition="EGF", concentration="10 ngmL", note="sample"),
    ]

    records = build_rename_plan(tmp_path, metadata(), files, lanes)

    assert [record.status for record in records] == ["OK", "OK"]
    cell_line_blocks = [
        CellLineBlock(
            experiment_key="E01",
            block_number=1,
            cell_line="CHO",
            modification="human BMPR2 wild-type; human HER2 wild-type",
            lane_start=1,
            lane_end=3,
            note="single cell-line block",
        )
    ]
    applied = apply_rename_plan(tmp_path, metadata(), files, lanes, records, cell_line_blocks=cell_line_blocks)

    assert [record.status for record in applied] == ["RENAMED", "RENAMED"]
    assert not (tmp_path / "target.tif").exists()
    assert not (tmp_path / "gapdh.jpg").exists()
    assert (tmp_path / applied[0].new_name).exists()
    assert (tmp_path / applied[1].new_name).exists()

    metadata_payload = json.loads((tmp_path / "wb_metadata.json").read_text(encoding="utf-8"))
    assert metadata_payload["experiment"]["cell_line"] == "HEK293T"
    assert metadata_payload["lanes"][0]["role"] == "NC"
    assert metadata_payload["cell_line_blocks"][0]["cell_line"] == "CHO"
    assert metadata_payload["cell_line_blocks"][0]["lane_start"] == 1

    with (tmp_path / "wb_rename_log.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["status"] == "RENAMED"
    assert rows[0]["sha256"]


def test_invalid_lane_role_blocks_plan(tmp_path: Path) -> None:
    (tmp_path / "blot.tif").write_bytes(b"image")

    records = build_rename_plan(
        tmp_path,
        metadata(),
        [FileAnnotation(original_name="blot.tif", blot_id="B01", file_kind="RAW")],
        [LaneAnnotation(lane_number=1, role="BAD", condition="EGF")],
    )

    assert records[0].status == "BLOCKED"
    assert "invalid role" in records[0].message


def test_file_experiment_key_selects_experiment_metadata(tmp_path: Path) -> None:
    (tmp_path / "a.tif").write_bytes(b"a")
    (tmp_path / "b.tif").write_bytes(b"b")
    files = [
        FileAnnotation(original_name="a.tif", experiment_key="E01", blot_id="B01", file_kind="RAW"),
        FileAnnotation(original_name="b.tif", experiment_key="E02", blot_id="B02", file_kind="RAW"),
    ]
    experiment_sets = {
        "E01": metadata(experiment_id="B2H2-E01", treatment_name="DOX"),
        "E02": metadata(experiment_id="B2H2-E02", treatment_name="BMP4"),
    }

    records = build_rename_plan(tmp_path, experiment_sets["E01"], files, [], experiment_sets)

    assert records[0].status == "OK"
    assert records[1].status == "OK"
    assert records[0].new_name.startswith("HEK293T_WT_DOX_")
    assert "DOX" in records[0].new_name
    assert records[1].new_name.startswith("HEK293T_WT_BMP4_")
    assert "BMP4" in records[1].new_name


def test_write_label_export_maps_current_files_to_labeled_filenames(tmp_path: Path) -> None:
    (tmp_path / "target.tif").write_bytes(b"target-image")
    files = [
        FileAnnotation(
            original_name="target.tif",
            experiment_key="E01",
            blot_id="B01",
            file_kind="CHEMI",
            protein_label="pERK1/2",
            protein_role="TGT",
        )
    ]
    lanes = [LaneAnnotation(lane_number=1, role="SMP", condition="EGF", concentration="50 ngmL")]
    records = build_rename_plan(tmp_path, metadata(), files, lanes)

    csv_path, json_path = write_label_export(tmp_path, metadata(), files, lanes, records)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["current_image_file"] == "target.tif"
    assert rows[0]["labeled_filename"] == records[0].new_name
    assert rows[0]["protein_role"] == "TGT"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["export_type"] == "label_map"
    assert payload["label_plan"][0]["new_name"] == records[0].new_name
