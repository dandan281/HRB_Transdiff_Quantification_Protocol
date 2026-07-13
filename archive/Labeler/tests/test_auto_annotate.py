from pathlib import Path

from wb_annotator.auto_annotate import (
    acquisition_label,
    auto_annotate_files,
    blot_group_key,
    experiment_group_key,
    infer_file_kind,
    infer_note,
)


def test_extracts_acquisition_label_and_group_key() -> None:
    name = "b2 vs b2h2 vs b2h2m_1(Chemiluminescence).raw16.tif"

    assert acquisition_label(name) == "Chemiluminescence"
    assert blot_group_key(name) == "b2 vs b2h2 vs b2h2m_1"


def test_infers_chemidoc_data_types() -> None:
    assert infer_file_kind("x(Chemiluminescence).tif") == "CHEMI"
    assert infer_file_kind("x(Colorimetric).jpg") == "COLOR"
    assert infer_file_kind("x(Composite).tif") == "MERGE"
    assert infer_note("x(Chemiluminescence).raw16.tif") == "Chemiluminescence; raw16 intensity"


def test_auto_labels_blot_ids_by_shared_group() -> None:
    files = [
        Path("b2 vs b2h2 vs b2h2m_1(Chemiluminescence).jpg"),
        Path("b2 vs b2h2 vs b2h2m_1(Chemiluminescence).raw16.tif"),
        Path("b2 vs b2h2 vs b2h2m_1(Colorimetric).tif"),
        Path("b2 vs b2h2 vs b2h2m_2(Chemiluminescence).jpg"),
        Path("b2 vs b2h2 vs b2h2m_2(Composite).tif"),
    ]

    annotations = auto_annotate_files(files)

    assert [item.blot_id for item in annotations] == ["B01", "B01", "B01", "B02", "B02"]
    assert [item.file_kind for item in annotations] == ["CHEMI", "CHEMI", "COLOR", "CHEMI", "MERGE"]


def test_auto_labels_experiment_sets_separately_from_blot_groups() -> None:
    files = [
        Path("b2 vs b2h2 vs b2h2m_1(Chemiluminescence).tif"),
        Path("b2 vs b2h2 vs b2h2m_2(Chemiluminescence).tif"),
        Path("tab2 vs tab2m titrat_1_h3(Chemiluminescence).tif"),
        Path("tab2 vs tab2m titrat_2_perk(Chemiluminescence).tif"),
    ]

    annotations = auto_annotate_files(files)

    assert experiment_group_key(files[0].name) == "b2 vs b2h2 vs b2h2m"
    assert experiment_group_key(files[2].name) == "tab2 vs tab2m titrat"
    assert [item.experiment_key for item in annotations] == ["E01", "E01", "E02", "E02"]
    assert [item.blot_id for item in annotations] == ["B01", "B02", "B03", "B04"]


def test_auto_labels_lab_specific_proteins() -> None:
    files = [
        Path("tab2 vs tab2m titrat_1_h3(Chemiluminescence).tif"),
        Path("tab2 vs tab2m titrat_2_perk(Chemiluminescence).tif"),
        Path("tab2 vs tab2m titrat_3_pakt(Colorimetric).tif"),
        Path("tab2 vs tab2m titrat_6_pp38(Chemiluminescence).tif"),
    ]

    annotations = auto_annotate_files(files)

    assert [(item.protein_label, item.protein_role) for item in annotations] == [
        ("H3", "LC"),
        ("pERK", "TGT"),
        ("pAKT", "TGT"),
        ("pP38", "TGT"),
    ]
