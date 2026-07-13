from wb_annotator.naming import canonical_filename, is_windows_safe_filename, slugify
from wb_annotator.schema import ExperimentMetadata, FileAnnotation


def metadata(**overrides: str) -> ExperimentMetadata:
    values = {
        "date": "2026-06-22",
        "experiment_id": "E001",
        "cell_line": "HEK293T",
        "modification": "OE-MYC",
        "treatment_name": "DOX",
        "dose_series": "0/10/100 nM",
        "treatment_time": "24 h",
        "target_protein": "pERK1/2",
        "loading_control": "GAPDH",
    }
    values.update(overrides)
    return ExperimentMetadata(**values)


def test_slugify_removes_windows_invalid_characters_and_transliterates() -> None:
    assert slugify("pAKT/β:Ser473?") == "pAKT-beta-Ser473"
    assert slugify("  a  b__c  ") == "a-b-c"
    assert slugify("CON") == "CON-value"


def test_canonical_filename_matches_wb_pattern_and_is_safe() -> None:
    name = canonical_filename(
        metadata(),
        FileAnnotation(original_name="old.raw16.tif", blot_id="B01", file_kind="TGT", protein_label="pERK1/2", protein_role="TGT"),
        1,
    )

    assert name == (
        "HEK293T_OE-MYC_DOX_0-10-100-nM_TGT-pERK1-2_R01_20260622.raw16.tif"
    )
    assert is_windows_safe_filename(name)


def test_canonical_filename_uses_no_treatment_and_zero_defaults() -> None:
    name = canonical_filename(
        metadata(treatment_name="", dose_series=""),
        FileAnnotation(original_name="loading.jpg", blot_id="B07", file_kind="LC", protein_label="H3", protein_role="LC"),
        3,
    )

    assert name == "HEK293T_OE-MYC_NoTreatment_0_LC-H3_R07_20260622.jpg"
    assert is_windows_safe_filename(name)
