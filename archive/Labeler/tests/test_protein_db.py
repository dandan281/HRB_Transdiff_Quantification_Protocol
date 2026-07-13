from wb_annotator.protein_db import infer_protein_from_text


def test_loading_control_aliases() -> None:
    assert infer_protein_from_text("sample_h3.tif").label == "H3"
    assert infer_protein_from_text("sample_h3.tif").role == "LC"
    assert infer_protein_from_text("total-S6 blot.tif").label == "S6"
    assert infer_protein_from_text("GAPDH exposure.jpg").label == "GAPDH"


def test_target_effector_aliases_and_lab_typos() -> None:
    assert infer_protein_from_text("condition_perk(Chemiluminescence).tif").label == "pERK"
    assert infer_protein_from_text("condition_P-ERC.tif").label == "pERK"
    assert infer_protein_from_text("condition_mPAkt.tif").label == "pAKT"
    assert infer_protein_from_text("condition_PP38-ish.tif").label == "pP38"
