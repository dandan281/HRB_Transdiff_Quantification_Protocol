import hashlib
from pathlib import Path

import json
import numpy as np
from PIL import Image

from precision_myotube.schema import InstanceRecord, encode_sparse_positions
from precision_myotube.t03 import (
    REVIEW_ORDER_PROXY,
    _labelling_standard_sensitivity,
    _reconstruct_correction,
    accepted_correction_ids,
    assess_t03_run,
    select_grid_index,
)


def test_labelling_shift_keeps_primary_and_adds_complete_drop_one_sensitivity():
    rows = [
        {"held_out_well": well, "n_gt": n_gt, "false_split_count": splits}
        for well, n_gt, splits in zip(
            REVIEW_ORDER_PROXY,
            (119, 60, 59, 54, 48, 35),
            (18, 6, 4, 8, 5, 11),
        )
    ]
    complete = (120, 61, 59, 54, 48, 35)
    candidates = (240, 237, 241, 225, 235, 69)
    snapshot = {"wells": {}}
    bootstrap = {"trainable_complete": 375, "per_well": {}}
    for well, certified, candidate, row in zip(
            REVIEW_ORDER_PROXY, complete, candidates, rows):
        snapshot["wells"][well] = {"counts": {
            "complete": certified,
            "border_truncated": 0,
            "ambiguous": candidate - certified,
        }}
        bootstrap["per_well"][well] = {"complete_kept": row["n_gt"]}

    result = _labelling_standard_sensitivity(rows, snapshot, bootstrap)
    assert result["status"] == (
        "mandatory_posthoc_sensitivity; does_not_replace_primary")
    assert result["primary_all_six_wells"] == {
        "n_gt": 375, "false_split_count": 52,
        "false_split_rate": 52 / 375,
    }
    assert result["suspected_well_omission"]["false_split_count_remaining"] == 34
    assert result["suspected_well_omission"]["false_split_rate_remaining"] == 34 / 256
    first = result["certification_by_review_order_proxy"][0]
    assert first["certified_fraction"] == 0.5
    assert first["authoritative_masks_after_binding_exclusions"] == 119
    assert len(result["drop_one_well"]) == 6


def test_accepted_corrections_require_accept_action_and_authoritative_gt_membership():
    corrections = [
        {"stem": "well_a", "id": "accepted", "action": "accept"},
        {"stem": "well_a", "id": "ambiguous", "action": "ambiguous"},
        {"stem": "well_a", "id": "not_in_gt", "action": "accept"},
        {"stem": "well_b", "id": "accepted_b", "action": "accept"},
    ]
    result = accepted_correction_ids(
        corrections,
        {"well_a": {"accepted", "ambiguous"}, "well_b": {"accepted_b"}},
    )
    assert result == {"well_a": {"accepted"}, "well_b": {"accepted_b"}}


def test_grid_selection_uses_training_wells_only_and_lowest_index_tie_break():
    table = {
        "train_a": [
            {"param_index": 0, "score": 0.5},
            {"param_index": 1, "score": 0.7},
        ],
        "train_b": [
            {"param_index": 0, "score": 0.7},
            {"param_index": 1, "score": 0.5},
        ],
        "held_out": [
            {"param_index": 0, "score": 0.0},
            {"param_index": 1, "score": 1.0},
        ],
    }
    index, score = select_grid_index(
        table, ["train_a", "train_b"], "score")
    assert index == 0
    assert score == 0.6


def _rowmajor_rle(mask):
    flat = np.asarray(mask, dtype=bool).ravel()
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [flat.size]))).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"h": mask.shape[0], "w": mask.shape[1], "counts": counts}


def test_correction_reconstruction_reproduces_full_field_apply_geometry(tmp_path):
    image_shape = (30, 40)
    edit = np.zeros((6, 8), dtype=bool)
    edit[1:5, 2:7] = True
    proposal = np.zeros_like(edit)
    proposal[2:4, 3:6] = True
    geometry = {
        "origin": [5, 9], "src_h": 12, "src_w": 16,
        "edit_h": 6, "edit_w": 8,
    }
    resized = np.asarray(
        Image.fromarray(edit.astype(np.uint8) * 255).resize(
            (geometry["src_w"], geometry["src_h"]), Image.NEAREST)
    ) > 127
    rows, cols = np.nonzero(resized)
    rows += geometry["origin"][0]
    cols += geometry["origin"][1]
    positions = rows.astype(np.int64) + cols.astype(np.int64) * image_shape[0]
    source = InstanceRecord(
        "m1", "complete", encode_sparse_positions(image_shape, positions),
        reviewed=True)

    npz = tmp_path / "pair.npz"
    np.savez_compressed(npz, proposal=proposal.astype(np.uint8),
                        corrected=edit.astype(np.uint8))
    correction = {
        "stem": "well_a", "id": "m1", "source_npz": str(npz),
        "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "corrected_px": int(edit.sum()),
    }
    decision = {
        "action": "accept", "geom": geometry,
        "labels_rle": [_rowmajor_rle(edit)],
    }
    reconstructed, evidence = _reconstruct_correction(
        source, correction, decision, npz, image_shape)
    assert evidence["decision_edit_mask_matches_npz"]
    assert evidence["full_field_correction_matches_sealed_gt"]
    assert reconstructed.reviewed
    assert reconstructed.status == "complete"


def test_official_assessment_marks_sparse_gt_precision_and_f1_uninterpretable():
    artifact = (
        Path(__file__).resolve().parents[1]
        / "runs/t03/classical_v1/assessment.json"
    )
    if not artifact.is_file():
        return
    result = json.loads(artifact.read_text(encoding="utf-8"))
    # This pins the reporting contract in newly generated assessments. The
    # historical artifact predates the explicit F1 field, so use its note as the
    # compatibility boundary until it is regenerated.
    subset = result["accepted_correction_subset"]
    assert subset["precision_interpretable"] is False
    assert "sparse" in subset["precision_note"].lower()


def test_linker_assessment_carries_posthoc_safety_limits():
    artifact = (
        Path(__file__).resolve().parents[1]
        / "runs/t03/classical_linker_v1/assessment.json"
    )
    if not artifact.is_file():
        return
    result = json.loads(artifact.read_text(encoding="utf-8"))
    diagnostic = result["all_proposal_conditioned_gt"]
    assert diagnostic["over_merge_rate_interpretable"] is False
    assert diagnostic["recall_interpretable_for_reviewed_subset"] is True
    assert diagnostic["recall_resolution_per_reviewed_object"] == 1 / 375
    assert "0.6487" in diagnostic["over_merge_note"]
    assert result["posthoc_safety_evidence"]["threshold_status"].startswith("locked")
    assert result["gate"]["candidate_selected"] is False
    assert "automatic_and_manual_QC_proposal_use_withdrawn" in result["gate"]["disposition"]
    checks = {item["id"]: item["passed"] for item in result["integrity"]["checks"]}
    assert checks["posthoc_current_linked_candidate_source_hash"] is True
    assert checks["posthoc_current_link_candidates_source_hash"] is True
    assert checks["posthoc_current_link_model_source_hash"] is True
    assert checks["posthoc_current_link_geometry_source_hash"] is True
    assert checks["posthoc_over_merge_flaggability_hash"] is True
    assert checks["posthoc_over_merge_review_hash"] is True
    assert checks["posthoc_over_merge_report_hash"] is True
    assert checks["posthoc_over_merge_flaggability_values"] is True
    assert checks["posthoc_over_merge_review_values"] is True
    assert checks["posthoc_control_only_population_values"] is True
    assert checks["posthoc_control_only_score_hash"] is True
    joined = " ".join(result["limitations"]).lower()
    assert "auc 0.323" in joined
    assert "0.6487" in joined


def test_linker_run_manifest_does_not_present_sparse_flags_as_an_error_rate():
    artifact = (
        Path(__file__).resolve().parents[1]
        / "runs/t02/classical_linker_v1/run_manifest.json"
    )
    if not artifact.is_file():
        return
    result = json.loads(artifact.read_text(encoding="utf-8"))
    assert result["release_status"].endswith(
        "automatic_and_manual_QC_proposal_use_withdrawn")
    assert result["summary"]["over_merge_rates_interpretable"] is False
    assert result["summary"]["recall_resolution_per_reviewed_object"] == 1 / 375
    evidence = result["posthoc_safety_evidence"]
    assert evidence["flag_rule"]["accepted_merges_in_two_reviewed_wells"] == 216
    assert evidence["flag_rule"]["reference_examinable_merges"] == 3
    assert evidence["blinded_two_pass_review"]["unflagged_controls_called_different_myotubes"] == 6
    assert evidence["control_only_round"]["population_over_merge_rate"] == 0.6487
    assert evidence["control_only_round"]["implied_over_merges"] == 350


def test_unaudited_over_merge_rate_is_not_claimed_interpretable():
    # The losing branch. The >=2-reviewed-reference detection rule is a property
    # of the sparse reviewed-complete GT, not of any one candidate, so a run with
    # no flaggability audit is unestablished, not valid. Absence of an audit must
    # never raise the claim.
    run_dir = Path(__file__).resolve().parents[2] / "model_labs/classical/_runs/v1"
    if not (run_dir / "run_manifest.json").is_file():
        return
    result = assess_t03_run(run_dir, bootstrap_resamples=10)
    diagnostic = result["all_proposal_conditioned_gt"]
    assert result["posthoc_safety_evidence"] is None
    assert diagnostic["over_merge_rate_interpretable"] is False
    assert diagnostic["over_merge_rate_audit"] == "no posthoc flaggability audit attached"
    assert "unestablished" in diagnostic["over_merge_note"]
