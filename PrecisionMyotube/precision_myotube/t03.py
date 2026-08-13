"""Official, fail-closed T03 assessment for sealed model-candidate runs.

T03 is deliberately stricter than trusting a candidate's own run manifest.  This
module independently verifies hashes, fold isolation, prediction authority,
parameter selection, and benchmark metrics.  It also separates proposal-conditioned
ground truth from accepted human corrections so a candidate that shares the
proposal generator cannot receive an unqualified near-ceiling score.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from .benchmark import _aggregate_metrics, benchmark_instances
from .schema import (InstanceRecord, InstanceSet, encode_sparse_positions,
                     rle_foreground_positions)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOOTSTRAP = (
    ROOT / "PrecisionMyotube/annotation_work/bootstrap_v1/bootstrap_manifest.json"
)
DEFAULT_CORRECTIONS = (
    ROOT / "PrecisionMyotube/annotation_work/bootstrap_v1/corrections.jsonl"
)
DEFAULT_SNAPSHOT = (
    ROOT / "PrecisionMyotube/annotation_work/six_well_snapshot.json"
)
IDENTICAL_IOU = 1.0 - 1e-12
REVIEW_ORDER_PROXY = (
    "19_B06_act104_trka",
    "22_B03_act104_egfrc",
    "29_C05_br223_egfrc",
    "32_C08_br223_igf1r",
    "33_C09_br223_trka",
    "23_B02_ctrl",
)
SUSPECTED_EARLY_STANDARD_WELL = REVIEW_ORDER_PROXY[0]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _root_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _check(checks: list[dict], check_id: str, passed: bool, detail) -> None:
    checks.append({"id": check_id, "passed": bool(passed), "detail": detail})


def select_grid_index(table: dict[str, list[dict]], train_wells: list[str],
                      metric: str) -> tuple[int, float]:
    """Reproduce fold-honest grid selection with the declared lowest-index tie break."""
    if not train_wells:
        raise ValueError("parameter selection requires at least one training well")
    indices = sorted({int(row["param_index"]) for row in table[train_wells[0]]})
    by_well = {
        well: {int(row["param_index"]): float(row[metric]) for row in table[well]}
        for well in train_wells
    }
    scores = {
        index: float(np.mean([by_well[well][index] for well in train_wells]))
        for index in indices
    }
    selected = max(indices, key=lambda index: (scores[index], -index))
    return selected, scores[selected]


def accepted_correction_ids(corrections: list[dict],
                            gt_ids_by_well: dict[str, set[str]]) -> dict[str, set[str]]:
    """Return only accepted corrections that are actually authoritative T03 GT.

    A correction record with ``action=ambiguous`` is useful failure-mode evidence,
    but it is not reviewed-complete ground truth and must never enter a recall
    denominator.
    """
    result = {well: set() for well in gt_ids_by_well}
    for record in corrections:
        well = str(record["stem"])
        instance_id = str(record["id"])
        if (record.get("action") == "accept" and well in gt_ids_by_well and
                instance_id in gt_ids_by_well[well]):
            result[well].add(instance_id)
    return result


def _metric_matches_manifest(actual: dict, declared: dict) -> tuple[bool, list[str]]:
    mismatches = []
    for key in (
        "n_gt", "n_pred", "tp", "precision", "recall", "f1",
        "precision_weighted_score", "mean_matched_iou", "false_split_count",
        "false_split_rate", "over_merge_count", "over_merge_rate",
        "length_mdape", "width_mdape", "automatic_coverage",
    ):
        left, right = actual.get(key), declared.get(key)
        if left is None or right is None:
            if left != right:
                mismatches.append(f"{key}: recomputed={left}, declared={right}")
        elif isinstance(left, (int, np.integer)) and isinstance(right, int):
            if int(left) != int(right):
                mismatches.append(f"{key}: recomputed={left}, declared={right}")
        elif not np.isclose(float(left), float(right), atol=0.0015, rtol=0.0):
            mismatches.append(f"{key}: recomputed={left}, declared={right}")
    return not mismatches, mismatches


def _labelling_standard_sensitivity(rows: list[dict], snapshot: dict,
                                    bootstrap: dict) -> dict:
    """Post-hoc disclosure without replacing the predeclared six-well primary.

    The first reviewed well has a materially higher certification fraction.  Dropping
    it from the primary after observing that shift would change the estimand post hoc,
    so the primary remains all six wells and a complete drop-one-well table is carried
    beside it.  This function is descriptive; it performs no relabelling or tuning.
    """
    by_well = {str(row["held_out_well"]): row for row in rows}
    total_gt = sum(int(row["n_gt"]) for row in rows)
    total_false_splits = sum(int(row["false_split_count"]) for row in rows)
    certification = []
    for order, well in enumerate(REVIEW_ORDER_PROXY, start=1):
        counts = snapshot["wells"][well]["counts"]
        candidates = sum(int(counts[key]) for key in (
            "complete", "border_truncated", "ambiguous"))
        certified = int(counts["complete"])
        kept = int(bootstrap["per_well"][well]["complete_kept"])
        certification.append({
            "review_order_proxy": order,
            "well": well,
            "triaged_candidates": candidates,
            "certified_complete_before_binding_exclusions": certified,
            "certified_fraction": certified / candidates if candidates else None,
            "authoritative_masks_after_binding_exclusions": kept,
            "authoritative_fraction_of_375": (
                kept / int(bootstrap["trainable_complete"])),
        })

    drop_one = []
    for well in REVIEW_ORDER_PROXY:
        row = by_well[well]
        kept_gt = total_gt - int(row["n_gt"])
        kept_splits = total_false_splits - int(row["false_split_count"])
        drop_one.append({
            "omitted_well": well,
            "n_gt_remaining": kept_gt,
            "false_split_count_remaining": kept_splits,
            "false_split_rate_remaining": kept_splits / kept_gt if kept_gt else None,
            "omitted_well_n_gt": int(row["n_gt"]),
            "omitted_well_false_split_count": int(row["false_split_count"]),
        })

    suspected = next(row for row in drop_one
                     if row["omitted_well"] == SUSPECTED_EARLY_STANDARD_WELL)
    return {
        "status": "mandatory_posthoc_sensitivity; does_not_replace_primary",
        "ruling": (
            "Keep the predeclared all-six-well pooled primary unchanged. Report this "
            "whole-well drop-one sensitivity beside it for every candidate; candidate "
            "comparison must also report paired drop-one deltas once a second candidate exists."
        ),
        "primary_all_six_wells": {
            "n_gt": total_gt,
            "false_split_count": total_false_splits,
            "false_split_rate": (
                total_false_splits / total_gt if total_gt else None),
        },
        "suspected_early_standard_well": SUSPECTED_EARLY_STANDARD_WELL,
        "suspected_well_omission": suspected,
        "drop_one_well": drop_one,
        "certification_by_review_order_proxy": certification,
        "review_order_source": (
            "decisions.json filesystem mtimes; proxy only, not a logged review order"),
        "interpretation": (
            "The first reviewed well was certified at 0.500 versus 0.204-0.257 in "
            "the next four treated wells. This is consistent with a settling annotation "
            "standard, but biology is not excluded because the first well has a unique "
            "treatment combination. The control was reviewed last and is confounded with order."
        ),
        "no_relabelling_performed": True,
    }


def _corrected_ground_truth(source: InstanceSet,
                            records: list[InstanceRecord]) -> InstanceSet:
    return InstanceSet(
        image_shape=source.image_shape,
        image_id=source.image_id,
        instances=records,
        annotation_policy=source.annotation_policy,
        provenance={
            **source.provenance,
            "t03_subset": "accepted human correction-pair masks reconstructed at full-field coordinates",
            "selected_ids": sorted(record.id for record in records),
        },
    )


def _decode_rowmajor(rle: dict) -> np.ndarray:
    height, width = int(rle["h"]), int(rle["w"])
    flat = np.zeros(height * width, dtype=bool)
    cursor = 0
    foreground = False
    for raw_count in rle["counts"]:
        count = int(raw_count)
        if foreground:
            flat[cursor:cursor + count] = True
        cursor += count
        foreground = not foreground
    if cursor != flat.size:
        raise ValueError(f"invalid row-major RLE length: {cursor}, expected {flat.size}")
    return flat.reshape(height, width)


def _reconstruct_correction(source_record: InstanceRecord, correction: dict,
                            decision: dict, npz_path: Path,
                            image_shape: tuple[int, int]
                            ) -> tuple[InstanceRecord, dict]:
    """Place a cropped correction back into its full-field coordinates.

    Correction NPZ files contain edit-canvas masks.  The frozen decision's geometry
    supplies the full-field origin and the source-crop dimensions.  This reproduces
    ``qc_review.apply`` exactly: nearest-neighbor resize from the edit canvas to the
    source crop, then placement at the recorded origin.
    """
    archive = np.load(npz_path)
    proposal = np.asarray(archive["proposal"], dtype=bool)
    corrected = np.asarray(archive["corrected"], dtype=bool)
    if not proposal.any() or not corrected.any():
        raise ValueError(f"{correction['stem']}/{correction['id']}: empty correction mask")

    rles = decision.get("labels_rle")
    if rles is None and decision.get("mask_rle"):
        rles = [decision["mask_rle"]]
    decision_union = np.zeros(corrected.shape, dtype=bool)
    for rle in rles or []:
        decision_union |= _decode_rowmajor(rle)
    decision_matches_npz = np.array_equal(decision_union, corrected)

    from PIL import Image
    geometry = decision["geom"]
    source_height = int(geometry["src_h"])
    source_width = int(geometry["src_w"])
    origin_row, origin_col = map(int, geometry["origin"])
    resized = np.asarray(
        Image.fromarray(corrected.astype(np.uint8) * 255).resize(
            (source_width, source_height), Image.NEAREST)
    ) > 127
    local_rows, local_cols = np.nonzero(resized)
    full_rows = local_rows + origin_row
    full_cols = local_cols + origin_col
    corrected_positions = (
        full_rows.astype(np.int64) +
        full_cols.astype(np.int64) * image_shape[0]
    )
    source_positions = rle_foreground_positions(source_record.rle)
    full_field_aligns = np.array_equal(
        np.sort(corrected_positions), np.sort(source_positions))
    if not decision_matches_npz or not full_field_aligns:
        raise ValueError(
            f"{correction['stem']}/{correction['id']}: frozen correction does not "
            "reproduce the sealed full-field evaluation mask")
    record = InstanceRecord(
        id=source_record.id,
        status="complete",
        rle=encode_sparse_positions(image_shape, corrected_positions),
        source="accepted_human_correction_pair",
        reviewed=True,
        notes=f"reconstructed from {correction['source_npz']}",
    )
    evidence = {
        "well": correction["stem"],
        "instance_id": correction["id"],
        "npz_hash_matches": _sha256(npz_path) == correction["sha256"],
        "decision_edit_mask_matches_npz": bool(decision_matches_npz),
        "full_field_correction_matches_sealed_gt": bool(full_field_aligns),
        "corrected_area_matches_manifest": (
            int(corrected.sum()) == int(correction["corrected_px"])),
        "corrected_edit_px": int(corrected.sum()),
        "corrected_full_field_px": int(resized.sum()),
        "sealed_gt_px": int(source_positions.size),
        "origin_row": origin_row,
        "origin_col": origin_col,
        "source_height": source_height,
        "source_width": source_width,
    }
    return record, evidence


def assess_t03_run(run_dir: str | Path, *,
                   bootstrap_path: str | Path = DEFAULT_BOOTSTRAP,
    corrections_path: str | Path = DEFAULT_CORRECTIONS,
                   snapshot_path: str | Path = DEFAULT_SNAPSHOT,
                   bootstrap_resamples: int = 10_000,
                   seed: int = 20260723) -> dict:
    """Assess one sealed T02 run under the binding T03 statistical contract."""
    run_dir = _root_path(run_dir)
    bootstrap_path = _root_path(bootstrap_path)
    corrections_path = _root_path(corrections_path)
    snapshot_path = _root_path(snapshot_path)
    run_manifest_path = run_dir / "run_manifest.json"
    grid_path = run_dir / "grid_scores.json"

    run = _load_json(run_manifest_path)
    bootstrap = _load_json(bootstrap_path)
    grid = _load_json(grid_path)
    corrections = _load_jsonl(corrections_path)
    snapshot = _load_json(snapshot_path)
    checks: list[dict] = []

    posthoc_safety = run.get("posthoc_safety_evidence")
    if posthoc_safety:
        for source_name, source_spec in run.get("source_hashes", {}).items():
            current_source_hash = (
                source_spec.get("current_source_sha256") or
                source_spec.get("current_reporting_source_sha256")
            )
            if not current_source_hash:
                continue
            current_source_path = _root_path(source_spec["path"])
            actual_current_source_hash = (
                _sha256(current_source_path) if current_source_path.is_file() else None
            )
            _check(
                checks,
                f"posthoc_current_{source_name}_source_hash",
                actual_current_source_hash == current_source_hash,
                {
                    "path": source_spec["path"],
                    "executed_source_sha256": source_spec.get("sha256"),
                    "expected_current_source_sha256": current_source_hash,
                    "actual_current_source_sha256": actual_current_source_hash,
                    "change_scope": source_spec.get("current_source_change_scope"),
                },
            )
        flag_spec = posthoc_safety["flag_rule"]
        review_spec = posthoc_safety["blinded_two_pass_review"]
        evidence_specs = (
            ("posthoc_over_merge_flaggability_hash",
             flag_spec["artifact"], flag_spec["sha256"]),
            ("posthoc_over_merge_review_hash",
             review_spec["score_artifact"], review_spec["score_sha256"]),
            ("posthoc_over_merge_report_hash",
             posthoc_safety["evidence_report"],
             posthoc_safety["evidence_report_sha256"]),
        )
        for check_id, evidence_path, expected_hash in evidence_specs:
            resolved = _root_path(evidence_path)
            actual_hash = _sha256(resolved) if resolved.is_file() else None
            _check(checks, check_id, actual_hash == expected_hash, {
                "path": evidence_path,
                "expected": expected_hash,
                "actual": actual_hash,
            })

        flag_payload = _load_json(_root_path(flag_spec["artifact"]))
        flag_summary = flag_payload["summary"]
        _check(
            checks,
            "posthoc_over_merge_flaggability_values",
            (
                int(flag_summary["n_accepted_merges"]) ==
                int(flag_spec["accepted_merges_in_two_reviewed_wells"]) and
                int(flag_summary["n_eligible"]) ==
                int(flag_spec["reference_examinable_merges"]) and
                int(flag_summary["n_flagged"]) ==
                int(flag_spec["flagged_merges"])
            ),
            flag_summary,
        )

        review_payload = _load_json(_root_path(review_spec["score_artifact"]))
        review_controls = review_payload["controls"]
        review_intra = review_payload["intra_rater"]
        review_calibration = review_payload["confidence_calibration"]
        _check(
            checks,
            "posthoc_over_merge_review_values",
            (
                int(review_payload["n_scored"]) == int(review_spec["n_objects"]) and
                int(review_controls["n"]) == int(review_spec["n_unflagged_controls"]) and
                int(review_controls["different_myotubes"]) ==
                int(review_spec["unflagged_controls_called_different_myotubes"]) and
                float(review_intra["cohens_kappa"]) ==
                float(review_spec["cohens_kappa"]) and
                int(review_intra["n_flips"]) == int(review_spec["n_flips"]) and
                float(review_calibration["auc_probability_predicts_same"]) ==
                float(posthoc_safety["confidence_calibration"]
                      ["auc_probability_predicts_human_same_myotube"])
            ),
            {
                "controls": review_controls,
                "intra_rater": review_intra,
                "confidence_calibration": review_calibration,
            },
        )

        control_spec = posthoc_safety.get("control_only_round")
        if control_spec:
            control_evidence = (
                ("posthoc_control_only_key_hash", "key_artifact", "key_sha256"),
                ("posthoc_control_only_decisions_hash", "decisions_artifact",
                 "decisions_sha256"),
                ("posthoc_control_only_score_hash", "score_artifact", "score_sha256"),
                ("posthoc_control_only_cases_hash", "cases_artifact", "cases_sha256"),
                ("posthoc_control_only_report_hash", "evidence_report",
                 "evidence_report_sha256"),
            )
            for check_id, path_key, hash_key in control_evidence:
                evidence_path = control_spec[path_key]
                resolved = _root_path(evidence_path)
                actual_hash = _sha256(resolved) if resolved.is_file() else None
                _check(checks, check_id, actual_hash == control_spec[hash_key], {
                    "path": evidence_path,
                    "expected": control_spec[hash_key],
                    "actual": actual_hash,
                })

            control_payload = _load_json(_root_path(control_spec["score_artifact"]))
            counts = control_payload["counts"]
            primary = control_payload["primary"]
            interval = primary["ci95_stratified_bootstrap"]
            sensitivity = control_payload["sensitivity_to_excluded_cases"]
            _check(
                checks,
                "posthoc_control_only_population_values",
                (
                    bool(control_payload["predeclared"]["prespecified_at_build_time"]) and
                    int(counts["n_cases"]) == int(control_spec["n_cases"]) and
                    int(counts["n_resolved"]) == int(control_spec["n_resolved"]) and
                    int(counts["n_different_myotubes"]) ==
                    int(control_spec["n_different_myotubes"]) and
                    int(counts["n_same_myotube"]) ==
                    int(control_spec["n_same_myotube"]) and
                    float(primary["population_over_merge_rate"]) ==
                    float(control_spec["population_over_merge_rate"]) and
                    float(interval["lower"]) == float(control_spec["ci95"][0]) and
                    float(interval["upper"]) == float(control_spec["ci95"][1]) and
                    int(primary["implied_over_merges"]) ==
                    int(control_spec["implied_over_merges"]) and
                    float(sensitivity["rate_if_all_excluded_were_same_myotube"]) ==
                    float(control_spec["sensitivity_bounds"][0]) and
                    float(sensitivity["rate_if_all_excluded_were_different"]) ==
                    float(control_spec["sensitivity_bounds"][1])
                ),
                {"counts": counts, "primary": primary, "sensitivity": sensitivity},
            )

    _check(
        checks, "bootstrap_manifest_hash",
        _sha256(bootstrap_path) == run.get("input_manifest_sha256"),
        {
            "expected": run.get("input_manifest_sha256"),
            "actual": _sha256(bootstrap_path),
        },
    )
    _check(checks, "candidate_reported_no_failures", not run.get("failures"),
           run.get("failures", []))
    _check(checks, "corrections_not_used_for_training_or_tuning",
           run.get("correction_pairs_used") is False,
           run.get("correction_pairs_used"))
    _check(checks, "synthetic_pairs_not_used",
           run.get("synthetic_pairs_used") is False,
           run.get("synthetic_pairs_used"))

    expected_wells = set(bootstrap["per_well"])
    folds = run.get("folds", [])
    held_out = [str(fold["held_out_well"]) for fold in folds]
    _check(checks, "six_unique_whole_well_folds",
           len(folds) == 6 and set(held_out) == expected_wells and
           len(set(held_out)) == len(held_out),
           {"expected": sorted(expected_wells), "actual": held_out})

    grid_metric = str(grid.get("selection_metric"))
    _check(checks, "selection_metric_matches_manifest",
           grid_metric == run.get("parameter_selection", {}).get("metric"),
           {
               "grid": grid_metric,
               "manifest": run.get("parameter_selection", {}).get("metric"),
           })

    rows: list[dict] = []
    gt_sets: dict[str, InstanceSet] = {}
    gt_ids_by_well: dict[str, set[str]] = {}
    decisions_by_well: dict[str, dict] = {}
    all_matches: list[dict] = []

    for fold in folds:
        well = str(fold["held_out_well"])
        train_wells = [str(value) for value in fold["train_wells"]]
        expected_train = expected_wells - {well}
        _check(checks, f"{well}:whole_well_split",
               set(train_wells) == expected_train and well not in train_wells,
               {"expected_train": sorted(expected_train), "actual_train": train_wells})

        selected_index, selected_score = select_grid_index(
            grid["table"], train_wells, grid_metric)
        _check(checks, f"{well}:fold_honest_parameter_selection",
               selected_index == int(fold["selected_param_index"]) and
               np.isclose(selected_score, float(fold["train_mean_selection_score"]),
                          atol=0.0015, rtol=0.0),
               {
                   "recomputed_index": selected_index,
                   "declared_index": fold["selected_param_index"],
                   "recomputed_train_mean": selected_score,
                   "declared_train_mean": fold["train_mean_selection_score"],
               })

        pred_path = _root_path(fold["prediction"]["instances"])
        pred_manifest_path = _root_path(fold["prediction"]["manifest"])
        gt_path = _root_path(fold["eval_gt"]["path"])
        pred_sha = _sha256(pred_path)
        gt_sha = _sha256(gt_path)
        pred_manifest = _load_json(pred_manifest_path)

        _check(checks, f"{well}:prediction_hash",
               pred_sha == fold["prediction"]["instances_sha256"] ==
               pred_manifest.get("instances_sha256"),
               {
                   "actual": pred_sha,
                   "run_manifest": fold["prediction"]["instances_sha256"],
                   "prediction_manifest": pred_manifest.get("instances_sha256"),
               })
        _check(checks, f"{well}:evaluation_gt_hash",
               gt_sha == fold["eval_gt"]["sha256"],
               {"actual": gt_sha, "expected": fold["eval_gt"]["sha256"]})

        prediction = InstanceSet.load(pred_path)
        ground_truth = InstanceSet.load(gt_path)
        gt_sets[well] = ground_truth
        gt_ids_by_well[well] = {record.id for record in ground_truth.instances}
        source_instances = _root_path(bootstrap["per_well"][well]["source_instances"])
        decisions_path = source_instances.with_name(
            source_instances.name.replace(".qc.instances.json", ".decisions.json"))
        decisions_by_well[well] = _load_json(decisions_path)["decisions"]
        frozen_prefix = snapshot["wells"][well]["decisions_sha256"]
        _check(checks, f"{well}:frozen_decisions_hash",
               _sha256(decisions_path).startswith(frozen_prefix),
               {
                   "actual": _sha256(decisions_path),
                   "frozen_prefix": frozen_prefix,
                   "path": str(decisions_path.relative_to(ROOT)),
               })

        embedded_provenance = prediction.provenance
        provenance = pred_manifest.get("provenance", {})
        selected_on = provenance.get("thresholds", {}).get("selected_on_wells", [])
        _check(checks, f"{well}:prediction_is_unreviewed",
               all(not record.reviewed for record in prediction.instances),
               {"n_predictions": len(prediction.instances)})
        _check(checks, f"{well}:prediction_status_is_scored_complete",
               all(record.status == "complete" for record in prediction.instances),
               {"statuses": sorted({record.status for record in prediction.instances})})
        _check(checks, f"{well}:hash_bound_sidecar_provenance",
               prediction.image_id == well and
               provenance.get("data_hash") == run.get("input_manifest_sha256") and
               provenance.get("model") == run.get("candidate") and
               set(selected_on) == expected_train and well not in selected_on,
               {
                   "image_id": prediction.image_id,
                   "model": provenance.get("model"),
                   "data_hash": provenance.get("data_hash"),
                   "selected_on_wells": selected_on,
               })
        _check(checks, f"{well}:embedded_instance_provenance_present",
               bool(embedded_provenance),
               {
                   "embedded_provenance": embedded_provenance,
                   "severity": "compliance_warning",
                   "mitigation": (
                       "complete provenance exists in the sidecar manifest whose "
                       "instances_sha256 binds it to this exact InstanceSet"),
               })
        _check(checks, f"{well}:gt_count_and_authority",
               len(ground_truth.instances) == int(fold["eval_gt"]["n_gt"]) and
               all(record.reviewed and record.status == "complete"
                   for record in ground_truth.instances),
               {
                   "actual_count": len(ground_truth.instances),
                   "expected_count": fold["eval_gt"]["n_gt"],
               })
        excluded = set(fold["eval_gt"].get("excluded_ids", []))
        _check(checks, f"{well}:binding_exclusions_absent",
               not (excluded & gt_ids_by_well[well]),
               {"excluded_ids": sorted(excluded)})

        metrics = benchmark_instances(gt_path, pred_path)
        agrees, mismatches = _metric_matches_manifest(
            metrics, fold["held_out_metrics"])
        _check(checks, f"{well}:independent_metric_reproduction", agrees, mismatches)
        row = {"held_out_well": well, **metrics}
        rows.append(row)
        all_matches.extend({"held_out_well": well, **match}
                           for match in metrics["matched_instances"])

    accepted_ids = accepted_correction_ids(corrections, gt_ids_by_well)
    accepted_flat = {(well, instance_id)
                     for well, ids in accepted_ids.items() for instance_id in ids}
    ambiguous_records = [record for record in corrections
                         if record.get("action") != "accept"]
    _check(checks, "correction_record_partition",
           len(corrections) == 40 and len(accepted_flat) == 25 and
           len(ambiguous_records) == 15,
           {
               "total_records": len(corrections),
               "accepted_in_authoritative_gt": len(accepted_flat),
               "nonaccepted_records": len(ambiguous_records),
           })

    correction_integrity = []
    corrected_records_by_well: dict[str, list[InstanceRecord]] = {
        well: [] for well in expected_wells
    }
    correction_by_key = {
        (str(record["stem"]), str(record["id"])): record
        for record in corrections if record.get("action") == "accept"
    }
    for well, instance_id in sorted(accepted_flat):
        record = correction_by_key[(well, instance_id)]
        npz_path = _root_path(record["source_npz"])
        source_record = next(item for item in gt_sets[well].instances
                             if item.id == instance_id)
        decision = decisions_by_well[well][instance_id]
        corrected_record, evidence = _reconstruct_correction(
            source_record, record, decision, npz_path, gt_sets[well].image_shape)
        corrected_records_by_well[well].append(corrected_record)
        correction_integrity.append(evidence)
    _check(checks, "accepted_correction_reconstruction_integrity",
           all(item["npz_hash_matches"] and
               item["decision_edit_mask_matches_npz"] and
               item["full_field_correction_matches_sealed_gt"] and
               item["corrected_area_matches_manifest"]
               for item in correction_integrity),
           correction_integrity)

    corrected_rows = []
    corrected_availability = []
    with tempfile.TemporaryDirectory(prefix="precision_myotube_t03_") as temp:
        temp_dir = Path(temp)
        for fold in folds:
            well = str(fold["held_out_well"])
            records = corrected_records_by_well.get(well, [])
            corrected_availability.append({
                "held_out_well": well,
                "n_authoritative_corrected_gt": len(records),
                "evaluable": bool(records),
            })
            if not records:
                continue
            subset = _corrected_ground_truth(gt_sets[well], records)
            subset_path = temp_dir / f"{well}.corrected.instances.json"
            subset.save(subset_path)
            metrics = benchmark_instances(
                subset_path, _root_path(fold["prediction"]["instances"]))
            corrected_rows.append({"held_out_well": well, **metrics})

    overall = _aggregate_metrics(
        rows, bootstrap_resamples=bootstrap_resamples, seed=seed)
    labelling_standard_sensitivity = _labelling_standard_sensitivity(
        rows, snapshot, bootstrap)
    corrected = _aggregate_metrics(
        corrected_rows, bootstrap_resamples=bootstrap_resamples, seed=seed)
    corrected["precision_interpretable"] = False
    corrected["f1_interpretable"] = False
    corrected["precision_note"] = (
        "All field predictions are compared with a deliberately sparse corrected-GT "
        "subset; recall, matched IoU, and matched measurement error are meaningful, "
        "but subset precision is not detector precision. F1 inherits that invalid "
        "precision denominator and is therefore not detector F1."
    )
    corrected["availability_by_well"] = corrected_availability

    corrected_matches = [
        {"held_out_well": row["held_out_well"], **match}
        for row in corrected_rows for match in row["matched_instances"]
    ]
    total_gt = sum(row["n_gt"] for row in rows)
    total_corrected = len(accepted_flat)
    all_ious = np.asarray([match["iou"] for match in all_matches], dtype=float)
    corrected_ious = np.asarray(
        [match["iou"] for match in corrected_matches], dtype=float)
    circularity = {
        "total_gt": total_gt,
        "sealed_eval_gt_masks": total_gt,
        "sealed_eval_gt_masks_with_accepted_corrections": total_corrected,
        "sealed_eval_gt_masks_without_recorded_correction": total_gt - total_corrected,
        "uncorrected_fraction_of_sealed_eval_gt": (
            (total_gt - total_corrected) / total_gt),
        "matched_sealed_eval_gt": len(all_matches),
        "matched_corrected_subset": len(corrected_matches),
        "recall_corrected_subset": len(corrected_matches) / total_corrected,
        "median_iou_corrected": (
            float(np.median(corrected_ious)) if corrected_ious.size else None),
        "pixel_identical_matches": (
            int((all_ious >= IDENTICAL_IOU).sum()) if all_ious.size else 0),
        "pixel_identical_fraction_of_matches": (
            float((all_ious >= IDENTICAL_IOU).mean()) if all_ious.size else None),
    }

    compliance_warnings = [
        item for item in checks
        if (not item["passed"] and
            isinstance(item.get("detail"), dict) and
            item["detail"].get("severity") == "compliance_warning")
    ]
    integrity_passed = all(
        item["passed"] or item in compliance_warnings for item in checks)
    corrected_units = len(corrected_rows)
    density_declared = bool(run.get("density_strata"))
    global_development_used_all_wells = bool(
        run.get("global_development_used_all_six_wells", False)
    )
    decision_reasons = []
    if len(folds) < 6:
        decision_reasons.append("fewer than six held-out folds")
    if not integrity_passed:
        decision_reasons.append("one or more sealed-artifact integrity checks failed")
    if corrected_units < 3:
        decision_reasons.append(
            "non-circular corrected-complete evidence spans fewer than three held-out wells")
    if not density_declared:
        decision_reasons.append("the sealed run does not declare density strata")
    if global_development_used_all_wells:
        decision_reasons.append(
            "linker architecture/window/operating-point development used all six "
            "wells; only its scaler and coefficients are LOWO-refit")
    decision_reasons.append(
        "G-SO2 must disclose the 0.500 first-well certification fraction versus "
        "0.204-0.257 in the next four treated wells and carry the drop-one-well "
        "sensitivity beside the unchanged predeclared primary")
    decision_reasons.append(
        "no genuinely independent second candidate is available; a linked variant "
        "of the classical floor does not satisfy the two-candidate comparison")

    candidate_is_linker = run.get("candidate") == "classical_ridge_graph_linker"
    recall_resolution = 1.0 / overall["n_gt"] if overall["n_gt"] else None
    run_limitations = list(run.get("limitations", []))
    limitations = [
        "single operator and proposal-conditioned retrospective reference masks",
        "certification rate shifted from 0.500 in the first reviewed well to "
        "0.204-0.257 in the next four treated wells; review order is inferred from "
        "file mtimes and biology is not formally excluded",
        "all six held-out wells are from one plate",
        "conventional precision is depressed by sparse reviewed-complete GT",
        "the classical floor cannot represent overlapping crossing instances",
        "the accepted corrected subset contains 25 masks from only two wells",
        "no prospective plate or independent biological replication is present",
        *(
            ["candidate architecture/window/operating-point development is not "
             "nested LOWO; only scaler and coefficients are fold-refit"]
            if global_development_used_all_wells else []
        ),
        *run_limitations,
    ]
    limitations = list(dict.fromkeys(limitations))
    if candidate_is_linker:
        control_round = (posthoc_safety or {}).get("control_only_round")
        if control_round:
            disposition = (
                "rejected_development_baseline_only; "
                "automatic_and_manual_QC_proposal_use_withdrawn"
            )
            decision_reasons.append(
                "the predeclared uniform six-well review measured population "
                "over-merge rate 0.6487 (95% CI 0.4497-0.8318), implying about "
                "350 wrong merges among 540 accepted; the linker branch is closed"
            )
            next_dependency = (
                "Do not use linked output automatically or as a manual-QC proposal "
                "source. T02 has no viable independent second candidate; Omnipose is "
                "the only remaining path and stays parked until the NVIDIA driver is stable."
            )
        else:
            disposition = (
                "retain_as_reproducible_manual_QC_candidate; "
                "automatic_use_not_released"
            )
            decision_reasons.append(
                "posthoc raw-image review found the sparse-reference over-merge count "
                "cannot estimate linker safety; the automatic over-merge cost is unquantified"
            )
            next_dependency = (
                "Before any automatic-use reconsideration, estimate accepted-merge error "
                "with a uniformly sampled, fully reviewed control-only round across all six "
                "wells; keep P>=0.90 locked and do not tune on that review."
            )
    else:
        disposition = "retain_as_reproducible_floor; manual_QC_only"
        next_dependency = (
            "Either complete a genuinely independent second candidate or formally "
            "revise the plan to a single-candidate feasibility assessment; add locked "
            "density metadata and independent corrected/prospective evidence."
        )

    return {
        "assessment": "T03_official_candidate_assessment",
        "assessment_version": "1.2",
        "candidate": run.get("candidate"),
        "candidate_version": run.get("candidate_version"),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "artifact_hashes": {
            "assessor_source_sha256": _sha256(Path(__file__)),
            "statistical_analysis_plan_sha256": _sha256(
                ROOT / "PrecisionMyotube/STATISTICAL_ANALYSIS_PLAN.md"),
            "run_manifest_sha256": _sha256(run_manifest_path),
            "grid_scores_sha256": _sha256(grid_path),
            "bootstrap_manifest_sha256": _sha256(bootstrap_path),
            "corrections_jsonl_sha256": _sha256(corrections_path),
        },
        "integrity": {
            "passed": integrity_passed,
            "compliance_warnings": compliance_warnings,
            "checks": checks,
        },
        "statistical_scope": {
            "plate": "PLATE_23",
            "held_out_evaluation_unit": "whole well/field",
            "n_held_out_units": len(rows),
            "n_independent_biological_replicates": 1,
            "biological_treatment_inference_supported": False,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": seed,
            "density_stratification": (
                "declared" if density_declared else "not evaluable: no locked density labels"),
            "fully_nested_candidate_development": not global_development_used_all_wells,
            "candidate_development_scope": run.get("lowo_scope"),
        },
        "all_proposal_conditioned_gt": {
            "interpretation": (
                "Secondary diagnostic only because the classical candidate shares the "
                "proposal family used to create most reference masks. Precision and F1 "
                "are additionally not detector metrics because reviewed-complete GT is "
                "a sparse subset of field structure. Recall is descriptive for the "
                "reviewed subset, not a population-performance claim."
            ),
            "precision_interpretable": False,
            "f1_interpretable": False,
            "recall_interpretable_for_reviewed_subset": True,
            "recall_resolution_per_reviewed_object": recall_resolution,
            "recall_note": (
                "The denominator is 375 reviewed masks, so one matched object changes "
                "pooled recall by about 0.0027. The linker changes 348 to 349 matches; "
                "this is low-resolution, well-sensitive, descriptive evidence and not "
                "an established recall benefit."
                if candidate_is_linker else
                "Recall is descriptive for the reviewed, proposal-conditioned subset "
                "and does not establish population detector sensitivity."
            ),
            "over_merge_rate_interpretable": False,
            "over_merge_rate_audit": (
                "predeclared uniform six-well population audit attached"
                if (posthoc_safety or {}).get("control_only_round") else
                "posthoc flaggability audit attached"
                if posthoc_safety else
                "no posthoc flaggability audit attached"
            ),
            "over_merge_note": (
                "The stored sparse-reference count remains a ceiling diagnostic, but "
                "a separate predeclared uniform six-well raw-image review measured the "
                "accepted-merge population error rate as 0.6487 (95% CI "
                "0.4497-0.8318), about 350 wrong merges among 540 accepted."
                if (posthoc_safety or {}).get("control_only_round") else
                "The stored count is a sparse-reference flag count, not a total "
                "over-merge count or safety rate. Only 3 of 216 accepted merges in the "
                "two reviewed wells had at least two reviewed masks and were examinable; "
                "all three were flagged. Neither predictions nor sparse GT is a valid "
                "denominator for the linker's over-merge probability."
                if posthoc_safety else
                "No posthoc flaggability audit is attached, so this rate is "
                "unestablished rather than valid. The >=2-reviewed-reference rule that "
                "saturated the linker audit is a property of the sparse "
                "reviewed-complete GT, not of any one candidate, and ceilings every run "
                "scored against this reference set."
            ),
            **overall,
            "fields": rows,
        },
        "labelling_standard_sensitivity": labelling_standard_sensitivity,
        "accepted_correction_subset": {
            "interpretation": (
                "Primary meaningful classical-floor evidence. Only accepted corrections "
                "that remain reviewed-complete in the sealed evaluation GT are included."
            ),
            **corrected,
            "fields": corrected_rows,
        },
        "proposal_circularity": circularity,
        "posthoc_safety_evidence": posthoc_safety,
        "gate": {
            "t03_complete": False,
            "g_so2_passed": False,
            "candidate_selected": False,
            "disposition": disposition,
            "reasons": decision_reasons,
            "next_dependency": next_dependency,
        },
        "limitations": limitations,
    }


def write_t03_assessment(run_dir: str | Path, out_path: str | Path, **kwargs) -> dict:
    result = assess_t03_run(run_dir, **kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
