"""Overlap-aware instance benchmark, model selection, and scientific release gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import measure_mask
from .schema import InstanceSet


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def benchmark_instances(gt_path: str | Path, prediction_path: str | Path,
                        iou_threshold: float = 0.5, coverage_threshold: float = 0.2) -> dict:
    gt_set, pred_set = InstanceSet.load(gt_path), InstanceSet.load(prediction_path)
    if gt_set.image_shape != pred_set.image_shape:
        raise ValueError("ground truth and predictions have different image shapes")
    if gt_set.image_id != pred_set.image_id:
        raise ValueError("ground truth and predictions have different image IDs")
    gt = [(record, bbox, mask) for record, bbox, mask in gt_set.cropped_masks()
          if record.reviewed and record.status == "complete"]
    pred = [(record, bbox, mask) for record, bbox, mask in pred_set.cropped_masks()
            if record.status == "complete"]
    gt_boxes = [bbox for _, bbox, _ in gt]
    pred_boxes = [bbox for _, bbox, _ in pred]
    pairs = []
    gt_links: dict[int, set[int]] = {}
    pred_links: dict[int, set[int]] = {}
    for i, ((_, _, gm), gb) in enumerate(zip(gt, gt_boxes)):
        ga = int(gm.sum())
        for j, ((_, _, pm), pb) in enumerate(zip(pred, pred_boxes)):
            if not _overlap(gb, pb):
                continue
            r0, c0, r1, c1 = (max(gb[0], pb[0]), max(gb[1], pb[1]),
                              min(gb[2], pb[2]), min(gb[3], pb[3]))
            g_crop = gm[r0 - gb[0]:r1 - gb[0], c0 - gb[1]:c1 - gb[1]]
            p_crop = pm[r0 - pb[0]:r1 - pb[0], c0 - pb[1]:c1 - pb[1]]
            inter = int(np.count_nonzero(g_crop & p_crop))
            if not inter:
                continue
            pa = int(pm.sum())
            iou = inter / (ga + pa - inter)
            cov_gt, cov_pred = inter / ga, inter / pa
            pairs.append((iou, i, j))
            if cov_gt >= coverage_threshold:
                gt_links.setdefault(i, set()).add(j)
            if cov_pred >= coverage_threshold:
                pred_links.setdefault(j, set()).add(i)
    matches = []
    used_gt, used_pred = set(), set()
    for iou, i, j in sorted(pairs, reverse=True):
        if iou < iou_threshold or i in used_gt or j in used_pred:
            continue
        matches.append((i, j, iou)); used_gt.add(i); used_pred.add(j)
    tp, n_gt, n_pred = len(matches), len(gt), len(pred)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_splits = sum(len(v) >= 2 for v in gt_links.values())
    over_merges = sum(len(v) >= 2 for v in pred_links.values())
    matched_instances = []
    length_errors = []
    width_errors = []
    for i, j, iou in sorted(matches):
        gt_record, _, gt_mask = gt[i]
        pred_record, _, pred_mask = pred[j]
        gt_geometry = measure_mask(gt_mask, 1.0, endpoint_exclusion_um=10.0)
        pred_geometry = measure_mask(pred_mask, 1.0, endpoint_exclusion_um=10.0)
        length_error = (
            abs(pred_geometry.length_um - gt_geometry.length_um) / gt_geometry.length_um
            if gt_geometry.length_um > 0 else None)
        width_error = (
            abs(pred_geometry.width_median_um - gt_geometry.width_median_um)
            / gt_geometry.width_median_um if gt_geometry.width_median_um > 0 else None)
        if length_error is not None:
            length_errors.append(length_error)
        if width_error is not None:
            width_errors.append(width_error)
        matched_instances.append({
            "ground_truth_id": gt_record.id,
            "prediction_id": pred_record.id,
            "iou": iou,
            "ground_truth_length_px": gt_geometry.length_um,
            "prediction_length_px": pred_geometry.length_um,
            "length_absolute_percentage_error": length_error,
            "ground_truth_width_px": gt_geometry.width_median_um,
            "prediction_width_px": pred_geometry.width_median_um,
            "width_absolute_percentage_error": width_error,
        })
    length_mdape = float(np.median(length_errors)) if length_errors else None
    width_mdape = float(np.median(width_errors)) if width_errors else None
    measurement_values = length_errors + width_errors
    return {
        "n_gt": n_gt, "n_pred": n_pred, "tp": tp, "precision": precision,
        "recall": recall, "f1": f1, "precision_weighted_score": (2 * precision + recall) / 3,
        "automatic_coverage": recall,
        "mean_matched_iou": float(np.mean([x[2] for x in matches])) if matches else 0.0,
        "false_split_count": false_splits,
        "false_split_rate": false_splits / n_gt if n_gt else 0.0,
        "over_merge_count": over_merges,
        "over_merge_rate": over_merges / n_pred if n_pred else 0.0,
        "length_mdape": length_mdape,
        "width_mdape": width_mdape,
        "measurement_mdape": (
            float(np.median(measurement_values)) if measurement_values else None),
        "matched_instances": matched_instances,
        "iou_threshold": iou_threshold, "coverage_threshold": coverage_threshold,
    }


def _aggregate_point_metrics(rows: list[dict]) -> dict:
    n_gt = sum(row["n_gt"] for row in rows)
    n_pred = sum(row["n_pred"] for row in rows)
    tp = sum(row["tp"] for row in rows)
    false_splits = sum(row["false_split_count"] for row in rows)
    over_merges = sum(row["over_merge_count"] for row in rows)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    matched = sum(row["tp"] * row["mean_matched_iou"] for row in rows)
    length_errors = [
        match["length_absolute_percentage_error"]
        for row in rows for match in row.get("matched_instances", [])
        if match["length_absolute_percentage_error"] is not None]
    width_errors = [
        match["width_absolute_percentage_error"]
        for row in rows for match in row.get("matched_instances", [])
        if match["width_absolute_percentage_error"] is not None]
    return {
        "n_fields": len(rows), "n_gt": n_gt, "n_pred": n_pred, "tp": tp,
        "precision": precision, "recall": recall,
        "automatic_coverage": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision_weighted_score": (2 * precision + recall) / 3,
        "mean_matched_iou": matched / tp if tp else 0.0,
        "false_split_count": false_splits,
        "false_split_rate": false_splits / n_gt if n_gt else 0.0,
        "over_merge_count": over_merges,
        "over_merge_rate": over_merges / n_pred if n_pred else 0.0,
        "length_mdape": float(np.median(length_errors)) if length_errors else None,
        "width_mdape": float(np.median(width_errors)) if width_errors else None,
        "measurement_mdape": (
            float(np.median(length_errors + width_errors))
            if length_errors or width_errors else None),
    }


_INTERVAL_METRICS = (
    "precision", "recall", "f1", "precision_weighted_score",
    "mean_matched_iou", "false_split_rate", "over_merge_rate",
    "length_mdape", "width_mdape", "measurement_mdape",
)


def _aggregate_metrics(rows: list[dict], *, bootstrap_resamples: int = 2_000,
                       seed: int = 20260722) -> dict:
    """Aggregate fields and quantify uncertainty by resampling whole fields.

    Object-level resampling would treat nested myotubes as independent and yield
    falsely narrow intervals.  The bootstrap therefore resamples complete held-out
    evaluation units (fields/wells).  These intervals quantify internal model-
    evaluation variability; they are not biological treatment-effect intervals.
    """
    result = _aggregate_point_metrics(rows)
    macro = {}
    for key in _INTERVAL_METRICS:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        macro[key] = {
            "mean_across_fields": float(np.mean(values)) if values else None,
            "median_across_fields": float(np.median(values)) if values else None,
            "n_fields": len(values),
        }
    result["macro_by_field"] = macro
    result["confidence_intervals"] = {}
    result["statistical_design"] = {
        "independent_evaluation_unit": "whole held-out field/well",
        "n_independent_evaluation_units": len(rows),
        "objects_resampled_as_independent": False,
        "bootstrap_resamples": bootstrap_resamples if len(rows) >= 3 else 0,
        "seed": seed,
        "scope": "internal model-evaluation variability; not biological treatment inference",
    }
    if len(rows) < 3:
        result["statistical_design"]["warning"] = (
            "fewer than 3 independent evaluation units; point estimates are descriptive only"
        )
        return result

    rng = np.random.default_rng(seed)
    samples = {key: [] for key in _INTERVAL_METRICS}
    for _ in range(bootstrap_resamples):
        indices = rng.integers(0, len(rows), size=len(rows))
        estimate = _aggregate_point_metrics([rows[index] for index in indices])
        for key in _INTERVAL_METRICS:
            if estimate.get(key) is not None:
                samples[key].append(estimate[key])
    for key, values in samples.items():
        if not values:
            continue
        lower, upper = np.quantile(values, [0.025, 0.975])
        result["confidence_intervals"][key] = {
            "lower": float(lower), "upper": float(upper),
            "confidence": 0.95,
            "method": "percentile bootstrap over whole held-out fields/wells",
        }
    return result


def benchmark_manifest(path: str | Path) -> dict:
    """Benchmark a candidate across plate and density strata from one locked manifest."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for sample in payload.get("samples", []):
        if "plate" not in sample or "density" not in sample:
            raise ValueError("every benchmark sample requires plate and density")
        gt, pred = Path(sample["ground_truth"]), Path(sample["prediction"])
        if not gt.is_absolute():
            gt = manifest_path.parent / gt
        if not pred.is_absolute():
            pred = manifest_path.parent / pred
        metrics = benchmark_instances(gt, pred)
        image_id = sample.get("image_id", gt.stem)
        rows.append({"image_id": image_id,
                     "evaluation_unit_id": str(sample.get("evaluation_unit_id", image_id)),
                     "biological_replicate_id": (
                         str(sample["biological_replicate_id"])
                         if sample.get("biological_replicate_id") is not None else None),
                     "plate": str(sample["plate"]), "density": str(sample["density"]),
                     **metrics})
    if not rows:
        raise ValueError("benchmark manifest has no samples")
    evaluation_ids = [row["evaluation_unit_id"] for row in rows]
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("benchmark evaluation_unit_id values must be unique")
    overall = _aggregate_metrics(rows)
    strata = []
    for kind in ("plate", "density"):
        for value in sorted({row[kind] for row in rows}):
            metric = _aggregate_metrics([row for row in rows if row[kind] == value])
            collapsed = (metric["precision"] < 0.85 or metric["over_merge_rate"] > 0.05 or
                         metric["recall"] < max(0.0, overall["recall"] - 0.20))
            strata.append({"kind": kind, "value": value, "collapsed": collapsed, **metric})
    present_plates = {row["plate"] for row in rows}
    present_densities = {row["density"] for row in rows}
    missing_strata = [
        *[{"kind": "plate", "value": str(value)}
          for value in payload.get("required_plates", []) if str(value) not in present_plates],
        *[{"kind": "density", "value": str(value)}
          for value in payload.get("required_densities", []) if str(value) not in present_densities],
    ]
    biological_ids = sorted({row["biological_replicate_id"] for row in rows
                             if row["biological_replicate_id"] is not None})
    overall["statistical_design"].update({
        "declared_experimental_unit": payload.get("experimental_unit"),
        "biological_replicate_ids": biological_ids,
        "n_biological_replicates": len(biological_ids),
        "biological_inference_supported": (
            bool(payload.get("experimental_unit")) and len(biological_ids) >= 3),
    })
    return {"model": payload.get("model", manifest_path.stem), **overall,
            "plate_collapse": any(row["collapsed"] for row in strata) or bool(missing_strata),
            "missing_required_strata": missing_strata,
            "strata": strata, "fields": rows}


def select_model(metric_files: list[str | Path]) -> dict:
    candidates = []
    for path in metric_files:
        metrics = json.loads(Path(path).read_text(encoding="utf-8"))
        name = metrics.get("model", Path(path).stem)
        stratum_collapse = any(
            row.get("precision", 0.0) < 0.85 or row.get("over_merge_rate", 1.0) > 0.05 or
            row.get("collapsed", False) for row in metrics.get("strata", [])
        )
        intervals = metrics.get("confidence_intervals", {})
        precision_evidence = intervals.get("precision", {}).get(
            "lower", metrics.get("precision", 0.0))
        over_merge_evidence = intervals.get("over_merge_rate", {}).get(
            "upper", metrics.get("over_merge_rate", 1.0))
        disqualified = (precision_evidence < 0.85 or
                        over_merge_evidence > 0.05 or
                        metrics.get("plate_collapse", False) or stratum_collapse)
        candidates.append({
            "model": name, "disqualified": disqualified,
            "selection_bounds": {
                "precision_lower_or_point": precision_evidence,
                "over_merge_upper_or_point": over_merge_evidence,
                "used_confidence_bounds": bool(intervals),
            },
            **metrics,
        })
    eligible = [x for x in candidates if not x["disqualified"]]
    winner = max(
        eligible,
        key=lambda x: (
            x.get("precision_weighted_score", 0.0),
            -(x.get("measurement_mdape")
              if x.get("measurement_mdape") is not None else float("inf")),
            x.get("precision", 0.0),
            x["model"],
        ),
    ) if eligible else None
    return {"winner": winner["model"] if winner else None,
            "decision": "automatic_candidate" if winner else "manual_qc_only",
            "candidates": candidates}


RELEASE_GATES = {
    "precision": (">=", 0.95), "over_merge_rate": ("<=", 0.02),
    "false_split_rate_auto": ("<=", 0.05), "false_split_rate_reviewed": ("<=", 0.02),
    "length_mdape": ("<=", 0.10), "width_mdape": ("<=", 0.10),
    "total_nuclei_error": ("<=", 0.03), "nucleus_assignment_accuracy": (">=", 0.95),
    "conversion_efficiency_abs_error": ("<=", 0.03),
}


def check_release(metrics: dict) -> dict:
    results = {}
    intervals = metrics.get("confidence_intervals", {})
    for key, (operator, threshold) in RELEASE_GATES.items():
        value = metrics.get(key)
        interval = intervals.get(key)
        conservative_bound = None
        if interval:
            conservative_bound = interval.get("lower" if operator == ">=" else "upper")
        passed = (
            value is not None and conservative_bound is not None and
            (conservative_bound >= threshold if operator == ">="
             else conservative_bound <= threshold)
        )
        results[key] = {
            "value": value, "operator": operator, "threshold": threshold,
            "confidence_interval": interval,
            "conservative_bound": conservative_bound,
            "passed": bool(passed),
            "rule": "the adverse 95% confidence bound, not only the point estimate, must pass",
        }

    design = metrics.get("statistical_design", {})
    declared_unit = design.get("declared_experimental_unit") or design.get("experimental_unit")
    n_biological = int(design.get("n_biological_replicates", 0) or 0)
    objects_independent = design.get("objects_resampled_as_independent")
    statistical_evidence = {
        "experimental_unit_declared": bool(declared_unit),
        "experimental_unit": declared_unit,
        "at_least_3_biological_replicates": n_biological >= 3,
        "n_biological_replicates": n_biological,
        "objects_not_treated_as_independent_n": objects_independent is False,
        "prospective_design_locked_before_unblinding": bool(
            metrics.get("prospective_design_locked_before_unblinding", False)),
        "all_release_metrics_have_confidence_intervals": all(
            key in intervals and intervals[key].get("lower") is not None and
            intervals[key].get("upper") is not None for key in RELEASE_GATES),
    }
    statistical_evidence["passed"] = all(
        value for key, value in statistical_evidence.items()
        if key not in {"experimental_unit", "n_biological_replicates"})
    all_passed = all(x["passed"] for x in results.values()) and statistical_evidence["passed"]
    return {"passed": all_passed,
            "release_mode": "authoritative_instance_metrics" if all_passed
                            else "field_metrics_and_manual_qc_only",
            "statistical_evidence": statistical_evidence,
            "gates": results}
