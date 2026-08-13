import json

import numpy as np
import pytest

from precision_myotube.benchmark import (benchmark_instances, benchmark_manifest,
                                         check_release, select_model)
from precision_myotube.schema import InstanceRecord, InstanceSet, encode_rle


def _set(path, reviewed=True):
    mask = np.zeros((30, 40), bool); mask[5:15, 4:28] = True
    data = InstanceSet(mask.shape, "x", [
        InstanceRecord("m1", "complete", encode_rle(mask), reviewed=reviewed)
    ])
    data.save(path)


def test_perfect_instance_benchmark(tmp_path):
    gt, pred = tmp_path / "gt.json", tmp_path / "pred.json"
    _set(gt); _set(pred, reviewed=False)
    metrics = benchmark_instances(gt, pred)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["automatic_coverage"] == 1.0
    assert metrics["over_merge_rate"] == 0.0
    assert metrics["length_mdape"] == 0.0
    assert metrics["width_mdape"] == 0.0
    assert metrics["matched_instances"][0]["ground_truth_id"] == "m1"


def test_model_gate_and_release_fallback(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"model": "bad", "precision": 0.5,
                               "over_merge_rate": 0.2, "precision_weighted_score": 0.8}))
    assert select_model([bad])["decision"] == "manual_qc_only"
    release = check_release({"precision": 0.99})
    assert not release["passed"]
    assert release["release_mode"] == "field_metrics_and_manual_qc_only"


def test_benchmark_manifest_reports_plate_and_density_strata(tmp_path):
    gt, pred = tmp_path / "gt.json", tmp_path / "pred.json"
    _set(gt); _set(pred, reviewed=False)
    manifest = tmp_path / "benchmark.json"
    manifest.write_text(json.dumps({
        "model": "perfect",
        "required_plates": ["PLATE_26"],
        "required_densities": ["dense"],
        "samples": [{"image_id": "field", "plate": "PLATE_26", "density": "dense",
                     "ground_truth": "gt.json", "prediction": "pred.json"}],
    }))
    metrics = benchmark_manifest(manifest)
    assert metrics["precision"] == 1.0
    assert not metrics["plate_collapse"]
    assert {(row["kind"], row["value"]) for row in metrics["strata"]} == {
        ("plate", "PLATE_26"), ("density", "dense")}


def test_missing_required_stratum_fails_closed(tmp_path):
    gt, pred = tmp_path / "gt.json", tmp_path / "pred.json"
    _set(gt); _set(pred, reviewed=False)
    manifest = tmp_path / "benchmark.json"
    manifest.write_text(json.dumps({
        "model": "incomplete",
        "required_plates": ["PLATE_26", "PLATE_27"],
        "required_densities": ["dense", "sparse"],
        "samples": [{"plate": "PLATE_26", "density": "dense",
                     "ground_truth": "gt.json", "prediction": "pred.json"}],
    }))
    metrics = benchmark_manifest(manifest)
    assert metrics["plate_collapse"]
    assert metrics["missing_required_strata"] == [
        {"kind": "plate", "value": "PLATE_27"},
        {"kind": "density", "value": "sparse"},
    ]


def test_manifest_bootstraps_whole_evaluation_units(tmp_path):
    samples = []
    for index in range(3):
        gt, pred = tmp_path / f"gt{index}.json", tmp_path / f"pred{index}.json"
        _set(gt); _set(pred, reviewed=False)
        samples.append({
            "image_id": f"field{index}", "evaluation_unit_id": f"well{index}",
            "biological_replicate_id": f"plate{index}",
            "plate": f"PLATE_{index}", "density": "dense",
            "ground_truth": gt.name, "prediction": pred.name,
        })
    manifest = tmp_path / "benchmark.json"
    manifest.write_text(json.dumps({
        "model": "perfect", "experimental_unit": "independent differentiation plate",
        "samples": samples,
    }))
    metrics = benchmark_manifest(manifest)
    assert metrics["confidence_intervals"]["precision"] == {
        "lower": 1.0, "upper": 1.0, "confidence": 0.95,
        "method": "percentile bootstrap over whole held-out fields/wells",
    }
    design = metrics["statistical_design"]
    assert design["n_independent_evaluation_units"] == 3
    assert design["n_biological_replicates"] == 3
    assert design["biological_inference_supported"]
    assert not design["objects_resampled_as_independent"]


def test_release_requires_adverse_confidence_bounds_and_design_lock():
    point = {
        "precision": 0.97, "over_merge_rate": 0.01,
        "false_split_rate_auto": 0.03, "false_split_rate_reviewed": 0.01,
        "length_mdape": 0.05, "width_mdape": 0.05,
        "total_nuclei_error": 0.01, "nucleus_assignment_accuracy": 0.98,
        "conversion_efficiency_abs_error": 0.01,
    }
    intervals = {
        key: {"lower": value - 0.005, "upper": value + 0.005, "confidence": 0.95}
        for key, value in point.items()
    }
    metrics = {
        **point, "confidence_intervals": intervals,
        "statistical_design": {
            "declared_experimental_unit": "independent differentiation plate",
            "n_biological_replicates": 3,
            "objects_resampled_as_independent": False,
        },
        "prospective_design_locked_before_unblinding": True,
    }
    release = check_release(metrics)
    assert release["passed"]
    assert release["gates"]["precision"]["conservative_bound"] == pytest.approx(0.965)

    metrics["confidence_intervals"]["precision"]["lower"] = 0.94
    release = check_release(metrics)
    assert not release["passed"]
    assert not release["gates"]["precision"]["passed"]
