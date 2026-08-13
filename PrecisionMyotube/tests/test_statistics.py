import json

import pytest

from precision_myotube.statistics import (
    analyze_statistics_manifest, benjamini_hochberg, summarize_hierarchical,
    wilson_interval)


def test_technical_observations_are_not_counted_as_independent_n():
    records = [
        {"condition": "control", "plate": "p1", "well": "w1", "value": 0.0},
        {"condition": "control", "plate": "p1", "well": "w1", "value": 2.0},
        {"condition": "control", "plate": "p1", "well": "w2", "value": 3.0},
        {"condition": "control", "plate": "p2", "well": "w3", "value": 5.0},
    ]
    result = summarize_hierarchical(
        records, value_key="value", condition_key="condition",
        biological_unit_key="plate", technical_unit_key="well",
        bootstrap_resamples=100, seed=7)
    group = result["groups"]["control"]
    # w1 first collapses to 1; p1 then collapses equally over w1=1 and w2=3.
    assert group["biological_unit_values"] == {"p1": 2.0, "p2": 5.0}
    assert group["n_biological_units"] == 2
    assert result["n_raw_observations"] == 4
    assert result["n_technical_units"] == 3
    assert group["mean_confidence_interval"] is None
    assert not group["inference_eligible"]


def test_paired_effect_bootstraps_complete_biological_units_deterministically():
    records = []
    for plate, control, treated in (("p1", 1.0, 2.0), ("p2", 2.0, 4.0),
                                    ("p3", 3.0, 6.0), ("p4", 4.0, 8.0)):
        records.extend([
            {"condition": "control", "plate": plate, "well": plate + "c", "value": control},
            {"condition": "treated", "plate": plate, "well": plate + "t", "value": treated},
        ])
    kwargs = dict(
        value_key="value", condition_key="condition", biological_unit_key="plate",
        technical_unit_key="well",
        comparisons=[{"a": "control", "b": "treated", "paired": True}],
        bootstrap_resamples=500, seed=13)
    first = summarize_hierarchical(records, **kwargs)
    second = summarize_hierarchical(records, **kwargs)
    comparison = first["comparisons"][0]
    assert comparison["estimate"] == pytest.approx(2.5)
    assert comparison["paired_biological_unit_ids"] == ["p1", "p2", "p3", "p4"]
    assert comparison["confidence_interval"] == second["comparisons"][0]["confidence_interval"]
    assert comparison["p_value"] is None
    assert comparison["inference_eligible"]


def test_manifest_preserves_design_and_exclusion_declarations(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "analysis_id": "conversion_primary",
        "outcome_label": "conversion efficiency",
        "value_key": "value",
        "condition_key": "condition",
        "biological_unit_key": "batch",
        "technical_unit_key": "well",
        "pre_specified": True,
        "exclusion_rule": "exclude only integrity failures",
        "bootstrap_resamples": 10,
        "records": [
            {"condition": "control", "batch": "only_batch", "well": "w1", "value": 0.1}
        ],
    }))
    result = analyze_statistics_manifest(path)
    assert result["analysis_id"] == "conversion_primary"
    assert result["pre_specified"] is True
    assert result["exclusion_rule"] == "exclude only integrity failures"
    assert not result["groups"]["control"]["inference_eligible"]


def test_overlapping_biological_ids_require_paired_comparison():
    records = [
        {"condition": "control", "plate": "p1", "value": 1.0},
        {"condition": "treated", "plate": "p1", "value": 2.0},
    ]
    with pytest.raises(ValueError, match="declare paired=true"):
        summarize_hierarchical(
            records, value_key="value", condition_key="condition",
            biological_unit_key="plate",
            comparisons=[{"a": "control", "b": "treated", "paired": False}],
            bootstrap_resamples=10)


def test_wilson_interval_and_bh_adjustment():
    interval = wilson_interval(50, 100)
    assert interval["lower"] < 0.5 < interval["upper"]
    assert "not biological replicates" in interval["unit_warning"]
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.04, 0.04])


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_bh_rejects_invalid_p_values(bad):
    with pytest.raises(ValueError):
        benjamini_hochberg([bad])
