"""Guardrails for the control-only safety round.

This round exists to answer the one question the sparse-reference flag cannot:
what fraction of *ordinary* accepted merges join two different myotubes. Its
validity rests entirely on the sample being uniform over accepted merges, so the
tests that matter here are the ones that stop a non-uniform sample from being
dressed up as a population estimate.
"""
from __future__ import annotations

from annotation_tools.qc_review.cli import _control_only_spec


def _payload(**overrides):
    payload = {
        "wells": [
            {"well": "well_a", "n_accepted_merge_components": 100},
            {"well": "well_b", "n_accepted_merge_components": 20},
        ],
    }
    payload.update(overrides)
    return payload


def _key(counts):
    key = {}
    n = 0
    for well, count in counts.items():
        for _ in range(count):
            n += 1
            key[f"uid_{n:03d}"] = {"well": well}
    return key


def test_inclusion_probabilities_are_recorded_per_well():
    spec = _control_only_spec(_payload(), _key({"well_a": 10, "well_b": 10}))
    by_well = {s["well"]: s for s in spec["strata"]}
    assert by_well["well_a"]["inclusion_probability"] == 0.1
    assert by_well["well_b"]["inclusion_probability"] == 0.5
    assert spec["accepted_merges_across_six_wells"] == 120


def test_estimator_is_well_size_weighted_not_a_mean_of_well_rates():
    # Equal draws over unequal wells. Averaging the per-well rates would weight a
    # 20-merge well the same as a 100-merge one -- the mean-of-wells mistake the
    # linker recall reporting already made once. The declared estimator must say
    # weighted, and must not license reporting the unweighted mean instead.
    spec = _control_only_spec(_payload(), _key({"well_a": 10, "well_b": 10}))
    assert "weighted" in spec["estimator"]
    assert "never instead of it" in spec["estimator"]
    assert spec["design"].startswith("stratified")


def test_unresolved_verdicts_never_count_as_safe_merges():
    spec = _control_only_spec(_payload(), _key({"well_a": 10, "well_b": 10}))
    assert "excluded from both numerator and denominator" in spec["unresolved_handling"]
    assert "never" in spec["unresolved_handling"]


def test_the_threshold_is_declared_locked_against_this_round():
    spec = _control_only_spec(_payload(), _key({"well_a": 10, "well_b": 10}))
    assert "LOCKED" in spec["threshold_status"]
    assert "must not be used to select or tune it" in spec["threshold_status"]
    assert spec["prespecified_at_build_time"] is True


def test_a_well_that_was_not_sampled_is_visible_as_zero_not_dropped():
    # A silently missing stratum would make the weighted estimator quietly
    # renormalise over five wells while the report still says six.
    spec = _control_only_spec(_payload(), _key({"well_a": 10}))
    by_well = {s["well"]: s for s in spec["strata"]}
    assert set(by_well) == {"well_a", "well_b"}
    assert by_well["well_b"]["sampled"] == 0
    assert by_well["well_b"]["inclusion_probability"] == 0.0
