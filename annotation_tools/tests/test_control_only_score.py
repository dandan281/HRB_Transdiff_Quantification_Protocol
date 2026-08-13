"""Reading rules for the control-only safety round.

The round's whole value is that its estimator was fixed before anyone saw a
verdict. These tests pin the parts that would let it drift afterwards: refusing a
packet that never predeclared one, refusing a flag-enriched sample, weighting by
well size rather than averaging well rates, and never letting an unresolved
verdict count as a safe merge.
"""
from __future__ import annotations

import pytest

from annotation_tools.qc_review.control_only_score import (
    score_control_only, stratified_bootstrap, weighted_population_rate)


SPEC = {
    "strata": [
        {"well": "big", "accepted_merges_in_well": 300, "sampled": 10},
        {"well": "small", "accepted_merges_in_well": 100, "sampled": 10},
    ],
    "estimator": "well-size-weighted mean",
    "interval": "stratified bootstrap over wells, 500 resamples, seed 7",
    "unresolved_handling": "ambiguous_2d excluded from both numerator and denominator",
}


def _packet(verdicts_by_well, *, n_flagged=0, spec=SPEC):
    key, decisions = {}, {}
    n = 0
    for well, verdicts in verdicts_by_well.items():
        for verdict in verdicts:
            n += 1
            uid = f"uid_{n:03d}"
            key[uid] = {"well": well, "case_kind": "control",
                        "fragment_ids": [1, 2], "accepted_pairs": []}
            decisions[uid] = {"decision": verdict}
    return ({"batch_id": "b", "threshold": 0.9, "n_over_merge_cases": n_flagged,
             "control_only_round": spec, "key": key},
            {"batch_id": "b", "threshold": 0.9, "reviewer": "r", "decisions": decisions})


def test_rate_is_weighted_by_well_size_not_averaged_over_wells():
    # big well 0% wrong, small well 100% wrong. The mean of the well rates is 0.50;
    # the population rate is 100/400 = 0.25. Reporting 0.50 would trebble the small
    # well's influence, which is the mean-of-wells error this design exists to avoid.
    key, export = _packet({"big": ["same_myotube"] * 4,
                           "small": ["different_myotubes"] * 4})
    report = score_control_only(key, export)
    assert report["primary"]["population_over_merge_rate"] == 0.25


def test_unresolved_verdicts_leave_the_denominator_and_never_count_as_safe():
    key, export = _packet({"big": ["different_myotubes", "ambiguous_2d"],
                           "small": ["different_myotubes", "ambiguous_2d"]})
    report = score_control_only(key, export)
    assert report["counts"]["n_resolved"] == 2
    assert report["counts"]["n_unresolved_ambiguous_2d"] == 2
    # Both resolved verdicts were `different`, so the rate is 1.0 -- an unresolved
    # case pulling it down towards 0.5 would be counting it as a safe merge.
    assert report["primary"]["population_over_merge_rate"] == 1.0


def test_an_undecided_case_is_excluded_but_the_gap_is_disclosed():
    key, export = _packet({"big": ["different_myotubes", None],
                           "small": ["same_myotube"]})
    report = score_control_only(key, export)
    sens = report["sensitivity_to_excluded_cases"]
    assert report["counts"]["n_undecided"] == 1
    assert sens["undecided_handling_was_predeclared"] is False
    assert sens["rate_if_all_excluded_were_same_myotube"] <= \
        report["primary"]["population_over_merge_rate"] <= \
        sens["rate_if_all_excluded_were_different"]


def test_a_flag_enriched_packet_is_refused():
    key, export = _packet({"big": ["same_myotube"], "small": ["same_myotube"]},
                          n_flagged=3)
    with pytest.raises(SystemExit, match="flag-enriched"):
        score_control_only(key, export)


def test_a_packet_with_no_predeclared_estimator_is_refused():
    key, export = _packet({"big": ["same_myotube"], "small": ["same_myotube"]})
    del key["control_only_round"]
    with pytest.raises(SystemExit, match="no estimator was predeclared"):
        score_control_only(key, export)


def test_an_export_that_does_not_cover_the_packet_is_refused():
    key, export = _packet({"big": ["same_myotube"], "small": ["same_myotube"]})
    export["decisions"].popitem()
    with pytest.raises(SystemExit, match="absent from the export"):
        score_control_only(key, export)


def test_threshold_and_batch_mismatches_are_refused():
    key, export = _packet({"big": ["same_myotube"], "small": ["same_myotube"]})
    export["threshold"] = 0.95
    with pytest.raises(SystemExit, match="threshold mismatch"):
        score_control_only(key, export)


def test_bootstrap_interval_brackets_the_point_estimate():
    strata = [{"well": "a", "population": 300, "verdicts": [1, 1, 0, 1],
               "rate": 0.75},
              {"well": "b", "population": 100, "verdicts": [0, 0, 1, 0],
               "rate": 0.25}]
    point = weighted_population_rate(strata)
    interval = stratified_bootstrap(strata, resamples=2000, seed=7)
    assert interval["lower"] <= point <= interval["upper"]
    assert interval["resamples"] == 2000


def test_a_well_with_no_resolved_verdict_leaves_the_denominator_entirely():
    # Otherwise its population weight would silently be attributed to a rate it
    # never measured.
    strata = [{"well": "a", "population": 300, "verdicts": [1, 1], "rate": 1.0},
              {"well": "b", "population": 100, "verdicts": [], "rate": None}]
    assert weighted_population_rate(strata) == 1.0
