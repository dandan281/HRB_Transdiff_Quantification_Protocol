"""TA03b tests — the nine groups the request enumerates, on synthetic data only.

Nothing here reads the plate. The point is to pin the contract, and a test that needs
the real masks could not run the fail-closed cases at all.
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from tier_a_audit import selector as sel
from tier_a_audit import scorer as sc


# --------------------------------------------------------------- helpers / fixtures


def rec(well="w1", field="f1", nid=1, ratio=1.0, row=10.0, col=10.0):
    return sel.NucleusRecord(
        well=well, field=field, nucleus_id=nid, centroid_row=row, centroid_col=col,
        ring_intensity=ratio * 440.0, ring_ratio=ratio, call_2d=ratio >= 1.0,
        stratum=sel.stratum_of(ratio), valid_by_area=True)


def frame(n_per_stratum=20, wells=("w1",), fields=("f1",)):
    out, nid = [], 0
    mids = {"lt_0.5": 0.25, "0.5_0.8": 0.65, "0.8_1.0": 0.9,
            "1.0_1.25": 1.1, "1.25_2.0": 1.5, "ge_2.0": 3.0}
    for w in wells:
        for f in fields:
            for _, r in mids.items():
                for _ in range(n_per_stratum):
                    nid += 1
                    out.append(rec(w, f, nid, r, row=nid % 100, col=(nid * 7) % 100))
    return out


def matched_rows(n_field=2, n_per=30, sens=0.95, spec=0.98, seed=0):
    """Synthetic matched set with a known confusion matrix."""
    rng = np.random.Generator(np.random.PCG64(seed))
    rows, nid = [], 0
    for f in range(n_field):
        for i in range(n_per):
            nid += 1
            truth = bool(i % 2)
            call = (rng.random() < sens) if truth else (rng.random() > spec)
            rows.append({"well": "w1", "field": f"f{f}", "nucleus_id": nid,
                         "inclusion_probability": 0.5, "stratum": "1.0_1.25",
                         "call_2d": bool(call), "truth_3d": truth,
                         "reference_id": f"r{nid}", "distance_px": 1.0,
                         "mask_overlap": 0.9})
    return rows


MIX = {"lt_0.5": 0.2, "0.5_0.8": 0.2, "0.8_1.0": 0.2,
       "1.0_1.25": 0.2, "1.25_2.0": 0.1, "ge_2.0": 0.1}
XFORM = {"basis": "DAPI nucleus channel", "model": "affine",
         "residual_px": 0.8, "source_hashes": {"overview": "abc"}}


# ----------------------------------------------- 1. deterministic selection + hashes


def test_selection_is_deterministic_across_runs():
    f = frame()
    a = sel.select(f, 5, seed=42)
    b = sel.select(f, 5, seed=42)
    assert [s.record.key for s in a] == [s.record.key for s in b]


def test_selection_changes_with_seed():
    f = frame()
    a = sel.select(f, 5, seed=1)
    b = sel.select(f, 5, seed=2)
    assert [s.record.key for s in a] != [s.record.key for s in b]


def test_selection_is_independent_of_input_ordering():
    f = frame()
    shuffled = list(reversed(f))
    assert ([s.record.key for s in sel.select(f, 5, seed=7)]
            == [s.record.key for s in sel.select(shuffled, 5, seed=7)])


def test_changing_one_stratum_target_does_not_reshuffle_others():
    """Per-cell seeding is the point: cells must be independent streams."""
    f = frame()
    base = sel.select(f, {"lt_0.5": 5, "1.0_1.25": 5}, seed=3)
    more = sel.select(f, {"lt_0.5": 5, "1.0_1.25": 9}, seed=3)
    keep = lambda xs: [s.record.key for s in xs if s.record.stratum == "lt_0.5"]
    assert keep(base) == keep(more)


def test_manifest_hash_is_stable():
    f = frame()
    s = sel.select(f, 4, seed=5)
    m1 = sel.build_manifest(s, {"w1": {}}, seed=5, threshold=440.0, source_hashes={})
    m2 = sel.build_manifest(s, {"w1": {}}, seed=5, threshold=440.0, source_hashes={})
    assert m1["manifest_sha256"] == m2["manifest_sha256"]


# ------------------------------------- 2. inclusion probabilities + post-strat weights


def test_inclusion_probability_is_taken_over_frame():
    f = frame(n_per_stratum=20)
    s = sel.select(f, {"0.8_1.0": 5}, seed=0)
    assert len(s) == 5
    assert all(abs(x.inclusion_probability - 0.25) < 1e-12 for x in s)


def test_exhausted_stratum_takes_all_at_probability_one():
    f = frame(n_per_stratum=3)
    s = sel.select(f, {"ge_2.0": 10}, seed=0)
    assert len(s) == 3
    assert all(x.inclusion_probability == 1.0 for x in s)
    assert all("frame_exhausted" in x.selection_reason for x in s)


def test_design_weights_invert_inclusion_probability():
    rows = [{"inclusion_probability": 0.25, "stratum": "1.0_1.25"},
            {"inclusion_probability": 0.5, "stratum": "1.0_1.25"}]
    assert sc.design_weights(rows, None) == [4.0, 2.0]


def test_post_stratification_matches_target_mixture():
    rows = ([{"inclusion_probability": 0.5, "stratum": "lt_0.5"}] * 10
            + [{"inclusion_probability": 0.5, "stratum": "ge_2.0"}] * 2)
    mix = {"lt_0.5": 0.3, "ge_2.0": 0.7}
    w = sc.design_weights(rows, mix)
    assert abs(sum(w[:10]) - 0.3) < 1e-12
    assert abs(sum(w[10:]) - 0.7) < 1e-12


def test_mixture_must_sum_to_one():
    rows = [{"inclusion_probability": 0.5, "stratum": "lt_0.5"}]
    with pytest.raises(sc.ScoringError, match="sum to 1.0"):
        sc.design_weights(rows, {"lt_0.5": 0.9})


# ------------------------------------------------------- 3. boundary values at edges


@pytest.mark.parametrize("ratio,expected", [
    (0.4999, "lt_0.5"), (0.5, "0.5_0.8"), (0.7999, "0.5_0.8"), (0.8, "0.8_1.0"),
    (0.9999, "0.8_1.0"), (1.0, "1.0_1.25"), (1.2499, "1.0_1.25"), (1.25, "1.25_2.0"),
    (1.9999, "1.25_2.0"), (2.0, "ge_2.0"), (1e9, "ge_2.0"), (0.0, "lt_0.5"),
])
def test_stratum_edges_are_half_open_upward(ratio, expected):
    assert sel.stratum_of(ratio) == expected


def test_ratio_of_exactly_one_is_a_positive_call_stratum():
    """A ratio of 1.0 is above threshold; it must not land in the negative band."""
    assert sel.stratum_of(1.0) == "1.0_1.25"


def test_nan_ratio_is_an_error_not_a_bucket():
    with pytest.raises(sel.SelectionError, match="NaN"):
        sel.stratum_of(float("nan"))


# ------------------------------- 4. duplicate / overlap / hash / missingness closure


def test_duplicate_nucleus_is_rejected():
    f = [rec(nid=1), rec(nid=1)]
    with pytest.raises(sel.SelectionError, match="duplicate"):
        sel.select(f, 1, seed=0)


def test_out_of_frame_centroid_is_rejected():
    f = [rec(nid=1, row=5000.0, col=1.0)]
    with pytest.raises(sel.SelectionError, match="out-of-frame"):
        sel.select(f, 1, seed=0, image_shape={("w1", "f1"): (100, 100)})


def test_unknown_stratum_name_is_rejected():
    with pytest.raises(sel.SelectionError, match="unknown stratum"):
        sel.select(frame(), {"not_a_stratum": 3}, seed=0)


def test_write_refuses_to_overwrite_existing_selection(tmp_path):
    s = sel.select(frame(), 2, seed=0)
    m = sel.build_manifest(s, {}, seed=0, threshold=440.0, source_hashes={})
    sel.write_selection(tmp_path / "run", s, m)
    with pytest.raises(sel.SelectionError, match="append-only"):
        sel.write_selection(tmp_path / "run", s, m)


def test_scorer_rejects_reused_reference_nucleus():
    rows = matched_rows(n_field=2, n_per=4)
    rows[1]["reference_id"] = rows[0]["reference_id"]
    with pytest.raises(sc.ScoringError, match="one-to-one"):
        sc.score({"manifest_sha256": "x", "n_selected": 8}, rows, [],
                 population_mixture={"1.0_1.25": 1.0}, transform_record=XFORM)


def test_scorer_requires_nucleus_channel_transform():
    bad = {**XFORM, "basis": "Desmin ridge overlay"}
    with pytest.raises(sc.ScoringError, match="not a nucleus channel"):
        sc.score({"manifest_sha256": "x", "n_selected": 8}, matched_rows(), [],
                 population_mixture={"1.0_1.25": 1.0}, transform_record=bad)


def test_scorer_requires_transform_provenance():
    for missing in ("model", "residual_px", "source_hashes"):
        bad = {k: v for k, v in XFORM.items() if k != missing}
        with pytest.raises(sc.ScoringError, match=missing):
            sc.score({"manifest_sha256": "x", "n_selected": 8}, matched_rows(), [],
                     population_mixture={"1.0_1.25": 1.0}, transform_record=bad)


def test_bad_inclusion_probability_is_rejected():
    with pytest.raises(sc.ScoringError, match="inclusion probability"):
        sc.design_weights([{"inclusion_probability": 0.0, "stratum": "lt_0.5",
                            "well": "w", "field": "f", "nucleus_id": 1}], None)


# ------------------------------------------- 5. calibration / validation leakage stop


def test_calibration_validation_overlap_is_rejected():
    rows = matched_rows(n_field=2, n_per=5)
    cal = {(rows[0]["well"], rows[0]["field"], rows[0]["nucleus_id"])}
    with pytest.raises(sc.ScoringError, match="calibration/validation overlap"):
        sc.score({"manifest_sha256": "x", "n_selected": 10}, rows, [],
                 population_mixture={"1.0_1.25": 1.0}, transform_record=XFORM,
                 calibration_ids=cal)


def test_disjoint_calibration_is_accepted():
    rows = matched_rows(n_field=2, n_per=5)
    cal = {("w1", "f9", 999)}
    r = sc.score({"manifest_sha256": "x", "n_selected": 10}, rows, [],
                 population_mixture={"1.0_1.25": 1.0}, transform_record=XFORM,
                 calibration_ids=cal, n_boot=50)
    assert r["n_matched"] == 10


# ------------------------------------------ 6. clustered rather than object resampling


def test_bootstrap_requires_at_least_two_fields():
    rows = matched_rows(n_field=1, n_per=20)
    with pytest.raises(sc.ScoringError, match="needs >=2 fields"):
        sc.field_cluster_bootstrap(rows, None, n_boot=10, seed=0)


def test_bootstrap_resamples_whole_fields():
    """A field must enter or leave as a block, so per-field homogeneity shows up."""
    rows = []
    for f, truth in (("f0", True), ("f1", False)):
        for i in range(20):
            rows.append({"well": "w", "field": f, "nucleus_id": len(rows) + 1,
                         "inclusion_probability": 1.0, "stratum": "1.0_1.25",
                         "call_2d": truth, "truth_3d": truth,
                         "reference_id": f"r{len(rows)}"})
    ci = sc.field_cluster_bootstrap(rows, None, n_boot=400, seed=0)
    lo, hi = ci["sensitivity"]
    # f1 has no positives; draws that pick only f1 have undefined sensitivity, and draws
    # that pick only f0 give exactly 1.0. A nucleus-level bootstrap could not do this.
    assert hi == 1.0 and lo == 1.0


def test_bootstrap_is_deterministic():
    rows = matched_rows(n_field=3, n_per=10)
    a = sc.field_cluster_bootstrap(rows, None, n_boot=100, seed=11)
    b = sc.field_cluster_bootstrap(rows, None, n_boot=100, seed=11)
    assert a == b


# ------------------------------- 7. known synthetic confusion matrices + weighted metrics


def test_known_confusion_matrix_gives_exact_metrics():
    cells = {"tp": 90.0, "fn": 10.0, "tn": 95.0, "fp": 5.0}
    m = sc.metrics_from_cells(cells)
    assert abs(m["sensitivity"] - 0.90) < 1e-12
    assert abs(m["specificity"] - 0.95) < 1e-12
    assert abs(m["ppv"] - 90 / 95) < 1e-12
    assert abs(m["npv"] - 95 / 105) < 1e-12
    assert abs(m["fp_inflation"] - (1 - 90 / 95)) < 1e-12


def test_empty_denominator_is_none_not_zero():
    m = sc.metrics_from_cells({"tp": 0.0, "fn": 0.0, "tn": 5.0, "fp": 0.0})
    assert m["sensitivity"] is None and m["specificity"] == 1.0


def test_weighting_changes_the_estimate_when_strata_are_oversampled():
    """Boundary oversampling must be undone by the weights, or the estimate is biased."""
    rows = ([{"well": "w", "field": "f", "nucleus_id": i, "inclusion_probability": 1.0,
              "stratum": "0.8_1.0", "call_2d": False, "truth_3d": True,
              "reference_id": f"a{i}"} for i in range(50)]
            + [{"well": "w", "field": "f", "nucleus_id": 100 + i,
                "inclusion_probability": 0.1, "stratum": "ge_2.0", "call_2d": True,
                "truth_3d": True, "reference_id": f"b{i}"} for i in range(10)])
    unw = sc.metrics_from_cells(sc._cells(rows, [1.0] * len(rows)))
    w = sc.metrics_from_cells(sc._cells(rows, sc.design_weights(rows, None)))
    assert unw["sensitivity"] < w["sensitivity"]


# ------------------------------------------------------- 8. adverse-bound gate decisions


def test_gates_pass_only_on_the_adverse_bound_not_the_point():
    point = {"sensitivity": 0.99, "specificity": 0.99, "fp_inflation": 0.01}
    ci = {"sensitivity": (0.80, 1.0), "specificity": (0.99, 1.0), "fp_inflation": (0.0, 0.02)}
    g = sc.evaluate_gates(point, ci, None)
    assert g["sensitivity"]["passed"] is False       # lower bound 0.80 < 0.90
    assert g["specificity"]["passed"] is True
    assert g["_co_primary_all_passed"] is False


def test_all_co_primary_gates_can_pass():
    point = {"sensitivity": 0.97, "specificity": 0.98, "fp_inflation": 0.03}
    ci = {"sensitivity": (0.94, 0.99), "specificity": (0.96, 0.99), "fp_inflation": (0.01, 0.06)}
    g = sc.evaluate_gates(point, ci, None)
    assert g["_co_primary_all_passed"] is True
    assert g["_all_passed"] is False                  # negative control still missing


def test_missing_negative_control_fails_rather_than_passes():
    g = sc.evaluate_gates({}, {}, None)
    assert g["negative_control_fpr"]["passed"] is False
    assert g["negative_control_fpr"]["status"] == "not_evaluable"


def test_gate_thresholds_are_the_ratified_ones():
    assert sc.GATES["specificity"]["threshold"] == 0.95
    assert sc.GATES["sensitivity"]["threshold"] == 0.90
    assert sc.GATES["fp_inflation"]["threshold"] == 0.10
    assert sc.GATES["negative_control_fpr"]["threshold"] == 0.05


# --------------------------------------------- 9. read-only guard over Conversion_Efficiency


def test_read_only_guard_passes_when_nothing_changes(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    with sel.read_only_guard(tmp_path):
        (tmp_path / "a.txt").read_text()


def test_read_only_guard_detects_a_write(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(sel.SelectionError, match="read-only guard tripped"):
        with sel.read_only_guard(tmp_path):
            (tmp_path / "b.txt").write_text("new")


def test_read_only_guard_detects_a_modification(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x")
    with pytest.raises(sel.SelectionError, match="read-only guard tripped"):
        with sel.read_only_guard(tmp_path):
            p.write_text("much longer content")


def test_read_only_guard_detects_a_deletion(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x")
    with pytest.raises(sel.SelectionError, match="read-only guard tripped"):
        with sel.read_only_guard(tmp_path):
            p.unlink()


# ------------------------------------------------------------------ matching contract


def test_matching_is_one_to_one_and_best_first():
    selected = [{"well": "w", "field": "f", "nucleus_id": 1, "inclusion_probability": 1.0,
                 "stratum": "1.0_1.25", "call_2d": True},
                {"well": "w", "field": "f", "nucleus_id": 2, "inclusion_probability": 1.0,
                 "stratum": "1.0_1.25", "call_2d": True}]
    cands = {("w", "f", 1): [{"reference_id": "R", "distance_px": 5.0,
                              "mask_overlap": 0.9, "truth_positive": True}],
             ("w", "f", 2): [{"reference_id": "R", "distance_px": 1.0,
                              "mask_overlap": 0.9, "truth_positive": True}]}
    out = sc.match_nuclei(selected, cands, sc.MatchRule(10.0, 0.5))
    assert len(out.matched) == 1
    assert out.matched[0]["nucleus_id"] == 2          # closer claim wins
    assert out.attrition[0]["reason"] == "duplicate_candidate"


def test_matching_rejects_desmin_in_candidates():
    selected = [{"well": "w", "field": "f", "nucleus_id": 1, "inclusion_probability": 1.0,
                 "stratum": "1.0_1.25", "call_2d": True}]
    cands = {("w", "f", 1): [{"reference_id": "R", "distance_px": 1.0, "mask_overlap": 0.9,
                              "truth_positive": True, "desmin_mean": 900.0}]}
    with pytest.raises(sc.ScoringError, match="Desmin"):
        sc.match_nuclei(selected, cands, sc.MatchRule(10.0, 0.5))


def test_unmatched_nuclei_are_retained_as_attrition_with_reasons():
    selected = [{"well": "w", "field": "f", "nucleus_id": i, "inclusion_probability": 1.0,
                 "stratum": "1.0_1.25", "call_2d": True} for i in (1, 2, 3)]
    cands = {("w", "f", 2): [{"reference_id": "R2", "distance_px": 99.0,
                              "mask_overlap": 0.9, "truth_positive": True}],
             ("w", "f", 3): [{"reference_id": "R3", "distance_px": 1.0,
                              "mask_overlap": 0.01, "truth_positive": True}]}
    out = sc.match_nuclei(selected, cands, sc.MatchRule(10.0, 0.5))
    reasons = {a["nucleus_id"]: a["reason"] for a in out.attrition}
    assert reasons == {1: "unmatched_no_candidate", 2: "unmatched_beyond_distance",
                       3: "unmatched_below_overlap"}
    assert not out.matched


def test_match_rule_validates():
    with pytest.raises(sc.ScoringError):
        sc.MatchRule(-1.0, 0.5).validate()
    with pytest.raises(sc.ScoringError):
        sc.MatchRule(5.0, 1.5).validate()


# ------------------------------------------------------------------ end-to-end report


def test_end_to_end_report_is_complete_and_hashable(tmp_path):
    rows = matched_rows(n_field=3, n_per=20, seed=1)
    att = [{"well": "w1", "field": "f0", "nucleus_id": 999,
            "reason": "unmatched_no_candidate", "detail": ""}]
    r = sc.score({"manifest_sha256": "abc", "n_selected": 61}, rows, att,
                 population_mixture={"1.0_1.25": 1.0}, transform_record=XFORM,
                 n_boot=200, seed=0)
    assert r["n_matched"] == 60
    assert r["attrition"]["unmatched_no_candidate"] == 1
    assert 0 < r["match_rate"] < 1
    assert set(r["by_stratum"]) == {"1.0_1.25"}
    assert len(r["by_field"]) == 3
    assert "report_sha256" in r
    paths = sc.write_report(tmp_path, r, rows, att)
    assert (tmp_path / "scoring_report.json").is_file()
    assert (tmp_path / "scoring_report.txt").is_file()
    txt = (tmp_path / "scoring_report.txt").read_text()
    assert "adverse-bound gates" in txt and "match rate" in txt


def test_scoring_refuses_an_empty_matched_set():
    with pytest.raises(sc.ScoringError, match="nothing to score"):
        sc.score({"manifest_sha256": "x", "n_selected": 0}, [], [],
                 population_mixture={"1.0_1.25": 1.0}, transform_record=XFORM)


# ------------------------------------------------- 10. cluster-aware sample-size planning


import math

from tier_a_audit import planning as pl


def test_design_effect_matches_the_standard_formula():
    assert pl.design_effect(1, 0.5) == 1.0            # singleton clusters: no inflation
    assert abs(pl.design_effect(25, 0.10) - (1 + 24 * 0.10)) < 1e-12


def test_design_effect_rejects_bad_icc():
    with pytest.raises(sc.ScoringError, match="icc"):
        pl.design_effect(10, 1.0)
    with pytest.raises(sc.ScoringError, match="icc"):
        pl.design_effect(10, -0.1)


def test_clustering_always_widens_the_interval():
    plain, _, _ = pl.analytic_half_width(6, 25, 0.0, 0.9)
    clustered, _, _ = pl.analytic_half_width(6, 25, 0.10, 0.9)
    assert clustered > plain


def test_half_width_is_widest_at_rate_one_half():
    at_half, _, _ = pl.analytic_half_width(6, 25, 0.05, 0.5)
    for r in (0.1, 0.3, 0.7, 0.9, 0.99):
        other, _, _ = pl.analytic_half_width(6, 25, 0.05, r)
        assert other <= at_half + 1e-12


def test_more_nuclei_per_field_never_needs_more_fields():
    prev = None
    for m in (10, 25, 50, 100):
        n = pl.minimum_fields(m, icc=0.05)
        assert n is not None
        if prev is not None:
            assert n <= prev
        prev = n


def test_higher_icc_requires_more_fields():
    assert pl.minimum_fields(25, icc=0.20) > pl.minimum_fields(25, icc=0.01)


def test_beta_params_recover_the_requested_icc():
    p, icc = 0.8, 0.1
    a, b = pl._beta_params(p, icc)
    assert abs(a / (a + b) - p) < 1e-9              # mean preserved
    assert abs(1.0 / (1.0 + a + b) - icc) < 1e-9    # rho = 1/(1+a+b)


def test_zero_icc_gives_a_degenerate_field_effect():
    a, b = pl._beta_params(0.9, 0.0)
    assert math.isinf(a) and math.isinf(b)


def test_simulation_refuses_a_single_field_design():
    with pytest.raises(sc.ScoringError, match=">= 2 fields"):
        pl.simulate_half_width(1, 25, 0.05, 0.9, n_rep=2, n_boot=10)


def test_simulation_is_deterministic():
    a = pl.simulate_half_width(4, 10, 0.05, 0.9, n_rep=5, n_boot=40, seed=3)
    b = pl.simulate_half_width(4, 10, 0.05, 0.9, n_rep=5, n_boot=40, seed=3)
    assert a == b


def test_simulated_half_width_shrinks_with_more_fields():
    few = pl.simulate_half_width(4, 10, 0.05, 0.9, n_rep=40, n_boot=80, seed=1)
    many = pl.simulate_half_width(16, 10, 0.05, 0.9, n_rep=40, n_boot=80, seed=1)
    assert many["median_half_width"] < few["median_half_width"]


def test_plan_report_round_trips(tmp_path):
    r = pl.build_plan_report([2, 4], [10], icc_grid=(0.05,), rate=0.9,
                             simulate_at=[(4, 10, 0.05)], n_rep=5, n_boot=40)
    assert r["analytic_grid"] and r["simulations"]
    pl.write_plan(tmp_path, r)
    txt = (tmp_path / "sample_size_plan.txt").read_text()
    assert "minimum fields required" in txt and "limitations" in txt


def test_target_half_width_is_the_ratified_ten_points():
    assert pl.TARGET_HALF_WIDTH == 0.10
