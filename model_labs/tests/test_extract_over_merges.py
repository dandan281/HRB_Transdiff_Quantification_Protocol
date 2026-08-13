"""Tests for the over-merge extraction that feeds the blinded hand review.

The extraction's job is to name the exact predictions the benchmark flagged, so
the two properties worth pinning are (a) it implements the benchmark's own
over-merge rule rather than an approximation of it, and (b) the locked threshold
and the published counts cannot drift silently -- if either moves, the review
packet is auditing something other than the measurement it claims to audit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from classical.extract_over_merges import (COVERAGE_THRESHOLD, LOCKED_THRESHOLD,
                                           PUBLISHED_OVER_MERGES, find_over_merges,
                                           merge_components, relabel)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "model_labs/classical/_runs/linker_instance_v1.json"


# ------------------------------------------------------------------ locked contract


def test_threshold_is_locked_at_the_published_operating_point():
    """0.90 is predeclared. A review built at any other threshold would be evidence
    about a pipeline the project does not run."""
    assert LOCKED_THRESHOLD == 0.90


def test_coverage_threshold_matches_the_benchmark_default():
    from precision_myotube.benchmark import benchmark_instances
    import inspect

    default = inspect.signature(benchmark_instances).parameters["coverage_threshold"].default
    assert COVERAGE_THRESHOLD == default, (
        "the extraction must use the benchmark's own coverage rule; a different value "
        "would flag a different set of objects than the measurement being audited")


@pytest.mark.skipif(not RUN.is_file(), reason="linker_instance_v1.json not present")
def test_published_counts_match_the_source_run():
    payload = json.loads(RUN.read_text(encoding="utf-8"))
    per_well = {w["well"]: w["linked"][str(LOCKED_THRESHOLD)]["over_merge_count"]
                for w in payload["wells"]}
    for well, expected in PUBLISHED_OVER_MERGES.items():
        assert per_well[well] == expected
    assert sum(per_well.values()) == sum(PUBLISHED_OVER_MERGES.values()), \
        "every over-merge in the run must be accounted for in the review packet"


# ------------------------------------------------------------------ union-find


def test_merge_components_groups_transitively():
    comps = merge_components([1, 2, 3, 4, 5], [(1, 2), (2, 3)])
    groups = sorted(sorted(v) for v in comps.values())
    assert groups == [[1, 2, 3], [4], [5]]


def test_merge_components_leaves_unlinked_fragments_alone():
    comps = merge_components([4, 7], [])
    assert sorted(sorted(v) for v in comps.values()) == [[4], [7]]


def test_relabel_paints_every_member_with_the_component_root():
    assigned = np.array([[1, 1, 0], [2, 2, 0], [3, 0, 0]], dtype=np.int32)
    comps = merge_components([1, 2, 3], [(1, 2)])
    out = relabel(assigned, comps)
    root = out[0, 0]
    assert out[1, 0] == root, "a linked fragment takes the root label"
    assert out[2, 0] != root and out[2, 0] != 0, "an unlinked fragment keeps its own"
    assert (out == 0).sum() == (assigned == 0).sum(), "background is untouched"


# ------------------------------------------------------------------ the rule itself


def _ref(rid, r0, c0, h, w):
    return {"id": rid, "bbox": (r0, c0, r0 + h, c0 + w),
            "mask": np.ones((h, w), dtype=bool), "area": h * w}


def test_two_references_each_over_twenty_percent_is_an_over_merge():
    merged = np.zeros((40, 100), dtype=np.int32)
    merged[10:20, 10:90] = 5                       # 800 px prediction
    refs = [_ref("a", 10, 10, 10, 40),             # 400 px = 50% of the prediction
            _ref("b", 10, 50, 10, 40)]             # 400 px = 50%
    over, area = find_over_merges(merged, {5: [5]}, refs)
    assert area[5] == 800
    assert set(over) == {5}
    assert {r["reference_id"] for r in over[5]} == {"a", "b"}
    assert all(r["fraction_of_prediction"] >= COVERAGE_THRESHOLD for r in over[5])


def test_a_second_reference_below_the_coverage_floor_is_not_an_over_merge():
    """A prediction clipping the corner of a neighbour is not a merge of two objects."""
    merged = np.zeros((40, 100), dtype=np.int32)
    merged[10:20, 10:90] = 5                       # 800 px
    refs = [_ref("a", 10, 10, 10, 70),             # 700 px = 87.5%
            _ref("b", 10, 80, 10, 10)]             # 100 px = 12.5%, under 20%
    over, _area = find_over_merges(merged, {5: [5]}, refs)
    assert over == {}


def test_one_reference_spanning_two_predictions_is_not_an_over_merge():
    """That is a false split, the opposite error; it must not be counted here."""
    merged = np.zeros((40, 100), dtype=np.int32)
    merged[10:20, 10:50] = 5
    merged[10:20, 50:90] = 6
    over, _area = find_over_merges(merged, {5: [5], 6: [6]}, [_ref("a", 10, 10, 10, 80)])
    assert over == {}


def test_fractions_are_reported_against_both_denominators():
    """`fraction_of_prediction` decides the flag; `fraction_of_reference` is what tells
    a reviewer whether the reference was swallowed whole or merely clipped."""
    merged = np.zeros((40, 100), dtype=np.int32)
    merged[10:20, 10:90] = 5                       # 800 px prediction, cols 10-90
    refs = [_ref("a", 10, 10, 10, 40),             # 400 px, wholly inside the prediction
            _ref("b", 10, 50, 10, 50)]            # 500 px, runs past the prediction's end
    over, _area = find_over_merges(merged, {5: [5]}, refs)
    by_id = {r["reference_id"]: r for r in over[5]}
    assert by_id["a"]["fraction_of_reference"] == pytest.approx(1.0)   # swallowed whole
    assert by_id["b"]["fraction_of_reference"] == pytest.approx(400 / 500)  # only clipped
    assert by_id["b"]["fraction_of_prediction"] == pytest.approx(400 / 800)


def test_background_label_is_never_flagged():
    merged = np.zeros((20, 20), dtype=np.int32)
    over, area = find_over_merges(merged, {}, [_ref("a", 0, 0, 10, 10)])
    assert over == {} and area == {}
