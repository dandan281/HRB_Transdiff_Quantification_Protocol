"""Tests for the over-merge eligibility count.

This is the analysis that turned `over_merge_count = 3` from a rate into a ceiling,
so the counting rule needs to be exactly the benchmark's and the "accepted merge"
definition needs to exclude untouched single fragments.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from classical.over_merge_flaggability import COVERAGE_THRESHOLD, eligibility

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "model_labs/classical/_runs/over_merge_flaggability_v1.json"


def _ref(r0, c0, h, w):
    return {"id": f"r{r0}_{c0}", "bbox": (r0, c0, r0 + h, c0 + w),
            "mask": np.ones((h, w), dtype=bool)}


def test_coverage_threshold_matches_the_benchmark():
    from precision_myotube.benchmark import benchmark_instances
    import inspect

    assert COVERAGE_THRESHOLD == inspect.signature(
        benchmark_instances).parameters["coverage_threshold"].default


def test_only_multi_fragment_components_count_as_accepted_merges():
    """An untouched single fragment is not a merge and must not dilute the denominator."""
    kept = np.zeros((20, 60), dtype=np.int32)
    kept[5:10, 0:20] = 1          # fragment 1 -- merged with 2
    kept[5:10, 20:40] = 2
    kept[5:10, 45:55] = 3          # fragment 3 -- untouched
    merged = kept.copy()
    merged[merged == 2] = 1        # 1+2 joined
    out = eligibility(merged, kept, [])
    assert out["n_merges"] == 1


def test_a_merge_with_two_references_over_the_floor_is_eligible():
    kept = np.zeros((20, 60), dtype=np.int32)
    kept[5:10, 0:20] = 1
    kept[5:10, 20:40] = 2
    merged = np.where(kept > 0, 1, 0).astype(np.int32)   # 200 px
    refs = [_ref(5, 0, 5, 20), _ref(5, 20, 5, 20)]        # 100 px each = 50% apiece
    out = eligibility(merged, kept, refs)
    assert out["n_merges"] == 1 and out["n_eligible"] == 1
    assert out["buckets"] == {0: 0, 1: 0, 2: 1}


def test_a_merge_with_one_reference_is_ineligible_however_wrong_it_is():
    """The core point: a single overlapping reference means the rule is blind here."""
    kept = np.zeros((20, 60), dtype=np.int32)
    kept[5:10, 0:20] = 1
    kept[5:10, 20:40] = 2
    merged = np.where(kept > 0, 1, 0).astype(np.int32)
    out = eligibility(merged, kept, [_ref(5, 0, 5, 20)])
    assert out["n_eligible"] == 0
    assert out["buckets"] == {0: 0, 1: 1, 2: 0}
    assert out["n_ineligible"] == 1


def test_a_merge_with_no_references_is_ineligible():
    kept = np.zeros((20, 60), dtype=np.int32)
    kept[5:10, 0:20] = 1
    kept[5:10, 20:40] = 2
    merged = np.where(kept > 0, 1, 0).astype(np.int32)
    out = eligibility(merged, kept, [])
    assert out["buckets"] == {0: 1, 1: 0, 2: 0}
    assert out["eligible_fraction"] == 0.0


def test_a_reference_under_the_coverage_floor_does_not_count():
    kept = np.zeros((20, 60), dtype=np.int32)
    kept[5:10, 0:25] = 1
    kept[5:10, 25:50] = 2
    merged = np.where(kept > 0, 1, 0).astype(np.int32)    # 250 px
    refs = [_ref(5, 0, 5, 25),        # 125 px = 50%  -> counts
            _ref(5, 45, 5, 5)]        # 25 px  = 10%  -> under 20%, does not
    out = eligibility(merged, kept, refs)
    assert out["buckets"] == {0: 0, 1: 1, 2: 0}


def test_eligible_fraction_is_none_when_there_are_no_merges():
    kept = np.zeros((10, 10), dtype=np.int32)
    out = eligibility(kept, kept, [])
    assert out["n_merges"] == 0 and out["eligible_fraction"] is None


@pytest.mark.skipif(not RUN.is_file(), reason="flaggability run not present")
def test_the_recorded_run_says_every_eligible_merge_was_flagged():
    """Pins the headline: the detector saw 3 merges and flagged all 3, so the count is
    a ceiling. If this ever stops holding, the linker trade must be restated."""
    payload = json.loads(RUN.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["n_eligible"] == summary["n_flagged"]
    assert summary["flag_rate_among_eligible"] == 1.0
    assert summary["ineligible_fraction"] > 0.95
