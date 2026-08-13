"""The geometric constraints the linker declares, now actually enforced.

Each test below corresponds to a configuration the control-only safety round
found in real data, so a regression here is not hypothetical -- it is a merge the
operator has already called wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from annotation_tools.qc_review.link_candidates import (ENDPOINT_LOCAL_PX,
                                                        find_link_candidates)
from annotation_tools.qc_review.link_geometry import (MIN_AXIS_EXTENT_PX, Axis,
                                                      axes_agree,
                                                      constrained_merge,
                                                      fragment_axis)


def _bar(shape, row, col0, col1, thickness=3):
    mask = np.zeros(shape, dtype=bool)
    mask[row:row + thickness, col0:col1] = True
    return mask


def _vbar(shape, col, row0, row1, thickness=3):
    mask = np.zeros(shape, dtype=bool)
    mask[row0:row1, col:col + thickness] = True
    return mask


# --------------------------------------------------------------- axis estimate

def test_the_minimum_extent_is_derived_from_the_direction_window_not_chosen():
    # If someone widens the endpoint neighbourhood, the length below which an axis
    # is meaningless moves with it. Pinning the relation stops the two drifting.
    assert MIN_AXIS_EXTENT_PX == 2 * ENDPOINT_LOCAL_PX


def test_a_long_bar_has_an_axis_along_its_length():
    axis = fragment_axis(_bar((60, 120), 30, 10, 100))
    assert axis is not None
    assert abs(axis.direction[1]) > 0.99          # runs along columns
    assert axis.extent_px == pytest.approx(89, abs=2)
    assert axis.elongation > 5


def test_a_blob_has_no_estimable_axis():
    # 435 px and 3:1 was a real fragment in case 030; a disc is the limiting case.
    disc = np.zeros((40, 40), dtype=bool)
    rows, cols = np.ogrid[:40, :40]
    disc[(rows - 20) ** 2 + (cols - 20) ** 2 <= 36] = True
    assert fragment_axis(disc) is None


def test_a_fragment_shorter_than_two_direction_windows_has_no_axis():
    assert fragment_axis(_bar((40, 40), 20, 10, 10 + MIN_AXIS_EXTENT_PX - 2)) is None
    assert fragment_axis(_bar((40, 60), 20, 10, 10 + MIN_AXIS_EXTENT_PX + 6)) is not None


# ------------------------------------------------------------------- agreement

def test_axes_agree_is_undirected():
    forward = Axis(np.array([1.0, 0.0]), 50.0, 10.0)
    backward = Axis(np.array([-1.0, 0.0]), 50.0, 10.0)
    assert axes_agree(forward, backward, 0.70)


def test_perpendicular_axes_do_not_agree():
    horizontal = Axis(np.array([0.0, 1.0]), 50.0, 10.0)
    vertical = Axis(np.array([1.0, 0.0]), 50.0, 10.0)
    assert not axes_agree(horizontal, vertical, 0.70)


def test_the_real_63_degree_pair_from_case_018_is_rejected():
    # fragments 86 and 237 of 29_C05, merged at P = 0.999998
    frag86 = Axis(np.array([0.983, -0.184]), 60.0, 8.4)
    frag237 = Axis(np.array([0.273, -0.962]), 60.0, 15.0)
    assert not axes_agree(frag86, frag237, 0.70)


def test_an_unestimable_axis_is_refused_not_waved_through():
    real = Axis(np.array([1.0, 0.0]), 50.0, 10.0)
    assert not axes_agree(real, None, 0.70)
    assert not axes_agree(None, None, 0.70)


# ------------------------------------------------------- the transitive closure

def test_a_legal_chain_may_not_assemble_an_illegal_object():
    # The case-018 failure in miniature: neighbours are inside the window, the two
    # ends are not. The old union-find built this object; this must not.
    axes = {
        1: Axis(np.array([1.0, 0.0]), 50.0, 10.0),               # 0 degrees
        2: Axis(np.array([0.866, 0.5]), 50.0, 10.0),             # 30
        3: Axis(np.array([0.5, 0.866]), 50.0, 10.0),             # 60 -- 60 from #1
    }
    result = constrained_merge([1, 2, 3], [(0.99, 1, 2), (0.98, 2, 3)], axes,
                               cos_min=0.70)
    assert sorted(result.components.values()) == [[1, 2], [3]]
    assert result.refused[0]["fragments"] == [2, 3]
    assert result.refused[0]["blocking_pair"] == [1, 3]
    assert "outside the declared window" in result.refused[0]["reason"]


def test_the_higher_probability_join_wins_a_conflict():
    axes = {
        1: Axis(np.array([1.0, 0.0]), 50.0, 10.0),
        2: Axis(np.array([0.866, 0.5]), 50.0, 10.0),
        3: Axis(np.array([0.5, 0.866]), 50.0, 10.0),
    }
    high_first = constrained_merge([1, 2, 3], [(0.99, 2, 3), (0.50, 1, 2)], axes,
                                   cos_min=0.70)
    assert sorted(high_first.components.values()) == [[1], [2, 3]]


def test_merging_is_independent_of_edge_order():
    axes = {n: Axis(np.array([1.0, 0.0]), 50.0, 10.0) for n in (1, 2, 3)}
    edges = [(0.9, 1, 2), (0.9, 2, 3)]
    first = constrained_merge([1, 2, 3], edges, axes, cos_min=0.70)
    second = constrained_merge([1, 2, 3], list(reversed(edges)), axes, cos_min=0.70)
    assert first.components == second.components


def test_an_aligned_chain_still_merges_completely():
    # The fix must not simply refuse everything: collinear fragments are the case
    # the linker exists for.
    axes = {n: Axis(np.array([1.0, 0.02 * n]), 50.0, 10.0) for n in (1, 2, 3, 4)}
    result = constrained_merge([1, 2, 3, 4],
                               [(0.99, 1, 2), (0.98, 2, 3), (0.97, 3, 4)],
                               axes, cos_min=0.70)
    assert list(result.components.values()) == [[1, 2, 3, 4]]
    assert result.refused == []


def test_a_fragment_with_no_axis_cannot_be_merged_into_anything():
    axes = {1: Axis(np.array([1.0, 0.0]), 50.0, 10.0), 2: None}
    result = constrained_merge([1, 2], [(1.0, 1, 2)], axes, cos_min=0.70)
    assert sorted(result.components.values()) == [[1], [2]]
    assert "unverifiable" in result.refused[0]["reason"]


# ----------------------------------------------------- the candidate finder gate

def test_the_finder_no_longer_offers_a_perpendicular_partner():
    labels = np.zeros((160, 160), dtype=np.int32)
    labels[_bar((160, 160), 80, 10, 70)] = 1          # horizontal
    labels[_vbar((160, 160), 78, 82, 150)] = 2        # vertical, tip near the first
    strict = find_link_candidates(labels, [1], 1.0, gap_um=40.0, cos_min=0.70)
    assert strict["myotube_0001"] == []


def test_the_pre_fix_behaviour_stays_reachable_for_reproduction():
    # Sealed runs were produced without this gate; reproducing them must remain
    # possible or their provenance becomes unverifiable.
    labels = np.zeros((160, 160), dtype=np.int32)
    labels[_bar((160, 160), 80, 10, 70)] = 1
    labels[_vbar((160, 160), 78, 82, 150)] = 2
    loose = find_link_candidates(labels, [1], 1.0, gap_um=40.0, cos_min=0.70,
                                 require_axis_agreement=False)
    strict = find_link_candidates(labels, [1], 1.0, gap_um=40.0, cos_min=0.70)
    assert len(loose["myotube_0001"]) >= len(strict["myotube_0001"])


def test_the_finder_still_offers_a_collinear_partner():
    labels = np.zeros((160, 200), dtype=np.int32)
    labels[_bar((160, 200), 80, 10, 80)] = 1
    labels[_bar((160, 200), 80, 95, 180)] = 2
    found = find_link_candidates(labels, [1], 1.0, gap_um=40.0, cos_min=0.70)
    assert [c.candidate_id for c in found["myotube_0001"]] == ["myotube_0002"]
