"""Tests for the oracle tracer -- the walk's guarantees, pinned.

The X selftest is the contract this lane exists for: two fibres crossing must
come back as two objects sharing the junction, not four fragments and not one
blob. The other tests pin the small pieces that failed once already.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tracer_lab.centreline_targets import build_targets            # noqa: E402
from tracer_lab.oracle_trace import (                              # noqa: E402
    TraceParams, _axial_diff, score_against_gt, trace_field)


def _x_plus_bystander():
    n = 200
    a = np.column_stack([np.linspace(20, 180, n), np.linspace(20, 180, n)])
    b = np.column_stack([np.linspace(180, 20, n), np.linspace(20, 180, n)])
    c = np.column_stack([np.full(n, 30.0), np.linspace(30, 150, n)])
    return build_targets((200, 200), [a, b, c])


def test_x_crossing_identity():
    """Two crossing fibres + one that ENDS on a fibre: 3 objects, no losses."""
    fields = _x_plus_bystander()
    sc = score_against_gt(trace_field(fields), fields)
    assert sc["false_split_count"] == 0
    assert sc["false_merge_count"] == 0
    assert sc["identity_through_crossing"] == 1.0
    assert sc["recall_traces"] == 1.0
    assert sc["length_mdape"] < 0.05


def test_single_fibre_one_object():
    """Many seeds on one fibre must union into ONE object via merges."""
    n = 300
    t = np.linspace(0, 2 * np.pi, n)
    curve = np.column_stack([150 + 60 * np.sin(t), np.linspace(20, 280, n)])
    fields = build_targets((300, 300), [curve])
    res = trace_field(fields)
    sc = score_against_gt(res, fields)
    assert sc["n_objects"] == 1
    assert sc["length_mdape"] < 0.1


def test_axial_diff_wraps():
    """0 and pi are the same axial direction; pi/2 is maximally different."""
    assert _axial_diff(0.0, np.pi) == pytest.approx(0.0, abs=1e-9)
    assert _axial_diff(0.1, 0.1 + np.pi) == pytest.approx(0.0, abs=1e-9)
    assert _axial_diff(0.0, np.pi / 2) == pytest.approx(np.pi / 2)


def test_parallel_fibres_stay_separate():
    """Two fibres 12 px apart, never crossing: two objects, no merges.

    Guards the claim radius / colinear rule from fusing a parallel bundle.
    """
    n = 200
    a = np.column_stack([np.full(n, 80.0), np.linspace(20, 180, n)])
    b = np.column_stack([np.full(n, 92.0), np.linspace(20, 180, n)])
    fields = build_targets((200, 200), [a, b])
    sc = score_against_gt(trace_field(fields), fields)
    assert sc["n_objects"] == 2
    assert sc["false_merge_count"] == 0
