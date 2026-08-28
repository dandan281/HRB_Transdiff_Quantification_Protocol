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
    TraceParams, _axial_diff, score_against_gt, trace_field, weld_objects)


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


def _line(p0, p1, m=120):
    return np.column_stack([np.linspace(p0[0], p1[0], m),
                            np.linspace(p0[1], p1[1], m)])


def _weld(pieces, object_of, gt_traces, **kw):
    fields = build_targets((200, 200), gt_traces)
    res = {"paths": pieces, "object_of": object_of, "merge_events": [],
           "stop_reasons": {}}
    out = weld_objects(res, fields, weld_dist_px=10.0, weld_deg=15.0,
                       crossing_gate_px=12.0, **kw)
    return len(set(out["object_of"].values()))


def test_weld_joins_cut_at_crossing():
    """One fibre cut at a crossing into co-linear pieces: welded to one
    object — abutting or across an 8 px hole. The 2026-08-27 break class."""
    A, B = _line((100, 10), (100, 190)), _line((10, 10), (190, 190))
    a1, a2 = _line((100, 10), (100, 99), 90), _line((100, 101), (100, 190), 90)
    assert _weld([a1, a2, B], {1: 1, 2: 2, 3: 3}, [A, B]) == 2
    h1, h2 = _line((100, 10), (100, 96), 87), _line((100, 104), (100, 190), 87)
    assert _weld([h1, h2, B], {1: 1, 2: 2, 3: 3}, [A, B]) == 2


def test_weld_refuses_wrong_joins():
    """Parallel neighbours (lateral connector), a transverse end at the
    junction, and an open-field cut (no crossing) must NOT weld — the
    2026-07 linker failure and the fragments-joined class, kept banned."""
    B = _line((10, 10), (190, 190))
    p1, p2 = _line((100, 10), (100, 190)), _line((106, 10), (106, 190))
    assert _weld([p1, p2, B], {1: 1, 2: 2, 3: 3}, [p1, p2, B]) == 3
    A = _line((100, 10), (100, 190))
    t1 = _line((100, 10), (100, 98), 88)
    assert _weld([t1, B], {1: 1, 2: 2}, [A, B]) == 2
    F = _line((40, 10), (40, 190))
    c1, c2 = _line((40, 10), (40, 90), 80), _line((40, 92), (40, 190), 98)
    assert _weld([c1, c2, B], {1: 1, 2: 2, 3: 3}, [F, B]) == 3


def test_weld_off_is_identity():
    """weld_dist_px <= 0 must return the input unchanged — the frozen
    configuration stays bit-identical."""
    A, B = _line((100, 10), (100, 190)), _line((10, 10), (190, 190))
    fields = build_targets((200, 200), [A, B])
    res = trace_field(fields)
    assert weld_objects(res, fields, weld_dist_px=0.0) is res


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
