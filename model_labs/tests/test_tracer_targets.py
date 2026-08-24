"""Tests for the T04 tracer target builder.

Runs in `pm-annotate` -- numpy and scipy only, no torch, no GPU.

The properties pinned here are the ones whose violation would be silent, and
each one exists because getting it wrong produces a target that trains happily
and means nothing:

* **orientation survives a reversed trace.** The operator's click direction is
  arbitrary. A target that encodes it teaches the model to predict which end of
  a fibre a human started at.
* **a crossing needs two different fibres.** A fibre that curves back near
  itself is not a crossing, and a target that says otherwise deletes the
  orientation supervision exactly where fibres are densest.
* **orientation is not supervised at crossings.** Two directions cannot be
  averaged into one; their mean is a direction no fibre has.
* **resampling removes the operator's mouse speed.** Every downstream field
  weights by vertex, so uneven spacing weights the target by how fast a hand
  moved.
"""
from __future__ import annotations

import numpy as np
import pytest

from tracer_lab.centreline_targets import (TargetConfig, build_targets,
                                           polyline_tangents,
                                           resample_polyline)


def _line(start, end, n=60):
    return np.column_stack([np.linspace(start[0], end[0], n),
                            np.linspace(start[1], end[1], n)])


# ----------------------------------------------------------------- resampling


def test_resample_equalises_spacing():
    """A freehand drag emits points at mouse speed; the target must not."""
    pts = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 40.0]])
    out = resample_polyline(pts, 1.0)
    step = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert step.std() < 1e-9
    assert step.mean() == pytest.approx(1.0, abs=0.05)


def test_resample_survives_duplicate_points():
    pts = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 15.0]])
    out = resample_polyline(pts, 1.0)
    assert len(out) >= 2 and np.isfinite(out).all()


def test_tangent_window_is_the_fibre_scale_not_three_pixels():
    """A wide window must ignore tremor a 3-point difference would chase.

    Costing this wrong has already been paid for twice on this project; see the
    `myotube-fibre-scale-direction-windows` note.
    """
    n = 200
    straight = np.column_stack([np.zeros(n), np.arange(n, dtype=float)])
    jitter = straight.copy()
    rng = np.random.default_rng(0)
    jitter[:, 0] += rng.normal(0, 0.4, n)          # hand tremor, sub-pixel

    wide = polyline_tangents(jitter, 15.0)
    narrow = polyline_tangents(jitter, 2.0)
    truth = polyline_tangents(straight, 15.0)
    err = lambda t: float(np.abs(np.abs((t * truth).sum(axis=1)) - 1).mean())
    assert err(wide) < err(narrow)


# ---------------------------------------------------------------- orientation


def test_orientation_is_invariant_to_trace_direction():
    """The same fibre clicked backwards must produce the same target."""
    fwd = _line((30, 5), (30, 55))
    bwd = fwd[::-1].copy()
    a = build_targets((60, 60), [fwd])["orient"]
    b = build_targets((60, 60), [bwd])["orient"]
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_orientation_separates_perpendicular_fibres():
    """Angle doubling must keep 0 and 90 degrees distinguishable."""
    t = build_targets((60, 60), [_line((30, 5), (30, 55)),
                                 _line((5, 30), (55, 30))])
    horizontal = t["orient"][:, 30, 10]
    vertical = t["orient"][:, 10, 30]
    np.testing.assert_allclose(horizontal, [1.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(vertical, [-1.0, 0.0], atol=1e-3)


def test_orientation_is_unit_norm_on_every_fibre_pixel():
    t = build_targets((60, 60), [_line((30, 5), (30, 55)),
                                 _line((5, 5), (55, 55))])
    on = t["instance"] > 0
    norm = np.linalg.norm(t["orient"], axis=0)[on]
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)


# ------------------------------------------------------------------ crossings


def test_crossing_needs_two_distinct_traces():
    t = build_targets((60, 60), [_line((30, 5), (30, 55)),
                                 _line((5, 30), (55, 30))],
                      config=TargetConfig(cross_radius_px=5.0))
    assert t["crossing"][30, 30]
    assert not t["crossing"][30, 10]
    assert not t["crossing"][10, 30]


def test_a_single_fibre_near_itself_is_not_a_crossing():
    """A hairpin is one object. Calling it a crossing would delete the
    orientation supervision wherever fibres bend."""
    down = np.column_stack([np.arange(10, 50, dtype=float), np.full(40, 30.0)])
    up = np.column_stack([np.arange(49, 9, -1, dtype=float), np.full(40, 33.0)])
    hairpin = np.vstack([down, up])
    t = build_targets((60, 60), [hairpin],
                      config=TargetConfig(cross_radius_px=5.0))
    assert not t["crossing"].any()


def test_orientation_is_masked_exactly_at_crossings():
    t = build_targets((60, 60), [_line((30, 5), (30, 55)),
                                 _line((5, 30), (55, 30))],
                      config=TargetConfig(cross_radius_px=5.0))
    within = t["instance"] > 0
    assert not t["orient_valid"][t["crossing"]].any()
    assert t["orient_valid"][within & ~t["crossing"]].all()


def test_crossing_fraction_is_stable_across_radius():
    """Measured on PLATE_32 B02: 53.9% of traces at r=3, 55.3% at r=6.

    A quantity that doubles when the radius doubles is a property of the radius.
    This one is not, which is why it can be reported.
    """
    fibres = [_line((30, 5), (30, 55)), _line((5, 30), (55, 30)),
              _line((5, 5), (55, 55))]
    touched = []
    for r in (3.0, 6.0):
        t = build_targets((60, 60), fibres, config=TargetConfig(cross_radius_px=r))
        touched.append(sum(bool(t["crossing"][
            np.clip(np.rint(p[:, 0]).astype(int), 0, 59),
            np.clip(np.rint(p[:, 1]).astype(int), 0, 59)].any())
            for p in t["traces"]))
    assert touched[0] == touched[1] == 3


# ----------------------------------------------------------------- degenerate


def test_empty_and_too_short_inputs_do_not_raise():
    t = build_targets((32, 32), [np.zeros((1, 2)), np.zeros((0, 2))])
    assert t["n_traces"] == 0
    assert t["centre"].shape == (32, 32)
    assert not t["crossing"].any()
    assert not t["orient_valid"].any()
