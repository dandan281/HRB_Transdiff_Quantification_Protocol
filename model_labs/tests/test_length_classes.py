"""Contract tests for the length-class + freeline-smoothing convention.

The length classes are the operator's metric of record (the share of
myotubes per length band, not total length), and the smoothing is the
convention that reconciles our measurements with Fiji's freeline `Length`
(2026-08-27 report §7d). Both are shared by every plate-quantification
script, so their behaviour is pinned here.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tracer_lab.length_classes import (                          # noqa: E402
    BINS_UM, LABELS, arc_um, class_shares, format_shares, smooth_polyline)


def test_shares_bin_and_sum_to_one():
    """One fibre per class, plus boundary values landing in the upper bin.

    Shares are rounded to 4 dp for reporting, so they sum to 1 only within
    that rounding (measured worst case on real wells: 1e-4). Asserting
    exact equality here would pass only for inputs whose shares happen to
    be exact -- a test that passes by luck.
    """
    s = class_shares([100, 200, 400, 600, 1000])
    assert s["n"] == 5
    assert all(s[l] == pytest.approx(0.2) for l in LABELS)
    assert sum(s[l] for l in LABELS) == pytest.approx(1.0, abs=1e-3)
    seven = class_shares([60, 70, 80, 200, 400, 600, 1000])   # thirds, inexact
    assert sum(seven[l] for l in LABELS) == pytest.approx(1.0, abs=1e-3)
    # np.histogram bins are right-open: an exact edge belongs upward
    edge = class_shares([150.0, 300.0, 500.0, 800.0])
    assert edge["50-150"] == 0.0
    assert all(edge[l] == pytest.approx(0.25)
               for l in ("150-300", "300-500", "500-800", ">800"))


def test_short_fibres_excluded_and_empty_is_safe():
    """< 50 um is below the counting gate; an empty well must not divide by zero."""
    s = class_shares([10, 49.9, 60])
    assert s["n"] == 1 and s["50-150"] == pytest.approx(1.0)
    empty = class_shares([])
    assert empty["n"] == 0
    assert all(empty[l] == 0.0 for l in LABELS)
    assert sum(empty[l] for l in LABELS) == 0.0     # not NaN


def test_smoothing_pins_endpoints_and_leaves_straight_lines():
    """A straight trace must not shrink; endpoints never move."""
    straight = np.column_stack([np.zeros(60), np.linspace(0, 59, 60)])
    sm = smooth_polyline(straight)
    assert np.allclose(sm, straight, atol=1e-9)
    jitter = straight + np.column_stack(
        [np.tile([0.6, -0.6], 30), np.zeros(60)])
    sj = smooth_polyline(jitter)
    assert np.allclose(sj[0], jitter[0]) and np.allclose(sj[-1], jitter[-1])


def test_raw_arc_inflates_freehand_traces():
    """The §7d finding, pinned: raw point-to-point arc over-measures a
    hand-drawn (jittery) trace by ~10-15%, while smoothing recovers the
    true length. A regression here would silently re-inflate every human
    length in the plate tables."""
    n = 400
    truth = np.column_stack([np.zeros(n), np.linspace(0, n - 1, n)])
    rng = np.random.default_rng(0)
    hand = truth + np.column_stack([rng.normal(0, 0.45, n), np.zeros(n)])
    raw = arc_um(hand, 1.0)
    smoothed = arc_um(hand, 1.0, smoothed=True)
    true_len = arc_um(truth, 1.0)
    assert raw / true_len > 1.08          # raw inflates
    assert 0.99 <= smoothed / true_len <= 1.05   # smoothing recovers it
    assert smoothed < raw


def test_arc_um_scales_with_pixel_size():
    line = np.column_stack([np.zeros(10), np.arange(10.0)])
    assert arc_um(line, 1.0) == pytest.approx(9.0)
    assert arc_um(line, 0.65) == pytest.approx(9.0 * 0.65)


def test_short_polyline_survives_smoothing():
    """Traces shorter than the window are returned unchanged, not crashed."""
    tiny = np.array([[0.0, 0.0], [0.0, 3.0]])
    assert np.allclose(smooth_polyline(tiny), tiny)


def test_format_shares_is_readable():
    line = format_shares(class_shares([100, 200]))
    assert "50-150 50.0%" in line and "150-300 50.0%" in line


def test_bins_and_labels_stay_aligned():
    """Labels describe the bins; a change to one must change the other."""
    assert len(LABELS) == len(BINS_UM) - 1
    assert BINS_UM[0] == 50.0 and BINS_UM[-1] == np.inf
