"""Contract tests for the CE plate tool's pure parts (no JVM, no nd2).

Pinned: micrometre->pixel conversion of the recipe (the reason a Plate-44
pixel constant would gate away real nuclei on a Q-plate), the plateau rule
on a known curve, the robust sigma, and the well / treatment parsing.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Conversion_Efficiency"))

import quantify_ce_plate as q                                   # noqa: E402


def test_recipe_scales_with_pixel_size():
    """At Plate 44's 1.7246 um/px the recipe must reproduce its original
    pixel constants (8-700 px nuclei, 6 px shell, 25/150 px rolling balls);
    at a 0.65 um/px Q-plate the same physical sizes are ~7x more pixels."""
    p44 = q.px_constants(1.7245709)
    assert (p44["nuc_lo"], p44["nuc_hi"]) == (8, 699) or \
           (p44["nuc_lo"], p44["nuc_hi"]) == (8, 700)
    assert p44["shell"] == 6 and p44["rb_dapi"] == 25 and p44["rb_des"] == 150
    assert abs(p44["blur"] - 1.0) < 0.01
    q26 = q.px_constants(0.650017)
    assert 55 <= q26["nuc_lo"] <= 58            # 24 um^2 -> ~57 px
    assert 4900 <= q26["nuc_hi"] <= 4930        # 2080 um^2 -> ~4922 px
    assert q26["shell"] == 16 and q26["rb_des"] == 398


def test_plateau_pick_finds_the_flat_part():
    """A curve that is steep, then flat, then steep: the plateau is the
    flat middle, with the contiguous range reported around it."""
    x = np.exp(np.linspace(np.log(10), np.log(1000), 60))
    y = np.where(x < 60, 1000 / x, np.where(x <= 300, 16.0, 16.0 * (300 / x) ** 3))
    x_star, lo, hi, inst = q.plateau_pick(x, y)
    assert 60 <= x_star <= 300
    assert lo <= x_star <= hi and lo >= 40 and hi <= 400
    assert inst[np.argmin(np.abs(x - x_star))] < 0.05


def test_plateau_pick_respects_window():
    x = np.exp(np.linspace(np.log(10), np.log(1000), 40))
    y = np.full_like(x, 5.0)                    # flat everywhere
    x_star, lo, hi, _ = q.plateau_pick(x, y, lo=100, hi=200)
    assert 100 <= x_star <= 200 and lo >= 100 and hi <= 200


def test_robust_sigma_is_mad_scaled():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 3.0, 200_000)
    assert q.robust_sigma(a) == pytest.approx(3.0, rel=0.03)
    a[:2000] = 1e6                               # outliers do not move it
    assert q.robust_sigma(a) == pytest.approx(3.0, rel=0.05)


def test_well_and_treatment_parsing():
    assert q.well_token("21_B04_br223_egfrc") == "B04"
    assert q.treatment_from_stem("21_B04_br223_egfrc", "B04") == "br223_egfrc"
    assert q.treatment_from_stem("18_B07", "B07") == ""
    assert q.treatment_from_stem("23_B02_ctrl", "B02") == "ctrl"


def test_k_grid_and_fracs_are_the_conventions():
    assert q.FRACS == (0.25, 0.5)
    assert 5 in q.K_GRID and min(q.K_GRID) == 2 and max(q.K_GRID) == 12


def test_plateau_range_is_not_a_single_point_at_a_peak():
    """A unimodal count curve has zero derivative at its peak; the reported
    plateau must be a RANGE (the flat top), not one point -- measured
    '1293..1293' before the absolute floor existed."""
    x = np.exp(np.linspace(np.log(100), np.log(10000), 50))
    y = 1000 * np.exp(-((np.log(x) - np.log(1000)) ** 2) / (2 * 0.6 ** 2))
    x_star, lo, hi, _ = q.plateau_pick(x, y)
    assert 800 <= x_star <= 1250
    # the floor turns the zero-derivative point into the flat TOP: several
    # grid steps wide (its width is set by the curve's own curvature, so
    # the contract is 'a range', not any particular ratio)
    assert hi > lo and hi / lo > 1.05


def test_trivial_flat_at_one_is_excludable():
    """Fraction-positive vs cut is flat at 1 for small cuts. The unrestricted
    rule picks that flat (Plate 44: cut 22 -> 95 % positive); confining the
    search to cuts calling 5-80 % positive lands on the real shoulder."""
    cuts = np.exp(np.linspace(np.log(5), np.log(2000), 200))
    # flat at 1 until ~30, steep drop to a shoulder near 0.2 around 80-300,
    # then decay to 0
    f = np.where(cuts < 30, 1.0,
                 np.where(cuts < 80, 1.0 - 0.8 * (np.log(cuts) - np.log(30))
                          / (np.log(80) - np.log(30)),
                          np.where(cuts < 300, 0.2, 0.2 * (300 / cuts) ** 2)))
    unrestricted, _, _, _ = q.plateau_pick(cuts, f, smooth=7)
    assert unrestricted < 30                 # the trivial flat wins
    band = (f <= 0.80) & (f >= 0.05)
    restricted, lo, hi, _ = q.plateau_pick(
        cuts, f, lo=float(cuts[band][0]), hi=float(cuts[band][-1]), smooth=7)
    assert 80 <= restricted <= 300           # the shoulder
