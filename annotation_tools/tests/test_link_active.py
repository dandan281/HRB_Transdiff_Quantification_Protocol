"""Active-learning linker: intensity features, model, uncertainty, and de-dup.

Uses small synthetic fields so the properties are exact and fast. The one thing
these cannot cover is the AUC on the real 91 pairs; that is checked by running the
CLI against the bootstrap (see the session report), not in unit tests.
"""
import numpy as np
import pytest

from annotation_tools.qc_review.link_features import (
    FEATURE_KEYS, ObjectGeom, axis_features, compute_features, field_background,
    geometry_cache, object_geometry)
from annotation_tools.qc_review.link_model import (
    FEATURE_SETS, LinkPair, fit_linker, leave_one_well_out_auc, uncertainty)
from annotation_tools.qc_review.link_features import LinkFeatures

PIXEL_UM = 0.6493


# ------------------------------------------------------------------- features


def _bridged_field(bridge_value: int):
    """A 200x60 field: two fibres at cols 20-40 (bright), gap 40-60, fibre 60-80.

    The gap pixels are set to ``bridge_value`` so a test can make the bridge bright
    (a real join) or dark (background, a false join).
    """
    fiber = np.full((60, 200), 600, dtype=np.uint16)   # background
    fiber[28:32, 20:80] = 9000                          # fragment
    fiber[28:32, 120:180] = 9000                        # continuation
    fiber[28:32, 80:120] = bridge_value                 # the gap
    return fiber


def test_bridge_over_bg_high_when_stain_crosses_the_gap():
    bright = _bridged_field(bridge_value=8000)
    dark = _bridged_field(bridge_value=600)
    ep_a, ep_b = (30, 80), (30, 120)                    # across the gap
    fb = compute_features(bright, None, ep_a, ep_b, gap_um=26.0, min_cos=0.99,
                          pixel_um=PIXEL_UM)
    fd = compute_features(dark, None, ep_a, ep_b, gap_um=26.0, min_cos=0.99,
                          pixel_um=PIXEL_UM)
    assert fb.bridge_over_bg > fd.bridge_over_bg
    assert fd.bridge_over_bg == pytest.approx(1.0, abs=0.2)   # dark gap ~ background
    assert fb.bridge_over_bg > 5.0                            # bright bridge >> bg


def test_bridge_min_catches_a_single_dark_step():
    """One dark pixel mid-gap should drop bridge_over_bg even if the rest is bright."""
    fiber = _bridged_field(bridge_value=8000)
    fiber[28:32, 98:102] = 600                          # a dark notch in the bridge
    f = compute_features(fiber, None, (30, 80), (30, 120), gap_um=26.0,
                         min_cos=0.99, pixel_um=PIXEL_UM)
    assert f.bridge_over_bg == pytest.approx(1.0, abs=0.3)


def test_territory_frac_measures_fraction_inside_territory():
    fiber = _bridged_field(8000)
    territory = np.zeros_like(fiber, dtype=np.uint8)
    territory[28:32, 80:100] = 1                         # covers first half of the gap
    f = compute_features(fiber, territory, (30, 80), (30, 120), gap_um=26.0,
                         min_cos=0.99, pixel_um=PIXEL_UM)
    assert 0.3 < f.territory_frac < 0.7


def test_features_are_deterministic():
    fiber = _bridged_field(8000)
    a = compute_features(fiber, None, (30, 80), (30, 120), 26.0, 0.99, PIXEL_UM)
    b = compute_features(fiber, None, (30, 80), (30, 120), 26.0, 0.99, PIXEL_UM)
    assert a.vector() == b.vector()
    assert list(FEATURE_KEYS) == ["gap_um", "min_cos", "bridge_over_bg",
                                  "bridge_mean_over_bg", "territory_frac",
                                  "axis_cos", "displacement_along_axis"]


# --------------------------------------------- operator axis heuristic (2026-07-23)


def test_object_geometry_recovers_the_long_axis():
    horiz = np.zeros((60, 200), dtype=np.int32)
    horiz[28:32, 20:180] = 1                            # runs along the columns (x)
    g = object_geometry(horiz == 1)
    # principal axis should be ~ (0, 1): no row component, all column
    assert abs(g.axis[0]) < 0.2 and abs(g.axis[1]) > 0.9

    vert = np.zeros((200, 60), dtype=np.int32)
    vert[20:180, 28:32] = 1                             # runs along the rows (y)
    gv = object_geometry(vert == 1)
    assert abs(gv.axis[0]) > 0.9 and abs(gv.axis[1]) < 0.2


def test_displacement_along_axis_separates_end_to_end_from_side_by_side():
    """The heuristic: same fibre = offset ALONG the axis; parallel neighbour = across."""
    # two horizontal fibres, end to end (offset along x = along their axis)
    a = ObjectGeom(centroid=(30.0, 40.0), axis=(0.0, 1.0), n_px=200)
    end_to_end = ObjectGeom(centroid=(30.0, 140.0), axis=(0.0, 1.0), n_px=200)
    axis_cos, along = axis_features(a, end_to_end)
    assert axis_cos > 0.99                              # parallel
    assert along > 0.99                                 # offset is along the axis

    # a parallel fibre sitting beside it (offset along y = across the axis)
    side_by_side = ObjectGeom(centroid=(90.0, 40.0), axis=(0.0, 1.0), n_px=200)
    axis_cos2, along2 = axis_features(a, side_by_side)
    assert axis_cos2 > 0.99                             # still parallel...
    assert along2 < 0.1                                 # ...but the offset is across
    assert along > along2                               # the rule discriminates


def test_axis_cos_is_sign_and_flip_invariant():
    a = ObjectGeom((0.0, 0.0), (0.0, 1.0), 100)
    b = ObjectGeom((0.0, 0.0), (0.0, -1.0), 100)        # same line, opposite PCA sign
    axis_cos, _ = axis_features(a, b)
    assert axis_cos == pytest.approx(1.0)               # |cos|, so sign does not matter


def test_geometry_cache_computes_only_requested_ids():
    labels = np.zeros((100, 200), dtype=np.int32)
    labels[10:14, 20:120] = 5                           # horizontal
    labels[40:120, 50:54] = 9                           # vertical
    cache = geometry_cache(labels, [5, 9, 999])
    assert set(cache) == {5, 9}                         # 999 absent, silently skipped
    assert abs(cache[5].axis[1]) > 0.9                  # id 5 runs along x
    assert abs(cache[9].axis[0]) > 0.9                  # id 9 runs along y


def test_axis_features_default_to_zero_without_geometry():
    fiber = _bridged_field(8000)
    f = compute_features(fiber, None, (30, 80), (30, 120), 26.0, 0.99, PIXEL_UM)
    assert f.axis_cos == 0.0 and f.displacement_along_axis == 0.0


def test_field_background_ignores_bright_fibres():
    fiber = _bridged_field(600)
    assert field_background(fiber) == pytest.approx(600, abs=50)


# ----------------------------------------------------------------- model


def _pair(well, frag, cand, bridge, label):
    feats = LinkFeatures(gap_um=20.0, min_cos=0.9, bridge_over_bg=bridge,
                         bridge_mean_over_bg=bridge, territory_frac=0.5)
    return LinkPair(well, frag, cand, feats, label=label)


def _separable_pairs():
    """Positives have a bright bridge, negatives a dark one, across three wells."""
    pairs = []
    for w in ("wA", "wB", "wC"):
        for i in range(4):
            pairs.append(_pair(w, f"f{i}", f"c{i}", bridge=3.0 + 0.1 * i, label=1))
            pairs.append(_pair(w, f"f{i}", f"d{i}", bridge=1.0 + 0.1 * i, label=0))
    return pairs


def test_linker_separates_bright_from_dark_bridges():
    pairs = _separable_pairs()
    result = leave_one_well_out_auc(pairs, ("bridge_over_bg",))
    assert result["auc"] is not None and result["auc"] > 0.9
    assert result["skipped_wells"] == []


def test_fit_linker_needs_enough_positives():
    thin = [_pair("wA", "f0", "c0", 3.0, 1)] + \
           [_pair("wA", f"f{i}", f"d{i}", 1.0, 0) for i in range(10)]
    with pytest.raises(RuntimeError, match="positives"):
        fit_linker(thin, ("bridge_over_bg",))


def test_pair_key_includes_well_because_ids_repeat():
    a = _pair("wA", "myotube_0005", "myotube_0009", 2.0, 1)
    b = _pair("wB", "myotube_0005", "myotube_0009", 2.0, 0)
    assert a.key() != b.key()                            # same ids, different wells
    assert a.key() == ("wA", "myotube_0005", "myotube_0009")


# ------------------------------------------------------------ uncertainty


def test_uncertainty_peaks_at_one_half():
    assert uncertainty(0.5) == pytest.approx(1.0)
    assert uncertainty(0.0) == pytest.approx(0.0)
    assert uncertainty(1.0) == pytest.approx(0.0)
    assert uncertainty(0.75) == pytest.approx(0.5)
    # ordering: a 0.55 prediction is more uncertain than a 0.9 one
    assert uncertainty(0.55) > uncertainty(0.9)


def test_feature_sets_are_small_first():
    # the search must include the single strongest feature on its own
    assert FEATURE_SETS["bridge_only"] == ("bridge_over_bg",)
    assert len(FEATURE_SETS["bridge_only"]) < len(FEATURE_SETS["all"])
