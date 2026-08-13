"""Guardrails for synthetic gap augmentation.

The load-bearing properties are that the operator's mask is never altered, that
only the targeted instance loses signal, and that the gap distribution cannot
quietly be drawn from a held-out well. Everything else is rendering.
"""
from __future__ import annotations

import numpy as np
import pytest

from omnipose_lab.gap_augment import (Gap, apply_synthetic_gap, augment_tile,
                                      geodesic_order, measure_gaps,
                                      skeleton_profile)

PX_UM = 0.6493
BG = 100.0
FIBRE = 900.0


def _straight(shape=(60, 300), row=30, col0=20, col1=280, thickness=7):
    mask = np.zeros(shape, dtype=bool)
    mask[row - thickness // 2:row + thickness // 2 + 1, col0:col1] = True
    return mask


def _image_from(mask, level=FIBRE, background=BG):
    image = np.full(mask.shape, background, dtype=float)
    image[mask] = level
    return image


# ------------------------------------------------------------------- ordering

def test_geodesic_order_follows_a_bent_fibre_end_to_end():
    # An L-shaped fibre. Ordering by projection onto a principal axis would
    # interleave the two arms; the geodesic walk must not.
    mask = np.zeros((120, 120), dtype=bool)
    mask[60:63, 10:80] = True
    mask[10:63, 77:80] = True
    from skimage.morphology import skeletonize
    ordered = geodesic_order(skeletonize(mask))
    assert ordered is not None
    steps = np.abs(np.diff(ordered, axis=0)).max(axis=1)
    assert steps.max() <= 1          # consecutive pixels are 8-connected


def test_a_disconnected_skeleton_returns_nothing_rather_than_guessing():
    mask = np.zeros((60, 200), dtype=bool)
    mask[28:32, 10:60] = True
    mask[28:32, 140:190] = True
    from skimage.morphology import skeletonize
    assert geodesic_order(skeletonize(mask)) is None


# ------------------------------------------------------------------ measuring

def test_a_real_gap_is_found_with_roughly_the_right_length():
    mask = _straight()
    image = _image_from(mask)
    image[:, 150:180] = BG                       # 30 px gap in the signal
    ordered, profile = skeleton_profile(mask, image)
    gaps = measure_gaps(profile, BG)
    assert len(gaps) == 1
    assert gaps[0].length_um == pytest.approx(30 * PX_UM, rel=0.35)
    assert gaps[0].depth < 0.1


def test_a_fibre_with_no_dip_reports_no_gap():
    mask = _straight()
    assert measure_gaps(skeleton_profile(mask, _image_from(mask))[1], BG) == []


def test_a_tapering_tip_is_not_counted_as_an_internal_gap():
    mask = _straight()
    image = _image_from(mask)
    image[:, :45] = BG                            # dim at one end, not inside
    gaps = measure_gaps(skeleton_profile(mask, image)[1], BG)
    assert gaps == []


# ----------------------------------------------------------------- augmenting

def test_the_operator_mask_is_never_modified():
    mask = _straight()
    image = _image_from(mask)
    before = mask.copy()
    apply_synthetic_gap(image, mask, background=BG,
                        distribution=[Gap(20.0, 0.1)],
                        rng=np.random.default_rng(0))
    assert np.array_equal(mask, before)


def test_the_gap_actually_lowers_signal_inside_the_fibre():
    mask = _straight()
    image = _image_from(mask)
    record = apply_synthetic_gap(image, mask, background=BG,
                                 distribution=[Gap(20.0, 0.1)],
                                 rng=np.random.default_rng(0))
    assert record is not None
    assert record["pixels_attenuated"] > 0
    assert image[mask].min() < 0.5 * FIBRE


@pytest.mark.parametrize("thickness", [5, 11, 21])
def test_the_synthesized_gap_measures_back_at_the_length_requested(thickness):
    # The bug this pins: dilating the centreline band isotropically grows the gap
    # ALONG the fibre by its half-width at each end, so a wide fibre gets a much
    # longer gap than was sampled. It survived every other test in this file and
    # only showed up when the measurement was round-tripped -- at 7 um requested
    # the corpus re-measured at 16 um. Thickness is parametrised because the
    # error scaled with fibre width, which is what made it diagnosable.
    mask = _straight(shape=(80, 400), row=40, col0=20, col1=380,
                     thickness=thickness)
    image = _image_from(mask)
    requested = 30 * PX_UM
    record = apply_synthetic_gap(image, mask, background=BG,
                                 distribution=[Gap(requested, 0.05)],
                                 rng=np.random.default_rng(5))
    assert record is not None
    measured = max(g.length_um for g in
                   measure_gaps(skeleton_profile(mask, image)[1], BG))
    assert measured == pytest.approx(requested, rel=0.4)


def test_a_neighbouring_fibre_crossing_the_band_keeps_its_signal():
    # Otherwise the augmentation teaches that two objects vanish there, which is
    # a different and false lesson.
    target = _straight(shape=(120, 300), row=40)
    neighbour = np.zeros((120, 300), dtype=bool)
    neighbour[70:77, 20:280] = True
    image = _image_from(target | neighbour)
    apply_synthetic_gap(image, target, background=BG,
                        distribution=[Gap(30.0, 0.05)],
                        rng=np.random.default_rng(3))
    assert image[neighbour].min() == pytest.approx(FIBRE)


def test_the_gap_is_placed_away_from_both_tips():
    mask = _straight()
    image = _image_from(mask)
    for seed in range(15):
        fresh = _image_from(mask)
        record = apply_synthetic_gap(fresh, mask, background=BG,
                                     distribution=[Gap(10.0, 0.1)],
                                     rng=np.random.default_rng(seed))
        if record is None:
            continue
        margin = 0.15 * record["skeleton_px"]
        assert record["start_index"] >= margin - 1
        assert record["start_index"] + record["gap_px"] <= record["skeleton_px"] - margin + 1
    assert image is not None


def test_a_fibre_too_short_for_the_sampled_gap_is_left_alone():
    mask = _straight(shape=(40, 60), row=20, col0=10, col1=50)
    image = _image_from(mask)
    before = image.copy()
    record = apply_synthetic_gap(image, mask, background=BG,
                                 distribution=[Gap(200.0, 0.1)],
                                 rng=np.random.default_rng(0))
    assert record is None
    assert np.array_equal(image, before)


def test_augmentation_is_deterministic_under_a_seed():
    mask = _straight()
    dist = [Gap(10.0, 0.2), Gap(30.0, 0.05), Gap(50.0, 0.15)]
    first, second = _image_from(mask), _image_from(mask)
    a = apply_synthetic_gap(first, mask, background=BG, distribution=dist,
                            rng=np.random.default_rng(11))
    b = apply_synthetic_gap(second, mask, background=BG, distribution=dist,
                            rng=np.random.default_rng(11))
    assert a == b
    assert np.array_equal(first, second)


def test_an_empty_distribution_is_a_no_op_not_a_crash():
    mask = _straight()
    image = _image_from(mask)
    before = image.copy()
    assert apply_synthetic_gap(image, mask, background=BG, distribution=[],
                               rng=np.random.default_rng(0)) is None
    assert np.array_equal(image, before)


# ----------------------------------------------------------------------- tiles

def test_tile_augmentation_gaps_some_instances_and_not_others():
    labels = np.zeros((200, 300), dtype=np.int32)
    labels[_straight(shape=(200, 300), row=40)] = 1
    labels[_straight(shape=(200, 300), row=100)] = 2
    labels[_straight(shape=(200, 300), row=160)] = 3
    image = _image_from(labels > 0)
    out, records = augment_tile(image, labels, background=BG,
                                distribution=[Gap(20.0, 0.1)],
                                rng=np.random.default_rng(1), probability=0.5)
    assert np.array_equal(image, _image_from(labels > 0))   # input untouched
    assert 0 < len(records) < 3                              # not all, not none
    touched = {r["label"] for r in records}
    for label_id in (1, 2, 3):
        region = labels == label_id
        if label_id in touched:
            assert out[region].min() < 0.5 * FIBRE
        else:
            assert out[region].min() == pytest.approx(FIBRE)


def test_centred_instance_resolves_to_its_local_tile_id():
    # The bug this pins: `centred_on` is a WELL-level id, `tile.labels` is remapped
    # to contiguous local ids. Comparing them directly gives an empty mask, so the
    # augmentation ran and changed nothing -- 2 tiles augmented out of 61 instead
    # of 30, with no error anywhere.
    from omnipose_lab.data import Tile, local_label_id

    tile = Tile(well="w", row=0, col=0, size=(10, 10),
                image=np.zeros((10, 10)), labels=np.zeros((10, 10), dtype=np.int32),
                centred_on=33, whole_ids=(7, 19, 33, 41), n_instances=4,
                n_dropped_border=0)
    assert local_label_id(tile) == 3
    assert tile.centred_on != local_label_id(tile)      # the trap itself


def test_a_border_cut_centred_instance_resolves_to_nothing():
    from omnipose_lab.data import Tile, local_label_id

    tile = Tile(well="w", row=0, col=0, size=(10, 10),
                image=np.zeros((10, 10)), labels=np.zeros((10, 10), dtype=np.int32),
                centred_on=33, whole_ids=(7, 19), n_instances=2, n_dropped_border=1)
    assert local_label_id(tile) is None


def test_corpus_distribution_has_no_default_well_list():
    # A convenient "all six" default is how a held-out well ends up inside a
    # training statistic without anyone choosing that.
    import inspect

    from omnipose_lab.gap_augment import corpus_gap_distribution
    signature = inspect.signature(corpus_gap_distribution)
    assert signature.parameters["wells"].default is inspect.Parameter.empty
