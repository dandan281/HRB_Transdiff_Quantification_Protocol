"""Tests for the T02 candidate-2 (Omnipose) harness.

These run in `pm-annotate`, which has **no torch**, so nothing here may import the
framework. That is deliberate: the parts worth pinning are the data contract and
the ignore policy, and those are pure numpy/scipy. `train_fold` and `infer_fold`
keep their torch imports inside functions precisely so their module-level
contracts stay testable here.

The properties pinned are the ones whose violation would be silent:

* a reviewed target is never painted or eroded (paint must not eat ground truth);
* every ignored pixel really is replaced (an unpainted ambiguous fibre labelled
  background is the exact failure `_shared/training_masks.py` exists to prevent);
* every instance is whole in at least one tile (a truncated target teaches the
  `fragment_too_short` error class the operator flagged most);
* a fold never sees its held-out well.
"""
from __future__ import annotations

import numpy as np
import pytest

from omnipose_lab.data import (Tile, WellData, _border_cut_ids, _span,
                               dataset_hash, instance_tiles)
from omnipose_lab.ignore_policy import (POLICIES, ambiguous_as_background,
                                        measure_halo, paint_out,
                                        tile_exclusion_report)
from omnipose_lab.infer_fold import iter_masks_from_labels


# ------------------------------------------------------------------ fixtures


def synthetic_field(size: int = 300, seed: int = 0):
    """Background + two reviewed fibres + one bright `ambiguous` fibre."""
    rng = np.random.default_rng(seed)
    image = rng.integers(600, 700, size=(size, size)).astype(np.uint16)

    labels = np.zeros((size, size), dtype=np.int32)
    labels[40:60, 30:150] = 1                      # target 1
    labels[200:215, 100:260] = 2                   # target 2
    image[labels > 0] = 9000                       # targets are bright

    ignore = np.zeros((size, size), dtype=bool)
    ignore[120:135, 20:280] = True                 # an ambiguous fibre
    image[ignore] = 8000
    return image, labels, ignore


# --------------------------------------------------------------- paint policy


def test_paint_out_never_touches_a_reviewed_target():
    image, labels, ignore = synthetic_field()
    result = paint_out(image, ignore, labels, seed=0)
    target = labels > 0
    assert not (result.painted & target).any()
    assert np.array_equal(result.image[target], image[target])


def test_paint_out_replaces_every_ignored_pixel():
    image, labels, ignore = synthetic_field()
    result = paint_out(image, ignore, labels, seed=0)
    assert result.painted[ignore].all()
    # and the fibre signal is actually gone, not merely relabelled
    assert np.median(result.image[ignore]) < 1000
    assert result.stats["painted_px"] >= int(ignore.sum())


def test_paint_out_halo_extends_beyond_the_mask():
    image, labels, ignore = synthetic_field()
    tight = paint_out(image, ignore, labels, halo_px=0, seed=0)
    grown = paint_out(image, ignore, labels, halo_px=6, seed=0)
    assert grown.stats["painted_px"] > tight.stats["painted_px"]
    assert int(tight.stats["painted_px"]) == int(ignore.sum())


def test_paint_out_is_deterministic_per_seed():
    image, labels, ignore = synthetic_field()
    a = paint_out(image, ignore, labels, seed=7)
    b = paint_out(image, ignore, labels, seed=7)
    c = paint_out(image, ignore, labels, seed=8)
    assert np.array_equal(a.image, b.image)
    assert not np.array_equal(a.image, c.image)
    assert a.image.dtype == image.dtype


def test_control_arm_leaves_the_image_untouched():
    image, labels, ignore = synthetic_field()
    result = ambiguous_as_background(image, ignore, labels)
    assert np.array_equal(result.image, image)
    assert not result.painted.any()
    assert result.stats["policy"] == "ambiguous_as_background"
    assert set(POLICIES) == {"paint_out", "ambiguous_as_background"}


def test_measure_halo_reports_a_radius_and_background_level():
    image, labels, ignore = synthetic_field()
    halo = measure_halo(image, ignore, labels, max_radius=12)
    assert 1 <= halo["halo_px"] <= 12
    assert 500 < halo["background_reference"] < 800


# ------------------------------------------------- option 1 evidence (rejected)


def test_tile_exclusion_report_counts_clean_and_contaminated_tiles():
    size = 200
    labels = np.zeros((size, size), dtype=np.int32)
    ignore = np.zeros((size, size), dtype=bool)
    labels[10:30, 10:30] = 1                       # tile (0,0): clean
    labels[110:130, 10:30] = 2                     # tile (1,0): contaminated
    ignore[140:160, 40:60] = True
    report = tile_exclusion_report(labels, ignore, tile=100)
    assert report["candidate_tiles"] == 2
    assert report["ignore_free_tiles"] == 1
    assert report["n_instances"] == 2
    assert report["n_instances_whole_in_free_tile"] == 1


# ---------------------------------------------------------------------- tiling


def test_span_grows_to_minimum_and_caps_at_maximum():
    # small object -> grown to the floor
    start, end = _span(slice(100, 120), margin=10, minimum=64, maximum=256, limit=1000)
    assert end - start == 64
    # large object -> capped at the ceiling
    start, end = _span(slice(0, 900), margin=10, minimum=64, maximum=256, limit=1000)
    assert end - start == 256
    # near the edge -> shifted inside, not shrunk
    start, end = _span(slice(2, 20), margin=50, minimum=64, maximum=256, limit=1000)
    assert start == 0 and end - start == 118


def test_border_cut_ids_flags_only_partially_present_instances():
    well = np.zeros((100, 100), dtype=np.int32)
    well[10:20, 10:20] = 1                          # wholly inside the tile below
    well[50:60, 10:20] = 2                          # straddles the tile edge
    full_area = np.bincount(well.ravel())
    tile = well[0:55, 0:100]
    assert _border_cut_ids(tile, full_area) == {2}


def test_every_instance_is_whole_in_some_tile():
    image, labels, ignore = synthetic_field(size=400)
    data = WellData("w", image, labels, ignore, {})
    tiles = instance_tiles(data, tile_px=256, margin_px=32, min_tile_px=96)
    assert tiles
    covered = set()
    for tile in tiles:
        assert isinstance(tile, Tile)
        # tile labels are contiguous 1..N, so cross-well id collisions cannot leak
        values = sorted(set(np.unique(tile.labels).tolist()) - {0})
        assert values == list(range(1, len(values) + 1))
        covered.add(tile.centred_on)
    assert covered == {1, 2}


def test_border_cut_targets_are_painted_out_not_relabelled_background():
    """A target the tile edge cuts must vanish from the image, not just the label.

    Zeroing the label alone would assert `background` over a visibly bright fibre
    -- the same false-negative supervision the ignore policy exists to prevent.
    """
    size = 400
    rng = np.random.default_rng(1)
    image = rng.integers(600, 700, size=(size, size)).astype(np.uint16)
    labels = np.zeros((size, size), dtype=np.int32)
    labels[190:210, 180:220] = 1                    # compact, gets its own small tile
    labels[100:300, 380:400] = 2                    # long, far away, will be cut
    image[labels > 0] = 9000
    data = WellData("w", image, labels, np.zeros((size, size), bool), {})

    tiles = instance_tiles(data, tile_px=256, margin_px=16, min_tile_px=96)
    tile = next(t for t in tiles if t.centred_on == 1)
    if tile.n_dropped_border:
        # wherever a dropped target's pixels fell inside this tile, they are dark
        assert tile.labels.max() == tile.n_instances
        assert int(tile.image.max()) < 9000


def test_dataset_hash_is_deterministic_and_config_sensitive():
    fold = {"config": {"policy": "paint_out", "seed": 0},
            "images": [np.ones((4, 4), np.uint16)],
            "labels": [np.eye(4, dtype=np.int32)],
            "tiles": [{"well": "w", "row": 0, "col": 0}]}
    first = dataset_hash(fold)
    assert first == dataset_hash(fold)
    other = {**fold, "config": {"policy": "ambiguous_as_background", "seed": 0}}
    assert dataset_hash(other) != first


# ------------------------------------------------------------------- inference


def test_iter_masks_yields_one_mask_per_label_without_building_a_list():
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[5:10, 5:10] = 1
    labels[20:30, 20:40] = 2
    labels[45:48, 2:6] = 3
    masks = iter_masks_from_labels(labels)
    assert not isinstance(masks, list)              # a generator, not materialised
    collected = list(masks)
    assert len(collected) == 3
    for value, mask in enumerate(collected, start=1):
        assert mask.dtype == np.bool_
        assert mask.sum() == int((labels == value).sum())
        assert np.array_equal(mask, labels == value)


def test_iter_masks_skips_absent_label_values():
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:5, 2:5] = 3                            # ids 1 and 2 do not exist
    assert len(list(iter_masks_from_labels(labels))) == 1


# ----------------------------------------------------------------- fold safety


def test_build_fold_refuses_a_held_out_well_in_the_training_set():
    from omnipose_lab.data import build_fold

    with pytest.raises(ValueError, match="held-out"):
        build_fold(["a", "b"], "a")


# ---------------------------------------------------------------- run aggregation


def _fold(well, **metrics):
    base = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
            "precision_weighted_score": 0.0, "mean_matched_iou": 0.0,
            "length_mdape": None, "width_mdape": None,
            "false_split_rate": 0.0, "over_merge_rate": 0.0,
            "n_gt": 10, "n_pred": 10, "tp": 5}
    base.update(metrics)
    return {"held_out_well": well, "metrics": base}


def test_aggregate_uses_whole_wells_and_reports_macro_and_micro():
    from omnipose_lab.run_folds import aggregate

    folds = [_fold("a", precision=0.4, recall=0.6, n_gt=100, n_pred=50, tp=40),
             _fold("b", precision=0.8, recall=0.2, n_gt=10, n_pred=10, tp=8)]
    summary = aggregate(folds)
    assert summary["n_folds"] == 2
    assert summary["precision_macro_mean"] == 0.6          # (0.4+0.8)/2
    # micro pools the counts: 48 tp / 60 pred, 48 tp / 110 gt
    assert summary["total_tp"] == 48
    assert summary["micro_precision"] == round(48 / 60, 3)
    assert summary["micro_recall"] == round(48 / 110, 3)


def test_aggregate_skips_none_metrics():
    from omnipose_lab.run_folds import aggregate

    folds = [_fold("a", length_mdape=None), _fold("b", length_mdape=0.2)]
    summary = aggregate(folds)
    assert summary["length_mdape_macro_mean"] == 0.2       # the None is dropped


def test_paired_comparison_pairs_by_well_and_signs_delta_toward_paint_out():
    from omnipose_lab.run_folds import paired_comparison

    by_arm = {
        "paint_out": [_fold("a", mean_matched_iou=0.7), _fold("b", mean_matched_iou=0.5)],
        "ambiguous_as_background": [_fold("a", mean_matched_iou=0.6),
                                    _fold("b", mean_matched_iou=0.5)],
    }
    result = paired_comparison(by_arm)
    assert result["wells"] == ["a", "b"]
    iou = result["delta_paint_out_minus_control"]["mean_matched_iou"]
    assert iou["mean_delta"] == 0.05                       # ((0.7-0.6)+(0.5-0.5))/2
    assert iou["wells_favouring_paint_out"] == 1           # only well a improved


def test_paired_comparison_needs_both_arms():
    from omnipose_lab.run_folds import paired_comparison

    result = paired_comparison({"paint_out": [_fold("a")]})
    assert "both arms required" in result["note"]


# ----------------------------------------------------------------- shared eval GT


def test_eval_gt_matches_sealed_classical_run(tmp_path):
    """The shared GT builder must reproduce the sealed classical run byte-for-byte.

    T03 ranks candidates against each other, so both must be scored on identical
    ground truth. The classical candidate is sealed and its source is not edited to
    serve a later candidate, so equality is verified here rather than assumed.
    """
    import json

    from _shared.eval_gt import ROOT, build_eval_gt, load_bootstrap_manifest

    sealed_path = ROOT / "model_labs/classical/_runs/v1/run_manifest.json"
    bootstrap = ROOT / "PrecisionMyotube/annotation_work/bootstrap_v1/bootstrap_manifest.json"
    if not sealed_path.is_file() or not bootstrap.is_file():
        pytest.skip("sealed classical run or bootstrap_v1 not materialised here")

    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    manifest = load_bootstrap_manifest(bootstrap)
    expected = {f["held_out_well"]: f["eval_gt"] for f in sealed["folds"]}
    assert expected, "sealed run recorded no folds"

    for well, want in expected.items():
        got = build_eval_gt(manifest, well, tmp_path)
        assert got["n_gt"] == want["n_gt"], well
        assert got["sha256"] == want["sha256"], (
            f"{well}: rebuilt evaluation GT differs from the sealed classical run")
    assert sum(w["n_gt"] for w in expected.values()) == 375
