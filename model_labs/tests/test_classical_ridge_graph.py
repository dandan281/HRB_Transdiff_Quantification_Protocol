"""T02 candidate 1 - classical ridge/graph tracer, gates, and fold honesty."""
import numpy as np
import pytest

from classical.ridge_graph import (
    pair_junction_ends,
    FilterParams, TracerParams, assign_and_filter, assign_territory,
    filter_assigned, instance_positions, iter_masks, trace_fibers_parameterised)
from classical.run_folds import SELECTION_METRIC, select_params
from precision_myotube.fiber_gate import trace_fibers
from precision_myotube.schema import InstanceSet


PIXEL_UM = 0.6493


def _cross_field(shape=(160, 160), thickness=5):
    """Two straight fibres crossing at right angles (one junction)."""
    mask = np.zeros(shape, dtype=bool)
    mid = shape[0] // 2
    half = thickness // 2
    mask[mid - half:mid + half + 1, 10:-10] = True     # horizontal
    mask[10:-10, mid - half:mid + half + 1] = True     # vertical
    return mask


def _spurred_field(shape=(160, 160), thickness=5):
    """One long fibre with a short stub, so spur pruning has something to prune."""
    mask = np.zeros(shape, dtype=bool)
    mid = shape[0] // 2
    half = thickness // 2
    mask[mid - half:mid + half + 1, 10:-10] = True     # long horizontal
    mask[mid - 12:mid, 60:60 + thickness] = True       # short perpendicular stub
    return mask


def test_defaults_reproduce_canonical_tracer_grouping():
    """At canonical constants the parameterised tracer must group identically.

    The canonical `fiber_gate.trace_fibers` hard-codes SPUR_UM=10 and
    STRAIGHT_DOT=-0.5; this candidate re-implements it with those exposed. If the
    two ever diverge at the defaults, the candidate has silently stopped being a
    faithful parameterisation of the validated recipe.
    """
    mask = _cross_field()
    _, _, canonical = trace_fibers(mask, PIXEL_UM)
    parameterised = trace_fibers_parameterised(
        mask, PIXEL_UM, TracerParams(spur_um=10.0, straight_dot=-0.5))
    canonical_lengths = np.sort([length for length, _ in canonical])
    ours = np.sort(parameterised.lengths_um[1:])
    assert canonical_lengths.shape == ours.shape
    assert np.allclose(canonical_lengths, ours)


def test_crossing_is_traced_as_two_fibres_not_four_branches():
    """Anti-parallel pairing must carry both fibres straight through the junction."""
    trace = trace_fibers_parameterised(_cross_field(), PIXEL_UM, TracerParams())
    assert trace.n_fibers == 2, f"expected 2 traced fibres, got {trace.n_fibers}"


def test_assignment_covers_all_territory_including_pruned_spurs():
    """Every territory pixel is assigned; spur territory is never orphaned.

    The canonical `length_gated_territory` measures distance to the whole
    skeleton, so territory nearest a pruned spur is dropped. Stage C measures
    distance to retained fibre pixels only, which must assign 100%.
    """
    mask = _spurred_field()
    trace = trace_fibers_parameterised(mask, PIXEL_UM, TracerParams())
    assigned, _ = assign_territory(trace)
    assert trace.n_fibers >= 1
    assert (assigned > 0).sum() == mask.sum(), "unassigned territory remains"
    assert not (assigned > 0)[~mask].any(), "assignment leaked outside the territory"


def test_looser_junction_pairing_never_increases_fragmentation():
    """Relaxing `straight_dot` merges more at junctions, so fibre count falls."""
    mask = _cross_field()
    strict = trace_fibers_parameterised(mask, PIXEL_UM, TracerParams(straight_dot=-0.9))
    loose = trace_fibers_parameterised(mask, PIXEL_UM, TracerParams(straight_dot=0.3))
    assert loose.n_fibers <= strict.n_fibers


def test_length_and_area_gates_filter_instances():
    trace = trace_fibers_parameterised(_cross_field(), PIXEL_UM, TracerParams())
    assigned, areas = assign_territory(trace)
    kept, _ = filter_assigned(trace, assigned, areas, FilterParams(min_length_um=0.0,
                                                                   min_area_px=1))
    dropped, _ = filter_assigned(trace, assigned, areas,
                                 FilterParams(min_length_um=10_000.0, min_area_px=1))
    assert len(kept) >= 2 and dropped == []
    huge_area, _ = filter_assigned(trace, assigned, areas,
                                   FilterParams(min_length_um=0.0, min_area_px=10 ** 9))
    assert huge_area == []


def test_assign_and_filter_masks_are_mutually_exclusive():
    masks, debug = assign_and_filter(
        trace_fibers_parameterised(_cross_field(), PIXEL_UM, TracerParams()),
        FilterParams(min_length_um=0.0, min_area_px=1))
    assert len(masks) >= 2
    stacked = np.sum(np.stack(masks).astype(int), axis=0)
    assert stacked.max() <= 1, "classical floor must emit mutually exclusive masks"
    assert debug["n_instances"] == len(masks)


def test_empty_territory_is_handled():
    trace = trace_fibers_parameterised(np.zeros((32, 32), dtype=bool), PIXEL_UM)
    assert trace.n_fibers == 0
    masks, debug = assign_and_filter(trace)
    assert masks == [] and debug["n_instances"] == 0


def test_instance_positions_matches_naive_masking():
    """The one-pass grouping must equal `assigned == i`, in the same order.

    `instance_positions` exists to avoid allocating a 13 MB field per instance;
    it must stay pixel-identical to the naive version it replaced.
    """
    assigned = np.zeros((12, 10), dtype=np.int32)
    assigned[2, 1:4] = 3
    assigned[5, 0:6] = 1
    assigned[9, 7:9] = 7          # ids deliberately unsorted and non-contiguous
    kept = [1, 3, 7]
    grouped = instance_positions(assigned, kept)
    assert [fid for fid, _ in grouped] == kept
    for fiber_id, positions in grouped:
        rows, cols = np.nonzero(assigned == fiber_id)
        expected = np.sort(rows.astype(np.int64) + cols.astype(np.int64) * assigned.shape[0])
        assert np.array_equal(np.sort(positions), expected)


def test_instance_positions_skips_ids_with_no_pixels():
    assigned = np.zeros((8, 8), dtype=np.int32)
    assigned[3, 1:5] = 2
    assert [fid for fid, _ in instance_positions(assigned, [2, 5])] == [2]
    assert instance_positions(np.zeros((4, 4), dtype=np.int32), [1]) == []


def test_iter_masks_streams_one_at_a_time():
    assigned = np.zeros((8, 8), dtype=np.int32)
    assigned[1, 1:4] = 1
    assigned[5, 2:6] = 2
    masks = iter_masks(assigned, [1, 2])
    first = next(masks)
    assert first.dtype == bool and first.sum() == 3
    assert next(masks).sum() == 4


def test_filter_assigned_returns_ids_not_masks():
    """Guards the memory contract: full fields must never be materialised here."""
    trace = trace_fibers_parameterised(_cross_field(), PIXEL_UM, TracerParams())
    assigned, areas = assign_territory(trace)
    kept, _ = filter_assigned(trace, assigned, areas,
                              FilterParams(min_length_um=0.0, min_area_px=1))
    assert kept and all(isinstance(i, int) for i in kept)


# ------------------------------------------------------------------ fold honesty


def _table(scores: dict[str, list[float]]):
    """Build a grid-score table shaped like run_folds' internal table."""
    return {well: [{SELECTION_METRIC: value, "param_index": i}
                   for i, value in enumerate(values)]
            for well, values in scores.items()}


def test_selection_ignores_the_held_out_well():
    """A held-out well that loves param 1 must not drag selection away from 0."""
    grid = [None, None]
    table = _table({
        "train_a": [0.9, 0.1],
        "train_b": [0.9, 0.1],
        "held_out": [0.0, 9.9],      # would win outright if it were consulted
    })
    index, score = select_params(table, ["train_a", "train_b"], grid)
    assert index == 0
    assert score == pytest.approx(0.9)


def test_selection_ties_break_to_lowest_index():
    grid = [None, None, None]
    table = _table({"w1": [0.5, 0.5, 0.5], "w2": [0.5, 0.5, 0.5]})
    assert select_params(table, ["w1", "w2"], grid)[0] == 0


def test_selection_uses_the_mean_across_training_wells():
    """Param 1 wins on one well but param 0 wins on average."""
    grid = [None, None]
    table = _table({"w1": [0.6, 0.0], "w2": [0.6, 0.9], "w3": [0.6, 0.0]})
    index, score = select_params(table, ["w1", "w2", "w3"], grid)
    assert index == 0
    assert score == pytest.approx(0.6)


# ------------------------------------------------------- export / scoring contract


def test_exported_predictions_are_unreviewed_but_scoreable(tmp_path):
    """`status="complete"` is required or the benchmark scores zero detections.

    `benchmark_instances` counts only predictions whose status is "complete";
    the export default of "ambiguous" would silently yield n_pred == 0.
    """
    from _shared.predict_export import ModelProvenance, export_prediction
    from precision_myotube.benchmark import benchmark_instances
    from precision_myotube.schema import InstanceRecord, encode_sparse_positions

    shape = (64, 64)
    mask = np.zeros(shape, dtype=bool)
    mask[20:24, 5:60] = True
    rows, cols = np.nonzero(mask)
    positions = rows.astype(np.int64) + cols.astype(np.int64) * shape[0]
    gt_path = tmp_path / "gt.json"
    InstanceSet(shape, "field", [InstanceRecord(
        id="myotube_0001", status="complete", reviewed=True, source="qc_review",
        rle=encode_sparse_positions(shape, positions))]).save(gt_path)

    prov = ModelProvenance(model="classical_ridge_graph", version="test", seed=0)
    info = export_prediction(tmp_path, "field", shape, prov, masks=[mask],
                             write_convenience_tiff=False, status="complete")
    reloaded = InstanceSet.load(info["instances"])
    assert all(not r.reviewed for r in reloaded.instances), "must stay unreviewed"
    assert all(r.status == "complete" for r in reloaded.instances)

    metrics = benchmark_instances(gt_path, info["instances"])
    assert metrics["n_pred"] == 1 and metrics["tp"] == 1


def test_export_default_status_is_still_ambiguous(tmp_path):
    """The conservative default must not change for existing callers."""
    from _shared.predict_export import ModelProvenance, export_prediction

    shape = (32, 32)
    mask = np.zeros(shape, dtype=bool)
    mask[10:14, 4:28] = True
    info = export_prediction(tmp_path, "field", shape,
                             ModelProvenance(model="m", version="v"), masks=[mask],
                             write_convenience_tiff=False)
    assert all(r.status == "ambiguous" for r in InstanceSet.load(info["instances"]).instances)


# --------------------------------------------- junction_decider injection point


def test_default_decider_is_the_classical_rule():
    """`junction_decider=None` must leave the sealed floor bit-identical.

    The learned classifier is injected here rather than by forking the tracer,
    so an A/B comparison differs in the junction rule and nothing else. If the
    default ever stopped reproducing `pair_junction_ends`, the sealed floor
    would silently change underneath every prior result.
    """
    mask = _cross_field()
    baseline = trace_fibers_parameterised(mask, PIXEL_UM, TracerParams())
    explicit = trace_fibers_parameterised(
        mask, PIXEL_UM, TracerParams(),
        junction_decider=lambda node, ends, coords: pair_junction_ends(
            ends, TracerParams().straight_dot))
    assert baseline.n_fibers == explicit.n_fibers
    assert np.allclose(np.sort(baseline.lengths_um), np.sort(explicit.lengths_um))
    assert np.array_equal(baseline.fiber_id, explicit.fiber_id)


def test_decider_that_refuses_every_pairing_fragments_the_crossing():
    """A decider returning no pairs must leave branches unjoined -- the
    representation the branch-point gate relies on."""
    trace = trace_fibers_parameterised(_cross_field(), PIXEL_UM, TracerParams(),
                                       junction_decider=lambda node, ends, coords: [])
    paired = trace_fibers_parameterised(_cross_field(), PIXEL_UM, TracerParams())
    assert trace.n_fibers > paired.n_fibers


def test_decider_receives_the_node_and_its_ends():
    seen = []

    def spy(node, ends, coordinates):
        seen.append((node, len(ends)))
        return pair_junction_ends(ends, TracerParams().straight_dot)

    trace_fibers_parameterised(_cross_field(), PIXEL_UM, TracerParams(), junction_decider=spy)
    assert seen, "the decider must be called"
    assert any(degree >= 3 for _node, degree in seen), "a crossing must reach the decider"
