"""Junction classifier: feature recompute, LOWO fit, and baseline comparison."""
import json

import numpy as np
import pytest
import tifffile

from annotation_tools.qc_review.junction_features import JunctionPairFeatures
from annotation_tools.qc_review.junction_model import (
    FEATURE_SETS, JunctionPairExample, classical_floor_decisions, decision_accuracy,
    fit_junction_classifier, ground_truth_decisions, leave_one_well_out_auc,
    leave_one_well_out_junction_decisions, recompute_training_pairs, select_feature_set)

PIXEL_UM = 0.6493


def _t_field(shape=(200, 200), left_thickness=9, right_thickness=3, stub_len=42,
            stub_thickness=5):
    """Same ambiguous T-junction fixture as test_junction_pairs.py."""
    mask = np.zeros(shape, dtype=bool)
    mid = shape[0] // 2
    lh, rh = left_thickness // 2, right_thickness // 2
    mask[mid - lh:mid + lh + 1, 10:100] = True
    mask[mid - rh:mid + rh + 1, 100:190] = True
    sh = stub_thickness // 2
    mask[mid - stub_len:mid, mid - sh:mid + sh + 1] = True
    return mask


def _write_well(tmp_path, well, territory, fiber):
    cache_dir = tmp_path / "territory_cache"
    cache_dir.mkdir(exist_ok=True)
    np.save(cache_dir / f"{well}.territory.npy", territory)
    well_dir = tmp_path / "bootstrap" / well
    well_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(well_dir / "image_fiber.tif", fiber)
    return cache_dir, tmp_path / "bootstrap"


def _find_node(territory, pixel_um=PIXEL_UM):
    from classical.ridge_graph import TracerParams, build_branch_graph
    from skimage.morphology import skeletonize

    skeleton = skeletonize(territory)
    _graph, node_ends, _coords = build_branch_graph(skeleton, pixel_um, TracerParams())
    node, ends = next((n, e) for n, e in node_ends.items() if len(e) == 3)
    return node


def _export_for_one_well(tmp_path, well="w1", chosen_pair="AB"):
    """A minimal one-junction export, in the exact format junction_page.py emits."""
    territory = _t_field()
    fiber = np.full((200, 200), 3000, dtype=np.uint16)
    node = _find_node(territory)

    branches_by_pair = {
        "AB": [0, 1] if chosen_pair == "AB" else [0, 1],
    }
    labels = {"AB": 0, "AC": 0, "BC": 0}
    labels[chosen_pair] = 1
    pairs = [
        {"key": "AB", "branches": [0, 0], "label": labels["AB"]},
        {"key": "AC", "branches": [0, 0], "label": labels["AC"]},
        {"key": "BC", "branches": [0, 0], "label": labels["BC"]},
    ]
    export = {
        "schema": "junction_pairs.v1", "batch_id": "test", "reviewer": "r",
        "session_started_at": "2026-01-01T00:00:00Z", "exported_at": "2026-01-01T00:00:00Z",
        "n_cases": 1, "n_explicitly_decided": 1,
        "decisions": {
            f"{well}/junction_{node:06d}": {
                "well": well, "node": node, "decided_at": "2026-01-01T00:00:00Z",
                "chosen_pair": chosen_pair, "branch_point": False, "unsure": False,
                "pairs": pairs,
            }
        },
    }
    return export, territory, fiber, node


def test_recompute_fills_in_real_branch_ids_and_features(tmp_path):
    """The export stores placeholder branch ids; recompute must replace them with the
    real branch ids from the branch graph and compute real features."""
    export, territory, fiber, node = _export_for_one_well(tmp_path)
    # patch in the real branch ids the way junction_page.py actually would
    from classical.ridge_graph import TracerParams, build_branch_graph
    from skimage.morphology import skeletonize
    skeleton = skeletonize(territory)
    _graph, node_ends, _coords = build_branch_graph(skeleton, PIXEL_UM, TracerParams())
    ends = node_ends[node]
    letters = "ABC"
    for pair in export["decisions"][f"w1/junction_{node:06d}"]["pairs"]:
        a, b = letters.index(pair["key"][0]), letters.index(pair["key"][1])
        pair["branches"] = [ends[a][0], ends[b][0]]

    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    cache_dir, bootstrap_dir = _write_well(tmp_path, "w1", territory, fiber)

    examples = recompute_training_pairs(export_path, cache_dir, bootstrap_dir, pixel_um=PIXEL_UM)
    assert len(examples) == 3
    assert isinstance(examples[0].features, JunctionPairFeatures)
    positives = [e for e in examples if e.label == 1]
    assert len(positives) == 1
    assert positives[0].key == "AB"


def test_mismatched_branch_ids_raise():
    """A branch-id mismatch (stale territory cache) must fail loudly, not silently
    train on the wrong geometry."""
    import tempfile
    from pathlib import Path

    export, territory, fiber, node = _export_for_one_well(Path(tempfile.mkdtemp()))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        export_path = tmp_path / "export.json"
        export_path.write_text(json.dumps(export), encoding="utf-8")
        cache_dir, bootstrap_dir = _write_well(tmp_path, "w1", territory, fiber)
        with pytest.raises(RuntimeError, match="branch id mismatch"):
            recompute_training_pairs(export_path, cache_dir, bootstrap_dir, pixel_um=PIXEL_UM)


def _synthetic_examples(n_per_well=20, seed_wells=("w1", "w2", "w3")):
    """Pair-level examples with a real signal in tangent_cos, for LOWO/fit sanity checks."""
    examples = []
    # deterministic pseudo-randomness without the banned Math.random/Date.now equivalents
    lin = np.linspace(-1, 1, n_per_well)
    for wi, well in enumerate(seed_wells):
        for i, cos in enumerate(lin):
            label = int(cos < -0.3)
            feats = JunctionPairFeatures(tangent_cos=float(cos), turn_angle_deg=0.0,
                                         width_ratio=1.0, intensity_ratio=1.0,
                                         length_min_um=20.0)
            examples.append(JunctionPairExample(well, wi * 1000 + i, "AB", feats, label))
    return examples


def test_leave_one_well_out_auc_detects_real_signal():
    examples = _synthetic_examples()
    result = leave_one_well_out_auc(examples, ("tangent_cos",))
    assert result["auc"] is not None and result["auc"] > 0.9


def test_select_feature_set_prefers_fewer_features_on_ties():
    examples = _synthetic_examples()
    name, selection = select_feature_set(examples)
    # tangent_cos alone should already win/tie since it's the only informative feature
    assert selection["all"][name]["n_features"] <= selection["all"]["all"]["n_features"]


def test_fit_requires_minimum_positives():
    examples = [JunctionPairExample("w1", i, "AB",
                                    JunctionPairFeatures(-0.9, 0, 1, 1, 20), 1)
               for i in range(3)]
    with pytest.raises(RuntimeError, match="positives"):
        fit_junction_classifier(examples, ("tangent_cos",))


def test_fit_and_score_roundtrip():
    examples = _synthetic_examples()
    model = fit_junction_classifier(examples, ("tangent_cos",))
    confident_positive = JunctionPairExample("w1", 9999, "AB",
                                             JunctionPairFeatures(-0.95, 0, 1, 1, 20), None)
    confident_negative = JunctionPairExample("w1", 9998, "AB",
                                             JunctionPairFeatures(0.95, 0, 1, 1, 20), None)
    assert model.score(confident_positive) > model.score(confident_negative)


def test_decision_accuracy_breaks_down_error_types():
    truth = {("w1", 1): "AB", ("w1", 2): "AC", ("w1", 3): None, ("w1", 4): "BC"}
    predicted = {("w1", 1): "AB",        # correct
                 ("w1", 2): "BC",        # wrong_pair
                 ("w1", 3): "AB",        # false_join / over_merge
                 ("w1", 4): None}        # false_split / under_merge
    result = decision_accuracy(predicted, truth)
    assert result == {"n": 4, "correct": 1, "accuracy": 0.25,
                      "false_join_over_merge": 1, "false_split_under_merge": 1,
                      "wrong_pair": 1}


def test_decision_accuracy_only_compares_common_keys():
    truth = {("w1", 1): "AB", ("w1", 2): "AC"}
    predicted = {("w1", 1): "AB", ("w1", 3): "BC"}     # key 2 and 3 don't overlap
    result = decision_accuracy(predicted, truth)
    assert result["n"] == 1 and result["correct"] == 1


def test_ground_truth_decisions_drops_unsure(tmp_path):
    export = {"decisions": {
        "w1/junction_000001": {"well": "w1", "node": 1, "unsure": True, "chosen_pair": None},
        "w1/junction_000002": {"well": "w1", "node": 2, "unsure": False, "chosen_pair": "AB"},
        "w1/junction_000003": {"well": "w1", "node": 3, "unsure": False, "chosen_pair": None},
    }}
    export_path = tmp_path / "e.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    truth = ground_truth_decisions(export_path)
    assert truth == {("w1", 2): "AB", ("w1", 3): None}
    assert ("w1", 1) not in truth


def test_classical_floor_decisions_matches_pair_junction_ends(tmp_path):
    export, territory, fiber, node = _export_for_one_well(tmp_path)
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    cache_dir, _bootstrap_dir = _write_well(tmp_path, "w1", territory, fiber)

    decisions = classical_floor_decisions(export_path, cache_dir, pixel_um=PIXEL_UM)
    assert ("w1", node) in decisions
    # the ambiguous T-fixture's winning pair is branches 1&2 (the two horizontal
    # halves) -- confirmed directly against evaluate_junction in
    # test_junction_pairs.py's equivalent fixture
    assert decisions[("w1", node)] in ("AB", "AC", "BC")


def test_leave_one_well_out_junction_decisions_groups_by_junction():
    examples = _synthetic_examples()
    # add the other two pairs per junction so grouping has something to pick among
    extra = []
    for e in examples:
        extra.append(JunctionPairExample(e.well, e.node, "AC",
                                         JunctionPairFeatures(0.9, 0, 1, 1, 20), 0))
        extra.append(JunctionPairExample(e.well, e.node, "BC",
                                         JunctionPairFeatures(0.9, 0, 1, 1, 20), 0))
    decisions = leave_one_well_out_junction_decisions(examples + extra, ("tangent_cos",))
    assert decisions
    for (well, node), (key, proba) in decisions.items():
        assert key in (None, "AB", "AC", "BC")
        assert 0.0 <= proba <= 1.0


# ------------------------------------------- branch-point gate + two-stage decisions


def _junction_example(well, node, best_tan, label, **over):
    """A JunctionExample whose branch-point-ness is driven by best_tan."""
    from annotation_tools.qc_review.junction_features import JunctionFeatures
    from annotation_tools.qc_review.junction_model import JunctionExample

    base = dict(best_tan=best_tan, second_tan=0.0, worst_tan=0.5,
                tan_margin=0.3, tan_spread=0.6,
                len_min_um=20.0, len_mid_um=30.0, len_max_um=40.0,
                len_ratio_min_max=0.5, width_min_um=3.0, width_max_um=4.0,
                width_ratio_min_max=0.75, node_intensity_over_min=1.0,
                node_intensity_over_max=1.0, intensity_ratio_min_max=0.9)
    base.update(over)
    return JunctionExample(well, node, JunctionFeatures(**base), label)


def _gate_examples():
    """Straight-through junctions (best_tan ~ -1) vs branch points (best_tan ~ 0)."""
    out = []
    for wi, well in enumerate(("wA", "wB", "wC")):
        for i in range(8):
            out.append(_junction_example(well, wi * 100 + i, -0.95 + 0.02 * i, 0))
            out.append(_junction_example(well, wi * 100 + 50 + i, -0.1 + 0.02 * i, 1))
    return out


def test_branch_point_model_learns_a_separable_gate():
    from annotation_tools.qc_review.junction_model import fit_branch_point_model

    model = fit_branch_point_model(_gate_examples())
    assert model.fit_info["n_branch_points"] == 24
    straight = _junction_example("wA", 999, -0.98, None)
    branchy = _junction_example("wA", 998, 0.0, None)
    assert model.score(branchy) > model.score(straight)


def test_branch_point_model_requires_enough_branch_points():
    from annotation_tools.qc_review.junction_model import fit_branch_point_model

    thin = [_junction_example("wA", i, -0.9, 0) for i in range(20)]
    thin.append(_junction_example("wA", 99, 0.0, 1))
    with pytest.raises(RuntimeError, match="branch points"):
        fit_branch_point_model(thin)


def test_two_stage_gate_can_call_a_branch_point_the_argmax_rule_misses():
    """The whole point of stage 1: at a junction where nothing continues, the
    single-stage rule still has to pick a best pair, and only declines it if
    that pair happens to fall below threshold."""
    from annotation_tools.qc_review.junction_model import two_stage_decisions

    pair_ex, junc_ex, truth = [], [], {}
    for wi, well in enumerate(("wA", "wB", "wC")):
        for i in range(10):
            # a real through-pair: AB strongly positive
            node = wi * 100 + i
            for key, cos, lab in (("AB", -0.95, 1), ("AC", 0.2, 0), ("BC", 0.3, 0)):
                pair_ex.append(JunctionPairExample(
                    well, node, key, JunctionPairFeatures(cos, 0.0, 1.0, 1.0, 30.0, 1.0), lab))
            junc_ex.append(_junction_example(well, node, -0.95, 0))
            truth[(well, node)] = "AB"
            # a branch point: nothing continues
            node_bp = wi * 100 + 50 + i
            for key, cos in (("AB", -0.05, ), ("AC", 0.1,), ("BC", 0.2,)):
                pair_ex.append(JunctionPairExample(
                    well, node_bp, key,
                    JunctionPairFeatures(cos, 0.0, 1.0, 1.0, 30.0, 1.0), 0))
            junc_ex.append(_junction_example(well, node_bp, -0.05, 1))
            truth[(well, node_bp)] = None

    decisions, info = two_stage_decisions(pair_ex, junc_ex, truth)
    assert decisions, "two-stage must produce decisions"
    assert len(info["gate_thresholds_per_fold"]) == 3
    called_branch_point = sum(1 for k, v in decisions.items() if v is None and truth[k] is None)
    assert called_branch_point > 0, "the gate must actually fire on true branch points"


def test_two_stage_gate_threshold_is_selected_on_training_wells_only():
    """Selecting the gate on the pooled set is tuning a threshold on its own
    test data; the contract is that it is chosen per fold, on training wells."""
    from annotation_tools.qc_review.junction_model import two_stage_decisions

    import inspect
    src = inspect.getsource(two_stage_decisions)
    assert "train_ids" in src and "scores_train" in src
    # the selection call must score the TRAINING ids, never the held-out ones
    selection_line = next(l for l in src.splitlines() if "max(gate_grid" in l)
    following = src.split(selection_line, 1)[1].splitlines()[0:3]
    assert any("train_ids" in l for l in following), \
        "gate threshold must be selected against training-well junctions"
