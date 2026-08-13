import numpy as np
import pytest
import inspect

from precision_myotube.linked_candidate import (
    DEFAULT_MERGE_POLICY,
    DEFAULT_OUT,
    LINKER_LIMITATIONS,
    RELEASE_STATUS,
    _micro_summary,
    _model_spec,
    merge_prediction,
    prediction_to_label_image,
    run_linked_candidate,
)
from precision_myotube.schema import (
    InstanceRecord,
    InstanceSet,
    encode_rle,
)


def _prediction(overlap=False):
    first = np.zeros((8, 10), dtype=bool)
    second = np.zeros_like(first)
    third = np.zeros_like(first)
    first[1:3, 1:4] = True
    second[1:3, 5:7] = True
    third[5:7, 7:9] = True
    if overlap:
        second[2, 3] = True
    return InstanceSet(
        (8, 10),
        "well",
        [
            InstanceRecord("a", "complete", encode_rle(first), source="model", reviewed=False),
            InstanceRecord("b", "complete", encode_rle(second), source="model", reviewed=False),
            InstanceRecord("c", "complete", encode_rle(third), source="model", reviewed=False),
        ],
    )


def test_prediction_to_label_image_preserves_the_sealed_mutually_exclusive_masks():
    prediction = _prediction()
    labels = prediction_to_label_image(prediction)
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    for index, (_, mask) in enumerate(prediction.masks(), start=1):
        assert np.array_equal(labels == index, mask)


def test_prediction_to_label_image_refuses_to_flatten_overlap():
    with pytest.raises(ValueError, match="mutually exclusive"):
        prediction_to_label_image(_prediction(overlap=True))


def test_merge_prediction_is_transitive_stable_and_unreviewed():
    shape = (120, 120)
    masks = []
    for row, (start, stop) in zip((10, 40, 70), ((5, 35), (40, 70), (75, 105))):
        mask = np.zeros(shape, dtype=bool)
        mask[row:row + 3, start:stop] = True
        masks.append(mask)
    prediction = InstanceSet(
        shape,
        "well",
        [InstanceRecord(chr(97 + index), "complete", encode_rle(mask),
                        source="model", reviewed=False)
         for index, mask in enumerate(masks)],
    )
    linked, merge_map, refused = merge_prediction(
        prediction,
        [(0.95, 2, 3), (0.90, 1, 2)],
        provenance={"model": "linked"},
        cos_min=0.70,
    )
    assert refused == []
    assert len(linked.instances) == 1
    assert merge_map[0]["base_instance_indices"] == [1, 2, 3]
    assert merge_map[0]["base_instance_ids"] == ["a", "b", "c"]
    assert linked.instances[0].status == "complete"
    assert not linked.instances[0].reviewed
    assert linked.provenance == {"model": "linked"}
    expected = np.logical_or.reduce([mask for _, mask in prediction.masks()])
    assert np.array_equal(next(linked.masks())[1], expected)


def test_merge_prediction_rejects_unknown_or_self_edges():
    prediction = _prediction()
    for edge in ((1, 1), (1, 4)):
        with pytest.raises(ValueError, match="invalid linker edge"):
            merge_prediction(
                prediction, [(0.95, *edge)], provenance={}, cos_min=0.70
            )


def test_merge_prediction_refuses_axis_inconsistent_edge_and_records_it():
    shape = (100, 100)
    horizontal = np.zeros(shape, dtype=bool)
    vertical = np.zeros(shape, dtype=bool)
    horizontal[20:23, 10:60] = True
    vertical[30:80, 70:73] = True
    prediction = InstanceSet(
        shape,
        "well",
        [
            InstanceRecord("a", "complete", encode_rle(horizontal),
                           source="model", reviewed=False),
            InstanceRecord("b", "complete", encode_rle(vertical),
                           source="model", reviewed=False),
        ],
    )
    linked, merge_map, refused = merge_prediction(
        prediction, [(0.999, 1, 2)], provenance={}, cos_min=0.70
    )
    assert len(linked.instances) == 2
    assert [row["base_instance_indices"] for row in merge_map] == [[1], [2]]
    assert len(refused) == 1
    assert refused[0]["fragments"] == [1, 2]
    assert refused[0]["blocking_pair"] == [1, 2]
    assert refused[0]["reason"] == "axis pair outside the declared window"


class _Features:
    def __init__(self, values):
        self.values = values

    def vector(self, keys):
        return [self.values[key] for key in keys]


class _Pair:
    def __init__(self, well, fragment, candidate, label, values):
        self.well = well
        self.fragment_id = fragment
        self.candidate_id = candidate
        self.label = label
        self.features = _Features(values)


class _Scaler:
    mean_ = np.array([1.0, 2.0])
    scale_ = np.array([0.5, 0.25])


class _Logistic:
    classes_ = np.array([0, 1])
    coef_ = np.array([[0.1, 0.2]])
    intercept_ = np.array([-0.3])
    C = 1.0
    class_weight = "balanced"
    solver = "lbfgs"
    max_iter = 1000


class _Model:
    keys = ("x", "y")
    scaler = _Scaler()
    model = _Logistic()
    fit_info = {"n": 2, "n_positive": 1}


def test_model_spec_is_deterministic_and_binds_only_declared_training_wells():
    pairs = [
        _Pair("train_b", "f2", "c2", 0, {"x": 2.0, "y": 3.0}),
        _Pair("train_a", "f1", "c1", 1, {"x": 1.0, "y": 4.0}),
    ]
    first = _model_spec(_Model(), pairs, ["train_a", "train_b"])
    second = _model_spec(_Model(), list(reversed(pairs)), ["train_a", "train_b"])
    assert first == second
    assert first["training_wells"] == ["train_a", "train_b"]
    assert [row["well"] for row in first["training_rows"]] == ["train_a", "train_b"]
    assert len(first["training_rows_sha256"]) == 64


def test_new_run_manifest_defaults_cannot_restore_withdrawn_safety_claims():
    row = {
        "n_gt": 10,
        "n_pred": 20,
        "tp": 8,
        "false_split_count": 2,
        "over_merge_count": 1,
    }
    summary = _micro_summary([{"held_out_metrics": row}])
    assert summary["over_merge_rates_interpretable"] is False
    assert summary["recall_resolution_per_reviewed_object"] == 0.1
    assert "rejected_development_baseline_only" in RELEASE_STATUS
    joined = " ".join(LINKER_LIMITATIONS).lower()
    assert "must be judged against corrected evidence" not in joined
    assert "must not be used as a proposal source" in joined
    assert "0.6487" in joined
    assert "must not be used for manual-qc proposals" in joined


def test_new_candidate_defaults_are_constrained_and_sealed_candidate_gate_is_explicit():
    from annotation_tools.qc_review.link_model import recompute_training_pairs

    assert DEFAULT_MERGE_POLICY == "constrained_axis"
    assert DEFAULT_OUT.endswith("classical_linker_constrained_v2")
    source = inspect.getsource(run_linked_candidate)
    assert 'require_axis_agreement = merge_policy == "constrained_axis"' in source
    assert "require_axis_agreement=require_axis_agreement" in source
    assert "cannot use the sealed v1 run id" in source
    training_source = inspect.getsource(recompute_training_pairs)
    assert "require_axis_agreement=require_axis_agreement" in training_source
