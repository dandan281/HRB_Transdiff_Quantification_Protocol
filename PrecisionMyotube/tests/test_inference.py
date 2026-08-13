import json

import numpy as np
import pytest

from precision_myotube.inference import adapt_label_image, adapt_overlap_json
from precision_myotube.schema import decode_rle


def test_label_adapter_preserves_count_confidence_and_unreviewed_state(tmp_path):
    labels = np.zeros((10, 12), dtype=np.uint16)
    labels[1:4, 2:6] = 1
    labels[6:9, 7:11] = 2
    confidence = np.zeros(labels.shape, dtype=np.float32)
    confidence[labels == 1] = 0.8
    confidence[labels == 2] = 0.6
    labels_path = tmp_path / "labels.npy"
    confidence_path = tmp_path / "confidence.npy"
    np.save(labels_path, labels)
    np.save(confidence_path, confidence)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    result = adapt_label_image(
        labels_path, image_id="field", expected_shape=labels.shape,
        architecture="cellpose-sam", checkpoint=checkpoint,
        environment={"python": "3.11"}, thresholds={"cellprob": 0.0},
        confidence_path=confidence_path)

    assert len(result.instances) == 2
    assert all(record.status == "complete" and not record.reviewed
               for record in result.instances)
    assert [record.confidence for record in result.instances] == pytest.approx([0.8, 0.6])
    assert result.provenance["checkpoint_sha256"]
    assert result.provenance["thresholds"] == {"cellprob": 0.0}


def test_polygon_adapter_preserves_overlapping_instances(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps({
        "image_id": "field",
        "image_shape": [12, 12],
        "instances": [
            {"id": "a", "polygon": [1, 1, 8, 1, 8, 8, 1, 8], "confidence": 0.9},
            {"id": "b", "polygon": [4, 4, 10, 4, 10, 10, 4, 10], "confidence": 0.7}
        ]
    }), encoding="utf-8")

    result = adapt_overlap_json(path, image_id="field", expected_shape=(12, 12),
                                architecture="omnipose")
    first, second = [decode_rle(record.rle) for record in result.instances]
    assert np.count_nonzero(first & second) > 0
    assert all(not record.reviewed for record in result.instances)


def test_adapter_rejects_wrong_source_identity(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps({
        "image_id": "wrong", "image_shape": [5, 5],
        "instances": [{"polygon": [0, 0, 3, 0, 3, 3]}]
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        adapt_overlap_json(path, image_id="field", architecture="micro-sam")
