import json

import numpy as np

from precision_myotube.analysis import analyze
from precision_myotube.schema import InstanceRecord, InstanceSet, encode_rle


def _square(labels, label_id, y, x):
    labels[y:y + 3, x:x + 3] = label_id


def test_analysis_separates_field_fusion_from_authoritative_instances(tmp_path):
    shape = (80, 100)
    metadata = {
        "image_id": "synthetic", "image_shape": list(shape), "pixel_um": 1.0,
        "channels": {"fiber": 1, "dapi": 2, "other": [0]},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    territory = np.zeros(shape, bool)
    complete = np.zeros(shape, bool); complete[20:35, 10:70] = True
    truncated = np.zeros(shape, bool); truncated[60:80, 30:90] = True
    territory |= complete | truncated
    np.save(tmp_path / "myotube_territory.npy", territory)

    nuclei = np.zeros(shape, np.int32)
    _square(nuclei, 1, 24, 20)
    _square(nuclei, 2, 24, 45)
    _square(nuclei, 3, 65, 40)
    _square(nuclei, 4, 5, 5)
    np.save(tmp_path / "nuclei_masks.npy", nuclei)

    instances = InstanceSet(shape, "synthetic", [
        InstanceRecord("m1", "complete", encode_rle(complete), reviewed=True),
        InstanceRecord("m2", "complete", encode_rle(truncated), reviewed=True),
    ])
    instance_path = tmp_path / "instances.json"; instances.save(instance_path)
    result = analyze(tmp_path, instance_path, amin_um2=1, amax_um2=100)
    summary = result["summary"]
    assert summary["total_nuclei"] == 4
    assert summary["nuclei_in_myotube_50"] == 3
    assert summary["instances_authoritative"] == 1
    assert summary["instances_truncated"] == 1
    assert next(row for row in result["myotubes"] if row["id"] == "m1")["nuclei_count"] == 2
    statuses = {row["id"]: row["assignment_status"] for row in result["nuclei"]}
    assert statuses[1] == "assigned"
    assert statuses[2] == "assigned"
    assert statuses[3] == "assignment_ambiguous"
    assert statuses[4] == "outside_instances"
    assert any(flag["type"] == "border_status_conflict" for flag in result["qc_flags"])
    assert (tmp_path / "myotube_instances.json").exists()
    assert (tmp_path / "qc_history.jsonl").exists()


def test_overlapping_instances_make_nucleus_assignment_ambiguous(tmp_path):
    shape = (50, 50)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "image_id": "overlap", "image_shape": list(shape), "pixel_um": 1.0,
        "channels": {"fiber": 1, "dapi": 2, "other": [0]},
    }))
    a = np.zeros(shape, bool); a[10:30, 5:30] = True
    b = np.zeros(shape, bool); b[10:30, 20:45] = True
    np.save(tmp_path / "myotube_territory.npy", a | b)
    nuclei = np.zeros(shape, np.int32); nuclei[15:20, 22:27] = 1
    np.save(tmp_path / "nuclei_masks.npy", nuclei)
    instances = InstanceSet(shape, "overlap", [
        InstanceRecord("a", "complete", encode_rle(a), reviewed=True),
        InstanceRecord("b", "complete", encode_rle(b), reviewed=True),
    ])
    path = tmp_path / "instances.json"; instances.save(path)
    result = analyze(tmp_path, path, amin_um2=1, amax_um2=100)
    assert result["nuclei"][0]["assignment_status"] == "assignment_ambiguous"
    assert result["summary"]["nuclei_assignment_ambiguous"] == 1
