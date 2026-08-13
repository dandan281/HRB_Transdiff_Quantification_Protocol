import csv

import numpy as np

from precision_myotube.cli import main
from precision_myotube.schema import InstanceSet


def test_import_labels_preserves_per_instance_properties(tmp_path):
    labels = np.zeros((12, 14), dtype=np.uint16)
    labels[2:6, 2:7] = 1
    labels[7:11, 6:12] = 2
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)
    properties_path = tmp_path / "properties.csv"
    with properties_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "label", "id", "status", "reviewed", "source", "notes"])
        writer.writeheader()
        writer.writerow({"label": 1, "id": "tube_a", "status": "complete",
                         "reviewed": "yes", "source": "consensus", "notes": ""})
        writer.writerow({"label": 2, "id": "tube_b", "status": "border_truncated",
                         "reviewed": "true", "source": "consensus", "notes": "edge"})
    out = tmp_path / "instances.json"
    main(["import-labels", "--labels", str(labels_path), "--image-id", "field",
          "--properties", str(properties_path), "--out", str(out)])
    result = InstanceSet.load(out)
    assert [(r.id, r.status, r.reviewed) for r in result.instances] == [
        ("tube_a", "complete", True), ("tube_b", "border_truncated", True)]
