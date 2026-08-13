from __future__ import annotations

import json

import numpy as np
import tifffile

from _shared.schema_bridge import InstanceRecord, InstanceSet, encode_rle
from freeze_bootstrap import _export_well, _is_excluded_pair, _plan


def test_export_well_applies_binding_exclusion(tmp_path):
    root = tmp_path / "work"
    package = root / "pkg"
    package.mkdir(parents=True)
    shape = (12, 14)
    fiber = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    dapi = np.flipud(fiber)
    tifffile.imwrite(package / "fiber_raw16.tif", fiber)
    tifffile.imwrite(package / "dapi_raw16.tif", dapi)

    masks = []
    for bounds in [(1, 1, 5, 5), (4, 4, 9, 9), (0, 10, 3, 14)]:
        mask = np.zeros(shape, dtype=bool)
        r0, c0, r1, c1 = bounds
        mask[r0:r1, c0:c1] = True
        masks.append(mask)
    records = [
        InstanceRecord("myotube_0001", "complete", encode_rle(masks[0]), reviewed=True),
        InstanceRecord("myotube_0002", "complete", encode_rle(masks[1]), reviewed=True),
        InstanceRecord("myotube_0003", "border_truncated", encode_rle(masks[2]), reviewed=True),
    ]
    InstanceSet(shape, "well_a", records).save(package / "well_a.qc.instances.json")
    exclusions = {("well_a", "myotube_0002")}

    plan = _plan(root, [("pkg", "well_a")], exclusions)
    assert plan["trainable_complete"] == 1
    assert plan["per_well"]["well_a"]["excluded"] == ["myotube_0002"]

    result = _export_well(root, "pkg", "well_a", tmp_path / "out", exclusions)
    labels = tifffile.imread(tmp_path / "out" / "labels.tif")
    assert result["n_trainable_instances"] == 1
    assert set(np.unique(labels)) == {0, 1}
    mapping = [json.loads(line) for line in (tmp_path / "out" / "instance_mapping.jsonl").read_text().splitlines()]
    assert mapping == [{"source_id": "myotube_0001", "train_label": 1}]


def test_synthetic_pair_inherits_source_exclusion():
    exclusions = {("well_a", "myotube_0002")}
    assert _is_excluded_pair({"stem": "well_a", "id": "myotube_0002"}, exclusions)
    assert _is_excluded_pair(
        {"stem": "well_a", "id": "myotube_0001+myotube_0002"}, exclusions
    )
    assert not _is_excluded_pair({"stem": "well_a", "id": "myotube_0001"}, exclusions)
