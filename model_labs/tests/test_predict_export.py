"""CL03 shared core -- prediction export, provenance, and channel configs."""
import json

import numpy as np

from _shared.predict_export import (
    ModelProvenance, export_prediction, prediction_dir, masks_from_label_image,
)
from _shared import channel_config as cc
from _shared.synthetic import synthetic_field
from _shared.schema_bridge import InstanceSet


def _prov(model="omnipose"):
    return ModelProvenance(model=model, version="v0-smoke", architecture="test",
                           checkpoint_hash="abc", environment_hash="env1",
                           data_hash="data1", seed=0, channels="desmin_only")


def test_export_label_image_is_unreviewed_and_provenanced(tmp_path):
    _, _, labels = synthetic_field((96, 96), n=4, seed=1)
    prov = _prov()
    info = export_prediction(tmp_path, "img1", labels.shape, prov, label_image=labels)

    reloaded = InstanceSet.load(info["instances"])
    assert reloaded.instances, "should have instances"
    # C02.4: every model output is unreviewed and sourced to the model.
    assert all(not r.reviewed for r in reloaded.instances)
    assert all(r.source == "omnipose" for r in reloaded.instances)
    assert all(r.status == "ambiguous" for r in reloaded.instances)

    manifest = json.loads((tmp_path / "omnipose" / "v0-smoke" /
                           "img1.prediction_manifest.json").read_text())
    assert manifest["provenance"]["checkpoint_hash"] == "abc"
    assert manifest["provenance"]["data_hash"] == "data1"
    assert manifest["n_instances"] == len(reloaded.instances)


def test_prediction_path_is_model_version_scoped(tmp_path):
    a = prediction_dir(tmp_path, _prov("omnipose"))
    b = prediction_dir(tmp_path, _prov("microsam"))
    assert a != b
    assert a.parts[-2:] == ("omnipose", "v0-smoke")


def test_overlap_safe_masks_share_pixels_after_export(tmp_path):
    shape = (40, 40)
    horiz = np.zeros(shape, dtype=bool); horiz[18:22, :] = True
    vert = np.zeros(shape, dtype=bool); vert[:, 18:22] = True
    prov = _prov("microsam")
    info = export_prediction(tmp_path, "cross", shape, prov,
                             masks=[horiz, vert], write_convenience_tiff=False)
    reloaded = InstanceSet.load(info["instances"])
    masks = [m for _, m in reloaded.masks()]
    shared = int(np.logical_and(masks[0], masks[1]).sum())
    assert shared == 16   # crossing survives in the authoritative JSON


def test_channel_config_stack_and_normalization():
    fiber, dapi, _ = synthetic_field((64, 64), n=3, seed=2)
    stack = cc.build_stack({"fiber": fiber, "dapi": dapi}, cc.DESMIN_DAPI)
    assert stack.shape == (2, 64, 64)
    assert 0.0 <= stack.min() and stack.max() <= 1.0
    one = cc.build_stack({"fiber": fiber}, cc.DESMIN_ONLY)
    assert one.shape == (1, 64, 64)


def test_masks_from_label_image_excludes_background():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[6:9, 6:9] = 2
    masks = masks_from_label_image(labels)
    assert len(masks) == 2
