"""Framework-agnostic laboratory smoke test (CL03.2).

Verifies the full lab plumbing -- synthetic data layout, channel normalization,
mask handling, checkpoint writing, and canonical prediction export -- without
requiring the GPU framework to be installed. When the real framework *is*
present, pass a ``predictor`` callable to exercise it; otherwise a deterministic
threshold fallback stands in so the export path is still validated end to end.

The completion check (CL03.2) is: one short run completes and inference returns
masks that pass the canonical adapter/validator.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import channel_config as cc
from .synthetic import synthetic_field
from .predict_export import ModelProvenance, export_prediction
from .schema_bridge import InstanceSet


def _threshold_predictor(stack: np.ndarray) -> np.ndarray:
    """Deterministic stand-in 'model': label connected bright regions on ch0."""
    from scipy import ndimage as ndi
    fg = stack[0] > 0.5
    labels, _ = ndi.label(fg)
    return labels.astype(np.int32)


def run_smoke(model: str, out_dir: str | Path, *, framework_available: bool = False,
              predictor=None, channels: str = "desmin_only", seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fiber, dapi, gt_labels = synthetic_field((128, 128), n=5, seed=seed)
    cfg = cc.CONFIGS[channels]
    stack = cc.build_stack({"fiber": fiber, "dapi": dapi}, cfg)

    predict = predictor or _threshold_predictor
    pred_labels = predict(stack)
    if pred_labels.shape != fiber.shape:
        raise RuntimeError("predictor returned wrong shape")

    # Write a checkpoint stub to prove the lab can persist a model artifact.
    checkpoint = out_dir / f"{model}_smoke.checkpoint.json"
    checkpoint.write_text(json.dumps({"model": model, "trained": bool(framework_available),
                                      "note": "smoke checkpoint stub"}), encoding="utf-8")

    prov = ModelProvenance(
        model=model, version="v0-smoke", architecture=model,
        checkpoint_hash="smoke", environment_hash="smoke-env",
        data_hash="smoke-data", seed=seed, channels=channels,
        thresholds={"fg": 0.5}, used_prompts=False)
    info = export_prediction(out_dir, "synthetic_smoke", fiber.shape, prov,
                             label_image=pred_labels)

    # Canonical adapter/validator must accept the smoke prediction.
    reloaded = InstanceSet.load(info["instances"])
    reloaded.validate()

    log = {
        "model": model, "framework_available": bool(framework_available),
        "channels": channels, "gt_instances": int(gt_labels.max()),
        "predicted_instances": len(reloaded.instances),
        "all_unreviewed": all(not r.reviewed for r in reloaded.instances),
        "canonical_validation": "passed",
        "checkpoint": str(checkpoint),
        "instances": info["instances"], "manifest": info["manifest"],
    }
    (out_dir / f"{model}_smoke_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log
