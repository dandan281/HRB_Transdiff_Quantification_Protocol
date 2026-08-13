"""Reproducible candidate-model command generation.

These adapters deliberately do not declare a winner. All predictions must be converted to the
common InstanceSet schema and scored on the same locked annotations.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys


def candidate_commands(train_dir: str | Path, test_dir: str | Path,
                       output_path: str | Path) -> dict:
    train = str(Path(train_dir).resolve())
    test = str(Path(test_dir).resolve())
    cellpose = [
        sys.executable, "-m", "cellpose", "--train", "--dir", train,
        "--test_dir", test, "--img_filter", "_img", "--mask_filter", "_masks",
        "--pretrained_model", "cpsam",
        "--learning_rate", "0.00001", "--weight_decay", "0.1",
        "--n_epochs", "100", "--train_batch_size", "1",
    ]
    omnipose = [
        "omnipose", "--train", "--use_gpu", "--dir", train,
        "--test_dir", test, "--img_filter", "_img", "--mask_filter", "_masks",
        "--nchan", "1", "--all_channels", "--channel_axis", "0",
        "--pretrained_model", "None", "--diameter", "0", "--nclasses", "3",
        "--learning_rate", "0.1", "--RAdam", "--batch_size", "4",
        "--n_epochs", "4000", "--tyx", "256,256",
    ]
    result = {
        "cellpose_sam": {"command": cellpose, "shell": shlex.join(cellpose),
                         "status": "ready_when_annotations_pass_audit"},
        "omnipose": {"command": omnipose, "shell": shlex.join(omnipose),
                     "status": "ready_when_annotations_pass_audit"},
        "micro_sam": {
            "command": ["micro_sam.train", "-h"],
            "shell": "micro_sam.train -h",
            "status": "configure through the installed CLI/UI for the current micro-sam version; "
                      "train the additional instance decoder",
        },
        "classical_baseline": {
            "command": [sys.executable, "-m", "precision_myotube", "proposals"],
            "status": "review-only baseline; never authoritative without curation",
        },
        "selection_rule": "common held-out benchmark; precision weighted 2x recall",
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
