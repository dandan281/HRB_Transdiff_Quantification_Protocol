"""T02 candidate 2 - train real Omnipose on one leave-one-well-out fold.

Runs in the isolated `pm-omnipose` environment only. `verify_env.verify()` is a
hard precondition rather than a human courtesy check: on this workstation a
CPU-only torch silently replaces the CUDA build, and a wrong-arch CUDA build
reports `is_available() == True` and then dies at the first kernel launch. A
training run that started on such an environment would waste hours or, worse,
quietly produce a CPU-trained checkpoint labelled as GPU.

Configuration decisions worth knowing:

**Initialisation is transfer, not random.** ``init_model="bact_phase_affinity"``
fine-tunes from Omnipose's shipped bacterial weights, which is what
``DEVELOPMENT_PLAN.md`` S10 asks for and what 375 instances argues for. The choice
of *which* bacterial model is forced: it is the only one that does not silently
rewrite ``nchan`` and the head shape. See the ``init_model`` note in ``DEFAULTS``,
and the architecture assertion after the model is constructed -- that guard exists
because the wrong choice here fails by training the wrong network, not by raising.
Pass ``--init-model scratch`` for random initialisation.

``nclasses=2`` (3 output channels: 2 flow + distance). Omnipose adds ``dim-1``
internally. The boundary variant (``nclasses=3`` -> 4 channels) costs a channel
for nothing in this version: `omnipose.core.loss` computes a BCE boundary loss
and then **unconditionally overwrites** ``bd_loss`` with the derivative loss, so
the boundary head receives no gradient of its own.

``rescale=False``. Omnipose's rescaling maps object *diameter* onto
``diam_mean``, which is meaningless for long fibres whose length varies ~5x while
width barely moves. Training at native scale keeps the pixel size honest
(0.6493 um/px) and is what the inference path also assumes.

``min_train_masks=1``. The upstream default of 5 would silently discard training
tiles -- our tiles are built around single reviewed instances and many legitimately
contain fewer than five.

Seeds are set for python, numpy and torch, and cuDNN is put in deterministic mode.
That makes a rerun reproducible on the same GPU; it does not make results
bit-identical across different hardware, and the manifest records which GPU ran.

Usage (from pm-omnipose)::

    python model_labs/omnipose_lab/train_fold.py --held-out 23_B02_ctrl \\
        --policy paint_out --out model_labs/omnipose_lab/_runs/v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MODEL_NAME = "omnipose"
MODEL_VERSION = "v1"
BOOTSTRAP = "PrecisionMyotube/annotation_work/bootstrap_v1/bootstrap_manifest.json"

DEFAULTS = {
    "n_epochs": 300,
    "batch_size": 8,
    "tyx": 384,
    "learning_rate": 0.1,
    "weight_decay": 1e-5,
    "nclasses": 2,
    "seed": 0,
    # Omnipose recomputes flows per crop, which dominates wall-clock: measured
    # 0.09 s/batch of GPU work inside a 57 s epoch. Its worker-based dataloader
    # would move that off the critical path, but `cellpose_omni/core.py:1223` calls
    # `set_start_method('fork', force=True)` unconditionally in that branch, and
    # Windows has no fork -- `dataloader=True` raises "cannot find context for
    # 'fork'" here regardless of num_workers. Forcing spawn would also push flow
    # computation onto the CPU (`core.py:1217`), so the in-process path is both
    # the only one available and not obviously slower. Recorded, not worked around.
    "dataloader": False,
    "num_workers": 0,
    # Peak GPU was 7.17 GB at batch 2 / tyx 256 on a 12 GB card; batch 8 / tyx 384
    # exhausted it and torch surfaced that as `CUDA error: unknown error` rather
    # than a clean OOM -- budget memory by measurement, not by the error message.
    # Autocast would buy headroom back but cannot run here: cellpose_omni calls
    # `autocast()` with no arguments, and torch 2.11 requires `device_type`. Left
    # off rather than patched, so this stays stock Omnipose.
    "autocast": False,
    # Initialisation. `DEVELOPMENT_PLAN.md` S10 asks for "real Omnipose transfer/
    # fine-tuning", and `bact_phase_affinity` is the only shipped bacterial model
    # whose architecture matches this one exactly: it appears in neither
    # `C2_MODEL_NAMES` nor `BD_MODEL_NAMES`, so it leaves `nchan=1` and the
    # 3-channel (2 flow + 1 scalar) head intact. Verified against the weights --
    # 311/311 tensors, zero shape mismatches.
    #
    # `bact_phase_omni` does NOT work here, and fails in the worst way. It is in
    # both lists, so `CellposeModel.__init__` rewrites nchan to 2 and the head to
    # the 4-channel boundary variant BEFORE building the net (`models.py:453-461`),
    # loads the checkpoint without complaint, and then wants 2-channel input we do
    # not have plus the boundary head this harness deliberately rejects.
    #
    # Known caveat, accepted: this model was trained with `affinity_field`, so its
    # scalar channel is a summed connectivity graph rather than a distance field
    # (`omnipose/core.py:1094`). The mismatch is confined to the 33 parameters of
    # that output slice out of 6.6M -- the encoder and both flow channels transfer
    # normally. Set to None for random initialisation.
    "init_model": "bact_phase_affinity",
}


def set_seeds(seed: int) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_one_fold(held_out: str, *, policy: str, include_round2: bool,
                   out_dir: Path, config: dict,
                   augment_gaps: bool = False) -> dict:
    import torch
    from cellpose_omni import models

    from omnipose_lab.data import build_fold, dataset_hash
    from omnipose_lab.env import verify

    environment = verify()                     # hard precondition: real GPU kernel
    set_seeds(config["seed"])

    manifest_path = ROOT / BOOTSTRAP
    wells = sorted(json.loads(manifest_path.read_text(encoding="utf-8"))["per_well"])
    if held_out not in wells:
        raise ValueError(f"{held_out} is not a bootstrap well")
    train_wells = [w for w in wells if w != held_out]

    started = time.time()
    fold = build_fold(train_wells, held_out, policy=policy,
                      include_round2=include_round2, seed=config["seed"],
                      augment_gaps=augment_gaps)
    data_hash = dataset_hash(fold)
    prep_seconds = time.time() - started
    print(f"  fold data: {fold['n_tiles']} tiles, {fold['n_instances']} instance slots, "
          f"hash {data_hash[:12]} ({prep_seconds:.0f}s)")

    # Leakage guards. Cheap, and the contract explicitly requires them.
    assert held_out not in fold["train_wells"], "held-out well in training set"
    assert all(t["well"] != held_out for t in fold["tiles"]), "held-out tiles present"

    tag = (f"{MODEL_VERSION}-fold-{held_out}-{policy}"
           + ("-r2" if include_round2 else "")
           + ("-gap" if augment_gaps else ""))
    model_dir = out_dir / "checkpoints" / tag
    model_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the pretrained weights before building the net. `model_path` fetches
    # them from GitHub on first use into CELLPOSE_LOCAL_MODELS_PATH (default
    # ~/.cellpose/models), so on a compute node with no outbound network this is
    # where the run dies -- better here, loudly, than after the GPU is claimed.
    init_model = config.get("init_model") or None
    init_sha = None
    if init_model:
        init_sha = sha256_file(models.model_path(init_model, 0, True))
        print(f"  init: fine-tuning from {init_model} ({init_sha[:12]})")
    else:
        print("  init: from scratch, random weights")

    model = models.CellposeModel(
        gpu=True, omni=True, dim=2, nchan=1, nclasses=config["nclasses"],
        diam_mean=0.0,
        **({"model_type": init_model} if init_model
           else {"pretrained_model": False}))

    # A named model rewrites nchan and nclasses to fit its own checkpoint and then
    # loads cleanly, so the failure mode is a silent architecture swap rather than
    # an error. Assert we got the architecture we asked for.
    expected_out = config["nclasses"] + 1          # Omnipose adds dim-1 internally
    if (model.nchan, model.nclasses) != (1, expected_out):
        raise SystemExit(
            f"FAIL: init_model={init_model!r} silently changed the architecture.\n"
            f"      nchan        = {model.nchan} (want 1)\n"
            f"      out-channels = {model.nclasses} (want {expected_out})\n"
            "      Use a model in neither C2_MODEL_NAMES nor BD_MODEL_NAMES, "
            "or set init_model=None to train from scratch.")

    torch.cuda.reset_peak_memory_stats()
    train_started = time.time()
    failure = None
    try:
        checkpoint = model.train(
            [t.astype(np.float32) for t in fold["images"]],
            [t.astype(np.int32) for t in fold["labels"]],
            # Upstream defaults `train_links=None` and then iterates it; an
            # explicit list of Nones is the "no linked labels" case. We have no
            # self-contact links: each reviewed instance is one connected object.
            train_links=[None] * len(fold["images"]),
            # Tiles arrive already percentile-normalised at *field* scope
            # (`omnipose_lab.data.normalize_field`). Letting Omnipose normalise
            # each tile again would restore the per-image scaling that field-level
            # normalisation exists to avoid.
            channels=None, channel_axis=None, normalize=False,
            save_path=str(model_dir), save_every=max(50, config["n_epochs"]),
            n_epochs=config["n_epochs"], learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"], batch_size=config["batch_size"],
            SGD=True, rescale=False, min_train_masks=1,
            tyx=(config["tyx"], config["tyx"]),
            dataloader=config["dataloader"], num_workers=config["num_workers"],
            do_autocast=config["autocast"],
            netstr=tag)
    except Exception as exc:                    # record, never silently continue
        failure = f"{type(exc).__name__}: {exc}"
        print(f"  !! training FAILED: {failure}")
        raise
    finally:
        train_seconds = time.time() - train_started
        peak_gpu = float(torch.cuda.max_memory_allocated()) / 1e9

    checkpoint = Path(checkpoint)
    record = {
        "task": "T02", "candidate": MODEL_NAME, "candidate_version": MODEL_VERSION,
        "held_out_well": held_out, "train_wells": train_wells,
        "ignore_policy": policy, "include_round2": include_round2,
        "augment_gaps": augment_gaps,
        "n_synthetic_gap_tiles": fold.get("n_synthetic_gap_tiles", 0),
        # Initialisation is part of what the candidate IS, so it is recorded at the
        # top level rather than left implicit inside `config`: a manifest that does
        # not distinguish transfer from random init cannot support a T03 claim.
        "init_model": init_model,
        "init_model_sha256": init_sha,
        "config": config,
        "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT)
                      else str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "input_manifest": BOOTSTRAP,
        "input_manifest_sha256": sha256_file(manifest_path),
        "dataset_sha256": data_hash,
        "n_train_tiles": fold["n_tiles"], "n_train_instance_slots": fold["n_instances"],
        "n_border_painted": sum(t["n_dropped_border"] for t in fold["tiles"]),
        "per_well": {w: {k: v for k, v in s.items() if k != "round2_promoted_ids"}
                     for w, s in fold["per_well"].items()},
        "environment": environment,
        "environment_hash": environment["environment_hash"],
        "timing": {"data_prep_seconds": round(prep_seconds, 1),
                   "train_seconds": round(train_seconds, 1)},
        "peak_gpu_gb": round(peak_gpu, 2),
        "platform": platform.platform(),
        "failure": failure,
        # The 2,290 bootstrap synthetic PAIRS remain unused. Gap augmentation is a
        # separate thing: it perturbs the image around masks the operator certified
        # and never invents a mask, so it is disclosed on its own field above.
        "synthetic_pairs_used": False,
        "correction_pairs_used": False,
        "evidence_class": "development_bootstrap_single_operator_proposal_conditioned",
    }
    (model_dir / "train_manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(f"  trained {tag}: {train_seconds:.0f}s, peak GPU {peak_gpu:.2f} GB")
    print(f"  checkpoint {checkpoint}")
    return record


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Training knobs shared by the train_fold and run_folds CLIs.

    One definition on purpose. Stage 2 once ran a different configuration from
    the probe that sized it because `--dataloader`/`--num-workers` existed on
    this parser but not on run_folds' (16 GPU-hours, zero folds; session state
    2026-08-12). Any new training knob goes here, never on one script's parser.
    """
    parser.add_argument("--epochs", type=int, default=DEFAULTS["n_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--tyx", type=int, default=DEFAULTS["tyx"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--dataloader", action="store_true",
                        help="use Omnipose's worker dataloader; REQUIRED on Linux "
                             "for the epoch times the probe measured (fails on "
                             "Windows: fork)")
    parser.add_argument("--autocast", action="store_true",
                        help="mixed precision (broken upstream against torch 2.11)")
    parser.add_argument("--init-model", default=DEFAULTS["init_model"],
                        help="pretrained Omnipose model to fine-tune from, or "
                             "'scratch' for random initialisation. Must be a model "
                             "in neither C2_MODEL_NAMES nor BD_MODEL_NAMES or the "
                             "architecture check will reject it (default: "
                             f"{DEFAULTS['init_model']})")
    return parser


def config_from_args(args: argparse.Namespace) -> dict:
    """The one mapping from shared CLI args to a training config dict."""
    return {**DEFAULTS, "n_epochs": args.epochs, "batch_size": args.batch_size,
            "tyx": args.tyx, "learning_rate": args.lr, "seed": args.seed,
            "num_workers": args.num_workers,
            "dataloader": args.dataloader,
            "autocast": args.autocast,
            "init_model": None if args.init_model == "scratch" else args.init_model}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--held-out", required=True)
    parser.add_argument("--policy", default="paint_out",
                        choices=["paint_out", "ambiguous_as_background"])
    parser.add_argument("--include-round2", action="store_true")
    parser.add_argument("--augment-gaps", action="store_true",
                        help="append synthetically gapped copies of training tiles; "
                             "the gap distribution is measured from the training "
                             "wells of this fold only (see gap_augment)")
    parser.add_argument("--out", default="model_labs/omnipose_lab/_runs/v1")
    add_training_args(parser)
    args = parser.parse_args(argv)

    config = config_from_args(args)
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    record = train_one_fold(args.held_out, policy=args.policy,
                            augment_gaps=args.augment_gaps,
                            include_round2=args.include_round2,
                            out_dir=out_dir, config=config)
    print(json.dumps({k: record[k] for k in
                      ("held_out_well", "ignore_policy", "include_round2",
                       "augment_gaps", "n_synthetic_gap_tiles",
                       "n_train_tiles", "dataset_sha256", "checkpoint_sha256",
                       "timing", "peak_gpu_gb")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
