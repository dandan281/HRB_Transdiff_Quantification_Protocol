"""T02 candidate 2 - six-fold leave-one-well-out Omnipose run, both policy arms.

Mirrors the contract the classical floor already satisfies
(`coordination/requests/claude/2026-07-21-t02-start.md`): whole-well folds, no
crop shared between a fold's training and held-out wells, canonical unreviewed
`InstanceSet` export with full provenance, and a run manifest recording commands,
hashes, seeds, timing and failures.

The ablation
------------
`CellposeModel.train()` has no per-pixel loss mask, so pixels the operator never
certified cannot simply be excluded from the loss. `omnipose_lab.ignore_policy`
records why tile exclusion and loss masking were both rejected and why painting
those regions out to real local background was chosen. That choice is
consequential, so it is **run as an experiment rather than asserted**: every fold
is trained twice, once under `paint_out` and once under the naive
`ambiguous_as_background` control that treats uncertified pixels as background.

Both arms share the fold split, the seed, the evaluation ground truth and the
inference path, so a paired per-well comparison isolates the policy. If the arms
are indistinguishable, that is the honest finding and it will be reported as such.

Failures do not abort the run: a fold that dies is recorded in `failures` and the
remaining folds continue, because a partial six-fold result with a visible gap is
more useful than no result at all -- and the contract requires failures be visible.

Usage (from pm-omnipose)::

    python model_labs/omnipose_lab/run_folds.py --out model_labs/omnipose_lab/_runs/v1
    python model_labs/omnipose_lab/run_folds.py --arms paint_out --epochs 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
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
ARMS = ("paint_out", "ambiguous_as_background")

METRIC_KEYS = ("precision", "recall", "f1", "precision_weighted_score",
               "mean_matched_iou", "length_mdape", "width_mdape",
               "false_split_rate", "over_merge_rate",
               # The predeclared T03 primary is false_split_count POOLED across
               # wells (training plan §5(b)); the counts were previously dropped
               # from every aggregate even though infer_fold returns them.
               "false_split_count", "over_merge_count")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fmt(value, digits=3):
    return round(value, digits) if isinstance(value, float) else value


def aggregate(folds: list[dict]) -> dict:
    """Macro summary. Whole held-out wells are the unit -- never single instances.

    `STATISTICAL_ANALYSIS_PLAN.md` governs the interval estimates T03 produces;
    this is a descriptive per-arm summary only.
    """
    if not folds:
        return {}
    out: dict = {}
    for key in METRIC_KEYS:
        values = [f["metrics"][key] for f in folds if f["metrics"].get(key) is not None]
        out[f"{key}_macro_mean"] = _fmt(float(np.mean(values))) if values else None
        out[f"{key}_macro_median"] = _fmt(float(np.median(values))) if values else None
    out["n_folds"] = len(folds)
    for total in ("n_gt", "n_pred", "tp"):
        out[f"total_{total}"] = sum(f["metrics"][total] for f in folds)
    # Pooled counts, the form the T03 primary is stated in (classical floor: 52).
    # None (not 0) when any fold lacks the key, so a missing metric can never
    # masquerade as a perfect score.
    for total in ("false_split_count", "over_merge_count"):
        values = [f["metrics"].get(total) for f in folds]
        out[f"total_{total}"] = (sum(values) if all(v is not None for v in values)
                                 else None)
    micro_p = out["total_tp"] / out["total_n_pred"] if out["total_n_pred"] else 0.0
    micro_r = out["total_tp"] / out["total_n_gt"] if out["total_n_gt"] else 0.0
    out["micro_precision"] = _fmt(micro_p)
    out["micro_recall"] = _fmt(micro_r)
    return out


def paired_comparison(by_arm: dict[str, list[dict]]) -> dict:
    """Per-well paired difference between the two policy arms.

    Paired by well because the arms differ only in the ignore policy; an unpaired
    comparison would drown a real effect in between-well variance, which across
    these six wells is large (reviewed instance counts range 35-119).
    """
    if set(by_arm) != set(ARMS):
        return {"note": "both arms required for a paired comparison"}
    primary = {f["held_out_well"]: f for f in by_arm["paint_out"]}
    control = {f["held_out_well"]: f for f in by_arm["ambiguous_as_background"]}
    shared = sorted(set(primary) & set(control))
    if not shared:
        return {"note": "no well completed in both arms"}

    per_well, deltas = {}, {}
    for key in METRIC_KEYS:
        values = []
        for well in shared:
            a = primary[well]["metrics"].get(key)
            b = control[well]["metrics"].get(key)
            if a is None or b is None:
                continue
            values.append(a - b)
            per_well.setdefault(well, {})[key] = {"paint_out": a,
                                                  "ambiguous_as_background": b,
                                                  "delta": _fmt(a - b)}
        if values:
            deltas[key] = {"mean_delta": _fmt(float(np.mean(values))),
                           "median_delta": _fmt(float(np.median(values))),
                           "wells_favouring_paint_out": int(sum(v > 0 for v in values)),
                           "n_wells": len(values)}
    return {
        "wells": shared, "per_well": per_well, "delta_paint_out_minus_control": deltas,
        "interpretation": (
            "Positive delta favours paint_out. With six wells and one operator this "
            "is descriptive evidence about the training policy, not a hypothesis "
            "test; no p-value is computed and none should be inferred."),
    }


def _fold_result_path(out_dir: Path, arm: str, well: str) -> Path:
    return out_dir / "fold_results" / f"{arm}__{well}.json"


def run(out_dir: Path, arms: list[str], config: dict, include_round2: bool,
        wells_filter: list[str] | None, resume: bool = True,
        augment_gaps: bool = False) -> dict:
    from omnipose_lab.env import verify
    from omnipose_lab.infer_fold import DEFAULT_THRESHOLDS, infer_one_fold
    from omnipose_lab.train_fold import train_one_fold

    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fold_results").mkdir(parents=True, exist_ok=True)
    environment = verify()

    manifest_path = ROOT / BOOTSTRAP
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wells = sorted(manifest["per_well"])
    held_out_wells = [w for w in wells if not wells_filter or w in wells_filter]

    failures: list[dict] = []
    by_arm: dict[str, list[dict]] = {arm: [] for arm in arms}
    resumed: list[str] = []

    for arm in arms:
        for held_out in held_out_wells:
            label = f"{arm}/{held_out}"
            # Per-fold resume. Each fold is independent -- its own checkpoint,
            # its own sealed prediction -- so a completed fold is replayable from
            # its sidecar. This turns a mid-run crash (the machine has been
            # blue-screening; see coordination/reports/claude_resume_state.md) or a
            # deliberate stop into a resume instead of restarting the whole run.
            # The sidecar is only written after inference succeeds and its config
            # is checked, so a half-finished fold is never resumed.
            result_path = _fold_result_path(
                out_dir, arm + ("-gap" if augment_gaps else ""), held_out)
            if resume and result_path.is_file():
                cached = json.loads(result_path.read_text(encoding="utf-8"))
                if (cached.get("_config_seed") == config["seed"]
                        and cached.get("_config_epochs") == config["n_epochs"]
                        and cached.get("include_round2") == include_round2
                        # Without this, an un-augmented sidecar is silently
                        # resumed into the augmented arm and the ablation
                        # compares a run against itself.
                        and bool(cached.get("_augment_gaps", False)) == augment_gaps
                        # Same failure, different axis: initialisation is part of
                        # what the candidate is, so a from-scratch sidecar must not
                        # be replayed into a fine-tuned run. Sidecars written before
                        # this field existed have no key and are treated as stale
                        # rather than assumed to match.
                        and cached.get("_init_model", "__absent__")
                            == config.get("init_model")):
                    by_arm[arm].append(cached)
                    resumed.append(label)
                    m = cached["metrics"]
                    print(f"\n=== {label} === (resumed) P={m['precision']:.3f} "
                          f"R={m['recall']:.3f} IoU={m['mean_matched_iou']:.3f}")
                    continue
                print(f"\n=== {label} === (stale sidecar: config differs, re-running)")
            else:
                print(f"\n=== {label} ===")
            try:
                trained = train_one_fold(held_out, policy=arm,
                                         include_round2=include_round2,
                                         out_dir=out_dir, config=config,
                                         augment_gaps=augment_gaps)
            except Exception as exc:
                failures.append({"stage": "train", "arm": arm, "well": held_out,
                                 "error": f"{type(exc).__name__}: {exc}"})
                print(f"  !! train FAILED: {type(exc).__name__}: {exc}")
                continue
            try:
                scored = infer_one_fold(
                    held_out, Path(trained["checkpoint"]) if Path(trained["checkpoint"]).is_absolute()
                    else ROOT / trained["checkpoint"],
                    out_dir, policy=arm, include_round2=include_round2,
                    train_manifest=trained, thresholds=dict(DEFAULT_THRESHOLDS),
                    seed=config["seed"])
            except Exception as exc:
                failures.append({"stage": "infer", "arm": arm, "well": held_out,
                                 "error": f"{type(exc).__name__}: {exc}"})
                print(f"  !! infer FAILED: {type(exc).__name__}: {exc}")
                continue
            scored["train"] = {k: trained[k] for k in (
                "dataset_sha256", "n_train_tiles", "n_train_instance_slots",
                "n_border_painted", "train_wells", "timing", "peak_gpu_gb")}
            scored["_config_seed"] = config["seed"]
            scored["_config_epochs"] = config["n_epochs"]
            scored["_augment_gaps"] = augment_gaps
            scored["_init_model"] = config.get("init_model")
            scored["n_synthetic_gap_tiles"] = trained.get("n_synthetic_gap_tiles", 0)
            result_path.write_text(json.dumps(scored, indent=2, default=str),
                                   encoding="utf-8")
            by_arm[arm].append(scored)
            m = scored["metrics"]
            print(f"  {label}: P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f} IoU={m['mean_matched_iou']:.3f} "
                  f"n_pred={m['n_pred']}")

    run_manifest = {
        "task": "T02", "candidate": MODEL_NAME, "candidate_version": MODEL_VERSION,
        "candidate_role": "learned candidate 2 (real Omnipose transfer/fine-tuning)",
        "command": (f"python model_labs/omnipose_lab/run_folds.py --out "
                    f"{out_dir.relative_to(ROOT)} --arms {' '.join(arms)}"),
        "input_manifest": BOOTSTRAP,
        "input_manifest_sha256": sha256_file(manifest_path),
        "split_policy": ("whole-well leave-one-well-out, 6 folds; training tiles are "
                         "built per well so no crop is shared between a fold's "
                         "training and held-out wells"),
        "ignore_policy_decision": {
            "chosen": "paint_out",
            "rejected_tile_exclusion": ("measured: at a 512px tile only 19/239 tiles "
                                        "are ignore-free and 11/375 instances survive"),
            "rejected_loss_mask": ("omnipose.core.loss applies its per-pixel weight to "
                                   "5 of 9 terms; masking properly would fork the loss"),
            "evidence": "model_labs/omnipose_lab/_runs/policy_evidence.json",
            "ablation": "both arms trained and scored on identical folds",
        },
        "config": config, "arms": arms, "include_round2": include_round2,
        "augment_gaps": augment_gaps,
        "augment_gaps_note": ("synthetic signal gaps appended to training tiles; the "
                              "gap distribution is measured from each fold's TRAINING "
                              "wells only and no mask is ever altered. Off unless "
                              "--augment-gaps is given"),
        "round2_note": ("`retriage_round2` is a second, less conservative pass by the "
                        "same operator; excluded by default and reported separately"),
        "seed": config["seed"],
        "environment": environment, "environment_hash": environment["environment_hash"],
        "platform": platform.platform(),
        "resumed_folds": resumed,
        "folds_by_arm": by_arm,
        "summary_by_arm": {arm: aggregate(folds) for arm, folds in by_arm.items()},
        "ablation": paired_comparison(by_arm),
        "failures": failures,
        "synthetic_pairs_used": False,
        "correction_pairs_used": False,
        "correction_pairs_note": ("The 40 real correction pairs were not used for "
                                  "training or tuning; they remain an untouched "
                                  "refinement/evaluation set."),
        "evidence_class": "development_bootstrap_single_operator_proposal_conditioned",
        "limitations": [
            "single human operator; not consensus and not inter-rater agreement",
            "proposal-conditioned ground truth; retrospective, not prospective",
            "predictions are unreviewed proposals; instance counts are not "
            "authoritative independent-myotube counts",
            "precision is depressed by the sparse-GT effect: reviewed 'complete' is a "
            "small subset of the fibre-like structure in each field, so an unmatched "
            "prediction is not necessarily a false object",
            "no hyperparameter search was run; the configuration was fixed in advance "
            "and applied unchanged to every fold and both arms",
            "the paint policy alters training images; inference always runs on the "
            "unmodified field, so any shortcut it induced shows up as held-out loss",
        ],
        "seconds_total": round(time.time() - started, 1),
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(run_manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {path}")
    return run_manifest


def main(argv=None) -> int:
    # Training knobs come from train_fold.add_training_args — ONE definition for
    # both CLIs. This file once redeclared them by hand and drifted: the full run
    # silently used the in-process path while the probe that sized it used the
    # worker dataloader (16 GPU-hours, zero folds; session state 2026-08-12), and
    # --autocast was settable on train_fold but not here. Only run-level flags
    # (--arms/--wells/--no-resume/...) belong on this parser.
    from omnipose_lab.train_fold import add_training_args, config_from_args

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="model_labs/omnipose_lab/_runs/v1")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--wells", nargs="+", default=None,
                        help="restrict held-out wells (smoke runs)")
    parser.add_argument("--include-round2", action="store_true")
    parser.add_argument("--augment-gaps", action="store_true",
                        help="train on synthetically gapped copies as well as the "
                             "originals; a separate axis from --arms, which selects "
                             "the ignore policy")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore per-fold sidecars and re-run every fold")
    add_training_args(parser)
    args = parser.parse_args(argv)

    config = config_from_args(args)
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out

    manifest = run(out_dir, args.arms, config, args.include_round2, args.wells,
                   resume=not args.no_resume, augment_gaps=args.augment_gaps)
    print("\n=== T02 Omnipose: six-fold leave-one-well-out ===")
    for arm, summary in manifest["summary_by_arm"].items():
        if summary:
            print(f"  {arm:26s} P={summary['precision_macro_mean']} "
                  f"R={summary['recall_macro_mean']} "
                  f"IoU={summary['mean_matched_iou_macro_mean']} "
                  f"({summary['n_folds']} folds)")
    deltas = manifest["ablation"].get("delta_paint_out_minus_control")
    if deltas:
        print("  ablation (paint_out - control):",
              json.dumps({k: v["mean_delta"] for k, v in deltas.items()}))
    if manifest["failures"]:
        print("  FAILURES:", json.dumps(manifest["failures"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
