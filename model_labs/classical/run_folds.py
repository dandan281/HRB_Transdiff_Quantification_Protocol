"""T02 candidate 1 - six-fold leave-one-well-out runner for the classical floor.

Implements the fold protocol required by
`coordination/requests/claude/2026-07-21-t02-start.md`:

* whole-well leave-one-well-out, six folds, never splitting object crops;
* candidate parameters are chosen on the fold's five **training** wells only and
  then applied unchanged to the held-out well;
* predictions are exported through the canonical adapter as unreviewed
  `InstanceSet` JSON with full `ModelProvenance`;
* a run manifest records the command, hashes, seeds, thresholds, per-fold
  timing, and failures.

The classical candidate has no learned weights, so "training" here means only
the deterministic parameter fit over :data:`classical.ridge_graph.PARAM_GRID`.
That is still done fold-honestly: a fold never sees its held-out well's score
when selecting parameters, which is what keeps this a legitimate floor rather
than a number tuned on its own test set.

Evaluation ground truth
-----------------------
Scoring uses each well's reviewed `complete` masks with the two binding
`training_exclude.json` exclusions removed. The source `*.qc.instances.json`
still carries those two records as reviewed/complete, so a naive scorer would
silently evaluate against masks the plan forbids using. The filtered set is
written per well and hashed, so T03 can reproduce exactly what was scored.

Everything produced here is exploratory, single-operator, proposal-conditioned,
retrospective development evidence. It is not consensus, not inter-rater
agreement, and not prospective validation.

Usage::

    $env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
    python model_labs/classical/run_folds.py --out model_labs/classical/_runs/v1
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from classical.ridge_graph import (  # noqa: E402
    PARAM_GRID, FilterParams, TracerParams, assign_territory, filter_assigned,
    instance_positions, iter_masks, semantic_territory_cached,
    trace_fibers_parameterised)
from _shared.predict_export import ModelProvenance, export_prediction  # noqa: E402
from precision_myotube.benchmark import benchmark_instances  # noqa: E402
from precision_myotube.schema import InstanceSet  # noqa: E402

MODEL_NAME = "classical_ridge_graph"
MODEL_VERSION = "v1"
BOOTSTRAP = "PrecisionMyotube/annotation_work/bootstrap_v1/bootstrap_manifest.json"
SELECTION_METRIC = "precision_weighted_score"   # canonical (2P + R) / 3
SEED = 0                                        # no stochastic step; recorded for completeness


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fmt(value, digits=3):
    return round(value, digits) if isinstance(value, float) else value


# --------------------------------------------------------------------------- data


def load_wells(manifest_path: Path) -> tuple[dict, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wells = sorted(manifest["per_well"])          # deterministic fold order
    return manifest, wells


def pixel_um_for(package: str) -> float:
    readme = ROOT / "PrecisionMyotube/annotation_work" / package / "README.json"
    return float(json.loads(readme.read_text(encoding="utf-8"))["pixel_um"])


def build_eval_gt(manifest: dict, well: str, out_dir: Path) -> dict:
    """Reviewed-complete GT minus the binding exclusions, sealed and hashed."""
    info = manifest["per_well"][well]
    source = ROOT / info["source_instances"]
    actual = sha256_file(source)
    if actual != info["source_instances_sha256"]:
        raise RuntimeError(f"{well}: source instances hash mismatch "
                           f"(expected {info['source_instances_sha256']}, got {actual})")
    excluded = set(info.get("excluded", []))
    full = InstanceSet.load(source)
    kept = [r for r in full.instances
            if r.reviewed and r.status == "complete" and r.id not in excluded]
    if len(kept) != info["complete_kept"]:
        raise RuntimeError(f"{well}: expected {info['complete_kept']} evaluation masks, "
                           f"built {len(kept)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{well}.eval_gt.instances.json"
    InstanceSet(tuple(full.image_shape), full.image_id, kept,
                provenance={
                    "derived_from": info["source_instances"],
                    "derived_from_sha256": actual,
                    "excluded_ids": sorted(excluded),
                    "filter": "reviewed AND status==complete AND id not in training_exclude",
                    "evidence_class": "single_operator_proposal_conditioned",
                }).save(path)
    return {"path": path, "n_gt": len(kept), "sha256": sha256_file(path),
            "image_id": full.image_id, "image_shape": tuple(full.image_shape),
            "excluded_ids": sorted(excluded)}


# ------------------------------------------------------------------- scoring grid


def score_well(well: str, package: str, gt: dict, cache_dir: Path, work_dir: Path,
               grid: list) -> tuple[list[dict], dict]:
    """Score every grid point on one well. Stage A cached; stage B per tracer."""
    pixel_um = pixel_um_for(package)
    fiber = tifffile.imread(ROOT / "PrecisionMyotube/annotation_work/bootstrap_v1"
                            / well / "image_fiber.tif")
    started = time.time()
    territory, territory_debug = semantic_territory_cached(
        fiber, cache_dir / f"{well}.territory.npy")
    del fiber
    stage_a_seconds = time.time() - started

    by_tracer: dict[str, list[int]] = {}
    for index, (tracer, _) in enumerate(grid):
        by_tracer.setdefault(tracer.key(), []).append(index)

    rows: list[dict] = [None] * len(grid)          # type: ignore[list-item]
    scratch = work_dir / f"{well}.scratch.instances.json"
    for key in sorted(by_tracer):
        indices = by_tracer[key]
        tracer = grid[indices[0]][0]
        trace = trace_fibers_parameterised(territory, pixel_um, tracer)
        assigned, areas = assign_territory(trace)
        for index in indices:
            filters = grid[index][1]
            kept, debug = filter_assigned(trace, assigned, areas, filters)
            write_instances(assigned, kept, gt["image_shape"], gt["image_id"], scratch)
            metrics = benchmark_instances(gt["path"], scratch)
            rows[index] = {
                "param_index": index,
                "tracer": asdict(tracer), "filters": asdict(filters),
                "n_instances": debug["n_instances"],
                "assigned_fraction": _fmt(debug["assigned_fraction"]),
                "precision": _fmt(metrics["precision"]),
                "recall": _fmt(metrics["recall"]),
                "f1": _fmt(metrics["f1"]),
                SELECTION_METRIC: _fmt(metrics[SELECTION_METRIC]),
                "mean_matched_iou": _fmt(metrics["mean_matched_iou"]),
                "false_split_count": metrics["false_split_count"],
                "over_merge_count": metrics["over_merge_count"],
                "length_mdape": _fmt(metrics["length_mdape"]) if metrics["length_mdape"] else None,
                "width_mdape": _fmt(metrics["width_mdape"]) if metrics["width_mdape"] else None,
                "tp": metrics["tp"], "n_gt": metrics["n_gt"], "n_pred": metrics["n_pred"],
            }
        del trace, assigned, areas
    scratch.unlink(missing_ok=True)
    return rows, {"pixel_um": pixel_um, "stage_a_seconds": round(stage_a_seconds, 1),
                  "territory_selected_high_pct": territory_debug.get("selected_high_pct"),
                  "territory_coverage_pct": _fmt(float(territory.mean() * 100.0))}


def write_instances(assigned: np.ndarray, kept_ids: list[int], image_shape,
                    image_id: str, path: Path) -> None:
    """Minimal scratch InstanceSet used only for grid scoring (never handed off).

    Builds records from a single pass over the assignment map -- never one
    full-field boolean mask per instance, which on a 3636x3636 field with a
    thousand instances would cost ~13 GB.
    """
    from precision_myotube.schema import InstanceRecord, encode_sparse_positions
    records = [
        InstanceRecord(id=f"{MODEL_NAME}_{n:04d}", status="complete", reviewed=False,
                       source=MODEL_NAME,
                       rle=encode_sparse_positions(tuple(image_shape), positions))
        for n, (_, positions) in enumerate(instance_positions(assigned, kept_ids), start=1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    InstanceSet(tuple(image_shape), image_id, records).save(path)


# ------------------------------------------------------------------------- folds


def select_params(table: dict[str, list[dict]], train_wells: list[str],
                  grid: list) -> tuple[int, float]:
    """Best grid point by mean selection metric over the training wells only.

    Ties break toward the lower parameter index, which is a fixed, documented
    order -- never toward whatever happens to score well on the held-out well.
    """
    best_index, best_score = None, None
    for index in range(len(grid)):
        scores = [table[w][index][SELECTION_METRIC] for w in train_wells]
        mean_score = float(np.mean(scores))
        if best_score is None or mean_score > best_score + 1e-12:
            best_index, best_score = index, mean_score
    return int(best_index), float(best_score)


def run_folds(out_dir: Path, manifest_path: Path, grid: list) -> dict:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_territory_cache"
    work_dir = out_dir / "_work"
    gt_dir = out_dir / "eval_gt"
    predictions_dir = out_dir / "predictions"
    for d in (cache_dir, work_dir, gt_dir, predictions_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest, wells = load_wells(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    failures: list[dict] = []

    ground_truth = {w: build_eval_gt(manifest, w, gt_dir) for w in wells}
    print(f"evaluation GT: {sum(g['n_gt'] for g in ground_truth.values())} masks "
          f"across {len(wells)} wells")

    table: dict[str, list[dict]] = {}
    well_info: dict[str, dict] = {}
    for well in wells:
        package = manifest["per_well"][well]["package"]
        t0 = time.time()
        try:
            rows, info = score_well(well, package, ground_truth[well], cache_dir,
                                    work_dir, grid)
        except Exception as exc:                        # keep going, record it
            failures.append({"stage": "score_well", "well": well,
                             "error": f"{type(exc).__name__}: {exc}"})
            print(f"  !! {well} FAILED: {type(exc).__name__}: {exc}")
            continue
        table[well] = rows
        well_info[well] = info
        best = max(rows, key=lambda r: r[SELECTION_METRIC])
        print(f"  {well:22s} gt={ground_truth[well]['n_gt']:3d} "
              f"grid={len(rows)} best_in_well={best[SELECTION_METRIC]:.3f} "
              f"({time.time()-t0:.0f}s)")

    scored_wells = [w for w in wells if w in table]
    if len(scored_wells) < 2:
        raise RuntimeError("need at least two scored wells to run folds")

    folds = []
    for held_out in scored_wells:
        train_wells = [w for w in scored_wells if w != held_out]
        # Leakage guard: the held-out well must not influence selection.
        assert held_out not in train_wells, "held-out well leaked into the fitting set"
        index, train_score = select_params(table, train_wells, grid)
        tracer, filters = grid[index]

        pixel_um = well_info[held_out]["pixel_um"]
        territory_cache = cache_dir / f"{held_out}.territory.npy"
        if not territory_cache.is_file():           # score_well always writes it
            raise RuntimeError(f"missing cached territory for {held_out}")
        territory = np.load(territory_cache)
        trace = trace_fibers_parameterised(territory, pixel_um, tracer)
        assigned, areas = assign_territory(trace)
        kept, debug = filter_assigned(trace, assigned, areas, filters)

        gt = ground_truth[held_out]
        provenance = ModelProvenance(
            model=MODEL_NAME, version=f"{MODEL_VERSION}-fold-{held_out}",
            architecture="sato-ridge -> skan branch graph -> antiparallel junction "
                         "pairing -> nearest-fibre territory",
            checkpoint_hash="none-deterministic-no-weights",
            environment_hash=environment_hash(),
            data_hash=manifest_hash, seed=SEED,
            thresholds={**asdict(tracer), **asdict(filters),
                        "selection_metric": SELECTION_METRIC,
                        "selected_on_wells": train_wells},
            channels="desmin_only", used_prompts=False)
        # Stream masks one at a time; `write_convenience_tiff=False` guarantees the
        # exporter consumes the iterable exactly once.
        exported = export_prediction(predictions_dir, gt["image_id"], gt["image_shape"],
                                     provenance, masks=iter_masks(assigned, kept),
                                     write_convenience_tiff=False, status="complete")

        held_metrics = benchmark_instances(gt["path"], exported["instances"])
        # Leakage guard: predictions must be for the held-out field only.
        exported_set = InstanceSet.load(exported["instances"])
        assert exported_set.image_id == gt["image_id"], "prediction image_id mismatch"
        assert all(not r.reviewed for r in exported_set.instances), "predictions must be unreviewed"

        folds.append({
            "held_out_well": held_out, "train_wells": train_wells,
            "selected_param_index": index,
            "selected_tracer": asdict(tracer), "selected_filters": asdict(filters),
            "train_mean_selection_score": _fmt(train_score),
            "n_instances_predicted": debug["n_instances"],
            "assigned_fraction": _fmt(debug["assigned_fraction"]),
            "held_out_metrics": {k: _fmt(held_metrics[k]) for k in (
                "n_gt", "n_pred", "tp", "precision", "recall", "f1",
                SELECTION_METRIC, "mean_matched_iou", "false_split_count",
                "false_split_rate", "over_merge_count", "over_merge_rate",
                "length_mdape", "width_mdape", "automatic_coverage")},
            "prediction": {
                "instances": str(Path(exported["instances"]).relative_to(ROOT)),
                "instances_sha256": sha256_file(exported["instances"]),
                "manifest": str(Path(exported["manifest"]).relative_to(ROOT)),
            },
            "eval_gt": {"path": str(gt["path"].relative_to(ROOT)),
                        "sha256": gt["sha256"], "n_gt": gt["n_gt"],
                        "excluded_ids": gt["excluded_ids"]},
        })
        m = folds[-1]["held_out_metrics"]
        print(f"  fold hold-out {held_out:22s} params#{index:2d} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} IoU={m['mean_matched_iou']:.3f} "
              f"lenMdAPE={m['length_mdape']}")
        del trace, assigned, areas, kept, territory

    summary = aggregate(folds)
    run_manifest = {
        "task": "T02", "candidate": MODEL_NAME, "candidate_version": MODEL_VERSION,
        "candidate_role": "deterministic classical reproducible floor (contract candidate 1)",
        "command": f"python model_labs/classical/run_folds.py --out {out_dir.relative_to(ROOT)}",
        "input_manifest": BOOTSTRAP, "input_manifest_sha256": manifest_hash,
        "split_policy": "whole-well leave-one-well-out, 6 folds; no object crop is "
                        "shared between a fold's training and held-out wells",
        "parameter_selection": {
            "metric": SELECTION_METRIC,
            "grid_size": len(grid),
            "rule": "mean over the fold's five training wells; ties break to the "
                    "lowest parameter index",
            "held_out_never_used_for_selection": True,
        },
        "seed": SEED,
        "environment": environment_record(),
        "wells": well_info,
        "folds": folds,
        "summary": summary,
        "failures": failures,
        "synthetic_pairs_used": False,
        "correction_pairs_used": False,
        "correction_pairs_note": "The 40 real correction pairs were not used for "
                                 "training or tuning by this candidate; they remain "
                                 "an untouched refinement/evaluation set.",
        "evidence_class": "development_bootstrap_single_operator_proposal_conditioned",
        "limitations": [
            "single human operator; not consensus and not inter-rater agreement",
            "proposal-conditioned ground truth; retrospective, not prospective",
            "predictions are unreviewed proposals; instance counts are not "
            "authoritative independent-myotube counts",
            "precision is dominated by the sparse-GT effect: the reviewed "
            "'complete' set is a small subset of the fibre-like structure in each "
            "field, so unmatched predictions are not necessarily false objects",
            "the classical floor emits mutually exclusive masks and therefore "
            "cannot represent a crossing as two overlapping instances",
        ],
        "seconds_total": round(time.time() - started, 1),
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    (out_dir / "grid_scores.json").write_text(
        json.dumps({"selection_metric": SELECTION_METRIC, "table": table}, indent=2),
        encoding="utf-8")
    return run_manifest


def aggregate(folds: list[dict]) -> dict:
    if not folds:
        return {}
    keys = ("precision", "recall", "f1", "mean_matched_iou", "length_mdape",
            "width_mdape", "false_split_rate", "over_merge_rate")
    out = {}
    for key in keys:
        values = [f["held_out_metrics"][key] for f in folds
                  if f["held_out_metrics"].get(key) is not None]
        out[f"{key}_mean"] = _fmt(float(np.mean(values))) if values else None
    out["n_folds"] = len(folds)
    out["total_gt"] = sum(f["held_out_metrics"]["n_gt"] for f in folds)
    out["total_pred"] = sum(f["held_out_metrics"]["n_pred"] for f in folds)
    out["total_tp"] = sum(f["held_out_metrics"]["tp"] for f in folds)
    return out


def environment_record() -> dict:
    import networkx, scipy, skan, skimage
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "skimage": skimage.__version__, "skan": skan.__version__,
        "networkx": networkx.__version__, "tifffile": tifffile.__version__,
        "gpu_required": False,
        "note": "CPU-only, deterministic; no framework install touches "
                "Conversion_Efficiency/cpenv",
    }


def environment_hash() -> str:
    return hashlib.sha256(
        json.dumps(environment_record(), sort_keys=True).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="model_labs/classical/_runs/v1")
    parser.add_argument("--manifest", default=BOOTSTRAP)
    parser.add_argument("--limit-grid", type=int, default=None,
                        help="use only the first N grid points (smoke runs)")
    args = parser.parse_args(argv)

    grid = PARAM_GRID if args.limit_grid is None else PARAM_GRID[:args.limit_grid]
    manifest = run_folds(Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out,
                         ROOT / args.manifest, grid)
    print("\n=== T02 classical floor: six-fold leave-one-well-out ===")
    for fold in manifest["folds"]:
        m = fold["held_out_metrics"]
        print(f"  {fold['held_out_well']:22s} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} IoU={m['mean_matched_iou']:.3f}")
    print("  mean:", json.dumps(manifest["summary"]))
    if manifest["failures"]:
        print("  FAILURES:", json.dumps(manifest["failures"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
