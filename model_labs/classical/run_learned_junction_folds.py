"""Instance-level A/B: classical junction pairing vs the learned classifier.

Everything measured so far about the junction classifier is a *proxy* -- junction
decisions scored in isolation (64.5% vs the fixed rule's 23.8%). This runs the
substitution through the actual pipeline and scores the readout the project
cares about: instance counts, recall, matched IoU, and the false-split /
over-merge rates junction pairing directly drives.

Protocol
--------
* **Identical everything except the junction rule.** Both arms use the same
  cached stage-A territory, the same `TracerParams`/`FilterParams` (the sealed
  v1 run's fold-selected values by default), the same sealed eval GT. The only
  difference is `trace_fibers_parameterised(junction_decider=...)`.
* **Leave-one-well-out, honestly.** For each held-out well the pair model, the
  branch-point gate, AND the gate threshold are fitted on the *other five*
  wells' junction labels only. The scored well never contributes to the model
  deciding its own junctions.
* `straight_dot` differs between the sealed floor (0.0, fold-selected) and the
  labeling rounds (-0.5, canonical) but does **not** affect `build_branch_graph`
  -- it is only consumed by the pairing step -- so the branch graph, and hence
  every feature the classifier was trained on, is bit-identical either way.

Usage::

    $env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
    python model_labs/classical/run_learned_junction_folds.py \
      --out model_labs/classical/_runs/learned_junctions_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from classical.learned_junctions import make_learned_decider              # noqa: E402
from classical.ridge_graph import (FilterParams, TracerParams,            # noqa: E402
                                   assign_territory, filter_assigned,
                                   instance_positions, trace_fibers_parameterised)
from classical.run_folds import MODEL_NAME, write_instances               # noqa: E402
from precision_myotube.benchmark import benchmark_instances               # noqa: E402
from precision_myotube.schema import InstanceSet                          # noqa: E402

EXPORTS = ["PrecisionMyotube/annotation_work/junctions_round1/junctions_round1.junctions.json",
           "PrecisionMyotube/annotation_work/junctions_round2/junctions_active_r2.junctions.json"]
CACHE = "model_labs/classical/_runs/v1/_territory_cache"
BOOTSTRAP = "PrecisionMyotube/annotation_work/bootstrap_v1"
EVAL_GT = "model_labs/classical/_runs/v1/eval_gt"
RUN_MANIFEST = "model_labs/classical/_runs/v1/run_manifest.json"
METRICS = ("n_gt", "n_pred", "tp", "precision", "recall", "f1", "mean_matched_iou",
           "false_split_count", "false_split_rate", "over_merge_count", "over_merge_rate",
           "length_mdape", "width_mdape")


def fit_fold_models(examples, junction_examples, truth, held_out):
    """Pair model, branch-point gate and gate threshold, from the other wells only."""
    from annotation_tools.qc_review.junction_model import (
        FEATURE_KEYS, GATE_THRESHOLD_GRID, decision_accuracy, fit_branch_point_model,
        fit_junction_classifier)

    train_pairs = [e for e in examples if e.well != held_out and e.label is not None]
    train_junctions = [e for e in junction_examples if e.well != held_out]
    pair_model = fit_junction_classifier(train_pairs, FEATURE_KEYS)
    gate_model = fit_branch_point_model(train_junctions)

    by_junction: dict = {}
    for e in train_pairs:
        by_junction.setdefault((e.well, e.node), []).append(e)
    gate_scores = {j.id_key(): gate_model.score(j) for j in train_junctions}

    best_threshold, best_accuracy = 0.5, -1.0
    for threshold in GATE_THRESHOLD_GRID:
        decisions = {}
        for key, rows in by_junction.items():
            if gate_scores.get(key, 0.0) >= threshold:
                decisions[key] = None
            else:
                decisions[key] = max(rows, key=pair_model.score).key
        accuracy = decision_accuracy(decisions, truth)["accuracy"] or 0.0
        if accuracy > best_accuracy:
            best_threshold, best_accuracy = threshold, accuracy
    return pair_model, gate_model, best_threshold, best_accuracy


def score_arm(trace, filters, gt_path, image_shape, image_id, scratch):
    assigned, areas = assign_territory(trace)
    kept, debug = filter_assigned(trace, assigned, areas, filters)
    write_instances(assigned, kept, image_shape, image_id, scratch)
    metrics = benchmark_instances(gt_path, scratch)
    del assigned, areas
    return {k: metrics[k] for k in METRICS}, debug


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="model_labs/classical/_runs/learned_junctions_v1.json")
    parser.add_argument("--pixel-um", type=float, default=0.6493)
    parser.add_argument("--wells", nargs="*", default=None)
    args = parser.parse_args(argv)

    from annotation_tools.qc_review.junction_model import (
        ground_truth_decisions, recompute_examples)
    from scipy import ndimage as ndi

    started = time.time()
    manifest = json.loads((ROOT / RUN_MANIFEST).read_text(encoding="utf-8"))
    fold_params = {f["held_out_well"]: (f["selected_tracer"], f["selected_filters"])
                   for f in manifest["folds"]}
    wells = args.wells or sorted(fold_params)

    exports = [ROOT / e for e in EXPORTS]
    print("recomputing junction features from the labeled rounds ...")
    recomputed = recompute_examples(exports, ROOT / CACHE, ROOT / BOOTSTRAP,
                                    pixel_um=args.pixel_um)
    truth = ground_truth_decisions(exports)
    print(f"  {len(recomputed.pairs)} pair rows / {len(recomputed.junctions)} junctions")

    work = ROOT / "tmp" / "learned_junctions"
    work.mkdir(parents=True, exist_ok=True)
    results = []
    for well in wells:
        tracer_cfg, filter_cfg = fold_params[well]
        tracer = TracerParams(**tracer_cfg)
        filters = FilterParams(**filter_cfg)
        gt_path = ROOT / EVAL_GT / f"{well}.eval_gt.instances.json"
        gt = InstanceSet.load(gt_path)

        pair_model, gate_model, threshold, train_accuracy = fit_fold_models(
            recomputed.pairs, recomputed.junctions, truth, well)

        territory = np.load(ROOT / CACHE / f"{well}.territory.npy")
        fiber = tifffile.imread(ROOT / BOOTSTRAP / well / "image_fiber.tif")
        scratch = work / f"{well}.scratch.instances.json"

        t0 = time.time()
        classical_trace = trace_fibers_parameterised(territory, args.pixel_um, tracer)
        classical, classical_debug = score_arm(classical_trace, filters, gt_path,
                                               gt.image_shape, gt.image_id, scratch)
        del classical_trace

        distance_to_bg = ndi.distance_transform_edt(np.asarray(territory, dtype=bool))
        stats: dict = {}
        decider = make_learned_decider(pair_model, gate_model, threshold, fiber,
                                       distance_to_bg, args.pixel_um, tracer, stats=stats)
        learned_trace = trace_fibers_parameterised(territory, args.pixel_um, tracer,
                                                   junction_decider=decider)
        learned, learned_debug = score_arm(learned_trace, filters, gt_path,
                                           gt.image_shape, gt.image_id, scratch)
        del learned_trace, distance_to_bg, territory, fiber

        row = {"well": well, "tracer": tracer_cfg, "filters": filter_cfg,
               "gate_threshold": threshold,
               "gate_threshold_train_accuracy": round(train_accuracy, 4),
               "classical": classical, "learned": learned,
               "classical_n_fibers_traced": classical_debug["n_fibers_traced"],
               "learned_n_fibers_traced": learned_debug["n_fibers_traced"],
               "decider_stats": dict(stats), "seconds": round(time.time() - t0, 1)}
        results.append(row)
        fired = stats.get("learned_pair", 0) + stats.get("learned_branch_point", 0)
        print(f"  {well:22s} learned fired on {fired}/{stats.get('total', 0)} junctions "
              f"({stats.get('learned_branch_point', 0)} branch points)  "
              f"n_pred {classical['n_pred']} -> {learned['n_pred']}  "
              f"R {classical['recall']:.3f} -> {learned['recall']:.3f}  "
              f"IoU {classical['mean_matched_iou']:.3f} -> {learned['mean_matched_iou']:.3f}  "
              f"({row['seconds']:.0f}s)")
        scratch.unlink(missing_ok=True)

    def mean(arm, key):
        vals = [r[arm][key] for r in results if r[arm].get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {arm: {k: mean(arm, k) for k in METRICS} for arm in ("classical", "learned")}
    out = {"protocol": "leave-one-well-out; pair model, branch-point gate and gate "
                       "threshold all fitted on the other five wells only",
           "arms": "identical territory/params/eval-GT; only the junction rule differs",
           "exports": EXPORTS, "wells": results, "summary": summary,
           "evidence_class": "development_bootstrap_single_operator_proposal_conditioned",
           "seconds_total": round(time.time() - started, 1)}
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== instance-level means (leave-one-well-out) ===")
    print(f"{'metric':<22} {'classical':>11} {'learned':>11} {'delta':>11}")
    for k in METRICS:
        c, l = summary["classical"][k], summary["learned"][k]
        if c is None or l is None:
            continue
        print(f"{k:<22} {c:>11.4f} {l:>11.4f} {l - c:>+11.4f}")
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
