"""Instance-level A/B: classical floor alone vs classical floor + fragment linker.

Closes the same gap for the linker that
`run_learned_junction_folds.py` closed for the junction classifier. The linker
reports LOWO AUC 0.902 and best-F1 precision 0.68 on its own pair task, but has
never been scored on the readout the project cares about. Fragmentation
(`fragment_too_short`) is the error the project has repeatedly called dominant,
so this is the measurement that says whether fixing it moves the science.

Protocol
--------
* Arm A = the sealed classical floor at its fold-selected parameters.
* Arm B = the same instances, then the linker merges accepted fragment pairs
  (union-find) before scoring. Nothing else differs.
* **Leave-one-well-out.** For each held-out well the linker is refitted on the
  other wells' banked pairs only.
* Threshold swept, because the linker is documented as a high-precision
  assistant rather than a blind auto-merger -- a single operating point would
  hide the tradeoff.

Known domain shift, stated up front: the linker was trained on pairs drawn from
the annotation packages' `starting_labels.tif` proposals, and is applied here to
classical-floor instances. Both come from ridge-style segmentation of the same
fields and the features are generic (bridge stain, PCA axes), but they are not
the same masks. A weak result here is therefore not automatically a verdict on
the linker itself -- see the report.

Usage::

    $env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
    python model_labs/classical/run_linker_folds.py \
      --out model_labs/classical/_runs/linker_instance_v1.json
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

from classical.ridge_graph import (FilterParams, TracerParams,          # noqa: E402
                                   assign_territory, filter_assigned,
                                   trace_fibers_parameterised)
from classical.run_folds import write_instances                         # noqa: E402
from precision_myotube.benchmark import benchmark_instances             # noqa: E402
from precision_myotube.schema import InstanceSet                        # noqa: E402

PAIRS = "PrecisionMyotube/annotation_work/links_active_r3/banked/combined_pairs_r123.jsonl"
CACHE = "model_labs/classical/_runs/v1/_territory_cache"
BOOTSTRAP = "PrecisionMyotube/annotation_work/bootstrap_v1"
EVAL_GT = "model_labs/classical/_runs/v1/eval_gt"
RUN_MANIFEST = "model_labs/classical/_runs/v1/run_manifest.json"
ANNOTATION = "PrecisionMyotube/annotation_work"
PACKAGES = {"19_B06_act104_trka": "19_B06_act104_trka",
            "22_B03_act104_egfrc": "22_B03_act104_egfrc",
            "23_B02_ctrl": "23_B02_ctrl",
            "29_C05_br223_egfrc": "29_C05_br223_egfrc",
            "32_C08_br223_igf1r": "32_C08_smoke",
            "33_C09_br223_trka": "33_C09_br223_trka"}
METRICS = ("n_gt", "n_pred", "tp", "precision", "recall", "f1", "mean_matched_iou",
           "false_split_count", "false_split_rate", "over_merge_count", "over_merge_rate",
           "length_mdape", "width_mdape")
# The linker is a high-precision assistant, not a blind merger; sweep rather than
# pick one operating point.
THRESHOLDS = (0.5, 0.7, 0.9)


class _Union:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def merge_labels(assigned, kept_ids, accepted_pairs):
    """Relabel ``assigned`` so every accepted pair shares one id."""
    union = _Union(kept_ids)
    for a, b in accepted_pairs:
        union.union(a, b)
    remap = {i: union.find(i) for i in kept_ids}
    out = np.zeros_like(assigned)
    for src, dst in remap.items():
        out[assigned == src] = dst
    return out, sorted(set(remap.values()))


def score(assigned, kept_ids, filters, trace, gt_path, image_shape, image_id, scratch,
          areas=None):
    write_instances(assigned, kept_ids, image_shape, image_id, scratch)
    metrics = benchmark_instances(gt_path, scratch)
    return {k: metrics[k] for k in METRICS}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="model_labs/classical/_runs/linker_instance_v1.json")
    parser.add_argument("--pixel-um", type=float, default=0.6493)
    parser.add_argument("--wells", nargs="*", default=None)
    parser.add_argument("--gap-um", type=float, default=80.0)
    parser.add_argument("--cos-min", type=float, default=0.70)
    args = parser.parse_args(argv)

    from annotation_tools.qc_review.link_candidates import find_link_candidates
    from annotation_tools.qc_review.link_features import (compute_features, field_background,
                                                          geometry_cache)
    from annotation_tools.qc_review.link_model import (FEATURE_SETS, LinkPair, fit_linker,
                                                       recompute_training_pairs)

    started = time.time()
    manifest = json.loads((ROOT / RUN_MANIFEST).read_text(encoding="utf-8"))
    fold_params = {f["held_out_well"]: (f["selected_tracer"], f["selected_filters"])
                   for f in manifest["folds"]}
    wells = args.wells or sorted(fold_params)

    packages = {w: ROOT / ANNOTATION / p for w, p in PACKAGES.items()}
    print("recomputing linker training pairs from the banked rounds ...")
    train_pairs = recompute_training_pairs(ROOT / PAIRS, packages,
                                           gap_um=args.gap_um, cos_min=args.cos_min)
    keys = FEATURE_SETS["bridge_axis"]
    print(f"  {len(train_pairs)} usable pairs across "
          f"{len(set(p.well for p in train_pairs))} wells; features {keys}")

    work = ROOT / "tmp" / "linker_instance"
    work.mkdir(parents=True, exist_ok=True)
    results = []
    for well in wells:
        tracer_cfg, filter_cfg = fold_params[well]
        tracer, filters = TracerParams(**tracer_cfg), FilterParams(**filter_cfg)
        gt_path = ROOT / EVAL_GT / f"{well}.eval_gt.instances.json"
        gt = InstanceSet.load(gt_path)
        scratch = work / f"{well}.scratch.instances.json"

        fold_train = [p for p in train_pairs if p.well != well]
        model = fit_linker(fold_train, keys)

        t0 = time.time()
        territory = np.load(ROOT / CACHE / f"{well}.territory.npy")
        fiber = tifffile.imread(ROOT / BOOTSTRAP / well / "image_fiber.tif")
        trace = trace_fibers_parameterised(territory, args.pixel_um, tracer)
        assigned, areas = assign_territory(trace)
        kept, _debug = filter_assigned(trace, assigned, areas, filters)
        kept_only = np.where(np.isin(assigned, kept), assigned, 0).astype(np.int32)
        del territory

        baseline = score(kept_only, kept, filters, trace, gt_path,
                         gt.image_shape, gt.image_id, scratch)

        background = field_background(fiber)
        # Pinned OFF deliberately: this produced the sealed linker run, and a run
        # that no longer reproduces is not a baseline. Turning the 2026-08-04 axis
        # gate on here is a new candidate, and belongs under a new run id.
        found = find_link_candidates(kept_only, kept, args.pixel_um,
                                     gap_um=args.gap_um, cos_min=args.cos_min,
                                     require_axis_agreement=False)
        needed = {int(f.split("_")[-1]) for f in found} | \
                 {int(c.candidate_id.split("_")[-1]) for cs in found.values() for c in cs}
        geoms = geometry_cache(kept_only, needed)
        scored = []
        for fragment_id, candidates in found.items():
            fi = int(fragment_id.split("_")[-1])
            for cand in candidates:
                ci = int(cand.candidate_id.split("_")[-1])
                feats = compute_features(
                    fiber, None, cand.fragment_endpoint, cand.candidate_endpoint,
                    cand.gap_um, min(cand.cos_fragment, cand.cos_candidate), args.pixel_um,
                    background=background, fragment_geom=geoms.get(fi),
                    candidate_geom=geoms.get(ci))
                proba = model.score(LinkPair(well, fragment_id, cand.candidate_id, feats))
                scored.append((proba, fi, ci))
        del fiber

        arms = {}
        for threshold in THRESHOLDS:
            accepted = [(a, b) for p, a, b in scored if p >= threshold]
            merged, merged_ids = merge_labels(kept_only, kept, accepted)
            arms[str(threshold)] = {
                "n_merges": len(accepted), "n_instances": len(merged_ids),
                **score(merged, merged_ids, filters, trace, gt_path,
                        gt.image_shape, gt.image_id, scratch)}
            del merged
        del assigned, areas, kept_only, trace

        row = {"well": well, "tracer": tracer_cfg, "filters": filter_cfg,
               "n_train_pairs": len(fold_train),
               "n_train_positive": model.fit_info["n_positive"],
               "n_candidates_offered": len(scored),
               "classical": baseline, "linked": arms,
               "seconds": round(time.time() - t0, 1)}
        results.append(row)
        best = arms[str(THRESHOLDS[0])]
        print(f"  {well:22s} cands={len(scored):<5} "
              f"n_pred {baseline['n_pred']} -> {best['n_pred']} (thr .5, {best['n_merges']} merges)  "
              f"R {baseline['recall']:.3f} -> {best['recall']:.3f}  "
              f"fsplit {baseline['false_split_rate']:.3f} -> {best['false_split_rate']:.3f}  "
              f"({row['seconds']:.0f}s)")
        scratch.unlink(missing_ok=True)

    def mean(getter, key):
        vals = [getter(r)[key] for r in results if getter(r).get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {"classical": {k: mean(lambda r: r["classical"], k) for k in METRICS}}
    for threshold in THRESHOLDS:
        summary[f"linked@{threshold}"] = {
            k: mean(lambda r, t=threshold: r["linked"][str(t)], k) for k in METRICS}
        summary[f"linked@{threshold}"]["n_merges"] = mean(
            lambda r, t=threshold: r["linked"][str(t)], "n_merges")

    out = {"protocol": "leave-one-well-out; linker refitted on the other wells' banked pairs",
           "domain_shift_note": "linker trained on annotation-package proposal masks, "
                                "applied here to classical-floor instances",
           "candidate_window": {"gap_um": args.gap_um, "cos_min": args.cos_min},
           "pairs_source": PAIRS, "wells": results, "summary": summary,
           "evidence_class": "development_bootstrap_single_operator_proposal_conditioned",
           "seconds_total": round(time.time() - started, 1)}
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== instance-level means (leave-one-well-out) ===")
    cols = ["classical"] + [f"linked@{t}" for t in THRESHOLDS]
    print(f"{'metric':<20}" + "".join(f"{c:>14}" for c in cols))
    for k in METRICS:
        if summary["classical"][k] is None:
            continue
        print(f"{k:<20}" + "".join(f"{summary[c][k]:>14.4f}" for c in cols))
    print(f"{'n_merges':<20}{'-':>14}" +
          "".join(f"{summary[f'linked@{t}']['n_merges']:>14.1f}" for t in THRESHOLDS))
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
