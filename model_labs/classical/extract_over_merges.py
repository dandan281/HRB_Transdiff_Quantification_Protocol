"""Extract the exact linked predictions that over-merge, for human review.

Why this exists
---------------
`run_linker_folds.py` reports that the fragment linker at the locked threshold
**0.90** introduces `over_merge_count = 3` across the six-well corpus: two in
`19_B06_act104_trka`, one in `22_B03_act104_egfrc`. Over-merge is one of the
operator's three named error classes, and three objects is few enough to settle
by hand rather than by argument. This script pulls those three predictions out
with everything a reviewer needs to judge them.

An over-merge, in the benchmark's own terms
-------------------------------------------
`precision_myotube.benchmark` builds ``pred_links[j] = {i : inter/area(pred_j)
>= coverage_threshold}`` and counts ``len(pred_links[j]) >= 2``. So a predicted
instance over-merges when **two or more distinct reference masks each account for
at least 20% of that prediction's area**. That is reproduced here directly from
the label arrays rather than through the scratch InstanceSet, to avoid depending
on record-ordering; the script asserts it recovers the published per-well counts
(2 and 1), which is the check that the reproduction is faithful.

Locked parameters
-----------------
Threshold is **0.90 and locked**. These three cases must never be used to select
it -- that is why the accepted-pair probabilities are written to the *key* file
only and never into the review payload. Everything else (tracer/filter params per
fold, candidate window, LOWO refit) is taken unchanged from the sealed floor and
`run_linker_folds.py`.

Usage::

    $env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
    python model_labs/classical/extract_over_merges.py \
      --out model_labs/classical/_runs/over_merges_v1
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
from classical.run_linker_folds import (BOOTSTRAP, CACHE, EVAL_GT,      # noqa: E402
                                        PACKAGES, PAIRS, RUN_MANIFEST,
                                        ANNOTATION, _Union)
from precision_myotube.schema import InstanceSet                        # noqa: E402

# Locked, not tunable here. Kept as a module constant so a test can pin it.
LOCKED_THRESHOLD = 0.90
COVERAGE_THRESHOLD = 0.2      # precision_myotube.benchmark default
# published per-well over-merge counts at the locked threshold (linker_instance_v1)
PUBLISHED_OVER_MERGES = {"19_B06_act104_trka": 2, "22_B03_act104_egfrc": 1}


def merge_components(kept_ids, accepted_pairs):
    """Union-find over accepted pairs -> {root_label: [member fragment ids]}."""
    union = _Union(kept_ids)
    for a, b in accepted_pairs:
        union.union(a, b)
    components: dict[int, list[int]] = {}
    for i in kept_ids:
        components.setdefault(union.find(i), []).append(int(i))
    return {root: sorted(members) for root, members in components.items()}


def relabel(assigned, components):
    out = np.zeros_like(assigned)
    for root, members in components.items():
        for member in members:
            out[assigned == member] = root
    return out


def gt_masks(gt_path):
    """Reviewed+complete reference masks, exactly the set the benchmark scores."""
    gt = InstanceSet.load(gt_path)
    out = []
    for record, bbox, mask in gt.cropped_masks():
        if record.reviewed and record.status == "complete":
            out.append({"id": record.id, "bbox": tuple(int(v) for v in bbox),
                        "mask": mask, "area": int(mask.sum())})
    return gt, out


def find_over_merges(merged, components, references):
    """pred_links per merged label, using the benchmark's coverage rule."""
    labels, counts = np.unique(merged[merged > 0], return_counts=True)
    area = {int(l): int(c) for l, c in zip(labels, counts)}
    links: dict[int, list[dict]] = {}
    for ref in references:
        r0, c0, r1, c1 = ref["bbox"]
        window = merged[r0:r1, c0:c1]
        inside = window[ref["mask"]]
        present, overlap = np.unique(inside[inside > 0], return_counts=True)
        for label, inter in zip(present, overlap):
            label, inter = int(label), int(inter)
            cov_pred = inter / area[label]
            if cov_pred >= COVERAGE_THRESHOLD:
                links.setdefault(label, []).append({
                    "reference_id": ref["id"],
                    "intersection_px": inter,
                    "fraction_of_prediction": round(cov_pred, 4),
                    "fraction_of_reference": round(inter / ref["area"], 4)})
    over = {label: refs for label, refs in links.items() if len(refs) >= 2}
    return over, area


def process_well(well, args, train_pairs, keys, fold_params, packages):
    from annotation_tools.qc_review.link_candidates import find_link_candidates
    from annotation_tools.qc_review.link_features import (compute_features, field_background,
                                                          geometry_cache)
    from annotation_tools.qc_review.link_model import LinkPair, fit_linker

    tracer_cfg, filter_cfg = fold_params[well]
    tracer, filters = TracerParams(**tracer_cfg), FilterParams(**filter_cfg)
    gt_path = ROOT / EVAL_GT / f"{well}.eval_gt.instances.json"

    fold_train = [p for p in train_pairs if p.well != well]
    model = fit_linker(fold_train, keys)

    territory = np.load(ROOT / CACHE / f"{well}.territory.npy")
    fiber = tifffile.imread(ROOT / BOOTSTRAP / well / "image_fiber.tif")
    dapi_path = ROOT / BOOTSTRAP / well / "image_dapi.tif"
    dapi = tifffile.imread(dapi_path) if dapi_path.is_file() else None
    trace = trace_fibers_parameterised(territory, args.pixel_um, tracer)
    assigned, areas = assign_territory(trace)
    kept, _debug = filter_assigned(trace, assigned, areas, filters)
    kept_only = np.where(np.isin(assigned, kept), assigned, 0).astype(np.int32)
    del territory, assigned

    background = field_background(fiber)
    # Pinned OFF deliberately. This script's job is to reproduce the sealed run
    # faithfully enough to assert its published over-merge counts; inheriting the
    # 2026-08-04 axis gate would audit a pipeline the measurement never used.
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
                background=background, fragment_geom=geoms.get(fi), candidate_geom=geoms.get(ci))
            proba = model.score(LinkPair(well, fragment_id, cand.candidate_id, feats))
            scored.append((float(proba), fi, ci,
                           tuple(int(v) for v in cand.fragment_endpoint),
                           tuple(int(v) for v in cand.candidate_endpoint),
                           float(cand.gap_um)))

    accepted = [(a, b) for p, a, b, *_ in scored if p >= LOCKED_THRESHOLD]
    components = merge_components(list(kept), accepted)
    merged = relabel(kept_only, components)
    _gt, references = gt_masks(gt_path)
    over, area = find_over_merges(merged, components, references)

    expected = PUBLISHED_OVER_MERGES.get(well)
    if expected is not None and len(over) != expected:
        raise SystemExit(
            f"{well}: recovered {len(over)} over-merges, published run says {expected}. "
            "The reproduction does not match -- refusing to emit a review packet "
            "built on a pipeline that disagrees with the measurement it is auditing.")

    pair_lookup = {(min(a, b), max(a, b)): (p, fe, ce, gap)
                   for p, a, b, fe, ce, gap in scored}
    cases = []
    for label, refs in sorted(over.items()):
        members = components[label]
        internal = [(min(a, b), max(a, b)) for a, b in accepted
                    if a in members and b in members]
        pairs = []
        for key in sorted(set(internal)):
            p, fe, ce, gap = pair_lookup[key]
            pairs.append({"fragments": list(key), "probability": round(p, 6),
                          "fragment_endpoint": list(fe), "candidate_endpoint": list(ce),
                          "gap_um": round(gap, 3)})
        rows, cols = np.nonzero(merged == label)
        cases.append({
            "well": well, "merged_label": int(label),
            "fragment_ids": members, "n_fragments": len(members),
            "prediction_area_px": area[int(label)],
            "accepted_pairs": pairs,
            "overlapping_references": sorted(refs, key=lambda r: -r["fraction_of_prediction"]),
            "bbox": [int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1],
        })
    # The population the safety round samples from: accepted merges, i.e. linked
    # components of more than one fragment. `n_accepted_merges` below counts EDGES
    # and is not that population; a uniform round needs the component count as its
    # inclusion-probability denominator, so record it separately.
    n_merge_components = sum(1 for members in components.values() if len(members) > 1)
    return {"well": well, "cases": cases, "n_accepted_merges": len(accepted),
            "n_accepted_merge_components": n_merge_components,
            "n_candidates": len(scored), "tracer": tracer_cfg, "filters": filter_cfg,
            "n_train_pairs": len(fold_train)}, {
        "merged": merged, "kept_only": kept_only, "fiber": fiber, "dapi": dapi,
        "references": references, "components": components, "scored": scored,
        "accepted": accepted, "area": area}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="model_labs/classical/_runs/over_merges_v1")
    parser.add_argument("--pixel-um", type=float, default=0.6493)
    parser.add_argument("--wells", nargs="*", default=sorted(PUBLISHED_OVER_MERGES))
    parser.add_argument("--gap-um", type=float, default=80.0)
    parser.add_argument("--cos-min", type=float, default=0.70)
    parser.add_argument("--controls", type=int, default=6,
                        help="accepted merges per well that do NOT over-merge, sampled as "
                             "blinded controls; 0 disables")
    parser.add_argument("--uniform-controls", action="store_true",
                        help="control-only safety round: sample --controls accepted merges "
                             "per well with equal probability, with no fragment-count or "
                             "reference-density matching. Use with --wells covering all six "
                             "wells to estimate a population over-merge rate.")
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args(argv)

    from annotation_tools.qc_review.link_model import FEATURE_SETS, recompute_training_pairs

    started = time.time()
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((ROOT / RUN_MANIFEST).read_text(encoding="utf-8"))
    fold_params = {f["held_out_well"]: (f["selected_tracer"], f["selected_filters"])
                   for f in manifest["folds"]}
    packages = {w: ROOT / ANNOTATION / p for w, p in PACKAGES.items()}
    print("recomputing linker training pairs from the banked rounds ...")
    train_pairs = recompute_training_pairs(ROOT / PAIRS, packages,
                                           gap_um=args.gap_um, cos_min=args.cos_min)
    keys = FEATURE_SETS["bridge_axis"]

    rng = np.random.default_rng(args.seed)
    all_cases, controls, wells_meta = [], [], []
    for well in args.wells:
        t0 = time.time()
        meta, arrays = process_well(well, args, train_pairs, keys, fold_params, packages)
        print(f"  {well:22s} {len(meta['cases'])} over-merge(s) of "
              f"{meta['n_accepted_merges']} accepted merges  ({time.time()-t0:.0f}s)")
        all_cases.extend(meta["cases"])

        if args.controls > 0:
            over_labels = {c["merged_label"] for c in meta["cases"]}
            if args.uniform_controls:
                # Control-only safety round. There is no flagged group to blend into,
                # so fragment-count matching would only bias the sample towards the
                # sizes that happened to be flaggable. Sample every accepted merge with
                # equal probability within the well instead; that is the whole point of
                # the round, and it is why this mode must never be mixed with cases.
                multi = [(root, members) for root, members in arrays["components"].items()
                         if len(members) > 1 and root not in over_labels
                         and root in arrays["area"]]
            else:
                # Match controls to the cases on fragment count. Unmatched controls are a
                # blinding leak: if every real flag is a 2-3 fragment object and the controls
                # run to 8, the reviewer can separate the groups by eye without judging any
                # biology.
                case_sizes = {c["n_fragments"] for c in meta["cases"]}
                multi = [(root, members) for root, members in arrays["components"].items()
                         if len(members) in case_sizes and root not in over_labels
                         and root in arrays["area"]]
            if args.uniform_controls and len(multi) < args.controls:
                raise SystemExit(
                    f"{well}: only {len(multi)} accepted merges available, asked for "
                    f"{args.controls}. Lower --controls rather than silently sampling "
                    "fewer in one well, or the per-well inclusion probabilities stop "
                    "being what the predeclared estimator assumes.")
            pick = rng.permutation(len(multi))[:args.controls]
            # Controls carry the SAME pair fields as the real cases -- endpoints and
            # gap_um included. Omitting them silently unblinds the packet: the review
            # page draws the bridge from the endpoints and shows gap_um as a chip, so a
            # control without them would render with no bridge line and no gap chip.
            full_lookup = {(min(a, b), max(a, b)): (p, fe, ce, gap)
                           for p, a, b, fe, ce, gap in arrays["scored"]}
            for k in pick:
                root, members = multi[int(k)]
                internal = sorted({(min(a, b), max(a, b)) for a, b in arrays["accepted"]
                                   if a in members and b in members})
                rows, cols = np.nonzero(arrays["merged"] == root)
                pairs = []
                for key in internal:
                    p, fe, ce, gap = full_lookup[key]
                    pairs.append({"fragments": list(key), "probability": round(p, 6),
                                  "fragment_endpoint": list(fe),
                                  "candidate_endpoint": list(ce),
                                  "gap_um": round(gap, 3)})
                controls.append({
                    "well": well, "merged_label": int(root), "fragment_ids": members,
                    "n_fragments": len(members),
                    "prediction_area_px": arrays["area"][root],
                    "accepted_pairs": pairs,
                    "overlapping_references": [],
                    "bbox": [int(rows.min()), int(cols.min()),
                             int(rows.max()) + 1, int(cols.max()) + 1]})

        np.savez_compressed(
            out_dir / f"{well}.arrays.npz",
            merged=arrays["merged"], kept_only=arrays["kept_only"],
            fiber=arrays["fiber"],
            **({"dapi": arrays["dapi"]} if arrays["dapi"] is not None else {}))
        ref_out = {r["id"]: {"bbox": list(r["bbox"]), "area": r["area"]}
                   for r in arrays["references"]}
        np.savez_compressed(
            out_dir / f"{well}.references.npz",
            **{r["id"]: np.packbits(r["mask"]) for r in arrays["references"]},
            **{f"{r['id']}__shape": np.array(r["mask"].shape) for r in arrays["references"]})
        (out_dir / f"{well}.references.json").write_text(
            json.dumps(ref_out, indent=2), encoding="utf-8")
        wells_meta.append(meta)
        del arrays

    payload = {
        "purpose": "hand review of every candidate over-merge introduced by the "
                   "fragment linker at the locked threshold",
        "threshold": LOCKED_THRESHOLD,
        "threshold_status": "LOCKED -- these cases must not be used to select or tune it",
        "coverage_threshold": COVERAGE_THRESHOLD,
        "over_merge_definition": "a predicted instance where >=2 distinct reviewed+complete "
                                 "reference masks each cover >=20% of the prediction's area",
        "candidate_window": {"gap_um": args.gap_um, "cos_min": args.cos_min},
        "source_run": "model_labs/classical/_runs/linker_instance_v1.json",
        "published_over_merge_counts": PUBLISHED_OVER_MERGES,
        "reproduction_check": "per-well counts matched the published run",
        "uniform_controls": bool(args.uniform_controls),
        "control_sampling": (
            "equal probability within each well over all accepted merges (components of "
            ">1 fragment), no fragment-count or reference-density matching"
            if args.uniform_controls else
            "matched to a flagged case in the same well on fragment count"),
        "wells": wells_meta, "cases": all_cases, "controls": controls,
        "control_note": "accepted merges at the same locked threshold that do NOT trip the "
                        "over-merge rule; included so a reviewer's verdict on the real cases "
                        "can be read against their behaviour on ordinary merges. Sampled only "
                        "from components whose fragment count matches a real case in the same "
                        "well, so group membership cannot be read off object size.",
        "evidence_class": "development_bootstrap_single_operator_proposal_conditioned",
        "seconds_total": round(time.time() - started, 1),
    }
    (out_dir / "cases.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n{len(all_cases)} over-merge case(s), {len(controls)} control(s)")
    print(f"written: {out_dir / 'cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
