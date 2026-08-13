"""Active-learning junction round 2: widen the pool, serve the least-certain junctions.

Round 1 labelled 245 junctions -- the classical floor's narrower ambiguity pool
(``near_threshold_winner | width_or_intensity_conflict``) -- and reached pair-level
LOWO AUC 0.701 / junction-decision accuracy 41% (vs. the classical floor's 20.5%
on the same set). That is a real improvement but, by the fragment linker's own
precedent, a first-round result, not a finished model.

This round mirrors the linker's round 2 exactly:

1. **Widen the window.** Round 1 used the narrower pool; round 2 scores the full
   near-threshold-broad pool (``reasons=None`` in `junction_pairs`, ~615
   junctions project-wide), which is a strict superset (proven in
   `test_junction_pairs.test_round1_pool_is_a_subset_of_the_full_ambiguous_pool`).
2. **Rank by uncertainty.** Every new junction's three candidate pairs are scored
   by the round-1 classifier; the junction's uncertainty is that of its
   best-guess pair (``1 - 2*|proba-0.5|``). The model's own guess is never shown
   to the operator -- it only orders the queue, same principle as the linker and
   the blind-repeat tool.
3. **Never re-serve.** Every ``(well, node)`` key that appeared in ANY prior
   export -- decided, branch-point, or unsure -- is excluded, so the operator
   only ever sees genuinely new junctions.

Output is the same `junction_pairs.v1` schema `junction_page.py` already
produces/consumes, so `junction_model.recompute_training_pairs` (and a future
round 3) reads it unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .junction_pairs import MIN_BRANCH_UM, find_junction_cases

DEFAULT_MAX_JUNCTIONS = 150


@dataclass
class ServedJunction:
    well: str
    node: int
    branch_ids: tuple
    branch_lengths_um: tuple
    centroid_rc: tuple
    best_pair_key: str
    best_proba: float
    uncertainty: float


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def uncertainty(proba: float) -> float:
    """0 at a confident 0/1 prediction, 1 at a maximally uncertain 0.5."""
    return 1.0 - 2.0 * abs(proba - 0.5)


def already_offered(prior_exports: list) -> set:
    """Every ``(well, node)`` shown in any prior round's export, any outcome."""
    seen = set()
    for export_path in prior_exports:
        export = json.loads(Path(export_path).read_text(encoding="utf-8"))
        for row in export["decisions"].values():
            seen.add((row["well"], row["node"]))
    return seen


def score_new_candidates(well: str, territory: np.ndarray, fiber: np.ndarray,
                         pixel_um: float, model, exclude: set,
                         min_branch_um: float = MIN_BRANCH_UM) -> tuple[list[ServedJunction], dict]:
    """Every widened-pool junction in ``well`` not in ``exclude``, scored by ``model``.

    Reuses `find_junction_cases`'s own candidate walk (``reasons=None`` is the
    full near-threshold-broad pool) rather than re-deriving it, and its
    returned ``node_ends`` for the direction vectors feature scoring needs --
    no second pass over the branch graph.
    """
    from scipy import ndimage as ndi

    from .junction_features import compute_pair_features
    from .junction_model import JunctionPairExample

    cases, coordinates, node_ends = find_junction_cases(
        well, territory, fiber, pixel_um, min_branch_um=min_branch_um, reasons=None)
    territory = np.asarray(territory, dtype=bool)
    distance_to_bg = ndi.distance_transform_edt(territory)

    served: list[ServedJunction] = []
    n_new = 0
    for case in cases:
        if (well, case.node) in exclude:
            continue
        n_new += 1
        ends = node_ends[case.node]
        pair_probas = {}
        for letter_pair, (a, b) in zip(("AB", "AC", "BC"), ((0, 1), (0, 2), (1, 2))):
            branch_a, end_a, _dir_a = ends[a]
            branch_b, end_b, _dir_b = ends[b]
            feats = compute_pair_features(
                coordinates, branch_a, end_a, branch_b, end_b, distance_to_bg,
                fiber, pixel_um, case.branch_lengths_um[a], case.branch_lengths_um[b])
            example = JunctionPairExample(well, case.node, letter_pair, feats, None)
            pair_probas[letter_pair] = model.score(example)
        best_key, best_proba = max(pair_probas.items(), key=lambda kv: kv[1])
        served.append(ServedJunction(
            well=well, node=case.node, branch_ids=case.branch_ids,
            branch_lengths_um=case.branch_lengths_um, centroid_rc=case.centroid_rc,
            best_pair_key=best_key, best_proba=round(best_proba, 4),
            uncertainty=round(uncertainty(best_proba), 4)))
    return served, {"candidates": len(cases), "new": n_new}


def build_active_round(territory_cache_dir, bootstrap_dir, prior_exports: list, out_path,
                       *, reviewer: str, wells: list, batch_id: str = "junctions_active_r2",
                       pixel_um: float = 0.6493, max_junctions: int = DEFAULT_MAX_JUNCTIONS,
                       size: int = 460, radius_um: float = 60.0) -> dict:
    import tifffile

    from .junction_model import fit_junction_classifier, recompute_training_pairs, select_feature_set
    from .junction_page import BRANCH_RGB, build_junction_page, render_case

    territory_cache_dir = Path(territory_cache_dir)
    bootstrap_dir = Path(bootstrap_dir)

    # 1. train the current model on every prior round's labels
    train_examples = []
    for export_path in prior_exports:
        train_examples.extend(recompute_training_pairs(
            export_path, territory_cache_dir, bootstrap_dir, pixel_um=pixel_um))
    feature_name, selection = select_feature_set(train_examples)
    feature_keys = selection["all"][feature_name]["keys"]
    model = fit_junction_classifier(train_examples, feature_keys)

    # 2. never re-serve a junction already shown in a prior round
    exclude = already_offered(prior_exports)

    # 3. widened pool across all wells, scored and ranked by uncertainty
    served: list[ServedJunction] = []
    per_well_counts: dict[str, dict] = {}
    fiber_by_well: dict[str, np.ndarray] = {}
    coordinates_by_well: dict[str, list] = {}
    for well in wells:
        territory = np.load(territory_cache_dir / f"{well}.territory.npy")
        fiber = tifffile.imread(bootstrap_dir / well / "image_fiber.tif")
        new_served, counts = score_new_candidates(well, territory, fiber, pixel_um, model, exclude)
        served.extend(new_served)
        per_well_counts[well] = counts
        fiber_by_well[well] = fiber
        # coordinates are needed again at render time; find_junction_cases is cheap
        # (skeleton + skan summarize, seconds not minutes) so recomputing per well
        # once more at render time is acceptable rather than holding every well's
        # full coordinate list in memory across the whole scoring pass.
        del territory

    if not served:
        raise SystemExit("no new junction candidates at the widened pool")

    # 4. bound the round: keep the most-uncertain junctions if over budget, and SAY
    #    what was dropped rather than silently truncating.
    served.sort(key=lambda s: -s.uncertainty)
    n_total = len(served)
    dropped = 0
    if n_total > max_junctions:
        dropped = n_total - max_junctions
        served = served[:max_junctions]

    # 5. render, most-uncertain first, grouped back by well so each well's
    #    coordinates are only rebuilt once
    served_by_well: dict[str, list[ServedJunction]] = {}
    for sj in served:
        served_by_well.setdefault(sj.well, []).append(sj)

    cases = []
    for well, sjs in served_by_well.items():
        territory = np.load(territory_cache_dir / f"{well}.territory.npy")
        fiber = fiber_by_well[well]
        _cases, coordinates, _ends = find_junction_cases(well, territory, fiber, pixel_um,
                                                          reasons=None)
        del territory
        for sj in sjs:
            img, overlay, _bbox = render_case(fiber, None, coordinates, sj.branch_ids,
                                              sj.centroid_rc, size=size, radius_um=radius_um,
                                              pixel_um=pixel_um)
            case_id = f"junction_{sj.node:06d}"
            branches = [{"letter": letter, "rgb": ",".join(str(v) for v in rgb),
                        "branch_id": int(bid), "length_um": length}
                       for (letter, rgb), bid, length in
                       zip(BRANCH_RGB, sj.branch_ids, sj.branch_lengths_um)]
            letters = [b["letter"] for b in branches]
            pairs = [{"key": letters[a] + letters[b], "letters": [letters[a], letters[b]],
                      "branches": [int(sj.branch_ids[a]), int(sj.branch_ids[b])]}
                     for a, b in ((0, 1), (0, 2), (1, 2))]
            cases.append({"id": case_id, "well": well, "node": sj.node,
                         "uid": f"{well}/{case_id}", "dom_id": f"{well}__{case_id}",
                         "img": img, "overlay": overlay, "branches": branches, "pairs": pairs})

    # re-sort: the well-grouped render loop above lost the global uncertainty order
    order = {(sj.well, sj.node): -sj.uncertainty for sj in served}
    cases.sort(key=lambda c: order[(c["well"], c["node"])])

    note = (f"Active-learning round 2: {len(cases)} junctions, most-uncertain first. "
           "Widened to the full near-threshold-broad pool (not just round 1's narrower "
           "subset). Only junctions not offered in round 1 are shown.")
    page = build_junction_page(cases, out_path, batch_id=batch_id, reviewer=reviewer,
                               session_started_at=_now(), note=note)

    manifest = {
        "schema": "junctions_active.v1", "batch_id": batch_id, "reviewer": reviewer,
        "created_at_utc": _now(),
        "strategy": "uncertainty_sampling (proba near 0.5 first); prediction never shown",
        "classifier": {"feature_set": feature_name, "feature_selection": selection,
                      "fit_info": model.fit_info,
                      "trained_on_exports": [str(p) for p in prior_exports]},
        "pool": {"new_junctions_total": n_total, "served": len(cases),
                 "dropped_least_uncertain": dropped, "max_junctions": max_junctions,
                 "per_well": per_well_counts},
        "served_junctions": [
            {"well": sj.well, "node": sj.node, "best_pair_guess": sj.best_pair_key,
             "model_proba": sj.best_proba, "uncertainty": sj.uncertainty}
            for sj in served],
        "page": str(page),
        "evidence_class": "single_operator_proposal_conditioned",
        "limitations": [
            "single operator; not consensus or inter-rater agreement",
            "the classifier is trained on round 1's 245 junctions and extrapolates to "
            "the wider pool, so its uncertainty ordering is itself provisional",
            "junctions already offered in round 1 are excluded, so round-1 context is "
            "not re-shown",
        ],
    }
    manifest_path = Path(out_path).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  classifier: {feature_name} AUC {selection['chosen_auc']} "
         f"({model.fit_info['n_positive']} positives)")
    print(f"  pool: {n_total} new junctions across {len(wells)} wells, "
         f"served {len(cases)} ({dropped} dropped as least-uncertain)")
    print(f"  page: {page}")
    return manifest
