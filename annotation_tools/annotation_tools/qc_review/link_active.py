"""Active-learning linking round: serve the pairs a label would help most.

Round 1 labelled every collinear pair inside a *narrow* window (gap <= 40 um,
cos >= 0.8) -- 91 pairs, all now decided. The linker trained on them reaches only
AUC ~0.80 / precision 0.59, which is not deployable: a wrong join fuses two real
myotubes, worse than leaving one split. To lift precision the model needs labels
where it is currently *uncertain*, and there are none left in the narrow window.

So this round does two things the plan's §8.D calls for:

1. **Widen the window** (default gap 80 um, cos 0.7). This surfaces new candidate
   pairs -- longer gaps, gentler bends -- that the narrow round never offered,
   including partners for the fragments left unresolved in round 1.
2. **Rank by uncertainty.** Every new pair is scored by the linker trained on the
   91 labels; pairs whose predicted probability sits near 0.5 are served first,
   because their labels move a data-starved model the most. The prediction itself
   is never shown -- it orders the queue and nothing else, so the operator's
   judgement stays independent (same principle as the blind-repeat tool).

Already-labelled pairs are excluded by ``(well, fragment, candidate)`` key -- never
by bare id, because proposal ids repeat across wells. The operator therefore only
ever sees genuinely new decisions.

Output is the same `fragment_links.v1` page/export as round 1, so `apply_links`
consumes it unchanged. A manifest records the model, the selected feature set, and
every served pair's probability and uncertainty for downstream analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import json
from pathlib import Path

import numpy as np

WIDE_GAP_UM = 80.0
WIDE_COS_MIN = 0.70
TRAIN_GAP_UM = 40.0            # window the 91 labels were collected under
TRAIN_COS_MIN = 0.80
MAX_CANDIDATES_PER_FRAGMENT = 5   # A-E; the card stays a fast glance
DEFAULT_MAX_PAIRS = 160


@dataclass
class ServedPair:
    well: str
    fragment_id: str
    candidate_id: str
    gap_um: float
    min_cos: float
    proba: float
    uncertainty: float
    fragment_endpoint: tuple[int, int]
    candidate_endpoint: tuple[int, int]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def build_active_round(packages: dict[str, Path], round2_dir: Path,
                       pairs_jsonl: Path, out_path: Path, *, reviewer: str,
                       batch_id: str = "links_active_b02",
                       wide_gap_um: float = WIDE_GAP_UM,
                       wide_cos_min: float = WIDE_COS_MIN,
                       train_gap_um: float = TRAIN_GAP_UM,
                       train_cos_min: float = TRAIN_COS_MIN,
                       exclude_jsonls: list[Path] | None = None,
                       max_pairs: int = DEFAULT_MAX_PAIRS,
                       size: int = 900) -> dict:
    import tifffile

    from .link_candidates import find_link_candidates, load_fragments
    from .link_features import compute_features, field_background, geometry_cache
    from .link_model import (FEATURE_SETS, LinkPair, fit_linker,
                             recompute_training_pairs, select_feature_set,
                             uncertainty)
    from .link_page import CANDIDATE_RGB, build_link_page, render_case

    # 1. train the linker on the labelled pairs. `train_gap_um/cos` must be wide
    #    enough to re-find every training pair -- e.g. round 3 trains on the
    #    combined round-1 (narrow) + round-2 (wide) set, so it must use the wide
    #    window or the round-2 pairs silently drop out of training.
    train_pairs = recompute_training_pairs(pairs_jsonl, packages,
                                           gap_um=train_gap_um, cos_min=train_cos_min)
    feature_name, selection = select_feature_set(train_pairs)
    model = fit_linker(train_pairs, FEATURE_SETS[feature_name])

    # Exclusion = every pair ever *offered*, not just the usable training labels,
    # so a previously-shown conflict/unsure/silent pair is not served again. Falls
    # back to the training keys when no explicit exclusion files are given.
    if exclude_jsonls:
        already = set()
        for fp in exclude_jsonls:
            for line in Path(fp).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    already.add((r["well"], r["fragment_id"], r["candidate_id"]))
    else:
        already = {p.key() for p in train_pairs}

    # 2. wide-window candidates across all six wells, scored and de-duplicated
    fragments_by_well = load_fragments(round2_dir)
    served: list[ServedPair] = []
    per_well_counts: dict[str, dict] = {}
    for well in sorted(fragments_by_well):
        package = packages[well]
        fiber = tifffile.imread(package / "fiber_raw16.tif")
        terr_path = package / "semantic_territory.tif"
        territory = tifffile.imread(terr_path) if terr_path.is_file() else None
        labels = tifffile.imread(package / "starting_labels.tif")
        background = field_background(fiber)
        pixel_um = _pixel_um(package)

        ids = fragments_by_well[well]
        label_ids = [int(i.split("_")[-1]) for i in ids]
        found = find_link_candidates(labels, label_ids, pixel_um,
                                     gap_um=wide_gap_um, cos_min=wide_cos_min)
        needed = {int(fid.split("_")[-1]) for fid in found} | \
                 {int(c.candidate_id.split("_")[-1])
                  for cands in found.values() for c in cands}
        geoms = geometry_cache(labels, needed)
        n_new = 0
        for fragment_id, cands in found.items():
            for cand in cands:
                key = (well, cand.fragment_id, cand.candidate_id)
                if key in already:
                    continue                        # decided in round 1
                feats = compute_features(
                    fiber, territory, cand.fragment_endpoint, cand.candidate_endpoint,
                    cand.gap_um, min(cand.cos_fragment, cand.cos_candidate), pixel_um,
                    background=background,
                    fragment_geom=geoms.get(int(cand.fragment_id.split("_")[-1])),
                    candidate_geom=geoms.get(int(cand.candidate_id.split("_")[-1])))
                proba = model.score(LinkPair(well, cand.fragment_id,
                                             cand.candidate_id, feats))
                served.append(ServedPair(
                    well, cand.fragment_id, cand.candidate_id, cand.gap_um,
                    round(min(cand.cos_fragment, cand.cos_candidate), 3),
                    round(proba, 4), round(uncertainty(proba), 4),
                    tuple(cand.fragment_endpoint), tuple(cand.candidate_endpoint)))
                n_new += 1
        per_well_counts[well] = {"fragments": len(ids), "new_pairs": n_new}

    if not served:
        raise SystemExit("no new candidate pairs at the widened window")

    # 3. bound the round: keep the most-uncertain pairs if over budget, and SAY
    #    what was dropped rather than silently truncating.
    served.sort(key=lambda s: -s.uncertainty)
    n_total = len(served)
    dropped = 0
    if n_total > max_pairs:
        dropped = n_total - max_pairs
        served = served[:max_pairs]

    # 4. group the kept pairs back into per-fragment cards, cards ordered by their
    #    most-uncertain pair, candidates within a card most-uncertain first.
    by_fragment: dict[tuple[str, str], list[ServedPair]] = {}
    for pair in served:
        by_fragment.setdefault((pair.well, pair.fragment_id), []).append(pair)
    fragment_order = sorted(
        by_fragment,
        key=lambda k: -max(p.uncertainty for p in by_fragment[k]))

    cases, served_records = [], []
    dropped_over_cap = 0
    for well, fragment_id in fragment_order:
        ranked = sorted(by_fragment[(well, fragment_id)], key=lambda p: -p.uncertainty)
        dropped_over_cap += max(0, len(ranked) - MAX_CANDIDATES_PER_FRAGMENT)
        pairs = ranked[:MAX_CANDIDATES_PER_FRAGMENT]
        package = packages[well]
        stem = _stem(package, well)
        fiber = tifffile.imread(package / "fiber_raw16.tif")
        dapi_path = package / "dapi_raw16.tif"
        dapi = tifffile.imread(dapi_path) if dapi_path.is_file() else None
        labels = tifffile.imread(package / "starting_labels.tif")

        fragment_label = int(fragment_id.split("_")[-1])
        candidate_labels = [int(p.candidate_id.split("_")[-1]) for p in pairs]
        img, overlay, _bbox = render_case(fiber, dapi, labels, fragment_label,
                                          candidate_labels, size=size)
        cases.append({
            "id": fragment_id, "well": stem, "uid": f"{stem}/{fragment_id}",
            "dom_id": f"{stem}__{fragment_id}".replace("/", "_").replace(".", "_"),
            "img": img, "overlay": overlay,
            "candidates": [
                {"letter": CANDIDATE_RGB[i % len(CANDIDATE_RGB)][0],
                 "rgb": ",".join(str(v) for v in CANDIDATE_RGB[i % len(CANDIDATE_RGB)][1]),
                 "candidate_id": p.candidate_id, "gap_um": p.gap_um,
                 "cos_fragment": p.min_cos, "cos_candidate": p.min_cos}
                for i, p in enumerate(pairs)],
        })
        for p in pairs:
            served_records.append({
                "well": stem, "fragment_id": fragment_id, "candidate_id": p.candidate_id,
                "gap_um": p.gap_um, "min_cos": p.min_cos,
                "model_proba": p.proba, "uncertainty": p.uncertainty})

    note = (f"Active-learning round: {len(cases)} fragments, most-uncertain first. "
            f"Widened to gap &le; {wide_gap_um:.0f} &micro;m, collinearity &ge; "
            f"{wide_cos_min}. Only pairs not decided in round 1 are shown.")
    page = build_link_page(cases, out_path, batch_id=batch_id, reviewer=reviewer,
                           session_started_at=_now(), gap_um=wide_gap_um,
                           cos_min=wide_cos_min, note=note)

    manifest = {
        "schema": "links_active.v1", "batch_id": batch_id, "reviewer": reviewer,
        "created_at_utc": _now(),
        "strategy": "uncertainty_sampling (proba near 0.5 first); prediction never shown",
        "linker": {
            "trained_on_pairs": len([p for p in train_pairs if p.label is not None]),
            "n_positive": model.fit_info["n_positive"],
            "feature_set": feature_name, "feature_selection": selection,
            "fit_info": model.fit_info,
            "train_window": {"gap_um": TRAIN_GAP_UM, "cos_min": TRAIN_COS_MIN},
        },
        "candidate_window": {"gap_um": wide_gap_um, "cos_min": wide_cos_min},
        "pool": {"new_pairs_total": n_total, "served": len(served_records),
                 "dropped_least_uncertain": dropped, "max_pairs": max_pairs,
                 "dropped_over_5_per_fragment": dropped_over_cap,
                 "per_well": per_well_counts},
        "n_fragment_cards": len(cases),
        "served_pairs": served_records,
        "page": str(page),
        "evidence_class": "single_operator_proposal_conditioned",
        "limitations": [
            "single operator; not consensus or inter-rater agreement",
            "candidates only offered within the widened gap/collinearity window; a "
            "join beyond it still cannot be expressed",
            "the linker is trained on 91 narrow-window pairs and extrapolates to "
            "wider gaps, so its uncertainty ordering is itself provisional",
            "pairs already decided in round 1 are excluded, so a fragment's round-1 "
            "partner is not re-shown for context",
        ],
    }
    manifest_path = Path(out_path).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  linker: {feature_name} AUC {selection['chosen_auc']} "
          f"({model.fit_info['n_positive']} positives)")
    print(f"  pool: {n_total} new pairs, served {len(served_records)} "
          f"({dropped} dropped as least-uncertain), {len(cases)} fragment cards")
    print(f"  page: {page}")
    return manifest


def _pixel_um(package: Path) -> float:
    readme = package / "README.json"
    if readme.is_file():
        return float(json.loads(readme.read_text(encoding="utf-8")).get("pixel_um", 0.6493))
    return 0.6493


def _stem(package: Path, well: str) -> str:
    readme = package / "README.json"
    if readme.is_file():
        return json.loads(readme.read_text(encoding="utf-8")).get("image_id", well)
    return well
