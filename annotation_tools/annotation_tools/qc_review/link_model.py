"""Learn the fragment linker and rank new candidates by how much a label would help.

Two jobs:

1. **Fit** a small logistic model on the operator's confirmed link pairs, using
   the intensity features in `link_features` rather than geometry alone. The
   round-1 analysis is explicit that *combining* many features hurt -- with only
   27 positives the model overfits -- so the feature set is chosen by
   leave-one-well-out AUC over a few candidate sets, not by throwing everything in.

2. **Score** new candidate pairs and rank them by **uncertainty** (proximity of the
   predicted probability to 0.5). Labelling the cases the current model is least
   sure about is what moves a data-starved linker fastest; labelling more easy,
   confident pairs teaches it almost nothing. This is uncertainty-sampling active
   learning.

The model's prediction is **never shown to the operator**. It orders the queue and
nothing more. Showing a guess would anchor the very judgement we are trying to
collect independently -- the same reason the blind-repeat tool hides learned
actions.

Everything is single-operator, proposal-conditioned development evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .link_features import (FEATURE_KEYS, LinkFeatures, compute_features,
                            field_background, geometry_cache)

# Candidate feature sets, smallest first. Round 1 found `bridge_over_bg` the
# strongest single feature (AUC 0.82) and that adding geometry *lowered* AUC, so
# the search is biased toward small sets and ties break to fewer features.
FEATURE_SETS = {
    "bridge_only": ("bridge_over_bg",),
    "bridge_territory": ("bridge_over_bg", "territory_frac"),
    "bridge_geom": ("bridge_over_bg", "min_cos"),
    "intensity": ("bridge_over_bg", "bridge_mean_over_bg", "territory_frac"),
    # Operator axis heuristic (2026-07-23): parallel axes + offset along the axis.
    # `displacement_along_axis` targets false positives (side-by-side neighbours),
    # which is what holds precision at 0.59.
    "bridge_disp": ("bridge_over_bg", "displacement_along_axis"),
    "bridge_axis": ("bridge_over_bg", "axis_cos", "displacement_along_axis"),
    "bridge_territory_disp": ("bridge_over_bg", "territory_frac",
                              "displacement_along_axis"),
    "all": FEATURE_KEYS,
}

MIN_POSITIVES = 8               # below this a fitted model is not trustworthy


@dataclass
class LinkPair:
    well: str
    fragment_id: str
    candidate_id: str
    features: LinkFeatures
    label: int | None = None    # 1 join / 0 not; None for an unlabelled candidate

    def key(self) -> tuple[str, str, str]:
        # Proposal ids repeat across wells (384 collisions in the queue), so a pair
        # is only unique with its well. Never key by bare id.
        return (self.well, self.fragment_id, self.candidate_id)


# --------------------------------------------------------------- feature recompute


def recompute_training_pairs(pairs_jsonl: str | Path, packages: dict[str, Path],
                             gap_um: float, cos_min: float,
                             usable_only: bool = True,
                             require_axis_agreement: bool = True) -> list[LinkPair]:
    """Rebuild `LinkFeatures` for every labelled pair by re-deriving its endpoints.

    The stored `link_pairs.jsonl` kept only geometry (gap, cos), not endpoints or
    intensity. To train on intensity we re-run the deterministic candidate finder
    per well at the window the labels were collected under, and match each stored
    pair by ``(fragment_id, candidate_id)`` to recover its endpoint pair.

    ``usable_only`` (default True) drops rows the export marked non-usable --
    two-sided conflicts, ``unsure`` cards, and cards advanced with no explicit
    selection. Those carry a ``label`` field for completeness but are not valid
    training signal, so they must not reach the model. Rows without a ``usable``
    key (older exports) are kept.
    """
    import tifffile

    from .link_candidates import find_link_candidates

    rows = [json.loads(line) for line in Path(pairs_jsonl).read_text().splitlines() if line.strip()]
    if usable_only:
        rows = [r for r in rows if r.get("usable", True)]
    by_well: dict[str, list[dict]] = {}
    for row in rows:
        by_well.setdefault(row["well"], []).append(row)

    out: list[LinkPair] = []
    unmatched = 0
    for well, well_rows in sorted(by_well.items()):
        package = packages[well]
        fiber = tifffile.imread(package / "fiber_raw16.tif")
        terr_path = package / "semantic_territory.tif"
        territory = tifffile.imread(terr_path) if terr_path.is_file() else None
        labels = tifffile.imread(package / "starting_labels.tif")
        background = field_background(fiber)
        pixel_um = _package_pixel_um(package)

        fragment_ids = sorted({int(r["fragment_id"].split("_")[-1]) for r in well_rows})
        found = find_link_candidates(
            labels,
            fragment_ids,
            pixel_um,
            gap_um=gap_um,
            cos_min=cos_min,
            require_axis_agreement=require_axis_agreement,
        )
        index = {(c.fragment_id, c.candidate_id): c
                 for cands in found.values() for c in cands}
        needed = {int(r["fragment_id"].split("_")[-1]) for r in well_rows} | \
                 {int(r["candidate_id"].split("_")[-1]) for r in well_rows}
        geoms = geometry_cache(labels, needed)
        for row in well_rows:
            cand = index.get((row["fragment_id"], row["candidate_id"]))
            if cand is None:
                unmatched += 1
                continue
            feats = compute_features(
                fiber, territory, cand.fragment_endpoint, cand.candidate_endpoint,
                cand.gap_um, min(cand.cos_fragment, cand.cos_candidate), pixel_um,
                background=background,
                fragment_geom=geoms.get(int(row["fragment_id"].split("_")[-1])),
                candidate_geom=geoms.get(int(row["candidate_id"].split("_")[-1])))
            out.append(LinkPair(well, row["fragment_id"], row["candidate_id"],
                                feats, label=int(row["label"])))
    if unmatched:
        print(f"  warning: {unmatched} labelled pairs did not re-match the finder "
              f"(window drift); excluded from training")
    return out


def _package_pixel_um(package: Path) -> float:
    readme = package / "README.json"
    if readme.is_file():
        return float(json.loads(readme.read_text(encoding="utf-8")).get("pixel_um", 0.6493))
    return 0.6493


# ------------------------------------------------------------------- fit and score


def _design(pairs: list[LinkPair], keys) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([p.features.vector(keys) for p in pairs], dtype=float)
    y = np.array([p.label for p in pairs], dtype=int)
    return X, y


def leave_one_well_out_auc(pairs: list[LinkPair], keys) -> dict:
    """LOWO AUC for one feature set -- the project's split policy, on the linker.

    A pair is scored by a model trained on the *other* wells only, so the number
    is not inflated by a well appearing on both sides. Folds with a single class in
    training are skipped and reported.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    labelled = [p for p in pairs if p.label is not None]
    wells = sorted({p.well for p in labelled})
    scores, truth, skipped = [], [], []
    for held in wells:
        train = [p for p in labelled if p.well != held]
        test = [p for p in labelled if p.well == held]
        ytr = np.array([p.label for p in train])
        if ytr.min() == ytr.max() or not test:
            skipped.append(held)
            continue
        Xtr, _ = _design(train, keys)
        Xte, yte = _design(test, keys)
        scaler = StandardScaler().fit(Xtr)
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        model.fit(scaler.transform(Xtr), ytr)
        proba = model.predict_proba(scaler.transform(Xte))[:, 1]
        scores.extend(proba.tolist())
        truth.extend(yte.tolist())

    auc = _auc(truth, scores)
    return {"auc": auc, "n_scored": len(truth), "skipped_wells": skipped,
            "n_features": len(keys), "keys": list(keys)}


def _auc(truth, scores) -> float | None:
    truth = np.asarray(truth)
    if truth.min() == truth.max():
        return None
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(truth, scores)), 3)


def select_feature_set(pairs: list[LinkPair]) -> tuple[str, dict]:
    """Pick the feature set with the best LOWO AUC; ties break to fewer features."""
    results = {name: leave_one_well_out_auc(pairs, keys)
               for name, keys in FEATURE_SETS.items()}
    ranked = sorted(
        (r for r in results.values() if r["auc"] is not None),
        key=lambda r: (-r["auc"], r["n_features"]))
    if not ranked:
        raise RuntimeError("no feature set produced a valid LOWO AUC")
    best = ranked[0]
    best_name = next(n for n, r in results.items() if r["keys"] == best["keys"])
    return best_name, {"chosen": best_name, "chosen_auc": best["auc"],
                       "all": results}


@dataclass
class LinkerModel:
    keys: tuple[str, ...]
    scaler: object
    model: object
    fit_info: dict

    def score(self, pair: LinkPair) -> float:
        x = np.array([pair.features.vector(self.keys)], dtype=float)
        return float(self.model.predict_proba(self.scaler.transform(x))[0, 1])


def fit_linker(pairs: list[LinkPair], keys) -> LinkerModel:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    labelled = [p for p in pairs if p.label is not None]
    X, y = _design(labelled, keys)
    if int(y.sum()) < MIN_POSITIVES:
        raise RuntimeError(f"only {int(y.sum())} positives; need >= {MIN_POSITIVES}")
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    model.fit(scaler.transform(X), y)
    coefs = dict(zip(keys, (round(float(c), 3) for c in model.coef_[0])))
    info = {"n": int(len(y)), "n_positive": int(y.sum()), "keys": list(keys),
            "train_accuracy": round(float(model.score(scaler.transform(X), y)), 3),
            "coef_std_features": coefs}
    return LinkerModel(tuple(keys), scaler, model, info)


def uncertainty(proba: float) -> float:
    """0 at a confident 0/1 prediction, 1 at a maximally uncertain 0.5."""
    return 1.0 - 2.0 * abs(proba - 0.5)
