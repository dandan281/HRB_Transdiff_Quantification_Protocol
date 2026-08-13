"""Learn the junction classifier from round-1 labels; compare to the classical floor.

Round 1 (`junction_page.py`) collected 245 operator decisions on the classical
floor's most-ambiguous degree-3 junctions: 145 through-pairs, 59 genuine branch
points, 40 unsure (dropped). Each decided junction yields three pair-level rows
(one per candidate pair), so the usable training set is 615 rows / 145 positive
-- comfortably past `MIN_POSITIVES`, unlike the linker's initial round.

Two jobs, mirroring `link_model.py`:

1. **Fit** a small logistic model on the pair-level rows, using the features in
   `junction_features` (tangent_cos, turn_angle_deg, width_ratio,
   intensity_ratio, length_min_um). Feature set chosen by leave-one-well-out
   AUC, smallest set among ties -- same discipline as the linker, which found
   that piling on features overfits a modest label count.
2. **Compare to baseline.** The classical floor already makes a decision at
   every junction (`classical.ridge_graph.pair_junction_ends` with the
   canonical `straight_dot=-0.5`): pick the most anti-parallel pair if it
   clears the threshold, else declare no pairing (implicitly a branch point).
   `junction_decision_accuracy` scores that fixed rule against the operator's
   ground truth at the **junction** level (one of 3-pairs-or-branch-point),
   not just the pair level, because that is the actual downstream decision the
   classical floor makes.

Recomputing features from the export (rather than trusting anything cached at
export time) follows the linker's precedent: `junctions_round1.junctions.json`
stores only branch ids and labels, not geometry, so the branch graph is
rebuilt per well and matched by node id -- if the territory cache ever drifts
from what the round was built on, a branch-id mismatch raises loudly rather
than silently training on the wrong geometry.

Everything here is single-operator, proposal-conditioned development evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .junction_features import (FEATURE_KEYS, JUNCTION_FEATURE_KEYS, JunctionFeatures,
                                JunctionPairFeatures, compute_junction_features,
                                compute_pair_features, node_position)

# Candidate feature sets, smallest first; ties in LOWO AUC break to fewer features
# (the linker's precedent: with a few hundred rows, piling on features overfits).
FEATURE_SETS = {
    "tangent_only": ("tangent_cos",),
    "tangent_turn": ("tangent_cos", "turn_angle_deg"),
    "tangent_width": ("tangent_cos", "width_ratio"),
    "tangent_intensity": ("tangent_cos", "intensity_ratio"),
    "tangent_width_intensity": ("tangent_cos", "width_ratio", "intensity_ratio"),
    "no_node": ("tangent_cos", "turn_angle_deg", "width_ratio", "intensity_ratio",
                "length_min_um"),
    "all": FEATURE_KEYS,
}

MIN_POSITIVES = 8            # below this a fitted model is not trustworthy
LETTER_INDEX = {"A": 0, "B": 1, "C": 2}
DEFAULT_PIXEL_UM = 0.6493
# Gate thresholds the two-stage rule searches. Selected per fold on that fold's
# TRAINING wells only -- never on the well being scored.
GATE_THRESHOLD_GRID = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)


@dataclass
class JunctionPairExample:
    well: str
    node: int
    key: str                          # "AB" / "AC" / "BC"
    features: JunctionPairFeatures
    label: int | None                 # 1 continues-through / 0 does not; None = unsure (dropped)

    def id_key(self) -> tuple[str, int, str]:
        return (self.well, self.node, self.key)


@dataclass
class JunctionExample:
    """One whole junction, for the branch-point gate. ``label`` 1 = branch point."""
    well: str
    node: int
    features: JunctionFeatures
    label: int | None

    def id_key(self) -> tuple[str, int]:
        return (self.well, self.node)


@dataclass
class RecomputedExamples:
    pairs: list
    junctions: list


def load_decision_rows(exports) -> list[dict]:
    """Decision rows from one export path or several, merged.

    Every export-reading entry point takes ``exports`` in this form so a
    multi-round training set is expressed the same way everywhere. Later
    exports win on a repeated ``(well, node)``; the active-learning builder
    already guarantees rounds never overlap, so a collision means a round was
    rebuilt/relabelled and the newest decision is the right one to keep.
    """
    if isinstance(exports, (str, Path)):
        exports = [exports]
    merged: dict[tuple[str, int], dict] = {}
    for export_path in exports:
        export = json.loads(Path(export_path).read_text(encoding="utf-8"))
        for row in export["decisions"].values():
            merged[(row["well"], row["node"])] = row
    return list(merged.values())


# --------------------------------------------------------------- feature recompute


def recompute_examples(exports, territory_cache_dir: str | Path,
                       bootstrap_dir: str | Path, pixel_um: float = DEFAULT_PIXEL_UM,
                       params=None) -> RecomputedExamples:
    """Rebuild pair-level AND junction-level features from one or more exports.

    Groups decisions by well so each well's branch graph is built once (across
    all rounds together, not once per round, and once for both feature levels
    -- skeletonising a full field is the expensive step). A branch-id mismatch
    between an export and the recomputed graph means the territory cache
    changed since that round was built -- that must fail loudly, not silently
    train on different geometry than the operator actually saw.
    """
    import tifffile
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    from classical.junction_ambiguity import _branch_length_um
    from classical.ridge_graph import TracerParams, build_branch_graph

    params = params or TracerParams()
    by_well: dict[str, list[dict]] = {}
    for row in load_decision_rows(exports):
        by_well.setdefault(row["well"], []).append(row)

    territory_cache_dir = Path(territory_cache_dir)
    bootstrap_dir = Path(bootstrap_dir)
    pairs: list[JunctionPairExample] = []
    junctions: list[JunctionExample] = []
    unmatched = 0
    for well, rows in sorted(by_well.items()):
        territory = np.load(territory_cache_dir / f"{well}.territory.npy")
        fiber = tifffile.imread(bootstrap_dir / well / "image_fiber.tif")
        territory = np.asarray(territory, dtype=bool)
        skeleton = skeletonize(territory)
        _graph, node_ends, coordinates = build_branch_graph(skeleton, pixel_um, params)
        distance_to_bg = ndi.distance_transform_edt(territory)
        branch_lengths = [_branch_length_um(c, pixel_um) for c in coordinates]

        for row in rows:
            ends = node_ends.get(row["node"])
            if ends is None or len(ends) != 3:
                unmatched += 1
                continue
            node_rc = node_position(coordinates, ends)
            for pair in row["pairs"]:
                if pair["label"] is None:
                    continue                       # unsure: not a valid training label
                a, b = LETTER_INDEX[pair["key"][0]], LETTER_INDEX[pair["key"][1]]
                branch_a, end_a, _dir_a = ends[a]
                branch_b, end_b, _dir_b = ends[b]
                if [branch_a, branch_b] != pair["branches"]:
                    raise RuntimeError(
                        f"{well} node {row['node']} pair {pair['key']}: branch id mismatch "
                        f"on recompute (expected {pair['branches']}, got [{branch_a}, {branch_b}]) "
                        "-- the territory cache likely changed since the round was built")
                feats = compute_pair_features(
                    coordinates, branch_a, end_a, branch_b, end_b,
                    distance_to_bg, fiber, pixel_um,
                    branch_lengths[branch_a], branch_lengths[branch_b], node_rc=node_rc)
                pairs.append(JunctionPairExample(well, row["node"], pair["key"], feats,
                                                 pair["label"]))
            # a junction is a branch-point example only when it was fully decided
            labels = [p["label"] for p in row["pairs"]]
            if all(v is not None for v in labels):
                junctions.append(JunctionExample(
                    well, row["node"],
                    compute_junction_features(
                        coordinates, ends, distance_to_bg, fiber, pixel_um,
                        [branch_lengths[b] for b, _e, *_ in ends]),
                    int(all(v == 0 for v in labels))))
        del territory, fiber
    if unmatched:
        print(f"  warning: {unmatched} decided junctions did not re-match the branch graph; "
             "excluded from training")
    return RecomputedExamples(pairs=pairs, junctions=junctions)


def recompute_training_pairs(exports, territory_cache_dir: str | Path,
                             bootstrap_dir: str | Path, pixel_um: float = DEFAULT_PIXEL_UM,
                             params=None) -> list[JunctionPairExample]:
    """Pair-level examples only. Thin wrapper over :func:`recompute_examples`."""
    return recompute_examples(exports, territory_cache_dir, bootstrap_dir,
                              pixel_um=pixel_um, params=params).pairs


# ------------------------------------------------------------------- fit and score


def _design(examples: list[JunctionPairExample], keys) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([e.features.vector(keys) for e in examples], dtype=float)
    y = np.array([e.label for e in examples], dtype=int)
    return X, y


def leave_one_well_out_auc(examples: list[JunctionPairExample], keys) -> dict:
    """LOWO AUC for one feature set. A row is scored by a model trained on the
    other wells only, so the number is not inflated by a well appearing on
    both sides. Folds with a single class in training are skipped and reported.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    wells = sorted({e.well for e in examples})
    scores, truth, skipped = [], [], []
    for held in wells:
        train = [e for e in examples if e.well != held]
        test = [e for e in examples if e.well == held]
        # `not train` is the single-well case: holding it out leaves nothing to
        # fit on, so LOWO is undefined -- must be checked before touching ytr,
        # which would be a zero-size array.
        if not train or not test:
            skipped.append(held)
            continue
        ytr = np.array([e.label for e in train])
        if ytr.min() == ytr.max():
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

    auc = None if len(set(truth)) < 2 else round(float(roc_auc_score(truth, scores)), 3)
    return {"auc": auc, "n_scored": len(truth), "skipped_wells": skipped,
            "n_features": len(keys), "keys": list(keys)}


def select_feature_set(examples: list[JunctionPairExample]) -> tuple[str, dict]:
    """Pick the feature set with the best LOWO AUC; ties break to fewer features."""
    results = {name: leave_one_well_out_auc(examples, keys) for name, keys in FEATURE_SETS.items()}
    ranked = sorted((r for r in results.values() if r["auc"] is not None),
                    key=lambda r: (-r["auc"], r["n_features"]))
    if not ranked:
        n_wells = len({e.well for e in examples})
        raise RuntimeError(
            f"no feature set produced a valid LOWO AUC ({n_wells} well(s) in the labels; "
            "leave-one-well-out needs at least 2, each with both classes present)")
    best = ranked[0]
    best_name = next(n for n, r in results.items() if r["keys"] == best["keys"])
    return best_name, {"chosen": best_name, "chosen_auc": best["auc"], "all": results}


@dataclass
class JunctionClassifierModel:
    keys: tuple[str, ...]
    scaler: object
    model: object
    fit_info: dict

    def score(self, example: JunctionPairExample) -> float:
        x = np.array([example.features.vector(self.keys)], dtype=float)
        return float(self.model.predict_proba(self.scaler.transform(x))[0, 1])


def fit_junction_classifier(examples: list[JunctionPairExample], keys) -> JunctionClassifierModel:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = _design(examples, keys)
    if int(y.sum()) < MIN_POSITIVES:
        raise RuntimeError(f"only {int(y.sum())} positives; need >= {MIN_POSITIVES}")
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    model.fit(scaler.transform(X), y)
    coefs = dict(zip(keys, (round(float(c), 3) for c in model.coef_[0])))
    info = {"n": int(len(y)), "n_positive": int(y.sum()), "keys": list(keys),
            "train_accuracy": round(float(model.score(scaler.transform(X), y)), 3),
            "coef_std_features": coefs}
    return JunctionClassifierModel(tuple(keys), scaler, model, info)


# --------------------------------------------------------- junction-level decisions


def leave_one_well_out_junction_decisions(examples: list[JunctionPairExample], keys,
                                          proba_threshold: float = 0.5) -> dict:
    """Per-junction argmax decision from LOWO out-of-fold pair probabilities.

    Groups the flat pair rows back into junctions (well, node), takes the
    highest-probability pair; if that probability doesn't clear
    ``proba_threshold`` the junction is called a branch point. This is the
    classifier's answer to the same question the classical floor's fixed rule
    answers, so the two can be compared junction-for-junction.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    wells = sorted({e.well for e in examples})
    proba_by_id: dict[tuple[str, int, str], float] = {}
    for held in wells:
        train = [e for e in examples if e.well != held]
        test = [e for e in examples if e.well == held]
        if not train or not test:
            continue                       # single-well: LOWO undefined (see above)
        ytr = np.array([e.label for e in train])
        if ytr.min() == ytr.max():
            continue
        Xtr, _ = _design(train, keys)
        Xte, _ = _design(test, keys)
        scaler = StandardScaler().fit(Xtr)
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        model.fit(scaler.transform(Xtr), ytr)
        proba = model.predict_proba(scaler.transform(Xte))[:, 1]
        for e, p in zip(test, proba):
            proba_by_id[e.id_key()] = float(p)

    by_junction: dict[tuple[str, int], list[JunctionPairExample]] = {}
    for e in examples:
        by_junction.setdefault((e.well, e.node), []).append(e)

    decisions = {}
    for junction, rows in by_junction.items():
        scored = [(r.key, proba_by_id.get(r.id_key())) for r in rows]
        if any(p is None for _, p in scored):
            continue                                # this well's fold was skipped
        best_key, best_proba = max(scored, key=lambda kp: kp[1])
        decisions[junction] = (best_key if best_proba >= proba_threshold else None, best_proba)
    return decisions


# ------------------------------------------------------- branch-point gate (stage 1)


@dataclass
class BranchPointModel:
    keys: tuple
    scaler: object
    model: object
    fit_info: dict

    def score(self, example: JunctionExample) -> float:
        x = np.array([example.features.vector(self.keys)], dtype=float)
        return float(self.model.predict_proba(self.scaler.transform(x))[0, 1])


def fit_branch_point_model(examples: list, keys=JUNCTION_FEATURE_KEYS) -> BranchPointModel:
    """Binary 'is this junction a branch point?' model over junction-level features.

    Exists because the single-stage rule can only call a branch point when all
    three independent pair scores fail at once -- something a pointwise model
    trained on pair labels is not optimised to do, and measurably the dominant
    remaining error (98 of 112 true branch points were being joined).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    labelled = [e for e in examples if e.label is not None]
    X = np.array([e.features.vector(keys) for e in labelled], dtype=float)
    y = np.array([e.label for e in labelled], dtype=int)
    if int(y.sum()) < MIN_POSITIVES:
        raise RuntimeError(f"only {int(y.sum())} branch points; need >= {MIN_POSITIVES}")
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    model.fit(scaler.transform(X), y)
    info = {"n": int(len(y)), "n_branch_points": int(y.sum()), "keys": list(keys),
            "train_accuracy": round(float(model.score(scaler.transform(X), y)), 3)}
    return BranchPointModel(tuple(keys), scaler, model, info)


def two_stage_decisions(pair_examples: list, junction_examples: list, truth: dict,
                        pair_keys=FEATURE_KEYS, junction_keys=JUNCTION_FEATURE_KEYS,
                        gate_grid=GATE_THRESHOLD_GRID) -> tuple[dict, dict]:
    """LOWO junction decisions under the two-stage rule, gate tuned per fold.

    Stage 1 asks the branch-point model "does anything continue through here?";
    only if it says yes does stage 2's pairwise model pick which pair. The gate
    threshold is chosen **inside each fold, on that fold's training wells
    only** -- picking it on the pooled set would be tuning a threshold on its
    own test data and inflates the result by ~2 points.

    Returns ``(decisions, info)``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pairs_by_junction: dict[tuple[str, int], list] = {}
    for e in pair_examples:
        pairs_by_junction.setdefault((e.well, e.node), []).append(e)
    junction_by_id = {e.id_key(): e for e in junction_examples}
    wells = sorted({e.well for e in pair_examples})

    def fit_pair(train):
        X = np.array([e.features.vector(pair_keys) for e in train], dtype=float)
        y = np.array([e.label for e in train], dtype=int)
        s = StandardScaler().fit(X)
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
        m.fit(s.transform(X), y)
        return s, m

    def pair_scores(junctions, scaler, model):
        out = {}
        for j in junctions:
            rows = pairs_by_junction[j]
            X = np.array([r.features.vector(pair_keys) for r in rows], dtype=float)
            proba = model.predict_proba(scaler.transform(X))[:, 1]
            out[j] = [(r.key, float(p)) for r, p in zip(rows, proba)]
        return out

    def decide(junctions, pscores, gate, threshold):
        return {j: (None if gate.get(j, 0.0) >= threshold
                    else max(pscores[j], key=lambda kp: kp[1])[0])
                for j in junctions if j in pscores}

    decisions: dict = {}
    chosen: list = []
    for held in wells:
        train_pairs = [e for e in pair_examples if e.well != held and e.label is not None]
        train_junctions = [e for e in junction_examples if e.well != held]
        test_ids = [j for j in pairs_by_junction if j[0] == held and j in junction_by_id]
        train_ids = [j for j in pairs_by_junction if j[0] != held and j in junction_by_id]
        y = np.array([e.label for e in train_pairs])
        if not test_ids or not train_pairs or y.min() == y.max():
            continue
        pair_scaler, pair_model = fit_pair(train_pairs)
        gate_model = fit_branch_point_model(train_junctions, junction_keys)

        gate_train = {j: gate_model.score(junction_by_id[j]) for j in train_ids}
        gate_test = {j: gate_model.score(junction_by_id[j]) for j in test_ids}
        scores_train = pair_scores(train_ids, pair_scaler, pair_model)
        scores_test = pair_scores(test_ids, pair_scaler, pair_model)

        best = max(gate_grid, key=lambda t: decision_accuracy(
            decide(train_ids, scores_train, gate_train, t), truth)["accuracy"] or 0.0)
        chosen.append(best)
        decisions.update(decide(test_ids, scores_test, gate_test, best))
    return decisions, {"gate_thresholds_per_fold": chosen,
                       "gate_threshold_selected_on": "training wells only, per fold"}


def classical_floor_decisions(exports, territory_cache_dir: str | Path,
                              pixel_um: float = DEFAULT_PIXEL_UM, params=None) -> dict:
    """The classical floor's own fixed `pair_junction_ends` decision at every
    decided junction -- the baseline the classifier must beat.
    """
    from classical.ridge_graph import TracerParams, build_branch_graph, pair_junction_ends
    from skimage.morphology import skeletonize

    params = params or TracerParams()
    by_well: dict[str, list[dict]] = {}
    for row in load_decision_rows(exports):
        by_well.setdefault(row["well"], []).append(row)

    letters = "ABC"
    territory_cache_dir = Path(territory_cache_dir)
    decisions = {}
    for well, rows in sorted(by_well.items()):
        territory = np.load(territory_cache_dir / f"{well}.territory.npy")
        skeleton = skeletonize(np.asarray(territory, dtype=bool))
        _graph, node_ends, _coords = build_branch_graph(skeleton, pixel_um, params)
        for row in rows:
            ends = node_ends.get(row["node"])
            if ends is None or len(ends) != 3:
                continue
            pairs = pair_junction_ends(ends, params.straight_dot)
            key = None if not pairs else "".join(sorted(letters[i] for i in pairs[0]))
            decisions[(well, row["node"])] = key
        del territory
    return decisions


def ground_truth_decisions(exports) -> dict:
    """Operator ground truth per decided (non-unsure) junction: pair key or None."""
    out = {}
    for row in load_decision_rows(exports):
        if row["unsure"]:
            continue
        out[(row["well"], row["node"])] = row["chosen_pair"]     # None == branch point
    return out


def decision_accuracy(predicted: dict, truth: dict) -> dict:
    """Fraction of ``truth``'s junctions where ``predicted`` agrees, plus the
    two error-type breakdowns that matter for this pipeline: missing a real
    pass-through (under-merge) vs wrongly joining a branch point (over-merge).
    """
    common = sorted(set(predicted) & set(truth))
    correct = sum(1 for k in common if predicted[k] == truth[k])
    false_join = sum(1 for k in common if truth[k] is None and predicted[k] is not None)
    false_split = sum(1 for k in common if truth[k] is not None and predicted[k] is None)
    wrong_pair = sum(1 for k in common if truth[k] is not None and predicted[k] is not None
                     and truth[k] != predicted[k])
    return {"n": len(common), "correct": correct,
            "accuracy": round(correct / len(common), 3) if common else None,
            "false_join_over_merge": false_join, "false_split_under_merge": false_split,
            "wrong_pair": wrong_pair}
