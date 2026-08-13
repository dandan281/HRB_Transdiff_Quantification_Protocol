"""Build a fold-refit classical-floor + fragment-linker candidate.

The sealed ``classical_ridge_graph/v1`` floor is an immutable input.  This
module reads its six prediction sets, refits the fragment-linker coefficients
on the other five wells for each fold, applies the predeclared high-confidence
threshold, and writes a distinct hash-bound candidate run.  It never rewrites
the floor.  The feature family, candidate window, and operating-point policy
were developed globally before this instance-level run, so the result is not a
fully nested estimate of architecture selection.

The linker training labels are retrospective, single-operator development
evidence.  The population safety review rejected both automatic use and
manual-QC proposal use.  Outputs remain unreviewed development evidence only.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Iterable

import numpy as np

from .benchmark import benchmark_instances
from .schema import (
    InstanceRecord,
    InstanceSet,
    encode_sparse_positions,
    rle_foreground_positions,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_RUN = "model_labs/classical/_runs/v1"
DEFAULT_BOOTSTRAP = "PrecisionMyotube/annotation_work/bootstrap_v1/bootstrap_manifest.json"
DEFAULT_PAIRS = (
    "PrecisionMyotube/annotation_work/links_active_r3/banked/combined_pairs_r123.jsonl"
)
SEALED_V1_OUT = "PrecisionMyotube/runs/t02/classical_linker_v1"
DEFAULT_OUT = "PrecisionMyotube/runs/t02/classical_linker_constrained_v2"
DEFAULT_THRESHOLD = 0.90
DEFAULT_GAP_UM = 80.0
DEFAULT_COS_MIN = 0.70
DEFAULT_MERGE_POLICY = "constrained_axis"
MERGE_POLICIES = ("constrained_axis", "legacy_transitive_closure")
FEATURE_SET = "bridge_axis"
CANDIDATE = "classical_ridge_graph_linker"
CANDIDATE_VERSION = "v2-constrained"
SEALED_V1_VERSION = "v1"
SEED = 0
RELEASE_STATUS = (
    "rejected_development_baseline_only; "
    "automatic_and_manual_QC_proposal_use_withdrawn"
)
LINKER_LIMITATIONS = (
    "single operator; not consensus and not inter-rater agreement",
    "proposal-conditioned retrospective reference masks",
    "linker train/deploy domain shift AUC 0.639",
    "architecture/window/operating-point development is not nested LOWO; "
    "only scaler and logistic coefficients are fold-refit",
    "precision and F1 are not interpretable with sparse reviewed-complete GT",
    "reviewed-subset recall is descriptive and its resolution is one divided by "
    "the number of reviewed masks; a one-object change is not an established benefit",
    "linked instance counts are not authoritative independent-myotube counts",
    "sparse-reference over_merge_count and derived over_merge_rate fields are "
    "flag diagnostics, not estimates of the probability that an accepted merge is wrong",
    "the sealed v1 control-only safety round estimates a population over-merge rate "
    "of 0.6487 (95% stratified-bootstrap CI 0.4497-0.8318; 36 wrong among 55 "
    "resolved uniformly sampled merges across six wells)",
    "the axis-constrained merge policy fixes a closure defect but has no independent "
    "validation and does not rescue the rejected linker architecture",
    "raising the locked P>=0.90 threshold is not a supported safety mitigation",
    "the completed control-only safety round closes the linker branch; linked output "
    "must not be used as a proposal source for new reviewed masks",
    "linked output must not be used for manual-QC proposals, unattended analysis, "
    "new reviewed-mask proposals, or authoritative instance counts",
)


def _root_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_hash(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


class _Union:
    def __init__(self, values: Iterable[int]):
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        value = int(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        root_a, root_b = self.find(first), self.find(second)
        if root_a == root_b:
            return
        # Stable representative regardless of candidate iteration order.
        low, high = sorted((root_a, root_b))
        self.parent[high] = low


def prediction_to_label_image(prediction: InstanceSet) -> np.ndarray:
    """Convert a mutually exclusive prediction set to deterministic integer labels.

    The classical floor is mutually exclusive.  Refusing overlap here prevents
    this postprocessor from silently flattening a future overlap-aware candidate.
    """
    shape = tuple(prediction.image_shape)
    labels = np.zeros(shape, dtype=np.int32)
    height = shape[0]
    for label, record in enumerate(prediction.instances, start=1):
        positions = rle_foreground_positions(record.rle)
        rows = positions % height
        cols = positions // height
        if np.any(labels[rows, cols]):
            raise ValueError("fragment linker requires mutually exclusive base predictions")
        labels[rows, cols] = label
    return labels


def _materialize_components(
    prediction: InstanceSet,
    components: Iterable[Iterable[int]],
    provenance: dict,
) -> tuple[InstanceSet, list[dict]]:
    """Materialize component memberships without making another merge decision."""
    ordered_components = sorted(
        (tuple(sorted(int(value) for value in component)) for component in components),
        key=lambda values: values[0],
    )
    records = []
    merge_map = []
    for output_index, component in enumerate(ordered_components, start=1):
        positions = np.unique(np.concatenate([
            rle_foreground_positions(prediction.instances[index - 1].rle)
            for index in component
        ]))
        records.append(InstanceRecord(
            id=f"{CANDIDATE}_{output_index:04d}",
            status="complete",
            source=f"model:{CANDIDATE}",
            reviewed=False,
            notes=f"linked_component_size={len(component)}",
            rle=encode_sparse_positions(tuple(prediction.image_shape), positions),
        ))
        merge_map.append({
            "prediction_id": records[-1].id,
            "base_instance_indices": list(component),
            "base_instance_ids": [
                prediction.instances[index - 1].id for index in component
            ],
        })

    result = InstanceSet(
        tuple(prediction.image_shape),
        prediction.image_id,
        records,
        provenance=provenance,
    )
    result.validate()
    if any(record.reviewed for record in result.instances):
        raise AssertionError("linked predictions must remain unreviewed")
    return result, merge_map


def merge_prediction(
    prediction: InstanceSet,
    accepted_edges: Iterable[tuple[float, int, int]],
    *,
    provenance: dict,
    cos_min: float,
) -> tuple[InstanceSet, list[dict], list[dict]]:
    """Apply accepted edges without allowing axis-inconsistent transitive closure.

    Candidate scoring remains separate. This function enforces the already-declared
    ``cos_min`` across every pair of fragments that would share a linked object and
    returns declined edges as auditable evidence.
    """
    from annotation_tools.qc_review.link_geometry import (
        constrained_merge,
        fragment_axis,
    )

    count = len(prediction.instances)
    deduplicated: dict[tuple[int, int], float] = {}
    for probability, first, second in accepted_edges:
        first, second = int(first), int(second)
        if first == second or not 1 <= first <= count or not 1 <= second <= count:
            raise ValueError(f"invalid linker edge: {(first, second)}")
        key = tuple(sorted((first, second)))
        deduplicated[key] = max(float(probability), deduplicated.get(key, -np.inf))

    axes = {
        index: fragment_axis(mask)
        for index, (_, mask) in enumerate(prediction.masks(), start=1)
    }
    merge_result = constrained_merge(
        range(1, count + 1),
        [(probability, first, second)
         for (first, second), probability in deduplicated.items()],
        axes,
        cos_min=float(cos_min),
    )
    linked, merge_map = _materialize_components(
        prediction, merge_result.components.values(), provenance
    )
    return linked, merge_map, merge_result.refused


def _merge_prediction_legacy(
    prediction: InstanceSet,
    accepted_pairs: Iterable[tuple[int, int]],
    *,
    provenance: dict,
) -> tuple[InstanceSet, list[dict]]:
    """Exact pre-2026-08-04 closure retained only to reproduce sealed v1."""
    count = len(prediction.instances)
    union = _Union(range(1, count + 1))
    for first, second in accepted_pairs:
        first, second = int(first), int(second)
        if first == second or first not in union.parent or second not in union.parent:
            raise ValueError(f"invalid linker edge: {(first, second)}")
        union.union(first, second)

    groups: dict[int, list[int]] = {}
    for index in range(1, count + 1):
        groups.setdefault(union.find(index), []).append(index)
    return _materialize_components(prediction, groups.values(), provenance)


def _model_spec(model, training_pairs, train_wells: list[str]) -> dict:
    rows = sorted([
        {
            "well": pair.well,
            "fragment_id": pair.fragment_id,
            "candidate_id": pair.candidate_id,
            "label": int(pair.label),
            "features": [
                float(value) for value in pair.features.vector(model.keys)
            ],
        }
        for pair in training_pairs
    ], key=lambda row: (row["well"], row["fragment_id"], row["candidate_id"]))
    return {
        "kind": "standard_scaler_plus_logistic_regression",
        "feature_keys": list(model.keys),
        "fit_info": model.fit_info,
        "training_wells": list(train_wells),
        "training_rows_sha256": _json_hash(rows),
        "training_rows": rows,
        "scaler": {
            "mean": np.asarray(model.scaler.mean_, dtype=float).tolist(),
            "scale": np.asarray(model.scaler.scale_, dtype=float).tolist(),
        },
        "logistic_regression": {
            "classes": np.asarray(model.model.classes_).tolist(),
            "coef": np.asarray(model.model.coef_, dtype=float).tolist(),
            "intercept": np.asarray(model.model.intercept_, dtype=float).tolist(),
            "C": float(model.model.C),
            "class_weight": model.model.class_weight,
            "solver": model.model.solver,
            "max_iter": int(model.model.max_iter),
        },
    }


def _environment_record() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "scikit-image", "scikit-learn", "tifffile"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "accelerator": "CPU only",
    }
    return {**record, "environment_sha256": _json_hash(record)}


def _training_input_hashes(bootstrap: dict, pairs_path: Path) -> dict:
    paths = {"pairs": pairs_path}
    for well, info in sorted(bootstrap["per_well"].items()):
        package = ROOT / "PrecisionMyotube/annotation_work" / info["package"]
        for name in ("fiber_raw16.tif", "starting_labels.tif", "README.json"):
            path = package / name
            if path.is_file():
                paths[f"{well}/{name}"] = path
        territory = package / "semantic_territory.tif"
        if territory.is_file():
            paths[f"{well}/semantic_territory.tif"] = territory
    return {
        key: {"path": _relative(path), "sha256": _sha256(path)}
        for key, path in sorted(paths.items())
    }


def _source_hashes() -> dict:
    from annotation_tools.qc_review import (
        link_candidates,
        link_features,
        link_geometry,
        link_model,
    )

    paths = {
        "linked_candidate": Path(__file__),
        "link_candidates": Path(link_candidates.__file__),
        "link_features": Path(link_features.__file__),
        "link_geometry": Path(link_geometry.__file__),
        "link_model": Path(link_model.__file__),
    }
    return {
        key: {"path": _relative(path), "sha256": _sha256(path)}
        for key, path in paths.items()
    }


def _write_prediction(
    out_dir: Path,
    prediction: InstanceSet,
    provenance: dict,
    candidate_version: str,
) -> dict:
    target = (
        out_dir / "predictions" / CANDIDATE
        / f"{candidate_version}-fold-{prediction.image_id}"
    )
    target.mkdir(parents=True, exist_ok=True)
    instances_path = target / f"{prediction.image_id}.instances.json"
    prediction.save(instances_path)
    manifest = {
        "image_id": prediction.image_id,
        "image_shape": list(prediction.image_shape),
        "n_instances": len(prediction.instances),
        "provenance": provenance,
        "authoritative_export": instances_path.name,
        "instances_sha256": _sha256(instances_path),
        "note": (
            "Unreviewed linked predictions; counts are not authoritative "
            "independent-myotube counts."
        ),
    }
    manifest_path = target / f"{prediction.image_id}.prediction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "instances": _relative(instances_path),
        "instances_sha256": _sha256(instances_path),
        "manifest": _relative(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def _metric_subset(metrics: dict) -> dict:
    keys = (
        "n_gt", "n_pred", "tp", "precision", "recall", "f1",
        "precision_weighted_score", "mean_matched_iou",
        "false_split_count", "false_split_rate",
        "over_merge_count", "over_merge_rate",
        "length_mdape", "width_mdape", "automatic_coverage",
    )
    return {key: metrics[key] for key in keys}


def _micro_summary(folds: list[dict]) -> dict:
    metrics = [fold["held_out_metrics"] for fold in folds]
    n_gt = sum(row["n_gt"] for row in metrics)
    n_pred = sum(row["n_pred"] for row in metrics)
    tp = sum(row["tp"] for row in metrics)
    false_splits = sum(row["false_split_count"] for row in metrics)
    over_merges = sum(row["over_merge_count"] for row in metrics)
    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": tp,
        "recall": tp / n_gt if n_gt else 0.0,
        "precision": tp / n_pred if n_pred else 0.0,
        "false_split_count": false_splits,
        "false_split_rate": false_splits / n_gt if n_gt else 0.0,
        "over_merge_count": over_merges,
        "over_merge_rate_per_prediction": over_merges / n_pred if n_pred else 0.0,
        "over_merge_rate_per_sparse_gt": over_merges / n_gt if n_gt else 0.0,
        "over_merge_rates_interpretable": False,
        "over_merge_rate_warning": (
            "Sparse reviewed-complete GT cannot establish the probability that an "
            "accepted merge is wrong; these fields are flag diagnostics only."
        ),
        "recall_resolution_per_reviewed_object": 1 / n_gt if n_gt else None,
        "recall_interpretation": (
            "valid only as descriptive reviewed-subset coverage; a one-object change "
            "does not establish a recall benefit"
        ),
    }


def run_linked_candidate(
    out_dir: str | Path = DEFAULT_OUT,
    *,
    base_run: str | Path = DEFAULT_BASE_RUN,
    bootstrap_path: str | Path = DEFAULT_BOOTSTRAP,
    pairs_path: str | Path = DEFAULT_PAIRS,
    threshold: float = DEFAULT_THRESHOLD,
    gap_um: float = DEFAULT_GAP_UM,
    cos_min: float = DEFAULT_COS_MIN,
    merge_policy: str = DEFAULT_MERGE_POLICY,
) -> dict:
    """Materialize a distinct six-fold linked candidate without mutating the floor."""
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("link threshold must be in (0,1)")
    if merge_policy not in MERGE_POLICIES:
        raise ValueError(f"unknown merge policy: {merge_policy}")
    out_dir = _root_path(out_dir)
    sealed_v1_out = _root_path(SEALED_V1_OUT)
    if merge_policy == "constrained_axis" and out_dir.resolve() == sealed_v1_out.resolve():
        raise ValueError(
            "constrained-axis merging is a new candidate and cannot use the sealed v1 run id"
        )
    candidate_version = (
        SEALED_V1_VERSION
        if merge_policy == "legacy_transitive_closure"
        else CANDIDATE_VERSION
    )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty linked run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_run = _root_path(base_run)
    bootstrap_path = _root_path(bootstrap_path)
    pairs_path = _root_path(pairs_path)
    base_manifest_path = base_run / "run_manifest.json"
    base_grid_path = base_run / "grid_scores.json"
    base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    if _sha256(bootstrap_path) != base["input_manifest_sha256"]:
        raise ValueError("base run and bootstrap manifest hash do not match")
    if str(base.get("candidate")) != "classical_ridge_graph":
        raise ValueError("linked candidate requires the sealed classical floor")

    # Runtime imports keep ordinary canonical imports independent of model-lab extras.
    import tifffile
    from annotation_tools.qc_review.link_candidates import find_link_candidates
    from annotation_tools.qc_review.link_features import (
        compute_features,
        field_background,
        geometry_cache,
    )
    from annotation_tools.qc_review.link_model import (
        FEATURE_SETS,
        LinkPair,
        fit_linker,
        recompute_training_pairs,
    )

    packages = {
        well: ROOT / "PrecisionMyotube/annotation_work" / info["package"]
        for well, info in bootstrap["per_well"].items()
    }
    # The sealed v1 candidate definition predates the global-axis gate.  A run
    # with that gate enabled is deliberately a new candidate under a new run id.
    require_axis_agreement = merge_policy == "constrained_axis"
    training_pairs = recompute_training_pairs(
        pairs_path,
        packages,
        gap_um=float(gap_um),
        cos_min=float(cos_min),
        require_axis_agreement=require_axis_agreement,
    )
    feature_keys = FEATURE_SETS[FEATURE_SET]
    wells = sorted(bootstrap["per_well"])
    if set(wells) != {str(fold["held_out_well"]) for fold in base["folds"]}:
        raise ValueError("base folds do not match bootstrap wells")

    environment = _environment_record()
    source_hashes = _source_hashes()
    input_hashes = _training_input_hashes(bootstrap, pairs_path)
    started = time.time()
    folds = []

    for base_fold in base["folds"]:
        well = str(base_fold["held_out_well"])
        train_wells = sorted(str(value) for value in base_fold["train_wells"])
        expected_train = sorted(set(wells) - {well})
        if train_wells != expected_train:
            raise ValueError(f"{well}: base fold training wells are not LOWO")
        fold_train = [pair for pair in training_pairs if pair.well in train_wells]
        if any(pair.well == well for pair in fold_train):
            raise AssertionError(f"{well}: held-out linker labels leaked into fit")
        model = fit_linker(fold_train, feature_keys)
        model_spec = _model_spec(model, fold_train, train_wells)
        model_dir = out_dir / "linker_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"fold-{well}.json"
        model_path.write_text(json.dumps(model_spec, indent=2), encoding="utf-8")
        model_hash = _sha256(model_path)

        base_prediction_path = _root_path(base_fold["prediction"]["instances"])
        if _sha256(base_prediction_path) != base_fold["prediction"]["instances_sha256"]:
            raise ValueError(f"{well}: sealed base prediction hash changed")
        base_prediction = InstanceSet.load(base_prediction_path)
        labels = prediction_to_label_image(base_prediction)
        fiber_path = (
            ROOT / "PrecisionMyotube/annotation_work/bootstrap_v1"
            / well / "image_fiber.tif"
        )
        fiber = tifffile.imread(fiber_path)
        background = field_background(fiber)
        offered = find_link_candidates(
            labels,
            list(range(1, len(base_prediction.instances) + 1)),
            float(bootstrap["per_well"][well].get("pixel_um", 0.6493)),
            gap_um=float(gap_um),
            cos_min=float(cos_min),
            # False reproduces sealed classical_linker_v1; True belongs only to
            # the explicitly new constrained candidate/run.
            require_axis_agreement=require_axis_agreement,
        )
        needed = (
            {int(fragment.split("_")[-1]) for fragment in offered}
            | {
                int(candidate.candidate_id.split("_")[-1])
                for candidates in offered.values() for candidate in candidates
            }
        )
        geoms = geometry_cache(labels, needed)
        scores = []
        for fragment_id, candidates in sorted(offered.items()):
            first = int(fragment_id.split("_")[-1])
            for candidate in candidates:
                second = int(candidate.candidate_id.split("_")[-1])
                features = compute_features(
                    fiber,
                    None,
                    candidate.fragment_endpoint,
                    candidate.candidate_endpoint,
                    candidate.gap_um,
                    min(candidate.cos_fragment, candidate.cos_candidate),
                    float(bootstrap["per_well"][well].get("pixel_um", 0.6493)),
                    background=background,
                    fragment_geom=geoms.get(first),
                    candidate_geom=geoms.get(second),
                )
                probability = model.score(
                    LinkPair(well, fragment_id, candidate.candidate_id, features)
                )
                scores.append({
                    "first": first,
                    "second": second,
                    "fragment_id": fragment_id,
                    "candidate_id": candidate.candidate_id,
                    "probability": probability,
                    "accepted": probability >= float(threshold),
                })
        scores.sort(key=lambda row: (row["first"], row["second"]))
        accepted_edges = [
            (float(row["probability"]), row["first"], row["second"])
            for row in scores if row["accepted"]
        ]
        thresholds = {
            **base_fold["selected_tracer"],
            **base_fold["selected_filters"],
            "base_parameter_selection_metric": base["parameter_selection"]["metric"],
            "selected_on_wells": train_wells,
            "link_probability": float(threshold),
            "link_threshold_origin": (
                "predeclared P>=0.90 high-confidence operating point in "
                "2026-07-23 linker reports"
            ),
            "link_gap_um": float(gap_um),
            "link_cos_min": float(cos_min),
            "link_feature_set": FEATURE_SET,
            "merge_policy": merge_policy,
        }
        provenance = {
            "kind": "model_prediction",
            "model": CANDIDATE,
            "version": f"{candidate_version}-fold-{well}",
            "architecture": (
                "sealed classical_ridge_graph/v1 predictions -> "
                "LOWO logistic fragment-linker -> "
                + (
                    "axis-constrained component union"
                    if merge_policy == "constrained_axis"
                    else "sealed-v1 legacy transitive closure"
                )
            ),
            "checkpoint_hash": model_hash,
            "environment_hash": environment["environment_sha256"],
            "data_hash": base["input_manifest_sha256"],
            "seed": SEED,
            "thresholds": thresholds,
            "channels": "desmin_only",
            "used_prompts": False,
            "base_prediction": _relative(base_prediction_path),
            "base_prediction_sha256": _sha256(base_prediction_path),
            "linker_training_pairs_sha256": _sha256(pairs_path),
            "review_policy": "unreviewed predictions are never authoritative",
        }
        if merge_policy == "constrained_axis":
            linked, merge_map, refused_edges = merge_prediction(
                base_prediction,
                accepted_edges,
                provenance=provenance,
                cos_min=float(cos_min),
            )
        else:
            linked, merge_map = _merge_prediction_legacy(
                base_prediction,
                [(first, second) for _, first, second in accepted_edges],
                provenance=provenance,
            )
            refused_edges = []
        prediction_info = _write_prediction(
            out_dir, linked, provenance, candidate_version
        )

        decision_dir = out_dir / "linker_decisions"
        decision_dir.mkdir(parents=True, exist_ok=True)
        decision_path = decision_dir / f"{well}.json"
        unique_edges = {
            tuple(sorted((row["first"], row["second"])))
            for row in scores if row["accepted"]
        }
        decisions = {
            "well": well,
            "threshold": float(threshold),
            "n_directed_candidates": len(scores),
            "n_directed_accepted": len(accepted_edges),
            "n_unique_accepted_edges": len(unique_edges),
            "n_base_instances": len(base_prediction.instances),
            "n_linked_instances": len(linked.instances),
            "merge_policy": merge_policy,
            "n_refused_edges": len(refused_edges),
            "refused_edges": refused_edges,
            "scores": scores,
            "merge_map": merge_map,
        }
        decision_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")

        gt_path = _root_path(base_fold["eval_gt"]["path"])
        metrics = benchmark_instances(
            gt_path, _root_path(prediction_info["instances"])
        )
        folds.append({
            "held_out_well": well,
            "train_wells": train_wells,
            "selected_param_index": base_fold["selected_param_index"],
            "selected_tracer": base_fold["selected_tracer"],
            "selected_filters": base_fold["selected_filters"],
            "train_mean_selection_score": base_fold["train_mean_selection_score"],
            "n_instances_predicted": len(linked.instances),
            "assigned_fraction": base_fold.get("assigned_fraction"),
            "held_out_metrics": _metric_subset(metrics),
            "base_held_out_metrics": base_fold["held_out_metrics"],
            "linker": {
                "threshold": float(threshold),
                "threshold_selected_on": "predeclared before instance-level integration",
                "feature_set": FEATURE_SET,
                "gap_um": float(gap_um),
                "cos_min": float(cos_min),
                "n_train_pairs": len(fold_train),
                "n_train_positive": model.fit_info["n_positive"],
                "training_wells": train_wells,
                "model": _relative(model_path),
                "model_sha256": model_hash,
                "decisions": _relative(decision_path),
                "decisions_sha256": _sha256(decision_path),
                "n_directed_candidates": len(scores),
                "n_directed_accepted": len(accepted_edges),
                "n_unique_accepted_edges": len(unique_edges),
                "merge_policy": merge_policy,
                "n_refused_edges": len(refused_edges),
                "refused_edges": refused_edges,
            },
            "prediction": prediction_info,
            "eval_gt": base_fold["eval_gt"],
        })
        del labels, fiber

    shutil.copyfile(base_grid_path, out_dir / "grid_scores.json")
    run_manifest = {
        "task": "T02",
        "candidate": CANDIDATE,
        "candidate_version": candidate_version,
        "candidate_role": (
            "rejected legacy fragment-linker reproducer"
            if merge_policy == "legacy_transitive_closure" else
            "axis-constrained fragment-linker development candidate"
        ),
        "release_status": RELEASE_STATUS,
        "release_status_reason": (
            "The sealed v1 policy has a measured population over-merge rate of 0.6487 "
            "(95% CI 0.4497-0.8318), about 350 wrong accepted merges versus 11 "
            "recovered false splits. The constrained policy fixes a closure defect but "
            "has no independent validation and does not reopen the rejected branch."
        ),
        "command": (
            "python -m precision_myotube linked-candidate-run "
            f"--base-run {_relative(base_run)} --pairs {_relative(pairs_path)} "
            f"--out {_relative(out_dir)} --threshold {float(threshold)} "
            f"--gap-um {float(gap_um)} --cos-min {float(cos_min)} "
            f"--merge-policy {merge_policy}"
        ),
        "input_manifest": _relative(bootstrap_path),
        "input_manifest_sha256": _sha256(bootstrap_path),
        "base_run": _relative(base_run),
        "base_run_manifest_sha256": _sha256(base_manifest_path),
        "base_grid_scores_sha256": _sha256(base_grid_path),
        "base_floor_mutated": False,
        "split_policy": (
            "whole-well leave-one-well-out; base parameters inherited from the "
            "sealed fold and linker coefficients refitted on the other five wells only"
        ),
        "lowo_scope": (
            "linker coefficients and scaler are fit on the other five wells; "
            "feature family, candidate window, and P>=0.90 policy were fixed from "
            "earlier all-well development and are not nested within each fold"
        ),
        "global_development_used_all_six_wells": True,
        "parameter_selection": base["parameter_selection"],
        "linker_operating_point": {
            "threshold": float(threshold),
            "threshold_origin": (
                "predeclared high-confidence P>=0.90 point; not selected on "
                "held-out instance metrics"
            ),
            "gap_um": float(gap_um),
            "cos_min": float(cos_min),
            "feature_set": FEATURE_SET,
            "require_axis_agreement_at_candidate_generation": require_axis_agreement,
            "merge_policy": merge_policy,
        },
        "seed": SEED,
        "environment": environment,
        "source_hashes": source_hashes,
        "linker_training_inputs": input_hashes,
        "linker_training_pairs_used": True,
        "linker_training_pair_source": _relative(pairs_path),
        "linker_training_pair_source_sha256": _sha256(pairs_path),
        "linker_training_domain": "annotation-package proposal masks",
        "deployment_domain": "sealed classical-floor predictions",
        "domain_shift_auc": 0.639,
        "folds": folds,
        "summary": _micro_summary(folds),
        "merge_constraint": {
            "policy": merge_policy,
            "cos_min": float(cos_min),
            "require_axis_agreement_at_candidate_generation": require_axis_agreement,
            "n_unique_accepted_edges_before_constraint": sum(
                fold["linker"]["n_unique_accepted_edges"] for fold in folds
            ),
            "n_refused_edges": sum(
                fold["linker"]["n_refused_edges"] for fold in folds
            ),
            "refused_edges_by_well": {
                fold["held_out_well"]: fold["linker"]["refused_edges"]
                for fold in folds
            },
        },
        "failures": [],
        "synthetic_pairs_used": False,
        "correction_pairs_used": False,
        "correction_pairs_note": (
            "The 40 mask-correction pairs were not used for training or tuning. "
            "Separate operator fragment-link decisions were used fold-honestly."
        ),
        "density_strata": None,
        "evidence_class": (
            "development_bootstrap_single_operator_proposal_conditioned_retrospective"
        ),
        "limitations": list(LINKER_LIMITATIONS),
        "seconds_total": round(time.time() - started, 1),
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return run_manifest
