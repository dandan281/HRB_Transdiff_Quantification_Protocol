"""Deterministic pilot-task sampling and fail-closed G1 readiness evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import csv

import numpy as np
from scipy import ndimage as ndi

from .geometry import measure_mask
from .io import load_metadata, load_run_channel
from .schema import InstanceSet
from .io import sha256_file

REQUIRED_DENSITIES = {"sparse", "dense"}
REQUIRED_INTENSITIES = {"bright", "dim"}
REQUIRED_LENGTHS = {"short", "long"}


def _quantile_class(values: list[float], low: str, middle: str, high: str) -> list[str]:
    if not values:
        return []
    q25, q75 = np.quantile(np.asarray(values, dtype=float), (0.25, 0.75))
    return [low if value <= q25 else high if value >= q75 else middle for value in values]


def build_pilot_candidates(path: str | Path, output_path: str | Path) -> dict:
    """Derive review-target strata from proposal masks; no record is treated as truth."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = []
    field_coverages: dict[str, float] = {}
    for sample in manifest.get("samples", []):
        run = Path(sample["run_dir"])
        if not run.is_absolute():
            run = (manifest_path.parent / run).resolve()
        metadata = load_metadata(run)
        proposal_path = Path(sample.get("proposals", run / "instance_proposals.json"))
        if not proposal_path.is_absolute():
            proposal_path = (manifest_path.parent / proposal_path).resolve()
        instance_set = InstanceSet.load(proposal_path)
        if instance_set.image_id != metadata["image_id"]:
            raise ValueError(f"{run}: proposal image ID does not match run metadata")
        fiber = np.asarray(load_run_channel(run, "fiber"))
        territory_path = run / "myotube_territory.npy"
        if not territory_path.is_file():
            raise FileNotFoundError(f"{run}: missing myotube_territory.npy")
        field_key = f"{sample['plate']}::{metadata['image_id']}"
        field_coverages[field_key] = float(np.load(territory_path).mean())
        occupied = np.zeros(instance_set.image_shape, dtype=bool)
        cropped = list(instance_set.cropped_masks())
        for _, (r0, c0, r1, c1), mask in cropped:
            occupied[r0:r1, c0:c1] |= mask
        for record, (r0, c0, r1, c1), mask in cropped:
            geometry = measure_mask(mask, float(metadata["pixel_um"]))
            pr0, pc0 = max(0, r0 - 1), max(0, c0 - 1)
            pr1, pc1 = min(occupied.shape[0], r1 + 1), min(occupied.shape[1], c1 + 1)
            local = np.zeros((pr1 - pr0, pc1 - pc0), dtype=bool)
            local[r0 - pr0:r1 - pr0, c0 - pc0:c1 - pc0] = mask
            neighbor_contact = bool(np.any(
                ndi.binary_dilation(local) & occupied[pr0:pr1, pc0:pc1] & ~local))
            candidates.append({
                "image_id": metadata["image_id"],
                "field_key": field_key,
                "object_id": record.id,
                "plate": str(sample["plate"]),
                "run_dir": str(run),
                "proposal_path": str(proposal_path),
                "mean_intensity": float(np.mean(fiber[r0:r1, c0:c1][mask])),
                "length_um": geometry.length_um,
                "branch_count": geometry.branch_count,
                "neighbor_contact": neighbor_contact,
                "border_contact": bool(
                    r0 == 0 or c0 == 0 or r1 == occupied.shape[0] or c1 == occupied.shape[1]),
            })
    if not candidates:
        raise ValueError("pilot source manifest has no proposal candidates")
    intensity_classes = _quantile_class(
        [x["mean_intensity"] for x in candidates], "dim", "medium", "bright")
    length_classes = _quantile_class(
        [x["length_um"] for x in candidates], "short", "medium", "long")
    coverage_values = list(field_coverages.values())
    q33, q67 = np.quantile(np.asarray(coverage_values), (1 / 3, 2 / 3))
    for item, intensity, length_class in zip(candidates, intensity_classes, length_classes):
        coverage = field_coverages[item["field_key"]]
        item["density"] = "sparse" if coverage <= q33 else "dense" if coverage >= q67 else "medium"
        item["intensity"] = intensity
        item["length_class"] = length_class
        item["hard_case"] = bool(
            item["branch_count"] > 0 or item["neighbor_contact"] or item["border_contact"])
        item["hard_case_reasons"] = [
            name for name, present in (
                ("branch_or_crossing", item["branch_count"] > 0),
                ("neighbor_contact", item["neighbor_contact"]),
                ("border_contact", item["border_contact"]),
            ) if present
        ]
    result = {
        "schema_version": "1.0",
        "purpose": "model-assisted review targets; not biological labels",
        "source_manifest": str(manifest_path.resolve()),
        "candidates": candidates,
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_pilot_handoff(manifest_path: str | Path, runs_path: str | Path,
                        output_path: str | Path) -> dict:
    """Bind frozen pilot targets to Claude-owned annotation packages."""
    manifest_path, runs_path = Path(manifest_path).resolve(), Path(runs_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    tasks_by_field: dict[str, list[dict]] = {}
    for task in manifest.get("tasks", []):
        tasks_by_field.setdefault(task["field_key"], []).append(task)
    fields = []
    bound: set[str] = set()
    for sample in runs.get("samples", []):
        package = Path(sample.get("package_dir", ""))
        if not package.is_absolute():
            package = (runs_path.parent / package).resolve()
        readme_path = package / "README.json"
        properties_path = package / "instance_properties.csv"
        if not readme_path.is_file() or not properties_path.is_file():
            raise FileNotFoundError(f"incomplete annotation package: {package}")
        readme = json.loads(readme_path.read_text(encoding="utf-8"))
        field_key = f"{sample['plate']}::{readme['image_id']}"
        tasks = tasks_by_field.get(field_key, [])
        with properties_path.open(newline="", encoding="utf-8-sig") as handle:
            prompt_ids = {row["id"] for row in csv.DictReader(handle)}
        missing = sorted({task["object_id"] for task in tasks} - prompt_ids)
        if missing:
            raise ValueError(f"{field_key}: pilot targets absent from package: {missing}")
        bound.update(task["task_id"] for task in tasks)
        artifacts = {}
        for name in ("README.json", "fiber_raw16.tif", "dapi_raw16.tif",
                     "semantic_territory.tif", "starting_labels.tif",
                     "instance_properties.csv", "overlap_ignore.tif"):
            artifact = package / name
            if not artifact.is_file():
                raise FileNotFoundError(f"{package}: missing {name}")
            artifacts[name] = sha256_file(artifact)
        stem = field_key.replace("::", "__")
        fields.append({
            "field_key": field_key, "plate": str(sample["plate"]),
            "image_id": readme["image_id"], "package_dir": str(package),
            "package_artifact_sha256": artifacts,
            "pilot_target_count": len(tasks),
            "pilot_task_ids": [task["task_id"] for task in tasks],
            "pilot_object_ids": [task["object_id"] for task in tasks],
            "suggested_export_stem": stem,
        })
    expected = {task["task_id"] for task in manifest.get("tasks", [])}
    if bound != expected:
        raise ValueError(f"unbound pilot tasks: {sorted(expected - bound)}")
    result = {
        "schema_version": "1.0", "pilot_manifest": str(manifest_path),
        "pilot_manifest_sha256": sha256_file(manifest_path),
        "task_count": len(expected), "field_count": len(fields), "fields": fields,
    }
    output = Path(output_path)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _stable_key(item: dict, seed: str) -> str:
    identity = f"{seed}|{item['image_id']}|{item['object_id']}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _validate_candidate(item: dict, index: int) -> dict:
    required = ("image_id", "object_id", "plate", "density", "intensity", "length_class",
                "hard_case")
    missing = [name for name in required if name not in item]
    if missing:
        raise ValueError(f"candidate {index}: missing {missing}")
    result = dict(item)
    result["plate"] = str(result["plate"])
    result["density"] = str(result["density"]).lower()
    result["intensity"] = str(result["intensity"]).lower()
    result["length_class"] = str(result["length_class"]).lower()
    result["hard_case"] = bool(result["hard_case"])
    result["task_id"] = str(result.get("task_id") or
                            f"{result['plate']}::{result['image_id']}::{result['object_id']}")
    return result


def audit_pilot_tasks(tasks: list[dict], *, target: int = 100, minimum_hard: int = 25) -> dict:
    ids = [item["task_id"] for item in tasks]
    plates = {item["plate"] for item in tasks}
    densities = {item["density"] for item in tasks}
    intensities = {item["intensity"] for item in tasks}
    lengths = {item["length_class"] for item in tasks}
    hard = sum(bool(item["hard_case"]) for item in tasks)
    requirements = {
        "approximately_target_size": max(1, target - 10) <= len(tasks) <= target + 10,
        "hard_cases_ge_minimum": hard >= minimum_hard,
        "sparse_and_dense_present": REQUIRED_DENSITIES <= densities,
        "bright_and_dim_present": REQUIRED_INTENSITIES <= intensities,
        "short_and_long_present": REQUIRED_LENGTHS <= lengths,
        "plate26_excluded": "PLATE_26" not in plates,
        "multiple_development_plates": len(plates) >= 2,
        "unique_task_ids": len(ids) == len(set(ids)),
    }
    return {
        "task_count": len(tasks), "hard_case_count": hard,
        "plates": sorted(plates), "densities": sorted(densities),
        "intensities": sorted(intensities), "length_classes": sorted(lengths),
        "requirements": requirements, "ready_for_dual_annotation": all(requirements.values()),
    }


def select_pilot_tasks(path: str | Path, output_path: str | Path, *,
                       target: int = 100, minimum_hard: int = 25,
                       seed: str = "precision-myotube-pilot-v1") -> dict:
    """Select review targets, never biological instances or ground truth."""
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    candidates = [_validate_candidate(item, index)
                  for index, item in enumerate(payload.get("candidates", []), start=1)]
    candidates = [item for item in candidates if item["plate"] != "PLATE_26"]
    if len(candidates) < target:
        raise ValueError(f"only {len(candidates)} development candidates for target {target}")
    candidates.sort(key=lambda item: _stable_key(item, seed))

    selected: list[dict] = []
    selected_ids: set[str] = set()

    def take(predicate) -> bool:
        for item in candidates:
            if item["task_id"] not in selected_ids and predicate(item):
                selected.append(item)
                selected_ids.add(item["task_id"])
                return True
        return False

    # Establish required category and plate coverage before filling quotas.
    for density in sorted(REQUIRED_DENSITIES):
        take(lambda item, value=density: item["density"] == value)
    for intensity in sorted(REQUIRED_INTENSITIES):
        take(lambda item, value=intensity: item["intensity"] == value)
    for length_class in sorted(REQUIRED_LENGTHS):
        take(lambda item, value=length_class: item["length_class"] == value)
    for plate in sorted({item["plate"] for item in candidates})[:2]:
        take(lambda item, value=plate: item["plate"] == value)
    while sum(item["hard_case"] for item in selected) < minimum_hard:
        if not take(lambda item: item["hard_case"]):
            raise ValueError(f"candidate pool cannot supply {minimum_hard} hard cases")
    for item in candidates:
        if len(selected) >= target:
            break
        if item["task_id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["task_id"])

    audit = audit_pilot_tasks(selected, target=target, minimum_hard=minimum_hard)
    result = {
        "schema_version": "1.0",
        "purpose": "dual-annotation review targets; not ground truth",
        "source": str(source_path.resolve()),
        "selection_seed": seed,
        "target": target,
        "minimum_hard_cases": minimum_hard,
        "tasks": selected,
        "audit": audit,
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_g1(path: str | Path) -> dict:
    """Evaluate recorded gate evidence without inferring missing human approval."""
    evidence_path = Path(path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    pilot_path = evidence.get("pilot_manifest")
    pilot_audit = None
    if pilot_path:
        pilot_path = Path(pilot_path)
        if not pilot_path.is_absolute():
            pilot_path = evidence_path.parent / pilot_path
        if pilot_path.is_file():
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
            pilot_audit = pilot.get("audit") or audit_pilot_tasks(pilot.get("tasks", []))
    requirements = {
        "synthetic_overlap_roundtrip_passed":
            evidence.get("synthetic_overlap_roundtrip_passed") is True,
        "pilot_ready_and_approximately_100":
            bool(pilot_audit and pilot_audit.get("ready_for_dual_annotation")),
        "all_pilot_tasks_dual_annotated":
            evidence.get("all_pilot_tasks_dual_annotated") is True,
        "disagreements_categorized":
            evidence.get("disagreements_categorized") is True,
        "disagreements_adjudicated_or_ambiguous":
            evidence.get("disagreements_adjudicated_or_ambiguous") is True,
        "no_critical_tool_defects":
            evidence.get("critical_tool_defects_open") == 0,
        "biological_reviewers_approved":
            evidence.get("biological_reviewers_approved") is True,
        "schema_validation_passed":
            evidence.get("schema_validation_passed") is True,
    }
    return {
        "gate": "G1", "passed": all(requirements.values()),
        "decision": "begin_wave_2" if all(requirements.values()) else "repeat_or_complete_pilot",
        "requirements": requirements, "pilot_audit": pilot_audit,
    }
