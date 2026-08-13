"""Canonical validation and comparison of independent G1 pilot reviews."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import sha256_file
from .schema import InstanceSet, VALID_STATUSES, decode_rle

VALID_DISPOSITIONS = {"instance", "not_an_instance"}


def create_pilot_review_template(manifest_path: str | Path, handoff_path: str | Path,
                                 reviewer: str, output_path: str | Path) -> dict:
    manifest_path, handoff_path = Path(manifest_path).resolve(), Path(handoff_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff.get("pilot_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("handoff does not belong to the supplied pilot manifest")
    field_exports = {}
    for field in handoff["fields"]:
        stem = field["suggested_export_stem"]
        field_exports[field["field_key"]] = {
            "instances": f"{stem}.instances.json",
            "review_log": f"{stem}.instances.review_log.jsonl",
        }
    result = {
        "schema_version": "1.0", "pilot_manifest_sha256": sha256_file(manifest_path),
        "reviewer": reviewer, "field_exports": field_exports,
        "tasks": [{
            "task_id": task["task_id"], "field_key": task["field_key"],
            "source_object_id": task["object_id"], "disposition": None,
            "status": None, "final_instance_ids": [], "notes": "",
        } for task in manifest["tasks"]],
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_review(path: str | Path) -> tuple[Path, dict]:
    review_path = Path(path).resolve()
    return review_path, json.loads(review_path.read_text(encoding="utf-8"))


def validate_pilot_review(manifest_path: str | Path, review_path: str | Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review_path, review = _load_review(review_path)
    errors: list[str] = []
    reviewer = str(review.get("reviewer", "")).strip()
    if not reviewer:
        errors.append("reviewer is required")
    if review.get("pilot_manifest_sha256") != sha256_file(manifest_path):
        errors.append("pilot manifest hash mismatch")
    expected = {task["task_id"]: task for task in manifest.get("tasks", [])}
    rows = review.get("tasks", [])
    by_id = {row.get("task_id"): row for row in rows if row.get("task_id")}
    if len(by_id) != len(rows):
        errors.append("review task IDs must be present and unique")
    missing, extra = sorted(set(expected) - set(by_id)), sorted(set(by_id) - set(expected))
    if missing:
        errors.append(f"missing task decisions: {missing}")
    if extra:
        errors.append(f"unknown task decisions: {extra}")

    exports: dict[str, tuple[InstanceSet, dict[str, dict], Path]] = {}
    for field_key, item in review.get("field_exports", {}).items():
        try:
            instances_path = _resolve(review_path.parent, item["instances"])
            log_path = _resolve(review_path.parent, item["review_log"])
            instance_set = InstanceSet.load(instances_path)
            log_states = {}
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("type") == "instance":
                    log_states[entry["id"]] = entry
            exports[field_key] = (instance_set, log_states, instances_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{field_key}: invalid field export: {exc}")

    status_counts = {status: 0 for status in sorted(VALID_STATUSES)}
    status_counts["not_an_instance"] = 0
    for task_id, task in expected.items():
        row = by_id.get(task_id)
        if not row:
            continue
        if row.get("field_key") != task["field_key"]:
            errors.append(f"{task_id}: field_key mismatch")
        if row.get("source_object_id") != task["object_id"]:
            errors.append(f"{task_id}: source_object_id mismatch")
        disposition = row.get("disposition")
        final_ids = row.get("final_instance_ids", [])
        status = row.get("status")
        if disposition not in VALID_DISPOSITIONS:
            errors.append(f"{task_id}: invalid disposition {disposition!r}")
            continue
        if disposition == "not_an_instance":
            status_counts["not_an_instance"] += 1
            if status not in (None, "") or final_ids:
                errors.append(f"{task_id}: not_an_instance must have null status and no final IDs")
            continue
        if status not in VALID_STATUSES:
            errors.append(f"{task_id}: invalid instance status {status!r}")
            continue
        status_counts[status] += 1
        if not final_ids or len(final_ids) != len(set(final_ids)):
            errors.append(f"{task_id}: instance decision requires unique final_instance_ids")
            continue
        export = exports.get(task["field_key"])
        if not export:
            errors.append(f"{task_id}: missing field export for {task['field_key']}")
            continue
        instance_set, log_states, _ = export
        records = {record.id: record for record in instance_set.instances}
        for instance_id in final_ids:
            record, log = records.get(instance_id), log_states.get(instance_id)
            if record is None:
                errors.append(f"{task_id}: final instance {instance_id!r} absent from InstanceSet")
                continue
            if not record.reviewed or record.status != status:
                errors.append(f"{task_id}: {instance_id} status/review does not match decision")
            if not log or not log.get("reviewed") or log.get("reviewer") != reviewer:
                errors.append(f"{task_id}: {instance_id} reviewer provenance missing or inconsistent")
    return {
        "schema_version": "1.0", "review": str(review_path), "reviewer": reviewer,
        "passed": not errors, "task_count_expected": len(expected),
        "task_count_received": len(by_id), "status_counts": status_counts,
        "field_exports_loaded": len(exports), "errors": errors,
    }


def compare_pilot_reviews(manifest_path: str | Path, review_a_path: str | Path,
                          review_b_path: str | Path, *, mask_iou_threshold: float = 0.8) -> dict:
    validation_a = validate_pilot_review(manifest_path, review_a_path)
    validation_b = validate_pilot_review(manifest_path, review_b_path)
    if not validation_a["passed"] or not validation_b["passed"]:
        return {"passed": False, "ready_for_adjudication": False,
                "validation_a": validation_a, "validation_b": validation_b,
                "errors": ["both reviews must validate before comparison"]}
    if validation_a["reviewer"] == validation_b["reviewer"]:
        return {"passed": False, "ready_for_adjudication": False,
                "validation_a": validation_a, "validation_b": validation_b,
                "errors": ["independent reviews require different reviewer IDs"]}
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    path_a, review_a = _load_review(review_a_path)
    path_b, review_b = _load_review(review_b_path)
    rows_a = {row["task_id"]: row for row in review_a["tasks"]}
    rows_b = {row["task_id"]: row for row in review_b["tasks"]}
    set_cache: dict[tuple[str, str], InstanceSet] = {}

    def instance_set(review: dict, review_path: Path, field_key: str) -> InstanceSet:
        cache_key = (str(review_path), field_key)
        if cache_key not in set_cache:
            path = _resolve(review_path.parent, review["field_exports"][field_key]["instances"])
            set_cache[cache_key] = InstanceSet.load(path)
        return set_cache[cache_key]

    disagreements, agreements = [], 0
    for task in manifest["tasks"]:
        task_id, field_key = task["task_id"], task["field_key"]
        a, b = rows_a[task_id], rows_b[task_id]
        categories = []
        if a["disposition"] != b["disposition"]:
            categories.append("disposition")
        if a.get("status") != b.get("status"):
            categories.append("status")
        mask_iou = None
        if a["disposition"] == b["disposition"] == "instance":
            set_a, set_b = instance_set(review_a, path_a, field_key), instance_set(review_b, path_b, field_key)
            records_a = {record.id: record for record in set_a.instances}
            records_b = {record.id: record for record in set_b.instances}
            union_a = np.zeros(set_a.image_shape, dtype=bool)
            union_b = np.zeros(set_b.image_shape, dtype=bool)
            for instance_id in a["final_instance_ids"]:
                union_a |= decode_rle(records_a[instance_id].rle)
            for instance_id in b["final_instance_ids"]:
                union_b |= decode_rle(records_b[instance_id].rle)
            intersection = int(np.count_nonzero(union_a & union_b))
            union = int(np.count_nonzero(union_a | union_b))
            mask_iou = intersection / union if union else 0.0
            if mask_iou < mask_iou_threshold:
                categories.append("mask")
        if categories:
            disagreements.append({
                "task_id": task_id, "field_key": field_key,
                "categories": categories, "mask_iou": mask_iou,
                "review_a": a, "review_b": b, "adjudication": "pending",
            })
        else:
            agreements += 1
    return {
        "passed": True, "ready_for_adjudication": True,
        "reviewer_a": validation_a["reviewer"], "reviewer_b": validation_b["reviewer"],
        "task_count": len(manifest["tasks"]), "agreement_count": agreements,
        "disagreement_count": len(disagreements),
        "agreement_fraction": agreements / len(manifest["tasks"]),
        "mask_iou_threshold": mask_iou_threshold, "disagreements": disagreements,
        "validation_a": validation_a, "validation_b": validation_b,
    }
