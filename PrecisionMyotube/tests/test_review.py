import json

import numpy as np

from precision_myotube.io import sha256_file
from precision_myotube.review import (compare_pilot_reviews, create_pilot_review_template,
                                      validate_pilot_review)
from precision_myotube.schema import InstanceRecord, InstanceSet, encode_rle


def _review_fixture(tmp_path, reviewer, status="complete", shift=0):
    shape = (20, 25)
    mask = np.zeros(shape, bool); mask[5:12, 6 + shift:18 + shift] = True
    field = tmp_path / reviewer; field.mkdir()
    instances = field / "field.instances.json"
    InstanceSet(shape, "field", [InstanceRecord(
        "final_1", status, encode_rle(mask), source="manual", reviewed=True)]).save(instances)
    log = field / "field.review_log.jsonl"
    log.write_text(json.dumps({"type": "instance", "id": "final_1", "status": status,
                               "reviewed": True, "reviewer": reviewer}) + "\n")
    return field, instances, log


def _write_review(path, manifest, reviewer, instances, log, status="complete"):
    path.write_text(json.dumps({
        "schema_version": "1.0", "pilot_manifest_sha256": sha256_file(manifest),
        "reviewer": reviewer,
        "field_exports": {"PLATE_23::field": {
            "instances": str(instances), "review_log": str(log)}},
        "tasks": [{"task_id": "PLATE_23::field::proposal_1",
                   "field_key": "PLATE_23::field", "source_object_id": "proposal_1",
                   "disposition": "instance", "status": status,
                   "final_instance_ids": ["final_1"], "notes": ""}],
    }), encoding="utf-8")


def test_review_validation_and_comparison(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{
        "task_id": "PLATE_23::field::proposal_1", "field_key": "PLATE_23::field",
        "object_id": "proposal_1"}]}))
    _, ai, al = _review_fixture(tmp_path, "annotator_a")
    _, bi, bl = _review_fixture(tmp_path, "annotator_b", shift=1)
    review_a, review_b = tmp_path / "a.json", tmp_path / "b.json"
    _write_review(review_a, manifest, "annotator_a", ai, al)
    _write_review(review_b, manifest, "annotator_b", bi, bl)
    assert validate_pilot_review(manifest, review_a)["passed"]
    result = compare_pilot_reviews(manifest, review_a, review_b, mask_iou_threshold=0.95)
    assert result["passed"] and result["ready_for_adjudication"]
    assert result["disagreement_count"] == 1
    assert result["disagreements"][0]["categories"] == ["mask"]


def test_not_an_instance_requires_no_status_or_final_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{
        "task_id": "PLATE_23::field::proposal_1", "field_key": "PLATE_23::field",
        "object_id": "proposal_1"}]}))
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "pilot_manifest_sha256": sha256_file(manifest), "reviewer": "r",
        "field_exports": {}, "tasks": [{
            "task_id": "PLATE_23::field::proposal_1", "field_key": "PLATE_23::field",
            "source_object_id": "proposal_1", "disposition": "not_an_instance",
            "status": None, "final_instance_ids": []}]}))
    assert validate_pilot_review(manifest, review)["passed"]


def test_review_template_binds_all_tasks_and_fields(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{
        "task_id": "PLATE_23::field::proposal_1", "field_key": "PLATE_23::field",
        "object_id": "proposal_1"}]}))
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps({
        "pilot_manifest_sha256": sha256_file(manifest),
        "fields": [{"field_key": "PLATE_23::field",
                    "suggested_export_stem": "PLATE_23__field"}]}))
    result = create_pilot_review_template(
        manifest, handoff, "reviewer_a", tmp_path / "review.json")
    assert result["tasks"][0]["disposition"] is None
    assert result["field_exports"]["PLATE_23::field"]["instances"] == (
        "PLATE_23__field.instances.json")
