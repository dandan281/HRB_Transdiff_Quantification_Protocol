import json
from pathlib import Path

import numpy as np
import pytest

from precision_myotube import batch


def _manifest(tmp_path, fields):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": "1.0", "fields": fields}), encoding="utf-8")
    return path


def test_manifest_rejects_duplicate_image_ids(tmp_path):
    nd2 = tmp_path / "field.nd2"
    nd2.write_bytes(b"synthetic")
    field = {"image_id": "field", "nd2": str(nd2), "plate": "1", "well": "A01",
             "field": "1", "output_dir": str(tmp_path / "run")}
    path = _manifest(tmp_path, [field, {**field, "output_dir": str(tmp_path / "run2")}])
    with pytest.raises(ValueError, match="duplicate image_id"):
        batch.load_batch_manifest(path)


def test_resume_reruns_deleted_stage_and_downstream(monkeypatch, tmp_path):
    nd2 = tmp_path / "field.nd2"
    nd2.write_bytes(b"synthetic nd2")
    masks = tmp_path / "nuclei.npy"
    np.save(masks, np.ones((4, 5), dtype=np.int32))
    instances = tmp_path / "instances.json"
    instances.write_text("{}", encoding="utf-8")
    run = tmp_path / "run"
    field = {"image_id": "field", "nd2": str(nd2), "plate": "1", "well": "A01",
             "field": "1", "output_dir": str(run), "nuclei_masks": str(masks),
             "instances": str(instances), "fiber_ch": 0, "dapi_ch": 1}
    path = _manifest(tmp_path, [field])
    calls = []

    def prepare(source, output, **kwargs):
        output = Path(output); output.mkdir(exist_ok=True)
        for index in range(2):
            np.save(output / f"ch{index}_raw16.npy", np.zeros((4, 5), dtype=np.uint16))
        (output / "metadata.json").write_text(json.dumps({
            "image_id": "field", "source_sha256": batch.sha256_file(source),
            "image_shape": [4, 5], "pixel_um": 1.0,
            "channels": {"fiber": 0, "dapi": 1}}), encoding="utf-8")
        calls.append("prepare")

    def territory(output):
        for name in ("desmin_semantic_mask.npy", "myotube_territory.npy"):
            np.save(Path(output) / name, np.ones((4, 5), dtype=bool))
        (Path(output) / "territory_metadata.json").write_text("{}", encoding="utf-8")
        calls.append("territory")

    def proposals(output):
        (Path(output) / "instance_proposals.json").write_text("{}", encoding="utf-8")
        calls.append("proposals")

    def analysis(output, _instances):
        for name in ("analysis_summary.json", "qc_flags.json"):
            (Path(output) / name).write_text("{}", encoding="utf-8")
        for name in ("myotubes.csv", "nuclei.csv", "field_summary.csv"):
            (Path(output) / name).write_text("header\n", encoding="utf-8")
        calls.append("analysis")
        return {"summary": {}}

    def reports(output, _result):
        (Path(output) / "qc_overlay.png").write_bytes(b"png")
        (Path(output) / "review.html").write_text("ok", encoding="utf-8")
        calls.append("report")

    monkeypatch.setattr(batch, "prepare_run", prepare)
    monkeypatch.setattr(batch, "create_territory", territory)
    monkeypatch.setattr(batch, "create_component_proposals", proposals)
    monkeypatch.setattr(batch, "analyze", analysis)
    monkeypatch.setattr(batch, "create_reports", reports)

    first = batch.run_batch(path, summary_dir=tmp_path / "summary")
    assert first["counts"] == {"success": 1, "failed": 0, "review_required": 0}
    assert calls == ["prepare", "territory", "proposals", "analysis", "report"]

    calls.clear()
    (run / "myotube_territory.npy").unlink()
    second = batch.run_batch(path, resume=True, summary_dir=tmp_path / "summary")
    assert second["counts"]["success"] == 1
    assert calls == ["territory", "proposals", "analysis", "report"]


def test_mask_shape_mismatch_fails_before_analysis(monkeypatch, tmp_path):
    run = tmp_path / "run"; run.mkdir()
    nd2 = tmp_path / "field.nd2"; nd2.write_bytes(b"x")
    bad = tmp_path / "bad.npy"; np.save(bad, np.zeros((2, 2), dtype=np.int32))
    (run / "metadata.json").write_text(json.dumps({
        "image_id": "field", "source_sha256": batch.sha256_file(nd2),
        "image_shape": [4, 5], "pixel_um": 1.0,
        "channels": {"fiber": 0, "dapi": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match source"):
        batch._validate_mask_shape({"image_id": "field"}, run, bad, "nucleus mask")
