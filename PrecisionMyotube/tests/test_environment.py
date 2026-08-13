import json

import pytest

from precision_myotube.environment import fingerprint_environment


def test_environment_fingerprint_is_sorted_hashed_and_validation_bound(tmp_path):
    summary = tmp_path / "analysis_summary.json"
    summary.write_text(json.dumps({"image_id": "C08", "total_nuclei": 10114}))

    result = fingerprint_environment(
        tmp_path / "environment",
        label="validated-cellpose",
        validation_summary=summary,
        expected_total_nuclei=10114,
        package_lines=["torch==2", "cellpose==4"],
        torch_info={"imported": True, "cuda_available": False},
    )

    assert result["validation"]["passed"]
    assert len(result["environment_sha256"]) == 64
    assert (tmp_path / "environment" / "requirements.freeze.txt").read_text() == (
        "cellpose==4\ntorch==2\n"
    )
    saved = json.loads((tmp_path / "environment" / "fingerprint.json").read_text())
    assert saved["requirements_sha256"] == result["requirements_sha256"]


def test_environment_fingerprint_fails_on_validation_drift(tmp_path):
    summary = tmp_path / "analysis_summary.json"
    summary.write_text(json.dumps({"image_id": "C08", "total_nuclei": 10113}))

    with pytest.raises(ValueError, match="expected 10114, observed 10113"):
        fingerprint_environment(
            tmp_path / "environment",
            label="validated-cellpose",
            validation_summary=summary,
            expected_total_nuclei=10114,
            package_lines=["cellpose==4"],
            torch_info={"imported": True, "cuda_available": False},
        )
