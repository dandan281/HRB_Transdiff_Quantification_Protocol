import json

from precision_myotube.integrity import verify_run_integrity
from precision_myotube.io import sha256_file


def test_one_byte_artifact_modification_is_detected(tmp_path):
    source = tmp_path / "field.nd2"
    source.write_bytes(b"source")
    instances = tmp_path / "instances.json"
    nuclei = tmp_path / "nuclei.npy"
    territory = tmp_path / "territory.npy"
    instances.write_bytes(b"instances")
    nuclei.write_bytes(b"nuclei")
    territory.write_bytes(b"territory")
    (tmp_path / "metadata.json").write_text(json.dumps({
        "source_nd2": str(source), "source_sha256": sha256_file(source)
    }), encoding="utf-8")
    entry = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in {
            "instances": instances, "nuclei_masks": nuclei, "territory": territory}.items()
    }
    (tmp_path / "qc_history.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    assert verify_run_integrity(tmp_path)["passed"]

    territory.write_bytes(b"territorz")
    result = verify_run_integrity(tmp_path)
    assert not result["passed"]
    assert next(x for x in result["checks"] if x["kind"] == "territory")["passed"] is False
