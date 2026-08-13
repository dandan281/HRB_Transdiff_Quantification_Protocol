"""Verification of source and analysis artifact provenance."""
from __future__ import annotations

import json
from pathlib import Path

from .io import load_metadata, sha256_file


def verify_run_integrity(run_dir: str | Path) -> dict:
    run = Path(run_dir).resolve()
    metadata = load_metadata(run)
    checks = []

    source = Path(metadata["source_nd2"])
    expected_source = metadata["source_sha256"]
    actual_source = sha256_file(source) if source.is_file() else None
    checks.append({
        "kind": "source_nd2", "path": str(source), "expected_sha256": expected_source,
        "actual_sha256": actual_source, "passed": actual_source == expected_source,
    })

    history_path = run / "qc_history.jsonl"
    if history_path.is_file():
        lines = [line for line in history_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if lines:
            latest = json.loads(lines[-1])
            for kind in ("instances", "nuclei_masks", "territory"):
                item = latest[kind]
                path = Path(item["path"])
                actual = sha256_file(path) if path.is_file() else None
                checks.append({
                    "kind": kind, "path": str(path), "expected_sha256": item["sha256"],
                    "actual_sha256": actual, "passed": actual == item["sha256"],
                })
    return {"run": str(run), "passed": bool(checks) and all(x["passed"] for x in checks),
            "checks": checks}
