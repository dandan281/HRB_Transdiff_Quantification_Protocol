"""Reproducible software-environment fingerprints for validated pipeline lanes."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .io import sha256_file


def _installed_packages() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True, capture_output=True, text=True,
    )
    return sorted(
        (line.strip() for line in result.stdout.splitlines() if line.strip()),
        key=str.casefold,
    )


def _torch_fingerprint() -> dict:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the target environment
        return {"imported": False, "error": f"{type(exc).__name__}: {exc}"}

    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    devices = []
    for index in range(count):
        devices.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        })
    return {
        "imported": True,
        "torch_version": str(torch.__version__),
        "compiled_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_available": available,
        "device_count": count,
        "devices": devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def fingerprint_environment(
    output_dir: str | Path,
    *,
    label: str,
    validation_summary: str | Path | None = None,
    expected_total_nuclei: int | None = None,
    package_lines: Iterable[str] | None = None,
    torch_info: dict | None = None,
) -> dict:
    """Write a package lock and a machine-readable, validation-bound fingerprint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    packages = sorted(
        (line.strip() for line in (package_lines or _installed_packages()) if line.strip()),
        key=str.casefold,
    )
    requirements_path = output_dir / "requirements.freeze.txt"
    requirements_path.write_text("\n".join(packages) + "\n", encoding="utf-8")

    python_info = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
    }
    system_info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    accelerator = torch_info if torch_info is not None else _torch_fingerprint()

    validation = None
    if validation_summary is not None:
        summary_path = Path(validation_summary).resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        observed = summary.get("total_nuclei")
        validation = {
            "analysis_summary": str(summary_path),
            "analysis_summary_sha256": sha256_file(summary_path),
            "image_id": summary.get("image_id"),
            "expected_total_nuclei": expected_total_nuclei,
            "observed_total_nuclei": observed,
            "passed": expected_total_nuclei is None or observed == expected_total_nuclei,
        }

    stable_payload = {
        "label": label,
        "python": {k: python_info[k] for k in ("version", "implementation")},
        "system": system_info,
        "packages": packages,
        "accelerator": accelerator,
    }
    environment_sha256 = hashlib.sha256(
        json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    result = {
        "schema_version": "1.0",
        "label": label,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "environment_sha256": environment_sha256,
        "python": python_info,
        "system": system_info,
        "accelerator": accelerator,
        "package_count": len(packages),
        "requirements_file": requirements_path.name,
        "requirements_sha256": sha256_file(requirements_path),
        "validation": validation,
    }
    output_path = output_dir / "fingerprint.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if validation is not None and not validation["passed"]:
        raise ValueError(
            "validation nucleus count does not match: "
            f"expected {expected_total_nuclei}, observed {validation['observed_total_nuclei']}"
        )
    return result
