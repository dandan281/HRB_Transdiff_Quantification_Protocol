"""Re-export the environment verifier from `model_labs/omnipose/verify_env.py`.

That file stays where the development plan and `coordination/requests/claude/
2026-07-21-t02-start.md` reference it. It cannot be imported as
``omnipose.verify_env`` without colliding with the installed Omnipose library
(see :mod:`omnipose_lab`), so it is loaded by path instead.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

VERIFY_ENV_PATH = Path(__file__).resolve().parents[1] / "omnipose" / "verify_env.py"


def _load():
    spec = importlib.util.spec_from_file_location("pm_verify_env", VERIFY_ENV_PATH)
    if spec is None or spec.loader is None:                  # pragma: no cover
        raise ImportError(f"cannot load {VERIFY_ENV_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify() -> dict:
    """Assert a real GPU kernel launch; returns the environment record."""
    return _load().verify()
