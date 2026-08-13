"""Make the canonical `precision_myotube` importable read-only during tests.

The annotation lane consumes Codex's frozen schema as the single source of truth
but does not require it to be pip-installed. This mirrors the runtime bridge in
``annotation_tools._schema_bridge`` so tests pass from a bare checkout.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_PKG = Path(__file__).resolve().parents[1]

for candidate in (ROOT / "PrecisionMyotube", ANNOTATION_PKG):
    if (candidate).is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
