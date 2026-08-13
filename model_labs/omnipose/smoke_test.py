"""Omnipose laboratory smoke test (CL03.2).

Runs the shared plumbing smoke. If Omnipose is installed, wire a real short
train/infer into ``_omnipose_predictor``; otherwise the deterministic fallback
validates the data layout, mask handling, checkpoint writing, and canonical
prediction export end to end.

    python model_labs/omnipose/smoke_test.py --out model_labs/omnipose/_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the shared helpers and canonical schema importable from a bare checkout.
ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "PrecisionMyotube", ROOT / "model_labs"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from _shared.smoke import run_smoke  # noqa: E402


def _try_omnipose_predictor():
    try:
        import omnipose  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return None, False

    def predict(stack):
        # TODO(CL03/M04): replace with a real short Omnipose train+infer on the
        # synthetic field. Kept as a fallback until the pinned env is created.
        from _shared.smoke import _threshold_predictor
        return _threshold_predictor(stack)

    return predict, True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_labs/omnipose/_smoke")
    ap.add_argument("--channels", default="desmin_only")
    args = ap.parse_args(argv)

    predictor, available = _try_omnipose_predictor()
    log = run_smoke("omnipose", args.out, framework_available=available,
                    predictor=predictor, channels=args.channels)
    print(json.dumps(log, indent=2))
    return 0 if log["canonical_validation"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
