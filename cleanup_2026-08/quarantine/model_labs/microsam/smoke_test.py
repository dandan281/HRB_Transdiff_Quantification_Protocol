"""micro-sam laboratory smoke test (CL03.2).

Runs the shared plumbing smoke. If micro-sam is installed, wire a real short
automatic-inference (no prompts) into ``_microsam_predictor``; otherwise the
deterministic fallback validates the export path end to end.

The official bake-off candidate (M05) must use automatic inference with NO
expert prompts -- ``used_prompts=False`` is enforced in the provenance.

    python model_labs/microsam/smoke_test.py --out model_labs/microsam/_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "PrecisionMyotube", ROOT / "model_labs"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from _shared.smoke import run_smoke  # noqa: E402


def _try_microsam_predictor():
    try:
        import micro_sam  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return None, False

    def predict(stack):
        # TODO(CL03/M05): replace with real automatic (prompt-free) micro-sam
        # instance segmentation. Fallback until the pinned env is created.
        from _shared.smoke import _threshold_predictor
        return _threshold_predictor(stack)

    return predict, True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_labs/microsam/_smoke")
    ap.add_argument("--channels", default="desmin_only")
    args = ap.parse_args(argv)

    predictor, available = _try_microsam_predictor()
    log = run_smoke("microsam", args.out, framework_available=available,
                    predictor=predictor, channels=args.channels)
    print(json.dumps(log, indent=2))
    return 0 if log["canonical_validation"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
