"""Fine-tune Cellpose on the myotube training set (thin, version-tolerant CLI wrapper).

Uses the Cellpose command-line trainer (more stable across Cellpose 3/4 than the Python API).
Point it at the folder built by build_training_set.py:

    python train_cellpose.py --dir training --epochs 300 --pretrained cyto3

It shells out to `python -m cellpose --train ...`. The resulting model is written under
`training/models/`; give that path to the napari "Run Cellpose" widget or to cellpose eval.

Notes
  * `--chan 0` = grayscale single fiber channel (0 = use the whole image). Keep it identical to
    how build_training_set.py wrote `<well>.tif`.
  * Start from `cyto3`; it already knows elongated/touching objects, so fine-tuning on a handful
    of wells goes a long way. Hold out >= 1 plate for validation, don't just trust train loss.
  * This does NOT install Cellpose. Do that first in a dedicated env (see README).
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="training folder (<well>.tif + <well>_masks.tif)")
    ap.add_argument("--pretrained", default="cyto3", help="starting model (cyto3 recommended)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--chan", type=int, default=0, help="fiber channel (0 = grayscale/whole image)")
    ap.add_argument("--chan2", type=int, default=0, help="optional 2nd channel (0 = none)")
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--mask-filter", default="_masks")
    ap.add_argument("--test-dir", default=None, help="held-out folder for validation")
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not run")
    a = ap.parse_args()

    cmd = [sys.executable, "-m", "cellpose", "--train",
           "--dir", a.dir,
           "--pretrained_model", a.pretrained,
           "--chan", str(a.chan), "--chan2", str(a.chan2),
           "--mask_filter", a.mask_filter,
           "--n_epochs", str(a.epochs),
           "--learning_rate", str(a.learning_rate),
           "--verbose"]
    if a.test_dir:
        cmd += ["--test_dir", a.test_dir]

    print("cellpose train:\n  " + " ".join(cmd))
    if a.dry_run:
        return
    try:
        raise SystemExit(subprocess.call(cmd))
    except FileNotFoundError:
        sys.exit("Cellpose not found. Install it first: pip install cellpose  (see README).")


if __name__ == "__main__":
    main()
