"""Render the human-annotated ground truth in the SAME style as
`zeroshot.py`'s overlays, so the two can be compared side by side.

Without this, a zero-shot overlay is judged against a mental image of what the
GT looks like. Rendered identically -- same normalisation, same label colouring,
same crop -- the comparison is direct.

Also prints the geometry of each field, which is the reference any candidate has
to reproduce.

Run from the repo root::

    python model_labs/omnipose_lab/render_gt_overlay.py
    python model_labs/omnipose_lab/render_gt_overlay.py --crop 1024
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "model_labs",
           ROOT / "model_labs" / "omnipose_lab"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BOOTSTRAP = ROOT / "PrecisionMyotube/annotation_work/bootstrap_v1"


def main(argv=None) -> int:
    import tifffile
    from zeroshot import (PIXEL_UM, measure_instances, normalize_field,
                          save_overlay)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wells", nargs="+", default=None)
    ap.add_argument("--crop", type=int, default=0,
                    help="centre crop in px; match the zero-shot run's --crop "
                         "for a like-for-like picture (0 = full field)")
    ap.add_argument("--pixel-um", type=float, default=PIXEL_UM)
    ap.add_argument("--out", default="model_labs/omnipose_lab/_runs/gt_overlays")
    args = ap.parse_args(argv)

    wells = args.wells or sorted(p.name for p in BOOTSTRAP.iterdir() if p.is_dir())
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"human-annotated ground truth  crop={args.crop or 'full field'}  "
          f"pixel={args.pixel_um} um\n")
    hdr = (f"{'well':<24}{'n':>5}{'len_med':>9}{'len_p90':>9}"
           f"{'wid_med':>9}{'aspect':>8}")
    print(hdr)
    print("-" * len(hdr))

    for w in wells:
        img = tifffile.imread(BOOTSTRAP / w / "image_fiber.tif")
        lab = tifffile.imread(BOOTSTRAP / w / "labels.tif").astype(np.int32)
        if args.crop:
            c = args.crop
            y, x = (img.shape[0] - c) // 2, (img.shape[1] - c) // 2
            img, lab = img[y:y + c, x:x + c], lab[y:y + c, x:x + c]
        norm = normalize_field(img)
        tag = f"{w}__GT" + (f"__crop{args.crop}" if args.crop else "__full")
        save_overlay(norm, lab, out_dir / f"{tag}.png")

        rows = measure_instances(lab, args.pixel_um)
        if not rows:
            print(f"{w:<24}{0:>5}   (no instances in this crop)")
            continue
        L = np.array([r["length_um"] for r in rows])
        W = np.array([r["width_median_um"] for r in rows])
        A = np.array([r["aspect_ratio"] for r in rows
                      if r["aspect_ratio"] is not None])
        print(f"{w:<24}{len(rows):>5}{np.median(L):>9.1f}"
              f"{np.percentile(L, 90):>9.1f}{np.median(W):>9.2f}"
              f"{np.median(A):>8.1f}")

    print(f"\n-> {out_dir.relative_to(ROOT)}")
    print("compare against the zero-shot overlays rendered the same way; the "
          "aspect ratio column is the one that separates a fibre from a blob.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
