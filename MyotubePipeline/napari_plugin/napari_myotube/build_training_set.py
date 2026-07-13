"""Turn organized (image + ImageJ ROI) folders into a Cellpose training set.

You said you have quantified Plates 23 / 26 / 28 / 32 with ROIs + results. Organize them once
into a source folder, one subfolder per well, each holding the fiber-channel image and the
ground-truth ROI .zip:

    <src>/
      23_B02_ctrl/        image.tif   rois.zip
      26_C08_.../          image.tif   rois.zip
      ...

Then:

    python build_training_set.py --src <src> --out training --width 15

writes Cellpose pairs `training/<well>.tif` + `training/<well>_masks.tif`. Point Cellpose at
`training/` (see train_cellpose.py).

IMPORTANT — label quality is everything: the ROIs you feed in are the ground truth the model
copies. Use your CORRECTED tracings, not an over-segmented automated set (e.g. for 32_C08 the
true set is 246 fibers, not the 157-ROI "complete" auto output). `image.tif` must be the SAME
fiber channel + preprocessing the model will see at inference, at the ROI's full resolution.

Single-pair mode:
    python build_training_set.py --image img.tif --roi rois.zip --out training --stem well01
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _roi_io import read_imagej_zip, export_cellpose_pair   # noqa: E402

IMAGE_NAMES = ("image.tif", "image.tiff", "img.tif", "fiber.tif")
ROI_NAMES = ("rois.zip", "roi.zip", "RoiSet.zip")


def _first_existing(folder, names):
    for n in names:
        p = os.path.join(folder, n)
        if os.path.exists(p):
            return p
    # fall back to a single glob hit
    import glob
    for pat in ("*.zip",) if names is ROI_NAMES else ("*.tif", "*.tiff"):
        hits = sorted(glob.glob(os.path.join(folder, pat)))
        if len(hits) == 1:
            return hits[0]
    return None


def one_pair(image_path, roi_path, out_dir, stem, width) -> int:
    img = tifffile.imread(image_path)
    if img.ndim == 3 and img.shape[0] <= 4:        # (C, H, W) -> take max projection as fiber view
        img = img.max(axis=0)
    polys = read_imagej_zip(roi_path)
    n = export_cellpose_pair(img, polys, out_dir, stem, fiber_width_px=width)
    print(f"  {stem:32s} img{tuple(np.asarray(img).shape)}  rois={len(polys):4d}  labels={n}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="source folder with one subfolder per well")
    ap.add_argument("--image", help="single-pair mode: fiber-channel image")
    ap.add_argument("--roi", help="single-pair mode: ImageJ ROI .zip")
    ap.add_argument("--stem", help="single-pair mode: output name")
    ap.add_argument("--out", default="training", help="output training folder")
    ap.add_argument("--width", type=float, default=15.0,
                    help="fiber width (px) the centerline is dilated to for the mask")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.image and a.roi:
        stem = a.stem or os.path.splitext(os.path.basename(a.image))[0]
        one_pair(a.image, a.roi, a.out, stem, a.width)
        return
    if not a.src:
        ap.error("give --src <folder> (batch) or --image/--roi (single pair)")

    wells = sorted(d for d in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, d)))
    print(f"building training set from {len(wells)} wells -> {a.out}  (width={a.width}px)")
    total = ok = 0
    for w in wells:
        folder = os.path.join(a.src, w)
        img = _first_existing(folder, IMAGE_NAMES)
        roi = _first_existing(folder, ROI_NAMES)
        if not img or not roi:
            print(f"  {w:32s} SKIP (missing {'image' if not img else 'roi'})")
            continue
        try:
            total += one_pair(img, roi, a.out, w, a.width)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  {w:32s} ERROR {type(exc).__name__}: {exc}")
    print(f"done: {ok}/{len(wells)} wells, {total} labelled fibers -> {a.out}")


if __name__ == "__main__":
    main()
