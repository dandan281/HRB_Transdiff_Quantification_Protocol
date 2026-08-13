"""Fusion index for ONE well: of the Cellpose nuclei, how many lie INSIDE the
Desmin+ myotubes.

- nuclei  = per-nucleus Cellpose masks (already saved by count_well.py)
- myotubes= ridge detector on the Desmin channel (myotube_detect.detect_myotubes),
  then hole-filled so a nucleus sitting in a fiber's cytoplasmic void still counts
  as enclosed.
- a nucleus is INSIDE if >= --frac of its pixels fall in the myotube territory.

Writes a 3-class overlay (myotube fill | nucleus-inside | nucleus-outside) and
appends one line to fusion_results.jsonl.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import nd2
from scipy.ndimage import binary_fill_holes
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from myotube_detect import detect_myotubes


def stretch(a, p=99.5):
    lo, hi = np.percentile(a, 1), np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nd2", required=True)
    ap.add_argument("--masks", required=True)          # {stem}_masks.npy
    ap.add_argument("--desmin-ch", type=int, default=1)
    ap.add_argument("--nuclei-ch", type=int, default=2)
    ap.add_argument("--frac", type=float, default=0.5) # overlap to call "inside"
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.nd2))[0]

    with nd2.ND2File(a.nd2) as x:
        arr = x.asarray()
        desmin = arr[a.desmin_ch].astype(np.float32)
        dapi = arr[a.nuclei_ch].astype(np.float32)
    nuc = np.load(a.masks)                              # int label image

    myo, dbg = detect_myotubes(desmin)                 # thin/dim-fiber ridge mask
    myo_terr = binary_fill_holes(myo)                  # fill nucleus-sized voids

    # per-nucleus overlap fraction with the myotube territory
    n = int(nuc.max())
    flat = nuc.ravel()
    area = np.bincount(flat, minlength=n + 1).astype(np.float64)
    inside_px = np.bincount(flat, weights=myo_terr.ravel().astype(np.float64),
                            minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(area > 0, inside_px / area, 0.0)
    is_inside = frac >= a.frac
    is_inside[0] = False                               # background label
    labels_inside = np.where(is_inside)[0]
    n_inside = int(labels_inside.size)
    fusion = 100.0 * n_inside / n if n else 0.0

    # ---- visualization: 3-class overlay ----
    raw = stretch(dapi)
    des = stretch(desmin)
    rgb = np.zeros((*nuc.shape, 3), np.float32)
    rgb[..., 1] = np.maximum(0.35 * des, 0.15 * myo_terr)   # myotube territory = green
    inside_pix = is_inside[nuc]                        # bool image of inside nuclei
    outside_pix = (nuc > 0) & ~inside_pix
    rgb[inside_pix] = [1.0, 0.1, 0.9]                  # inside  = magenta
    rgb[outside_pix, 2] = 1.0                          # outside = blue
    rgb[outside_pix, 0] = 0.1
    rgb = np.clip(rgb, 0, 1)
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    im.thumbnail((1800, 1800))
    im.save(os.path.join(a.outdir, f"{stem}_fusion.png"))

    rec = {"well": stem, "nuclei_total": n, "nuclei_inside": n_inside,
           "nuclei_outside": n - n_inside,
           "fusion_index_pct": round(fusion, 2),
           "myotube_coverage_pct": round(dbg["coverage_pct"], 2),
           "frac_thresh": a.frac}
    with open(os.path.join(a.outdir, "fusion_results.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"FUSION_DONE {stem}  inside={n_inside}/{n}  = {fusion:.1f}%  "
          f"(myo cov {dbg['coverage_pct']:.1f}%)")


if __name__ == "__main__":
    main()
