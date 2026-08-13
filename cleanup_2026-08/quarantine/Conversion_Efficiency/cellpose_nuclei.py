"""Zero-shot nucleus counting with Cellpose-SAM (the 'cpsam' model in cellpose 4.x).

Run with the dedicated env:
    cpenv/Scripts/python.exe cellpose_nuclei.py --nd2 <file> --nuclei-ch 2 \
        --outdir cp_out --crop 1400 1400 800

Compares Cellpose's instance count to the watershed count from the main
pipeline, and saves outline overlays so the over-splitting can be inspected.
Flow-field instance segmentation gives ONE mask per nucleus regardless of the
internal chromatin texture that makes intensity-watershed over-split.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import nd2
from PIL import Image


def flatfield(img, sigma_frac=0.09):
    from scipy.ndimage import gaussian_filter
    s = max(60, int(round(img.shape[0] * sigma_frac)))
    illum = gaussian_filter(img, s)
    illum = np.maximum(illum, np.percentile(illum, 20) * 0.5)
    return (img / illum * illum.mean()).astype(np.float32)


def stretch(a, p=99.5):
    lo, hi = np.percentile(a, 1), np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nd2", required=True)
    ap.add_argument("--nuclei-ch", type=int, default=2)
    ap.add_argument("--outdir", default="cp_out")
    ap.add_argument("--crop", type=int, nargs=3, default=None,
                    help="y0 x0 size (omit = full field)")
    ap.add_argument("--diameter", type=float, default=None,
                    help="nucleus diameter px (None = cpsam native)")
    ap.add_argument("--flatfield", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    with nd2.ND2File(a.nd2) as x:
        dapi = x.asarray()[a.nuclei_ch].astype(np.float32)
    if a.crop:
        y0, x0, s = a.crop
        dapi = dapi[y0:y0 + s, x0:x0 + s]
    if a.flatfield:
        dapi = flatfield(dapi)

    import torch
    from cellpose import models
    gpu = torch.cuda.is_available()
    print(f"GPU={gpu} {torch.cuda.get_device_name(0) if gpu else 'CPU'}  "
          f"image={dapi.shape}")
    model = models.CellposeModel(gpu=gpu)          # cpsam (Cellpose-SAM)
    import time
    t0 = time.time()
    masks, flows, styles = model.eval(dapi, diameter=a.diameter)
    if gpu:
        torch.cuda.synchronize()
    secs = time.time() - t0
    n = int(masks.max())
    print(f"Cellpose-SAM nuclei = {n}   |   inference = {secs:.1f} s")

    # outline overlay on stretched raw
    from cellpose import utils
    raw = stretch(dapi)
    rgb = np.stack([raw, raw, raw], -1)
    for outl in utils.outlines_list(masks):
        outl = outl.astype(int)
        rgb[outl[:, 1], outl[:, 0]] = [255, 60, 60]
    Image.fromarray(rgb).save(os.path.join(a.outdir, "cellpose_outlines.png"))
    np.save(os.path.join(a.outdir, "cellpose_masks.npy"), masks.astype(np.int32))
    with open(os.path.join(a.outdir, "cellpose_count.json"), "w") as fh:
        json.dump({"source": os.path.basename(a.nd2), "crop": a.crop,
                   "flatfield": a.flatfield, "diameter": a.diameter,
                   "nuclei_cellpose": n, "gpu": gpu,
                   "inference_seconds": round(secs, 1)}, fh, indent=2)
    print("saved ->", a.outdir)


if __name__ == "__main__":
    main()
