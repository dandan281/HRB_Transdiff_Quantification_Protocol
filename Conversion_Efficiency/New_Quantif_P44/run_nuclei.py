"""Cellpose-SAM nuclei masks for every PLATE 44 well, at ONE plate-global
cellprob_threshold chosen by `pilot_cellprob.py`.

DAPI is **channel 0** on this plate, not channel 2 as on Q_Plates/PLATE_2x. The
channel index comes from `p44_layout`, never from a literal here -- reading ch2
would segment the AF546 receptor channel and produce nuclei counts that look
entirely reasonable and are entirely wrong.

Segmentation runs at this plate's native 1.72 um/px. `pilot_cellprob.py` records
the cross-check against resampling to 0.65 um/px.

Resumable: an existing `<well>_masks.npy` is reused.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/run_nuclei.py --cellprob 0
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import nd2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    AMAX_UM2, AMIN_UM2, DAPI_CH, UM, UM2, nd2_path, well_id, wells)

OUT = os.path.join(HERE, "nuclei")


def stretch(a, p=99.5):
    lo, hi = np.percentile(a, 1), np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def summarise(masks: np.ndarray) -> dict:
    n = int(masks.max())
    if n == 0:
        return {"nuclei_raw": 0, "nuclei_valid": 0, "median_area_um2": 0.0}
    area = np.bincount(masks.ravel(), minlength=n + 1)[1:] * UM2
    keep = (area >= AMIN_UM2) & (area <= AMAX_UM2)
    return {"nuclei_raw": n, "nuclei_valid": int(keep.sum()),
            "median_area_um2": round(float(np.median(area)), 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cellprob", type=float, required=True,
                    help="plate-global operating point from pilot_cellprob.py")
    ap.add_argument("--overlays", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import torch
    from cellpose import models
    gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=gpu)
    ws = wells()
    print(f"Cellpose-SAM  gpu={gpu}  cellprob={a.cellprob:+.1f}  "
          f"DAPI=ch{DAPI_CH}  {UM} um/px  {len(ws)} wells\n", flush=True)

    recs = []
    for i, stem in enumerate(ws, 1):
        mf = os.path.join(OUT, f"{stem}_masks.npy")
        if os.path.exists(mf):
            rec = {"well": stem, "well_id": well_id(stem), "cached": True,
                   **summarise(np.load(mf))}
            recs.append(rec)
            print(f"[{i:>2}/{len(ws)}] cached {stem:<10} "
                  f"valid={rec['nuclei_valid']:>6,}", flush=True)
            continue
        with nd2.ND2File(nd2_path(stem)) as x:
            dapi = x.asarray()[DAPI_CH].astype(np.float32)
        t0 = time.time()
        masks, _, _ = model.eval(dapi, cellprob_threshold=a.cellprob)
        secs = time.time() - t0
        masks = masks.astype(np.int32)
        np.save(mf, masks)
        rec = {"well": stem, "well_id": well_id(stem), "cached": False,
               "seconds": round(secs, 1), **summarise(masks)}
        recs.append(rec)
        print(f"[{i:>2}/{len(ws)}] {stem:<10} raw={rec['nuclei_raw']:>6,}  "
              f"valid={rec['nuclei_valid']:>6,}  "
              f"med={rec['median_area_um2']:6.1f}um2  {secs:5.1f}s", flush=True)
        if a.overlays:
            from PIL import Image
            from skimage.color import label2rgb
            over = label2rgb(masks, image=stretch(dapi), bg_label=0, alpha=0.5,
                             image_alpha=1)
            im = Image.fromarray((np.clip(over, 0, 1) * 255).astype(np.uint8))
            im.thumbnail((1600, 1600))
            im.save(os.path.join(OUT, f"{stem}_labeled.png"))

    with open(os.path.join(OUT, "nuclei_results.json"), "w") as fh:
        json.dump({"plate": "PLATE_44", "cellprob_threshold": a.cellprob,
                   "dapi_channel": DAPI_CH, "pixel_um": UM,
                   "nucleus_area_um2": [AMIN_UM2, AMAX_UM2],
                   "gpu": gpu, "per_well": recs}, fh, indent=2)
    raw = sum(r["nuclei_raw"] for r in recs)
    val = sum(r["nuclei_valid"] for r in recs)
    print(f"\nPLATE 44 total: {raw:,} raw -> {val:,} valid nuclei over "
          f"{len(recs)} wells (mean {val/len(recs):,.0f} valid/well)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
