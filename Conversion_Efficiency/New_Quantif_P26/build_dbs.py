"""Build the background-subtracted Desmin cache for PLATE_26 (raw camera units).

Identical preprocessing to conversion_v2.get_dbs on PLATE_23: white top-hat
(disk r=40) on the raw Desmin channel, kept in RAW units, cached as uint16. This
is the per-image-normalisation-free input every downstream step uses.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P26/build_dbs.py
"""
from __future__ import annotations
import os, glob
import numpy as np
import nd2
from skimage.morphology import white_tophat, disk

HERE = os.path.dirname(os.path.abspath(__file__))
ND2_DIR = "../Q_PLATES/Q_Plates/PLATE_26"
CACHE = os.path.join(HERE, "dbs_cache")
DESMIN_CH, TOPHAT_R = 1, 40


def main():
    os.makedirs(CACHE, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(ND2_DIR, "*.nd2"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        cf = os.path.join(CACHE, f"{stem}_dbs.npy")
        if os.path.exists(cf):
            print(f"  cached {stem}")
            continue
        with nd2.ND2File(f) as x:
            raw = x.asarray()[DESMIN_CH].astype(np.float32)
        d = white_tophat(raw, disk(TOPHAT_R))          # RAW units, no rescale
        np.save(cf, np.clip(d, 0, 65535).astype(np.uint16))
        print(f"  built  {stem}  (median dbs={np.median(d):.1f})")
    print("dbs_cache ready")


if __name__ == "__main__":
    main()
