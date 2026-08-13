"""CONVERSION EFFICIENCY v2 -- absolute-threshold Desmin, no per-image normalisation.

Why: the production detector (myotube_detect.py) normalises per image four times
(percentile rescale -> CLAHE -> percentile hysteresis -> percentile intensity
gate). The gate at the 90th percentile pins Desmin coverage near 11% in EVERY
well by construction, which erases between-well fold-change. Measured: the p90
gate is 408 raw units in the control but 1242 in B03 -- the wells with the most
Desmin get the harshest threshold.

v2 keeps RAW camera units end to end:
  raw Desmin -> white_tophat (slow-background removal, still raw units)
             -> threshold at k * SHARED background sigma (same absolute value
                for every well)
  nucleus is CONVERTED if >= frac of its pixels overlap that Desmin mask.

No fibre-length gate: conversion efficiency counts converted cells including
MONONUCLEATED Desmin+ cells, which the fusion-index fibre gate discards.
Both overlap fractions (25% / 50%) reported per convention.

The expensive top-hat is cached to NEW_Quantif/dbs_cache/ as uint16, so
subsequent k sweeps are cheap.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe NEW_Quantif/conversion_v2.py
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
import nd2
from skimage.morphology import (white_tophat, disk, remove_small_objects,
                                remove_small_holes)
from scipy.ndimage import binary_fill_holes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from real_fusion import nuclei_inside, UM2  # noqa: E402

ND2_DIR = "../Q_PLATES/Q_Plates/PLATE_23"
NUC_DIR = "plate23_nuclei"
CACHE = os.path.join(HERE, "dbs_cache")
DESMIN_CH, TOPHAT_R, MIN_OBJ = 1, 40, 180
KS = [3, 5, 8, 12]
FRACS = [0.25, 0.5]
AMIN, AMAX = 50.0, 500.0
CTRL = "23_B02_ctrl"


def robust_sigma(a):
    med = np.median(a)
    return 1.4826 * np.median(np.abs(a - med)) + 1e-12


def get_dbs(stem, path):
    """Background-subtracted Desmin in RAW units, cached (uint16)."""
    os.makedirs(CACHE, exist_ok=True)
    cf = os.path.join(CACHE, f"{stem}_dbs.npy")
    if os.path.exists(cf):
        return np.load(cf).astype(np.float32)
    with nd2.ND2File(path) as x:
        raw = x.asarray()[DESMIN_CH].astype(np.float32)
    d = white_tophat(raw, disk(TOPHAT_R))
    np.save(cf, np.clip(d, 0, 65535).astype(np.uint16))
    return d


def main():
    files = sorted(glob.glob(os.path.join(ND2_DIR, "*.nd2")))
    dbs, sig = {}, {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        dbs[stem] = get_dbs(stem, f)
        sig[stem] = float(robust_sigma(dbs[stem]))
        print(f"  prepared {stem}  sigma={sig[stem]:.1f}")
    sig_ref = float(np.median(list(sig.values())))       # ONE shared scale
    print(f"\nshared background sigma = {sig_ref:.2f}\n")

    wells = sorted(dbs)
    res = {}
    for k in KS:
        thr = k * sig_ref
        print(f"=== k={k}  (absolute threshold {thr:.0f} raw units) ===")
        hdr = (f"{'well':<24}{'desmin%':>9}{'nuclei':>8}"
               f"{'conv25':>8}{'eff25':>8}{'conv50':>8}{'eff50':>8}")
        print(hdr); print("-" * len(hdr))
        res[k] = {}
        for w in wells:
            m = dbs[w] > thr
            m = remove_small_objects(m, min_size=MIN_OBJ)
            m = remove_small_holes(m, area_threshold=MIN_OBJ)
            m = binary_fill_holes(m)          # Desmin voids where nuclei sit
            nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
            e = {"desmin_pct": round(100 * float(m.mean()), 2)}
            for fr in FRACS:
                is_in, ntot, _ = nuclei_inside(nuc, m, AMIN, AMAX, frac=fr)
                nin = int(is_in.sum())
                e[f"overlap_{int(100*fr)}pct"] = {
                    "converted": nin, "total": ntot,
                    "efficiency_pct": round(100 * nin / ntot, 2) if ntot else 0.0}
            res[k][w] = e
            a, b = e["overlap_25pct"], e["overlap_50pct"]
            print(f"{w:<24}{e['desmin_pct']:>8.2f}%{a['total']:>8,}"
                  f"{a['converted']:>8,}{a['efficiency_pct']:>7.1f}%"
                  f"{b['converted']:>8,}{b['efficiency_pct']:>7.1f}%")
        c = res[k][CTRL]
        for fr in FRACS:
            key = f"overlap_{int(100*fr)}pct"
            base = c[key]["efficiency_pct"]
            folds = "  ".join(
                f"{w.split('_',1)[1]}={res[k][w][key]['efficiency_pct']/base:.2f}x"
                for w in wells if w != CTRL)
            print(f"  fold vs ctrl @{int(100*fr)}% overlap (ctrl={base:.1f}%): {folds}")
        print()

    with open(os.path.join(HERE, "conversion_v2.json"), "w") as fh:
        json.dump({"shared_sigma": sig_ref, "ks": KS, "tophat_radius": TOPHAT_R,
                   "nucleus_area_um2": [AMIN, AMAX], "per_well_sigma": sig,
                   "results": {str(k): v for k, v in res.items()}}, fh, indent=2)
    print("-> NEW_Quantif/conversion_v2.json")


if __name__ == "__main__":
    main()
