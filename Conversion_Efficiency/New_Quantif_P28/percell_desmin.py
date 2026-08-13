"""CONVERSION EFFICIENCY v3 -- per-CELL Desmin positivity. PLATE_28.
Identical code/parameters to New_Quantif_P26/percell_desmin.py; only the nuclei
directory differs (plate28_nuclei).

For every nucleus, take a CYTOPLASMIC RING around it and record the mean
background-subtracted Desmin there. One SHARED threshold (Otsu on the pooled
distribution) splits positive from negative across all wells -- no per-image
normalisation, no per-well tuning, no calibration to any expected value.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P28/percell_desmin.py [--ring-um 10]
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
from skimage.segmentation import expand_labels
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from real_fusion import UM, UM2  # noqa: E402

CACHE = os.path.join(HERE, "dbs_cache")
NUC_DIR = "plate28_nuclei"
AMIN, AMAX = 50.0, 500.0
CTRL = "23_B02_ctrl"


def ring_intensity(nuc, dbs, ring_px):
    """Mean bg-subtracted Desmin in a cytoplasmic ring around each nucleus."""
    grown = expand_labels(nuc, distance=ring_px)
    ring = (grown > 0) & (nuc == 0)                  # ring only, excludes nucleus
    lab = grown[ring].ravel()
    val = dbs[ring].ravel().astype(np.float64)
    n = int(nuc.max())
    cnt = np.bincount(lab, minlength=n + 1).astype(np.float64)
    tot = np.bincount(lab, weights=val, minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    return mean, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring-um", type=float, default=3.0)
    a = ap.parse_args()
    ring_px = max(1, int(round(a.ring_um / UM)))
    print(f"cytoplasmic ring = {a.ring_um} um = {ring_px} px\n")

    wells = sorted(f.replace("_dbs.npy", "") for f in os.listdir(CACHE)
                   if f.endswith("_dbs.npy"))
    per_well = {}
    for w in wells:
        dbs = np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
        nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
        area_um2 = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2
        valid = (area_um2 >= AMIN) & (area_um2 <= AMAX)
        valid[0] = False
        mean, cnt = ring_intensity(nuc, dbs, ring_px)
        keep = valid[:mean.size] & (cnt > 0)
        per_well[w] = mean[keep]
        print(f"  {w:<24} {keep.sum():>6,} cells   "
              f"ring median={np.median(mean[keep]):8.1f}  "
              f"p90={np.percentile(mean[keep], 90):8.1f}")

    pooled = np.concatenate([per_well[w] for w in wells])
    logp = np.log10(np.maximum(pooled, 1.0))
    thr_log = float(threshold_otsu(logp))            # ONE shared threshold
    thr = float(10 ** thr_log)
    print(f"\npooled cells = {pooled.size:,}")
    print(f"shared Otsu threshold (log10) = {thr_log:.3f} -> {thr:.1f} raw units\n")

    hdr = f"{'well':<24}{'cells':>8}{'Desmin+':>9}{'conv eff':>10}{'fold':>7}"
    print(hdr); print("-" * len(hdr))
    res, effs = {}, {}
    for w in wells:
        v = per_well[w]
        npos = int((v > thr).sum())
        eff = 100 * npos / v.size
        effs[w] = eff
        res[w] = {"cells": int(v.size), "desmin_pos": npos,
                  "conversion_pct": round(eff, 2),
                  "ring_median": round(float(np.median(v)), 1)}
    for w in wells:
        fold = effs[w] / effs[CTRL]
        res[w]["fold_vs_ctrl"] = round(fold, 2)
        print(f"{w:<24}{res[w]['cells']:>8,}{res[w]['desmin_pos']:>9,}"
              f"{res[w]['conversion_pct']:>9.1f}%{fold:>6.2f}x")

    # ---- distribution figure: is the split bimodal?
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    ax[0].hist(logp, bins=200, color="#64748b")
    ax[0].axvline(thr_log, color="#ef4444", lw=2,
                  label=f"Otsu = {thr:.0f} raw units")
    ax[0].set_xlabel("log10 mean ring Desmin"); ax[0].set_ylabel("cells")
    ax[0].set_title(f"Pooled per-cell Desmin (n={pooled.size:,})")
    ax[0].legend()
    for w in wells:
        v = np.log10(np.maximum(per_well[w], 1.0))
        ax[1].hist(v, bins=120, histtype="step", lw=1.8,
                   density=True, label=w.split("_", 1)[1])
    ax[1].axvline(thr_log, color="#ef4444", lw=2)
    ax[1].set_xlabel("log10 mean ring Desmin"); ax[1].set_ylabel("density")
    ax[1].set_title("Per-well distributions (shared threshold)")
    ax[1].legend(fontsize=8)
    fig.suptitle("PLATE_28 per-cell Desmin positivity — conversion efficiency v3",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "percell_desmin.png"), dpi=120,
                bbox_inches="tight", facecolor="white")

    with open(os.path.join(HERE, "percell_desmin.json"), "w") as fh:
        json.dump({"ring_um": a.ring_um, "ring_px": ring_px,
                   "otsu_threshold_raw": thr, "nucleus_area_um2": [AMIN, AMAX],
                   "per_well": res}, fh, indent=2)
    print("\n-> New_Quantif_P28/percell_desmin.png / .json")


if __name__ == "__main__":
    main()
