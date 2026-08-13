"""CONVERSION EFFICIENCY v3 -- per-CELL Desmin positivity. PLATE 44.

Same method as PLATE_23/26/28/32: for each valid nucleus, take the mean
background-subtracted Desmin in a cytoplasmic ring around it, pool every cell on
the plate, and split positive from negative with ONE shared Otsu threshold on
the log10 distribution. No per-image normalisation, no per-well tuning, no
calibration against an expected answer.

Two things differ from the PLATE_2x scripts and both come from `p44_layout`:
the ring is 10 um = **6 px** here (15 px there), and areas use this plate's own
UM2 (2.974 um2/px, not 0.4225).

The threshold is in RAW camera units and is **not comparable to another plate's**
-- P44 is 12-bit and a different acquisition. Compare folds, never absolute
percentages, across plates.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/percell_desmin.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import expand_labels
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    AMAX_UM2, AMIN_UM2, CTRL, CTRL_IS_ASSUMED, RING_UM, UM, UM2, well_id)

CACHE = os.path.join(HERE, "dbs_cache")
NUC_DIR = os.path.join(HERE, "nuclei")


def ring_intensity(nuc: np.ndarray, dbs: np.ndarray, ring_px: int):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring-um", type=float, default=RING_UM,
                    help=f"cytoplasmic ring width (default {RING_UM}, the value "
                         "used on every other plate)")
    a = ap.parse_args()
    ring_px = max(1, int(round(a.ring_um / UM)))
    print(f"PLATE 44 | {UM} um/px | cytoplasmic ring = {a.ring_um} um = "
          f"{ring_px} px | nucleus {AMIN_UM2:.0f}-{AMAX_UM2:.0f} um2\n")

    wells = sorted((f.replace("_dbs.npy", "") for f in os.listdir(CACHE)
                    if f.endswith("_dbs.npy")),
                   key=lambda s: int(s.split("_")[0]))
    if not wells:
        raise SystemExit("no dbs cache; run build_dbs.py first")

    per_well: dict[str, np.ndarray] = {}
    for w in wells:
        nf = os.path.join(NUC_DIR, f"{w}_masks.npy")
        if not os.path.exists(nf):
            raise SystemExit(f"missing nuclei mask for {w}; run run_nuclei.py")
        dbs = np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
        nuc = np.load(nf)
        if dbs.shape != nuc.shape:
            raise SystemExit(f"{w}: dbs {dbs.shape} != nuclei {nuc.shape}")
        area_um2 = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2
        valid = (area_um2 >= AMIN_UM2) & (area_um2 <= AMAX_UM2)
        valid[0] = False
        mean, cnt = ring_intensity(nuc, dbs, ring_px)
        keep = valid[:mean.size] & (cnt > 0)
        per_well[w] = mean[keep]
        print(f"  {w:<10} {keep.sum():>7,} cells   "
              f"ring median={np.median(mean[keep]):8.1f}  "
              f"p90={np.percentile(mean[keep], 90):8.1f}")

    pooled = np.concatenate([per_well[w] for w in wells])
    logp = np.log10(np.maximum(pooled, 1.0))
    thr_log = float(threshold_otsu(logp))            # ONE shared threshold
    thr = float(10 ** thr_log)
    print(f"\npooled cells = {pooled.size:,}")
    print(f"shared Otsu threshold (log10) = {thr_log:.3f} -> {thr:.1f} raw units")
    print("  (raw camera units, 12-bit; NOT comparable to another plate)\n")

    res, effs = {}, {}
    for w in wells:
        v = per_well[w]
        npos = int((v > thr).sum())
        eff = 100 * npos / v.size if v.size else 0.0
        effs[w] = eff
        res[w] = {"well_id": well_id(w), "cells": int(v.size),
                  "desmin_pos": npos, "conversion_pct": round(eff, 2),
                  "ring_median": round(float(np.median(v)), 1)}

    ctrl_eff = effs.get(CTRL)
    hdr = f"{'well':<10}{'id':>5}{'cells':>9}{'Desmin+':>9}{'conv eff':>10}{'fold':>8}"
    print(hdr)
    print("-" * len(hdr))
    for w in sorted(wells, key=lambda s: effs[s]):
        fold = (effs[w] / ctrl_eff) if ctrl_eff else float("nan")
        res[w]["fold_vs_assumed_ctrl"] = round(fold, 2)
        mark = "  <- assumed control" if w == CTRL else ""
        print(f"{w:<10}{res[w]['well_id']:>5}{res[w]['cells']:>9,}"
              f"{res[w]['desmin_pos']:>9,}{res[w]['conversion_pct']:>9.1f}%"
              f"{fold:>7.2f}x{mark}")

    # ---- distribution figure: is the split bimodal? ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    ax[0].hist(logp, bins=200, color="#64748b")
    ax[0].axvline(thr_log, color="#ef4444", lw=2,
                  label=f"Otsu = {thr:.0f} raw units")
    ax[0].set_xlabel("log10 mean ring Desmin")
    ax[0].set_ylabel("cells")
    ax[0].set_title(f"Pooled per-cell Desmin (n={pooled.size:,})")
    ax[0].legend()
    for w in wells:
        v = np.log10(np.maximum(per_well[w], 1.0))
        ax[1].hist(v, bins=120, histtype="step", lw=0.9, density=True,
                   label=well_id(w))
    ax[1].axvline(thr_log, color="#ef4444", lw=2)
    ax[1].set_xlabel("log10 mean ring Desmin")
    ax[1].set_ylabel("density")
    ax[1].set_title("Per-well distributions (shared threshold)")
    ax[1].legend(fontsize=5, ncol=4)
    fig.suptitle("PLATE 44 per-cell Desmin positivity — conversion efficiency v3",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "percell_desmin.png"), dpi=120,
                bbox_inches="tight", facecolor="white")

    with open(os.path.join(HERE, "percell_desmin.json"), "w") as fh:
        json.dump({"plate": "PLATE_44", "pixel_um": UM,
                   "ring_um": a.ring_um, "ring_px": ring_px,
                   "otsu_threshold_raw": thr,
                   "otsu_threshold_log10": thr_log,
                   "threshold_units": "raw camera (12-bit); not cross-plate comparable",
                   "nucleus_area_um2": [AMIN_UM2, AMAX_UM2],
                   "pooled_cells": int(pooled.size),
                   "control_well": CTRL,
                   "control_is_assumed_by_position": CTRL_IS_ASSUMED,
                   "condition_labels": "absent -- no layout sheet provided for this plate",
                   "per_well": res}, fh, indent=2)
    print("\n-> New_Quantif_P44/percell_desmin.png / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
