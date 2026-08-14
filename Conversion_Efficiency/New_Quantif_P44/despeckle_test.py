"""PLATE 44 troubleshooting, step 2: does removing Desmin debris fix the readout?

`diagnose_threshold.py` measured the problem: with no valley in the per-cell
distribution the cut lands inside the background's right flank, B11 (no Desmin)
still scores 4.5 % positive, and in CONTROL wells only ~47 % of above-threshold
pixels belong to fibre-like objects -- the rest are blobs and sub-resolution
specks. A nucleus that happens to sit beside a speck gets a high ring mean.

This script tests the obvious fix: suppress small above-threshold components
before measuring the ring, then re-derive the pooled threshold and recompute
everything. It writes NOTHING over the main result -- it is an experiment whose
output is a comparison table.

Parameters are swept, not picked, and the sweep is expressed in um^2 so it is
plate-independent. 50 um^2 is the area floor the pipeline already applies to
nuclei, so it is a pre-existing constant rather than a new tuned knob.

Judged on three criteria that cannot be gamed by moving the threshold:
  * the false-positive floor measured on the B11 null well -- must fall;
  * whether a VALLEY appears in the pooled distribution -- the real goal;
  * whether replicate wells agree better (mean within-condition SD).

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/despeckle_test.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    AMAX_UM2, AMIN_UM2, CONDITION_ORDER, CONTROL_CONDITION, RING_PX,
    TECHNICAL_FAILURES, UM, UM2, condition_of, well_id, wells)
from percell_desmin import ring_intensity  # noqa: E402

CACHE = os.path.join(HERE, "dbs_cache")
NUC_DIR = os.path.join(HERE, "nuclei")
NULL_WELL = "14_B11"
MIN_AREA_UM2 = [0.0, 25.0, 50.0, 100.0, 200.0]     # 0 = current behaviour


def ring_values(dbs, nuc):
    area = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2
    valid = (area >= AMIN_UM2) & (area <= AMAX_UM2)
    valid[0] = False
    mean, cnt = ring_intensity(nuc, dbs, RING_PX)
    return mean[valid[:mean.size] & (cnt > 0)]


def despeckle(dbs, seed_thr, min_px):
    """Zero small above-seed components back to background, keep everything else.

    Only the *debris* is suppressed; the background distribution is preserved,
    so the pooled threshold is still derived from a comparable population.
    """
    if min_px <= 0:
        return dbs
    lab, n = ndi.label(dbs > seed_thr)
    if n == 0:
        return dbs
    sizes = np.bincount(lab.ravel())
    small = np.zeros(sizes.size, dtype=bool)
    small[1:] = sizes[1:] < min_px
    out = dbs.copy()
    bg = float(np.median(dbs[dbs > 0])) if (dbs > 0).any() else 0.0
    out[small[lab]] = bg
    return out


def has_valley(pooled, thr):
    lv = np.log10(np.maximum(pooled, 1.0))
    h, edges = np.histogram(lv, bins=200, range=(0.8, 3.8))
    centres = 0.5 * (edges[:-1] + edges[1:])
    sm = ndi.uniform_filter1d(h.astype(float), 7)
    pk = int(sm.argmax())
    right = sm[pk:]
    tr = [i for i in range(1, right.size - 1)
          if right[i] < right[i - 1] and right[i] < right[i + 1]]
    depth = None
    if tr:
        ti = pk + tr[0]
        depth = float(sm[ti] / sm.max())
        return True, float(10 ** centres[ti]), depth
    return False, None, None


def main() -> int:
    order = wells()
    base = json.load(open(os.path.join(HERE, "percell_desmin.json")))
    seed_thr = base["otsu_threshold_raw"]      # seed for finding debris
    print(f"seed threshold for component finding = {seed_thr:.1f} raw units\n")

    dbs_all = {w: np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
               for w in order}
    nuc_all = {w: np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
               for w in order}

    results = {}
    for min_um2 in MIN_AREA_UM2:
        min_px = int(round(min_um2 / UM2))
        vals = {}
        for w in order:
            d = despeckle(dbs_all[w], seed_thr, min_px)
            vals[w] = ring_values(d, nuc_all[w])
        pooled = np.concatenate([vals[w] for w in order])
        thr = float(10 ** threshold_otsu(np.log10(np.maximum(pooled, 1.0))))
        conv = {w: 100 * float((vals[w] > thr).mean()) for w in order}

        fp = conv[NULL_WELL]
        ctrl = [w for w in order if condition_of(w) == CONTROL_CONDITION]
        ctrl_mean = float(np.mean([conv[w] for w in ctrl]))
        sds = []
        for c in CONDITION_ORDER:
            v = [conv[w] for w in order if condition_of(w) == c
                 and well_id(w) not in TECHNICAL_FAILURES]
            if len(v) > 1:
                sds.append(float(np.std(v, ddof=1)))
        mean_sd = float(np.mean(sds))
        top = max((conv[w] for w in order
                   if well_id(w) not in TECHNICAL_FAILURES))
        valley, vpos, vdepth = has_valley(pooled, thr)

        results[min_um2] = {
            "min_px": min_px, "threshold_raw": round(thr, 1),
            "null_fp_pct": round(fp, 2), "control_mean_pct": round(ctrl_mean, 2),
            "control_minus_floor": round(ctrl_mean - fp, 2),
            "top_well_pct": round(top, 2),
            "top_over_control": round(top / ctrl_mean, 3) if ctrl_mean else None,
            "mean_within_condition_sd": round(mean_sd, 3),
            "valley": valley,
            "valley_raw": round(vpos, 1) if vpos else None,
            "valley_depth_frac_of_peak": round(vdepth, 3) if vdepth else None,
            "per_well_pct": {w: round(conv[w], 2) for w in order},
        }
        print(f"min area {min_um2:6.1f} um2 ({min_px:3d} px): thr={thr:6.1f}  "
              f"null_FP={fp:5.2f}%  control={ctrl_mean:5.2f}%  "
              f"ctrl-floor={ctrl_mean-fp:5.2f}pp  top={top:5.1f}%  "
              f"top/ctrl={top/ctrl_mean:.2f}x  within-cond SD={mean_sd:.2f}  "
              f"valley={'YES @%.0f' % vpos if valley else 'no'}")

    b, best = MIN_AREA_UM2[0], results[MIN_AREA_UM2[0]]
    print(f"\nbaseline (no despeckle): null floor {best['null_fp_pct']:.2f}%, "
          f"within-condition SD {best['mean_within_condition_sd']:.2f}")
    for m in MIN_AREA_UM2[1:]:
        r = results[m]
        print(f"  {m:5.1f} um2 -> floor {r['null_fp_pct']:5.2f}% "
              f"({r['null_fp_pct']-best['null_fp_pct']:+.2f}), "
              f"SD {r['mean_within_condition_sd']:.2f} "
              f"({r['mean_within_condition_sd']-best['mean_within_condition_sd']:+.2f}), "
              f"top/ctrl {r['top_over_control']:.2f}x "
              f"({r['top_over_control']-best['top_over_control']:+.2f})")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    xs = MIN_AREA_UM2
    ax[0].plot(xs, [results[m]["null_fp_pct"] for m in xs], "-o",
               color="#d03b3b", lw=2)
    ax[0].set_xlabel("min Desmin object area (um2)")
    ax[0].set_ylabel("false-positive floor (%)")
    ax[0].set_title("B11 null well: does despeckling\nremove the false floor?")
    ax[0].grid(alpha=0.3)

    ax[1].plot(xs, [results[m]["mean_within_condition_sd"] for m in xs], "-o",
               color="#2a78d6", lw=2)
    ax[1].set_xlabel("min Desmin object area (um2)")
    ax[1].set_ylabel("mean within-condition SD (pp)")
    ax[1].set_title("replicate agreement\n(lower = replicates agree better)")
    ax[1].grid(alpha=0.3)

    ax[2].plot(xs, [results[m]["control_mean_pct"] for m in xs], "-o",
               color="#52514e", lw=2, label="control")
    ax[2].plot(xs, [results[m]["top_well_pct"] for m in xs], "-o",
               color="#eb6834", lw=2, label="top well")
    ax[2].plot(xs, [results[m]["null_fp_pct"] for m in xs], "-o",
               color="#d03b3b", lw=2, label="B11 null")
    ax[2].set_xlabel("min Desmin object area (um2)")
    ax[2].set_ylabel("conversion (%)")
    ax[2].set_title("absolute levels")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.suptitle("PLATE 44 — does suppressing Desmin debris fix the readout?",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "despeckle_test.png"), dpi=130,
                bbox_inches="tight", facecolor="white")
    with open(os.path.join(HERE, "despeckle_test.json"), "w") as fh:
        json.dump({"plate": "PLATE_44", "seed_threshold_raw": seed_thr,
                   "null_well": NULL_WELL,
                   "sweep_min_area_um2": MIN_AREA_UM2,
                   "results": {str(k): v for k, v in results.items()}},
                  fh, indent=2)
    print("\n-> despeckle_test.png / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
