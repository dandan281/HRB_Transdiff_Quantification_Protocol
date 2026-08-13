"""v3b -- threshold calibration for per-cell Desmin positivity.

The pooled per-cell Desmin distribution is unimodal with a right shoulder, NOT
bimodal, so no threshold is self-determining: the absolute conversion % is set by
where we cut. Two consequences, both handled here:

  1. CALIBRATE. The control's expected conversion (~20%) is external knowledge.
     Solve for the threshold that puts B02 at that value, then read every other
     well off the same threshold. Absolute levels become anchored rather than
     arbitrary; the treated wells are then a genuine prediction, not a fit.
  2. SHOW THE SENSITIVITY. Sweep the threshold and report conversion + fold for
     every well, so it is explicit which conclusions survive the choice and which
     do not. Fold-changes have been stable where absolute levels were not.

Caches per-cell ring intensities to percell_values.npz so further sweeps are free.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe NEW_Quantif/percell_calibrate.py [--ctrl-target 20]
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from percell_desmin import ring_intensity, CACHE, NUC_DIR, AMIN, AMAX, CTRL  # noqa
from real_fusion import UM, UM2  # noqa: E402

VALS = os.path.join(HERE, "percell_values.npz")


def load_values(ring_px):
    if os.path.exists(VALS):
        z = np.load(VALS)
        return {k: z[k] for k in z.files}
    out = {}
    wells = sorted(f.replace("_dbs.npy", "") for f in os.listdir(CACHE)
                   if f.endswith("_dbs.npy"))
    for w in wells:
        dbs = np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
        nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
        area = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2
        valid = (area >= AMIN) & (area <= AMAX); valid[0] = False
        mean, cnt = ring_intensity(nuc, dbs, ring_px)
        out[w] = mean[valid[:mean.size] & (cnt > 0)]
        print(f"  measured {w}  n={out[w].size:,}")
    np.savez_compressed(VALS, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl-target", type=float, default=20.0)
    ap.add_argument("--ring-um", type=float, default=3.0)
    a = ap.parse_args()
    V = load_values(max(1, int(round(a.ring_um / UM))))
    wells = sorted(V)

    # ---- 1. calibrate on the control ----
    thr = float(np.percentile(V[CTRL], 100 - a.ctrl_target))
    print(f"\nthreshold putting {CTRL} at {a.ctrl_target:.0f}% = "
          f"{thr:.1f} raw units\n")
    hdr = f"{'well':<24}{'cells':>8}{'Desmin+':>9}{'conv eff':>10}{'fold':>7}"
    print(hdr); print("-" * len(hdr))
    res = {}
    base = a.ctrl_target
    for w in wells:
        eff = 100 * float((V[w] > thr).mean())
        res[w] = {"cells": int(V[w].size), "desmin_pos": int((V[w] > thr).sum()),
                  "conversion_pct": round(eff, 2),
                  "fold_vs_ctrl": round(eff / base, 2)}
        print(f"{w:<24}{res[w]['cells']:>8,}{res[w]['desmin_pos']:>9,}"
              f"{eff:>9.1f}%{eff/base:>6.2f}x")

    # ---- 2. sensitivity: sweep the threshold ----
    targets = [10, 15, 20, 25, 30]        # control anchored at each of these
    print(f"\n{'ctrl=':<8}" + "".join(f"{w.split('_',1)[1][:12]:>14}" for w in wells
                                      if w != CTRL))
    print("-" * (8 + 14 * (len(wells) - 1)))
    sweep = {}
    for t in targets:
        th = float(np.percentile(V[CTRL], 100 - t))
        row = {}
        for w in wells:
            if w == CTRL:
                continue
            e = 100 * float((V[w] > th).mean())
            row[w] = {"conversion_pct": round(e, 2), "fold": round(e / t, 2)}
        sweep[t] = {"threshold": th, "wells": row}
        print(f"{t:>5}%  " + "".join(
            f"{row[w]['conversion_pct']:>7.1f}%{row[w]['fold']:>6.2f}x"
            for w in wells if w != CTRL))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for w in wells:
        if w == CTRL:
            continue
        ax.plot(targets, [sweep[t]["wells"][w]["fold"] for t in targets],
                "-o", label=w.split("_", 1)[1])
    ax.axhspan(2, 3, color="#22c55e", alpha=0.12, label="expected 2-3x")
    ax.set_xlabel("control conversion anchored at (%)")
    ax.set_ylabel("fold-change vs control")
    ax.set_title("Fold-change stability vs threshold calibration")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(HERE, "percell_calibration.png"), dpi=120,
                bbox_inches="tight", facecolor="white")

    with open(os.path.join(HERE, "percell_calibrate.json"), "w") as fh:
        json.dump({"ctrl_target_pct": a.ctrl_target, "threshold_raw": thr,
                   "calibrated": res,
                   "sweep": {str(k): v for k, v in sweep.items()}}, fh, indent=2)
    print("\n-> NEW_Quantif/percell_calibration.png / percell_calibrate.json")


if __name__ == "__main__":
    main()
