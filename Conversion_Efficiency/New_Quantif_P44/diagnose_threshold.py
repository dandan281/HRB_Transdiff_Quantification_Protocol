"""PLATE 44 troubleshooting: why do the conversion numbers look odd?

Five diagnostics, cheapest and most decisive first. Everything here is READ-ONLY
-- it changes no result, it explains one.

1. **Empirical null.** B11's Desmin channel is effectively empty (dbs p99 = 329
   vs 1,066-2,331 plate-wide). Its per-cell ring distribution is therefore a
   measured NEGATIVE CONTROL: whatever fraction of B11 cells clears the
   threshold is a false-positive floor that every other well also pays.
2. **Background drift.** Each well's ring distribution has a background mode.
   If that mode moves between wells while the threshold stays global, a fixed
   cut slices different amounts of background tail per well -- apparent
   conversion differences that are pure background.
3. **Threshold sensitivity.** With no valley in the distribution, the Otsu cut is
   placed on a slope. How much does conversion move per unit of threshold?
4. **Separability.** Is there any real valley, or is the "positive" population
   only a shoulder of the background?
5. **Artifact morphology** (the operator's hypothesis): of the pixels above
   threshold, how many belong to fibre-like objects vs small round debris?

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/diagnose_threshold.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import pearsonr
from skimage.measure import label as sklabel, regionprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    CONTROL_CONDITION, RING_UM, TECHNICAL_FAILURES, UM, condition_of, well_id,
    wells)

VALUES = os.path.join(HERE, f"percell_values_r{RING_UM}.npz")
CACHE = os.path.join(HERE, "dbs_cache")
NULL_WELL = "14_B11"
ART_WELLS = ["23_B02", "22_B03", "57_E09", "32_C08"]   # control x2 + two top wells


def mode_log10(v: np.ndarray) -> float:
    """Peak of the background distribution, via a fine histogram."""
    lv = np.log10(np.maximum(v, 1.0))
    h, edges = np.histogram(lv, bins=220, range=(0.5, 4.0))
    return float(0.5 * (edges[h.argmax()] + edges[h.argmax() + 1]))


def main() -> int:
    z = np.load(VALUES)
    order = wells()
    vals = {w: z[w] for w in order}
    pc = json.load(open(os.path.join(HERE, "percell_desmin.json")))
    thr = pc["otsu_threshold_raw"]
    print(f"threshold in use: {thr:.1f} raw units "
          f"(log10 {np.log10(thr):.3f})\n")

    # -- 1. empirical null ---------------------------------------------------
    nullv = vals[NULL_WELL]
    fp = float((nullv > thr).mean())
    print("1. EMPIRICAL NULL (B11, Desmin channel effectively empty)")
    print(f"   {(nullv > thr).sum():,}/{nullv.size:,} = {100*fp:.1f}% of its "
          f"cells are called Desmin+ with essentially no Desmin present.")
    print(f"   -> a false-positive floor of ~{100*fp:.1f} pp applies to EVERY "
          f"well.")
    ctrl_wells = [w for w in order if condition_of(w) == CONTROL_CONDITION]
    ctrl_pct = np.mean([100 * (vals[w] > thr).mean() for w in ctrl_wells])
    print(f"   control mean = {ctrl_pct:.1f}% -> ~{100*fp/ctrl_pct*100:.0f}% of "
          f"the control signal is floor, leaving ~{ctrl_pct-100*fp:.1f} pp real.\n")

    # -- 2. background drift -------------------------------------------------
    modes = np.array([mode_log10(vals[w]) for w in order])
    convs = np.array([100 * (vals[w] > thr).mean() for w in order])
    keep = np.array([well_id(w) not in TECHNICAL_FAILURES for w in order])
    r_bg, p_bg = pearsonr(modes[keep], convs[keep])
    print("2. BACKGROUND DRIFT")
    print(f"   per-well background mode spans {10**modes.min():.0f}-"
          f"{10**modes.max():.0f} raw units "
          f"({10**(modes.max()-modes.min()):.2f}x)")
    print(f"   Pearson r(background mode, conversion%) = {r_bg:+.3f} "
          f"(p={p_bg:.2g}, n={keep.sum()})")
    if abs(r_bg) > 0.6:
        print("   -> STRONG: the ranking substantially tracks background level, "
              "not Desmin.\n")
    elif abs(r_bg) > 0.35:
        print("   -> MODERATE: background contributes to the ranking.\n")
    else:
        print("   -> weak; background drift is not the main driver.\n")

    # -- 3. threshold sensitivity -------------------------------------------
    print("3. THRESHOLD SENSITIVITY (no valley -> the cut sits on a slope)")
    pooled = np.concatenate([vals[w] for w in order])
    for mult, lbl in ((0.8, "-20%"), (0.9, "-10%"), (1.0, "as used"),
                      (1.1, "+10%"), (1.25, "+25%")):
        t = thr * mult
        allpct = 100 * (pooled > t).mean()
        cm = np.mean([100 * (vals[w] > t).mean() for w in ctrl_wells])
        top = 100 * (vals["57_E09"] > t).mean()
        print(f"   thr x{mult:<5.2f} ({lbl:>7}) = {t:6.1f}: plate {allpct:5.1f}%"
              f"  control {cm:5.1f}%  E09 {top:5.1f}%  "
              f"fold {top/cm if cm else float('nan'):.2f}x")
    d = (100 * (pooled > thr * 1.1).mean() - 100 * (pooled > thr * 0.9).mean())
    print(f"   -> a +/-10% threshold move changes plate conversion by "
          f"{abs(d):.1f} pp (on {100*(pooled>thr).mean():.1f}%), i.e. "
          f"{abs(d)/(100*(pooled>thr).mean())*100:.0f}% relative.\n")

    # -- 4. separability -----------------------------------------------------
    lv = np.log10(np.maximum(pooled, 1.0))
    h, edges = np.histogram(lv, bins=200, range=(0.8, 3.8))
    centres = 0.5 * (edges[:-1] + edges[1:])
    sm = ndi.uniform_filter1d(h.astype(float), 7)
    peak_i = int(sm.argmax())
    right = sm[peak_i:]
    # a genuine valley requires the smoothed count to rise again after falling
    troughs = [i for i in range(1, right.size - 1)
               if right[i] < right[i - 1] and right[i] < right[i + 1]]
    print("4. SEPARABILITY")
    print(f"   background peak at {10**centres[peak_i]:.0f} raw units")
    if troughs:
        ti = peak_i + troughs[0]
        print(f"   valley found at {10**centres[ti]:.0f} raw units "
              f"(threshold in use {thr:.0f})")
    else:
        print("   NO VALLEY between background and the positive shoulder -- the "
              "'positive' population is a tail, not a separate mode.")
        frac_at = sm[np.argmin(np.abs(centres - np.log10(thr)))] / sm.max()
        print(f"   the threshold sits where the histogram is still "
              f"{100*frac_at:.0f}% of peak height, i.e. inside the background's "
              f"own right flank.\n")

    # -- 5. artifact morphology ---------------------------------------------
    print("5. ARTIFACT MORPHOLOGY (are above-threshold pixels fibres or debris?)")
    art = {}
    for w in ART_WELLS:
        dbs = np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
        sat = float((dbs >= 4095).mean())
        m = dbs > thr
        lab = sklabel(m)
        props = regionprops(lab)
        areas = np.array([p.area for p in props], dtype=float)
        big = [p for p in props if p.area >= 9]
        # elongation via the object's own second moments; robust for thin shapes
        elong = np.array([(p.axis_major_length / max(p.axis_minor_length, 1e-6))
                          for p in big]) if big else np.array([])
        px_fib = sum(p.area for p, e in zip(big, elong) if e >= 3.0)
        px_blob = sum(p.area for p, e in zip(big, elong) if e < 3.0)
        px_small = float(areas[areas < 9].sum())
        tot = max(px_fib + px_blob + px_small, 1)
        art[w] = {"saturated_frac": sat, "n_objects": len(props),
                  "px_fibrelike": px_fib, "px_blobby": px_blob,
                  "px_specks": px_small,
                  "pct_fibrelike": round(100 * px_fib / tot, 1),
                  "pct_blobby": round(100 * px_blob / tot, 1),
                  "pct_specks": round(100 * px_small / tot, 1)}
        print(f"   {w:<10} {condition_of(w):<12} above-thr px: "
              f"fibre-like {art[w]['pct_fibrelike']:5.1f}%  "
              f"blobby {art[w]['pct_blobby']:5.1f}%  "
              f"specks(<9px) {art[w]['pct_specks']:5.1f}%  "
              f"saturated {100*sat:.4f}%")

    # -- figure --------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    ax[0].hist(np.log10(np.maximum(nullv, 1.0)), bins=120, density=True,
               color="#d03b3b", alpha=0.65, label=f"B11 null (no Desmin)")
    for w in ctrl_wells:
        ax[0].hist(np.log10(np.maximum(vals[w], 1.0)), bins=120, density=True,
                   histtype="step", lw=1.4, color="#2a78d6",
                   label="control" if w == ctrl_wells[0] else None)
    ax[0].axvline(np.log10(thr), color="#0b0b0b", lw=2, ls="--",
                  label=f"threshold {thr:.0f}")
    ax[0].set_xlabel("log10 mean ring Desmin"); ax[0].set_ylabel("density")
    ax[0].set_title(f"1. empirical null: {100*fp:.1f}% of B11 cells clear the cut\n"
                    "with essentially no Desmin present")
    ax[0].legend(fontsize=8)

    ax[1].scatter(10 ** modes[keep], convs[keep], s=70, color="#2a78d6",
                  edgecolor="#0b0b0b", zorder=3)
    ax[1].scatter(10 ** modes[~keep], convs[~keep], s=70, color="#d03b3b",
                  marker="x", zorder=3, label="B11 (null)")
    m_, b_ = np.polyfit(10 ** modes[keep], convs[keep], 1)
    xs = np.array([(10 ** modes).min(), (10 ** modes).max()])
    ax[1].plot(xs, m_ * xs + b_, "--", color="#898781")
    ax[1].set_xlabel("per-well background mode (raw units)")
    ax[1].set_ylabel("conversion (%)")
    ax[1].set_title(f"2. background drift vs result\nPearson r = {r_bg:+.2f}")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)

    ts = np.linspace(thr * 0.6, thr * 1.8, 60)
    for w in ctrl_wells:
        ax[2].plot(ts, [100 * (vals[w] > t).mean() for t in ts], color="#2a78d6",
                   lw=1.3)
    ax[2].plot(ts, [100 * (vals["57_E09"] > t).mean() for t in ts],
               color="#eb6834", lw=2, label="E09 (top well)")
    ax[2].plot(ts, [100 * (nullv > t).mean() for t in ts], color="#d03b3b",
               lw=2, label="B11 (null)")
    ax[2].axvline(thr, color="#0b0b0b", ls="--", lw=1.6, label="threshold used")
    ax[2].set_xlabel("threshold (raw units)")
    ax[2].set_ylabel("conversion (%)")
    ax[2].set_title("3. threshold sensitivity\n(blue = the three control wells)")
    ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)

    fig.suptitle("PLATE 44 — why the conversion readout is unstable",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "diagnose_threshold.png"), dpi=130,
                bbox_inches="tight", facecolor="white")

    with open(os.path.join(HERE, "diagnose_threshold.json"), "w") as fh:
        json.dump({"plate": "PLATE_44", "threshold_raw": thr,
                   "null_well": NULL_WELL,
                   "false_positive_floor_pct": round(100 * fp, 2),
                   "control_mean_pct": round(float(ctrl_pct), 2),
                   "control_minus_floor_pct": round(float(ctrl_pct - 100 * fp), 2),
                   "background_mode_raw_min": round(float(10 ** modes.min()), 1),
                   "background_mode_raw_max": round(float(10 ** modes.max()), 1),
                   "r_background_vs_conversion": round(float(r_bg), 3),
                   "p_background_vs_conversion": float(p_bg),
                   "valley_found": bool(troughs),
                   "artifact_morphology": art}, fh, indent=2)
    print("\n-> diagnose_threshold.png / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
