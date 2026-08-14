"""PLATE 44 troubleshooting, step 4: is the staining correlation TECHNICAL or REAL?

Step 3 found conversion tracks Desmin signal strength at r = +0.917 within
condition. That has two possible causes and they demand opposite responses:

  TECHNICAL  a well stained brighter / was exposed longer. Every structure scales
             up, a fixed absolute threshold catches more, conversion rises with
             no extra biology. -> the readout is broken and must be normalised.

  REAL       a well genuinely converted more, so it has MORE myotube. Brightness
             per fibre is unchanged; the AREA covered by Desmin rises. -> the
             readout is fine and the correlation is the result, not an artifact.

These separate cleanly:

  * `p99`        peak brightness -- a multiplicative gain term. Scales with
                 exposure/staining, NOT with how much myotube exists.
  * `cov_norm`   fraction of the field above half of that well's OWN p99 --
                 how much myotube there is, measured in units of that well's own
                 brightness, so it is invariant to a gain change.

Regressing conversion on both, with condition means removed so biology is held
constant, asks which one actually carries the signal.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/diagnose_brightness_vs_area.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    CONDITION_ORDER, TECHNICAL_FAILURES, condition_of, well_id, wells)

CACHE = os.path.join(HERE, "dbs_cache")
REL = 0.5          # "myotube" = above half this well's own peak brightness


def partial_r(y, x, z):
    """Correlation of y and x with z regressed out of both."""
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    return pearsonr(rx, ry)


def main() -> int:
    pc = json.load(open(os.path.join(HERE, "percell_desmin.json")))["per_well"]
    order = [w for w in wells() if well_id(w) not in TECHNICAL_FAILURES]

    p99, cov_norm, cov_abs, conv, ids, conds = [], [], [], [], [], []
    for w in order:
        dbs = np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
        p = float(np.percentile(dbs, 99))
        p99.append(p)
        cov_norm.append(float((dbs > REL * p).mean()) * 100)
        cov_abs.append(float((dbs > 118.9).mean()) * 100)
        conv.append(pc[w]["conversion_pct"])
        ids.append(well_id(w))
        conds.append(condition_of(w))
    p99 = np.array(p99); cov_norm = np.array(cov_norm)
    cov_abs = np.array(cov_abs); conv = np.array(conv)

    # centre within condition -> biology removed, only technical variation left
    def centre(v):
        out = np.array(v, dtype=float)
        for c in set(conds):
            idx = [i for i, cc in enumerate(conds) if cc == c]
            if len(idx) > 1:
                out[idx] -= out[idx].mean()
            else:
                out[idx] = np.nan
        return out

    cp, cc_n, cc_a, cv = centre(p99), centre(cov_norm), centre(cov_abs), centre(conv)
    m = ~np.isnan(cp)
    cp, cc_n, cc_a, cv = cp[m], cc_n[m], cc_a[m], cv[m]

    print("PLATE 44 — brightness (technical gain) vs coverage (real myotube)\n"
          + "=" * 66)
    print(f"n = {cv.size} wells, condition means removed (biology held constant)\n")

    r_p, pp = pearsonr(cp, cv)
    r_cn, pcn = pearsonr(cc_n, cv)
    r_ca, pca = pearsonr(cc_a, cv)
    print(f"conversion vs peak BRIGHTNESS p99          r = {r_p:+.3f} (p={pp:.2g})")
    print(f"conversion vs ABSOLUTE coverage (>119)     r = {r_ca:+.3f} (p={pca:.2g})")
    print(f"conversion vs NORMALISED coverage (>0.5*p99) r = {r_cn:+.3f} "
          f"(p={pcn:.2g})   <- gain-invariant 'how much myotube'")

    r_bright_ctrl, p_bc = partial_r(cv, cp, cc_n)
    r_cov_ctrl, p_cc = partial_r(cv, cc_n, cp)
    print(f"\npartial correlations (each controlling for the other)")
    print(f"  brightness | coverage held  r = {r_bright_ctrl:+.3f} (p={p_bc:.2g})")
    print(f"  coverage   | brightness held r = {r_cov_ctrl:+.3f} (p={p_cc:.2g})")

    print(f"\nhow much does brightness itself vary between replicate wells?")
    for c in CONDITION_ORDER:
        idx = [i for i, cc in enumerate(conds) if cc == c]
        if len(idx) > 1:
            v = p99[idx]
            print(f"  {c:<14} p99 {', '.join(f'{x:.0f}' for x in v)}  "
                  f"({v.max()/v.min():.2f}x spread)")

    if abs(r_bright_ctrl) > abs(r_cov_ctrl) and p_bc < 0.05:
        verdict = ("TECHNICAL: with the amount of myotube held constant, wells "
                   "that merely stained brighter still score higher. The "
                   "absolute threshold is converting staining gain into "
                   "apparent conversion.")
    elif abs(r_cov_ctrl) > abs(r_bright_ctrl) and p_cc < 0.05:
        verdict = ("REAL: with brightness held constant, wells with more "
                   "myotube coverage score higher. The correlation with signal "
                   "strength is the biology, not an artifact.")
    else:
        verdict = ("AMBIGUOUS: brightness and coverage are too collinear on "
                   "this plate to separate.")
    print(f"\nVERDICT: {verdict}")

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    for a, (x, r, t) in zip(ax, [
            (cp, r_p, "peak brightness p99\n(technical gain term)"),
            (cc_n, r_cn, f"normalised coverage >{REL:g}*p99\n(gain-invariant myotube amount)"),
            (cc_a, r_ca, "absolute coverage >119\n(confounds both)")]):
        a.axhline(0, color="#c3c2b7", lw=1); a.axvline(0, color="#c3c2b7", lw=1)
        a.scatter(x, cv, s=80, color="#2a78d6", edgecolor="#0b0b0b", zorder=3)
        xs = np.array([x.min(), x.max()])
        a.plot(xs, np.polyval(np.polyfit(x, cv, 1), xs), "--", color="#898781")
        a.set_xlabel(f"{t.splitlines()[0]} (deviation from condition mean)")
        a.set_ylabel("conversion deviation (pp)")
        a.set_title(f"{t}\nr = {r:+.2f}")
        a.grid(alpha=0.3)
    fig.suptitle("PLATE 44 — within condition, what actually predicts conversion?",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "diagnose_brightness_vs_area.png"), dpi=130,
                bbox_inches="tight", facecolor="white")
    with open(os.path.join(HERE, "diagnose_brightness_vs_area.json"), "w") as fh:
        json.dump({"plate": "PLATE_44", "n_wells": int(cv.size),
                   "relative_coverage_level": REL,
                   "r_brightness": round(float(r_p), 3),
                   "r_coverage_normalised": round(float(r_cn), 3),
                   "r_coverage_absolute": round(float(r_ca), 3),
                   "partial_r_brightness_given_coverage": round(float(r_bright_ctrl), 3),
                   "partial_p_brightness": float(p_bc),
                   "partial_r_coverage_given_brightness": round(float(r_cov_ctrl), 3),
                   "partial_p_coverage": float(p_cc),
                   "verdict": verdict}, fh, indent=2)
    print("\n-> diagnose_brightness_vs_area.png / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
