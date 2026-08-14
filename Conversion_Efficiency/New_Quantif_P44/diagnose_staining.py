"""PLATE 44 troubleshooting, step 3: is the ranking tracking Desmin STAINING
STRENGTH rather than biology?

Steps 1-2 ruled out three suspects:
  * background drift          r = +0.07  (not it)
  * saturation                0.0000 % of pixels (not it)
  * small-object debris       despeckling to 200 um^2 moved the null floor only
                              4.51 % -> 4.10 % and within-condition SD 2.03 ->
                              1.91 (real, but marginal)

That leaves the amount of Desmin SIGNAL per well. A global threshold is correct
policy (per-image normalisation was removed for flattening real differences),
but it converts any technical variation in staining or exposure directly into
apparent conversion. This script measures how much of the ranking that explains,
using the per-well Desmin statistics already recorded by `build_dbs.py`.

The decisive comparison: within a CONDITION, replicate wells are biologically
identical, so any conversion spread between them must be technical. If that
spread tracks the wells' Desmin signal strength, the readout is staining-limited.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/diagnose_staining.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    CONDITION_ORDER, CONTROL_CONDITION, TECHNICAL_FAILURES, condition_of,
    well_id, wells)


def main() -> int:
    man = {r["well"]: r for r in
           json.load(open(os.path.join(HERE, "dbs_cache", "dbs_manifest.json")))
           ["per_well"] if not r.get("cached")}
    pc = json.load(open(os.path.join(HERE, "percell_desmin.json")))["per_well"]
    order = [w for w in wells() if w in man]

    p99 = np.array([man[w]["dbs_p99"] for w in order])
    med = np.array([man[w]["dbs_median"] for w in order])
    conv = np.array([pc[w]["conversion_pct"] for w in order])
    ids = [well_id(w) for w in order]
    conds = [condition_of(w) for w in order]
    ok = np.array([well_id(w) not in TECHNICAL_FAILURES for w in order])

    r_all, p_all = pearsonr(p99[ok], conv[ok])
    rs_all = spearmanr(p99[ok], conv[ok])[0]
    print("PLATE 44 — Desmin staining strength vs conversion\n" + "=" * 52)
    print(f"dbs p99 (Desmin signal strength) spans {p99[ok].min():.0f}-"
          f"{p99[ok].max():.0f} raw units ({p99[ok].max()/p99[ok].min():.2f}x) "
          f"across biologically-different wells")
    print(f"Pearson  r(dbs_p99, conversion) = {r_all:+.3f} (p={p_all:.2g}, "
          f"n={ok.sum()})")
    print(f"Spearman r(dbs_p99, conversion) = {rs_all:+.3f}")
    print(f"Pearson  r(dbs_median, conversion) = "
          f"{pearsonr(med[ok], conv[ok])[0]:+.3f}")

    # --- the decisive test: WITHIN condition, replicates are identical -------
    dp, dc, lab = [], [], []
    for c in CONDITION_ORDER:
        idx = [i for i in range(len(order))
               if conds[i] == c and ok[i]]
        if len(idx) < 2:
            continue
        pm, cm = p99[idx].mean(), conv[idx].mean()
        for i in idx:
            dp.append(p99[i] - pm)
            dc.append(conv[i] - cm)
            lab.append(f"{ids[i]}·{c}")
    dp, dc = np.array(dp), np.array(dc)
    r_in, p_in = pearsonr(dp, dc)
    print(f"\nWITHIN-CONDITION (biology held constant, n={dp.size} wells across "
          f"{len(set(conds))} conditions)")
    print(f"Pearson r(Desmin signal deviation, conversion deviation) = "
          f"{r_in:+.3f} (p={p_in:.2g})")
    slope = np.polyfit(dp, dc, 1)[0]
    print(f"slope = {slope*1000:+.2f} pp of conversion per 1000 raw units of p99")
    if abs(r_in) > 0.6:
        verdict = ("STAINING-LIMITED: replicate wells that stained brighter "
                   "score higher, with biology held constant")
    elif abs(r_in) > 0.35:
        verdict = ("PARTIALLY staining-driven: a real component of the "
                   "replicate scatter is signal strength")
    else:
        verdict = ("replicate scatter is NOT explained by Desmin signal "
                   "strength")
    print(f"-> {verdict}")

    r2 = r_in ** 2
    print(f"   staining explains {100*r2:.0f}% of the within-condition variance;"
          f" the remaining {100*(1-r2):.0f}% is other technical + biological "
          f"noise")

    # how much of the BETWEEN-condition range could this manufacture?
    cond_mean_p99, cond_mean_conv, cnames = [], [], []
    for c in CONDITION_ORDER:
        idx = [i for i in range(len(order)) if conds[i] == c and ok[i]]
        if idx:
            cond_mean_p99.append(p99[idx].mean())
            cond_mean_conv.append(conv[idx].mean())
            cnames.append(c)
    cond_mean_p99 = np.array(cond_mean_p99)
    cond_mean_conv = np.array(cond_mean_conv)
    r_cond = pearsonr(cond_mean_p99, cond_mean_conv)[0]
    predicted = slope * (cond_mean_p99 - cond_mean_p99.mean())
    print(f"\nBETWEEN-CONDITION  r = {r_cond:+.3f}")
    print(f"   condition-mean Desmin signal spans "
          f"{cond_mean_p99.min():.0f}-{cond_mean_p99.max():.0f}")
    print(f"   at the within-condition slope that alone would manufacture a "
          f"{predicted.max()-predicted.min():.1f} pp spread; the observed "
          f"condition spread is {cond_mean_conv.max()-cond_mean_conv.min():.1f} pp")

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    ax[0].scatter(p99[ok], conv[ok], s=80, color="#2a78d6",
                  edgecolor="#0b0b0b", zorder=3)
    ax[0].scatter(p99[~ok], conv[~ok], s=90, color="#d03b3b", marker="x",
                  zorder=3, label="B11 (staining failure)")
    for x, y, l in zip(p99, conv, ids):
        ax[0].annotate(l, (x, y), fontsize=6.5, xytext=(4, 3),
                       textcoords="offset points")
    m_, b_ = np.polyfit(p99[ok], conv[ok], 1)
    xs = np.array([p99.min(), p99.max()])
    ax[0].plot(xs, m_ * xs + b_, "--", color="#898781")
    ax[0].set_xlabel("Desmin signal strength (dbs p99, raw units)")
    ax[0].set_ylabel("conversion (%)")
    ax[0].set_title(f"all wells — r = {r_all:+.2f}\n(confounds biology with staining)")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    ax[1].axhline(0, color="#c3c2b7", lw=1); ax[1].axvline(0, color="#c3c2b7", lw=1)
    ax[1].scatter(dp, dc, s=80, color="#eb6834", edgecolor="#0b0b0b", zorder=3)
    for x, y, l in zip(dp, dc, lab):
        ax[1].annotate(l.split("·")[0], (x, y), fontsize=6.5, xytext=(4, 3),
                       textcoords="offset points")
    xs2 = np.array([dp.min(), dp.max()])
    ax[1].plot(xs2, np.polyval(np.polyfit(dp, dc, 1), xs2), "--",
               color="#898781")
    ax[1].set_xlabel("Desmin signal deviation from condition mean")
    ax[1].set_ylabel("conversion deviation from condition mean (pp)")
    ax[1].set_title(f"WITHIN condition — biology held constant\n"
                    f"r = {r_in:+.2f}, explains {100*r2:.0f}% of replicate scatter")
    ax[1].grid(alpha=0.3)

    o = np.argsort(cond_mean_conv)
    ax[2].scatter(cond_mean_p99[o], cond_mean_conv[o], s=110, color="#1baf7a",
                  edgecolor="#0b0b0b", zorder=3)
    for x, y, l in zip(cond_mean_p99, cond_mean_conv, cnames):
        ax[2].annotate(l, (x, y), fontsize=7, xytext=(4, 3),
                       textcoords="offset points")
    ax[2].set_xlabel("condition-mean Desmin signal (dbs p99)")
    ax[2].set_ylabel("condition-mean conversion (%)")
    ax[2].set_title(f"BETWEEN conditions — r = {r_cond:+.2f}")
    ax[2].grid(alpha=0.3)

    fig.suptitle("PLATE 44 — is the readout tracking Desmin staining strength?",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "diagnose_staining.png"), dpi=130,
                bbox_inches="tight", facecolor="white")
    with open(os.path.join(HERE, "diagnose_staining.json"), "w") as fh:
        json.dump({"plate": "PLATE_44",
                   "r_all_wells": round(float(r_all), 3),
                   "spearman_all_wells": round(float(rs_all), 3),
                   "r_within_condition": round(float(r_in), 3),
                   "p_within_condition": float(p_in),
                   "variance_explained_within_condition_pct": round(100 * r2, 1),
                   "slope_pp_per_1000_p99": round(float(slope * 1000), 3),
                   "r_between_conditions": round(float(r_cond), 3),
                   "dbs_p99_min": float(p99[ok].min()),
                   "dbs_p99_max": float(p99[ok].max()),
                   "verdict": verdict}, fh, indent=2)
    print("\n-> diagnose_staining.png / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
