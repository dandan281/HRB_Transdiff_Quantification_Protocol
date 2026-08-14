"""PLATE 44 QC: is the conversion ranking driven by real Desmin, or by nucleus
density (the confound that invalidated P28)? Direct analogue of
`New_Quantif_P32/density_confound.py`, reading this plate's `visualize_final.json`.

Low |r| = density is not driving the result (trust it); high |r| (P28 was 0.99)
= the ranking is a density artifact.

PLATE 44 has replicate wells, so the correlation is reported BOTH at well level
(the P32-comparable number) and at condition level, because averaging replicates
changes the effective sample and either could mislead alone.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/density_confound.py
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
from p44_layout import CONTROL_CONDITION  # noqa: E402


def main() -> int:
    vf = json.load(open(os.path.join(HERE, "visualize_final.json")))
    pw, pc = vf["per_well"], vf["per_condition"]

    keep = {w: r for w, r in pw.items() if not r["technical_failure"]}
    dens = np.array([r["valid"] for r in keep.values()], dtype=float)
    conv = np.array([r["conversion_pct"] for r in keep.values()])
    labs = [r["well_id"] for r in keep.values()]
    is_ctrl = [r["condition"] == CONTROL_CONDITION for r in keep.values()]

    rp, pv = pearsonr(dens, conv)
    rs = spearmanr(dens, conv)[0]
    print(f"P44 well-level  Pearson r(density, conversion)  = {rp:+.3f} "
          f"(p={pv:.3g}, n={dens.size})")
    print(f"P44 well-level  Spearman r(density, conversion) = {rs:+.3f}")

    cd, cc, cl = [], [], []
    for c, r in pc.items():
        vals = [pw[w]["valid"] for w in pw
                if pw[w]["condition"] == c and not pw[w]["technical_failure"]]
        cd.append(float(np.mean(vals)))
        cc.append(r["mean_pct"])
        cl.append(c)
    cd, cc = np.array(cd), np.array(cc)
    rpc = pearsonr(cd, cc)[0]
    print(f"P44 condition-level Pearson r = {rpc:+.3f} (n={cd.size})")
    print("(P28 was r=0.99/1.00 = pure density artifact; P23 0.61; P32 0.34; "
          "P9 ~0.00)")

    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
    cols = ["#64748b" if c else "#db2777" for c in is_ctrl]
    ax[0].scatter(dens, conv, c=cols, s=110, edgecolor="#1e293b", zorder=3)
    for x, y, l in zip(dens, conv, labs):
        ax[0].annotate(l, (x, y), fontsize=7, xytext=(4, 4),
                       textcoords="offset points")
    m, b = np.polyfit(dens, conv, 1)
    xs = np.array([dens.min(), dens.max()])
    ax[0].plot(xs, m * xs + b, "--", color="#94a3b8", zorder=1)
    ax[0].set_xlabel("valid nuclei (density; field area constant)")
    ax[0].set_ylabel("conversion efficiency (%)")
    ax[0].set_title(f"well level — Pearson r = {rp:+.2f}, Spearman {rs:+.2f}\n"
                    f"(grey = {CONTROL_CONDITION} control; P28 artifact was 0.99)")
    ax[0].grid(alpha=0.3)

    ccols = ["#64748b" if c == CONTROL_CONDITION else "#db2777" for c in cl]
    ax[1].scatter(cd, cc, c=ccols, s=130, edgecolor="#1e293b", zorder=3)
    for x, y, l in zip(cd, cc, cl):
        ax[1].annotate(l, (x, y), fontsize=7.5, xytext=(4, 4),
                       textcoords="offset points")
    m2, b2 = np.polyfit(cd, cc, 1)
    xs2 = np.array([cd.min(), cd.max()])
    ax[1].plot(xs2, m2 * xs2 + b2, "--", color="#94a3b8", zorder=1)
    ax[1].set_xlabel("mean valid nuclei per well")
    ax[1].set_ylabel("condition mean conversion (%)")
    ax[1].set_title(f"condition level — Pearson r = {rpc:+.2f}")
    ax[1].grid(alpha=0.3)

    fig.suptitle("PLATE 44 conversion vs nucleus density", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "density_confound.png"), dpi=120,
                bbox_inches="tight", facecolor="white")
    with open(os.path.join(HERE, "density_confound.json"), "w") as fh:
        json.dump({"plate": "PLATE_44",
                   "well_level": {"pearson_r": round(float(rp), 3),
                                  "pearson_p": float(pv),
                                  "spearman_r": round(float(rs), 3),
                                  "n": int(dens.size)},
                   "condition_level": {"pearson_r": round(float(rpc), 3),
                                       "n": int(cd.size)},
                   "reference": {"P28": 0.99, "P23": 0.61, "P32": 0.34,
                                 "P9": 0.0}}, fh, indent=2)
    print("-> density_confound.png / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
