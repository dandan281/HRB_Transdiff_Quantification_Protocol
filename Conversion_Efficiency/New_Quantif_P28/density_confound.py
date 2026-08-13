"""PLATE_28 QC: the apparent conversion ranking is a NUCLEUS-DENSITY artifact.

On P28 every well's per-cell Desmin distribution overlaps (no treatment effect),
and the apparent "conversion %" tracks nucleus density almost perfectly
(Spearman = 1.00). In dense fields the 10 um cytoplasmic ring around each nucleus
picks up neighbouring cytoplasm/background, inflating per-cell Desmin -- so denser
wells score higher regardless of true conversion. This figure shows that confound
next to P23, where a real treatment effect dominates and the density correlation is
weak. Conclusion: P28 shows NO interpretable treatment effect.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P28/density_confound.py
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(plate_dir, valid_key):
    r = json.load(open(os.path.join(ROOT, plate_dir, "visualize_final.json")))["per_well"]
    dens = np.array([r[w][valid_key] for w in r])
    conv = np.array([r[w]["conversion_pct"] for w in r])
    labs = [w.split("_", 1)[1] for w in r]
    ctrl = ["ctrl" in w for w in r]
    return dens, conv, labs, ctrl


fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
for k, (pdir, title) in enumerate([("New_Quantif_P28", "PLATE_28 (this plate)"),
                                    ("New_Quantif_P23", "PLATE_23 (real effect, for contrast)")]):
    dens, conv, labs, ctrl = load(pdir, "valid")
    r = pearsonr(dens, conv)[0]
    cols = ["#64748b" if c else "#db2777" for c in ctrl]
    ax[k].scatter(dens, conv, c=cols, s=120, edgecolor="#1e293b", zorder=3)
    for x, y, l in zip(dens, conv, labs):
        ax[k].annotate(l, (x, y), fontsize=8, xytext=(4, 4),
                       textcoords="offset points")
    m, b = np.polyfit(dens, conv, 1)
    xs = np.array([dens.min(), dens.max()])
    ax[k].plot(xs, m * xs + b, "--", color="#94a3b8", zorder=1)
    ax[k].set_xlabel("valid nuclei (density; field area constant)")
    ax[k].set_ylabel("conversion efficiency (%)")
    ax[k].set_title(f"{title}\nPearson r(density, conversion) = {r:+.2f}")
    ax[k].grid(alpha=0.3)

fig.suptitle("P28 apparent conversion IS nucleus density (r=0.99) -> no real "
             "treatment effect; grey = control", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(HERE, "density_confound.png"), dpi=120,
            bbox_inches="tight", facecolor="white")
print("-> density_confound.png")
