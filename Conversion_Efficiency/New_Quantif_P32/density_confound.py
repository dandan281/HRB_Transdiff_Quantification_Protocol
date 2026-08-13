"""PLATE_32 QC: is the conversion ranking driven by real Desmin, or by nucleus
density (the confound that invalidated P28)? Plots conversion vs valid-nuclei
density for P32 and reports the correlation. Low |r| = density is not driving the
result (trust it); high |r| (as on P28, r=0.99) = the ranking is a density artifact.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P32/density_confound.py
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(plate_dir):
    r = json.load(open(os.path.join(ROOT, plate_dir, "visualize_final.json")))["per_well"]
    dens = np.array([r[w]["valid"] for w in r])
    conv = np.array([r[w]["conversion_pct"] for w in r])
    labs = [w.split("_", 1)[1] for w in r]
    ctrl = ["ctrl" in w for w in r]
    return dens, conv, labs, ctrl


dens, conv, labs, ctrl = load("New_Quantif_P32")
rp = pearsonr(dens, conv)[0]
rs = spearmanr(dens, conv)[0]
print(f"P32 Pearson r(density, conversion)  = {rp:+.3f}")
print(f"P32 Spearman r(density, conversion) = {rs:+.3f}")
print("(P28 was r=0.99/1.00 = pure density artifact; P23 was 0.61 = real effect)")

fig, ax = plt.subplots(figsize=(9, 6.5))
cols = ["#64748b" if c else "#db2777" for c in ctrl]
ax.scatter(dens, conv, c=cols, s=120, edgecolor="#1e293b", zorder=3)
for x, y, l in zip(dens, conv, labs):
    ax.annotate(l, (x, y), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
m, b = np.polyfit(dens, conv, 1)
xs = np.array([dens.min(), dens.max()])
ax.plot(xs, m * xs + b, "--", color="#94a3b8", zorder=1)
ax.set_xlabel("valid nuclei (density; field area constant)")
ax.set_ylabel("conversion efficiency (%)")
ax.set_title(f"PLATE_32 conversion vs nucleus density\n"
             f"Pearson r = {rp:+.2f}, Spearman = {rs:+.2f}   "
             f"(grey = control; P28 artifact was r=0.99)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "density_confound.png"), dpi=120,
            bbox_inches="tight", facecolor="white")
print("-> density_confound.png")
