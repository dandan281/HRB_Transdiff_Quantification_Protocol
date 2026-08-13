"""Visualise the B02 multinucleation SENSITIVITY that was previously table-only:
how the nuclei-per-myotube distribution shifts with the length gate (0/30/50/100 um)
and the overlap convention (25% / 50%). Reads b02_multinucleation.json -- no recompute.
The locked operating point (gate 50, overlap 50%) is boxed.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P23/b02_multinuc_sensitivity.py
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "b02_multinucleation.json")))["results"]
GATES = [0, 30, 50, 100]
CATS = ["1", "2", "3", "4", "5plus"]
CATCOLS = ["#3b82f6", "#22c55e", "#eab308", "#f97316", "#ef4444"]
CATNAMES = ["1", "2", "3", "4", "5+"]

fig, axes = plt.subplots(1, 3, figsize=(19, 6),
                         gridspec_kw={"width_ratios": [1.2, 1.2, 1]})

# panels 1-2: stacked composition vs gate, one per overlap fraction
for pi, frac in enumerate([50, 25]):
    ax = axes[pi]
    data = np.array([[R[f"frac{frac}_gate{g}"]["pct_of_nucleated"][c] for c in CATS]
                     for g in GATES])
    bottom = np.zeros(len(GATES)); x = np.arange(len(GATES))
    for k, (c, nm) in enumerate(zip(CATCOLS, CATNAMES)):
        ax.bar(x, data[:, k], bottom=bottom, color=c, label=f"{nm} nuclei",
               edgecolor="white", linewidth=0.6)
        bottom += data[:, k]
    ax.set_xticks(x); ax.set_xticklabels([f"≥{g}" for g in GATES])
    ax.set_xlabel("individual-myotube length gate (µm)")
    ax.set_ylabel("% of nucleated myotubes"); ax.set_ylim(0, 100)
    ax.set_title(f"overlap ≥ {frac}%")
    for xi, g in enumerate(GATES):
        ax.text(xi, 101, f"n={R[f'frac{frac}_gate{g}']['n_myotubes_with_nuclei']}",
                ha="center", fontsize=7.5, color="#334155")
    if frac == 50 and 50 in GATES:                    # box the locked operating point
        i = GATES.index(50)
        ax.add_patch(plt.Rectangle((i - 0.45, 0), 0.9, 100, fill=False,
                                   edgecolor="black", lw=2.2, ls="--"))
    if pi == 0:
        ax.legend(title="nuclei / myotube", fontsize=8)

# panel 3: mean nuclei + %>=2 vs gate, both overlaps
ax = axes[2]
for frac, ls in [(50, "-"), (25, "--")]:
    means = [R[f"frac{frac}_gate{g}"]["mean_nuclei_per_nucleated"] for g in GATES]
    ge2 = [R[f"frac{frac}_gate{g}"]["pct_ge2_of_nucleated"] for g in GATES]
    ax.plot(GATES, means, ls, marker="o", color="#7c3aed", label=f"mean ({frac}%)")
    ax.plot(GATES, np.array(ge2) / 25, ls, marker="s", color="#db2777",
            label=f"%≥2 ({frac}%)")
ax.axvline(50, color="black", ls=":", lw=1)
ax.set_xlabel("length gate (µm)")
ax.set_ylabel("mean nuclei/myotube  |  %≥2 ÷ 25")
ax.set_title("maturity vs gate (locked gate=50 dotted)")
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

fig.suptitle("B02 multinucleation sensitivity to length gate & overlap "
             "(locked operating point boxed: gate ≥50 µm, overlap ≥50%)",
             fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(HERE, "b02_multinuc_sensitivity.png"), dpi=120,
            bbox_inches="tight", facecolor="white")
print("-> b02_multinuc_sensitivity.png")
