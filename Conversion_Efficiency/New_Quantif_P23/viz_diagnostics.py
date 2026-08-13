"""Visualise the two remaining parts of the story that were JSON-only:
  A. THE BUG -- per-image normalisation flattened Desmin coverage to ~11% in every
     well; a shared absolute threshold lets the wells separate. (Reads the old
     per-image coverage from plate23_myotube/myotube_results.jsonl and the new
     shared-absolute coverage from absolute_desmin.json.)
  B. ROBUSTNESS -- fold-change is stable across every hyperparameter: intensity
     threshold k, cytoplasmic ring size, and nucleus area cut. (Reads conversion_v2,
     ring_sweep, amin_sweep.)

Pure plotting from existing JSON -- no recompute. Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe NEW_Quantif/viz_diagnostics.py
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORDER = ["23_B02_ctrl", "33_C09_br223_trka", "29_C05_br223_egfrc",
         "19_B06_act104_trka", "32_C08_br223_igf1r", "22_B03_act104_egfrc"]
SHORT = [w.split("_", 1)[1] for w in ORDER]
CTRL = "23_B02_ctrl"


# ---------- Figure A: the bug ----------
old_cov = {}
with open(os.path.join(ROOT, "plate23_myotube", "myotube_results.jsonl")) as fh:
    for line in fh:
        r = json.loads(line)
        old_cov[r["well"]] = r["coverage_pct"]
absd = json.load(open(os.path.join(HERE, "absolute_desmin.json")))
new_cov = {w: absd["coverage_pct"][w]["5"] for w in ORDER}   # k=5 shared absolute

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(ORDER)); bw = 0.38
ax.bar(x - bw / 2, [old_cov[w] for w in ORDER], bw, color="#94a3b8",
       edgecolor="#334155", label="OLD: per-image percentile gate")
ax.bar(x + bw / 2, [new_cov[w] for w in ORDER], bw, color="#db2777",
       edgecolor="#831843", label="NEW: shared absolute threshold (k=5)")
ax.set_xticks(x); ax.set_xticklabels(SHORT, rotation=15, ha="right")
ax.set_ylabel("Desmin coverage (% of field)")
ax.set_title("The bug: per-image normalisation pins every well near ~11%\n"
             "shared absolute threshold lets the wells separate (biology)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.annotate("all wells ~10-12%\nregardless of true Desmin",
            xy=(2.0, 11), xytext=(2.2, 20), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#334155"))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "diagnosis_the_bug.png"), dpi=120,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("-> diagnosis_the_bug.png")


# ---------- Figure B: robustness (fold-change stable across 3 knobs) ----------
cv = json.load(open(os.path.join(HERE, "conversion_v2.json")))["results"]
rs = json.load(open(os.path.join(HERE, "ring_sweep.json")))["results"]
am = json.load(open(os.path.join(HERE, "amin_sweep_g30.json")))["plate"]

fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(ORDER) - 1))

# B1: fold vs intensity threshold k (25% overlap, per-well conversion_v2)
ks = sorted(cv, key=lambda s: int(s))
for c, w in zip(cmap, [w for w in ORDER if w != CTRL]):
    folds = []
    for k in ks:
        eff = cv[k][w]["overlap_25pct"]
        base = cv[k][CTRL]["overlap_25pct"]
        folds.append(eff["efficiency_pct"] / base["efficiency_pct"])
    ax[0].plot([int(k) for k in ks], folds, "-o", color=c, label=w.split("_", 1)[1])
ax[0].set_xlabel("intensity threshold k (x background sigma)")
ax[0].set_title("fold vs threshold k")

# B2: fold vs ring size (per-cell, control anchored -- fold is the anchor-free part)
rings = sorted(rs, key=float)
for c, w in zip(cmap, [w for w in ORDER if w != CTRL]):
    ax[1].plot([float(r) for r in rings],
               [rs[r]["wells"][w]["fold"] for r in rings], "-o", color=c)
ax[1].set_xlabel("cytoplasmic ring size (um)")
ax[1].set_title("fold vs ring size")

# B3: plate conversion vs nucleus area cut (both overlaps)
amins = sorted(am, key=lambda s: int(s))
ax[2].plot([int(a) for a in amins], [am[a]["fusion_25_pct"] for a in amins],
           "-o", color="#db2777", label="25% overlap")
ax[2].plot([int(a) for a in amins], [am[a]["fusion_50_pct"] for a in amins],
           "-s", color="#6366f1", label="50% overlap")
ax[2].set_xlabel("nucleus lower area cut (um^2)")
ax[2].set_ylabel("plate conversion (%)")
ax[2].set_title("plate conversion vs area cut")
ax[2].legend(fontsize=8)

for a in ax[:2]:
    a.axhspan(2, 3, color="#22c55e", alpha=0.10)
    a.set_ylabel("fold-change vs control"); a.grid(alpha=0.3)
ax[2].grid(alpha=0.3)
ax[0].legend(fontsize=7, ncol=2)
fig.suptitle("Robustness: fold-change is stable across every hyperparameter "
             "(green band = expected 2-3x)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(HERE, "robustness_sweeps.png"), dpi=120,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("-> robustness_sweeps.png")
