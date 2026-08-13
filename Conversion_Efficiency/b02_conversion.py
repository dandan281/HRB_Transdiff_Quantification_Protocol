"""B02 (control) ONLY.
1) For each myotube length bin: how many nuclei live in myotubes of that length.
2) Conversion efficiency = myotube-positive nuclei / total nuclei, and how it
   changes with the myotube-length definition (cumulative, gate >= L).
Total nuclei = area-boundary-valid nuclei (50-500 um2).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nuclei_3d import well_pairs
from real_fusion import UM2, NUC_DIR

W = "23_B02_ctrl"
BINW = 25.0

# host-myotube length (um) of every Desmin+ nucleus in B02
area_um2, host_len = well_pairs(W)
n_pos = host_len.size

# total valid nuclei (area boundary 50-500 um2)
nuc = np.load(os.path.join(NUC_DIR, f"{W}_masks.npy"))
a = np.bincount(nuc.ravel()).astype(float) * UM2
a[0] = 0
total_valid = int(((a >= 50) & (a <= 500)).sum())

conv_eff = 100 * n_pos / total_valid
print(f"B02  Desmin+ (myotube+) nuclei = {n_pos}")
print(f"B02  total valid nuclei        = {total_valid}")
print(f"B02  CONVERSION EFFICIENCY     = {conv_eff:.2f}%\n")

# ---- (1) nuclei per myotube-length bin ----
edges = np.arange(0, np.ceil(host_len.max() / BINW) * BINW + BINW, BINW)
counts, _ = np.histogram(host_len, bins=edges)
print(f"{'myotube length (um)':22s}{'# nuclei in myotube':>20s}")
for i in range(len(counts)):
    if counts[i] or i < 8:
        print(f"  {edges[i]:3.0f}-{edges[i+1]:<3.0f}{'':13s}{counts[i]:>10d}")

# ---- (2) conversion efficiency vs length gate (cumulative) ----
gates = np.arange(0, 260, 10)
ce = [100 * (host_len >= g).sum() / total_valid for g in gates]

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ctr = 0.5 * (edges[:-1] + edges[1:])
ax[0].bar(ctr, counts, width=BINW * 0.9, color="#db2777", edgecolor="#831843")
ax[0].set_xlabel("myotube length (um)")
ax[0].set_ylabel("number of nuclei in myotube")
ax[0].set_title(f"B02 — nuclei per myotube length  (total myotube+ = {n_pos})")
ax[0].set_xlim(0, min(300, edges[-1])); ax[0].grid(axis="y", alpha=0.3)

ax[1].plot(gates, ce, "-o", color="#2563eb")
for g, marker in [(0, "all myotubes"), (50, "real (>=50um)"), (100, "mature (>=100um)")]:
    v = 100 * (host_len >= g).sum() / total_valid
    ax[1].plot(g, v, "*", ms=16, color="#ef4444")
    ax[1].annotate(f"{marker}\n{v:.1f}%", (g, v), textcoords="offset points",
                   xytext=(8, 8), fontsize=9)
ax[1].set_xlabel("myotube length gate  (count nuclei in myotubes >= L um)")
ax[1].set_ylabel("conversion efficiency (%)")
ax[1].set_title("B02 — conversion efficiency vs myotube-length definition")
ax[1].grid(alpha=0.3); ax[1].set_ylim(0, max(ce) * 1.15)

fig.suptitle(f"Plate 23  B02 (control) — conversion efficiency = "
             f"{conv_eff:.1f}%  ({n_pos}/{total_valid} nuclei)",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = "plate23_real_fusion/B02_conversion_efficiency.png"
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("\nsaved ->", out)
