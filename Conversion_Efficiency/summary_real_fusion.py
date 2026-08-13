"""Summary of the REAL-myotube fusion index: per-well fusion at each length gate
(grouped bars) + the plate total, showing the control dropping to the bottom
once fragments are gated out."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "plate23_real_fusion"
R = json.load(open(os.path.join(D, "real_fusion_results.json")))
gates = R["gates"]
wells = sorted(R["per_well"])
short = [w.split("_", 1)[1] for w in wells]

fig, ax = plt.subplots(1, 2, figsize=(18, 7))

# left: per-well fusion index at each gate
x = np.arange(len(wells))
bw = 0.8 / len(gates)
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(gates)))
for k, g in enumerate(gates):
    vals = [R["per_well"][w][str(g) if str(g) in R["per_well"][w] else g]["fusion_pct"]
            if False else R["per_well"][w][[kk for kk in R["per_well"][w]
            if float(kk) == g][0]]["fusion_pct"] for w in wells]
    ax[0].bar(x + k * bw, vals, bw, color=cmap[k], label=f">= {int(g)} um")
ax[0].set_xticks(x + 0.4 - bw / 2)
ax[0].set_xticklabels(short, rotation=15, ha="right", fontsize=9)
ax[0].set_ylabel("fusion index (% nuclei inside real myotubes)")
ax[0].set_title("Per-well fusion index vs fibre-length gate")
ax[0].legend(title="myotube length gate"); ax[0].grid(axis="y", alpha=0.3)

# right: plate total nuclei-inside + fusion% vs gate
ins = [R["totals"][[kk for kk in R["totals"] if float(kk) == g][0]]["inside"] for g in gates]
tot = R["totals"][[kk for kk in R["totals"] if float(kk) == gates[0]][0]]["total"]
fus = [100 * i / tot for i in ins]
ax2 = ax[1]
bars = ax2.bar([str(int(g)) for g in gates], ins, color="#db2777", edgecolor="#831843")
for b, i, f in zip(bars, ins, fus):
    ax2.text(b.get_x() + b.get_width() / 2, i + 40, f"{i:,}\n{f:.1f}%",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_xlabel("myotube fibre-length gate (um)")
ax2.set_ylabel("total nuclei inside real myotubes (plate 23)")
ax2.set_title(f"Plate 23 total — of {tot:,} nuclei")
ax2.grid(axis="y", alpha=0.3); ax2.set_ylim(0, max(ins) * 1.2)

fig.suptitle("Plate 23 — nuclei inside REAL myotubes (fragments gated out by fibre length)",
             fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(D, "real_fusion_summary.png")
fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
print("saved ->", out)
for g in gates:
    key = [kk for kk in R["totals"] if float(kk) == g][0]
    print(f"gate>={int(g):>3d}um  total inside = {R['totals'][key]['inside']:,}"
          f"  ({100*R['totals'][key]['inside']/tot:.2f}%)")
