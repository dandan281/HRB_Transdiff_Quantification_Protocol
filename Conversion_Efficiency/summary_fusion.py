"""Plate fusion-index summary: montage of the 6 fusion overlays + per-well
fusion-index bar chart + plate-level fusion index."""
import os, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "plate23_fusion"
recs = [json.loads(l) for l in open(os.path.join(D, "fusion_results.jsonl"))]
recs.sort(key=lambda r: r["well"])
tot_in = sum(r["nuclei_inside"] for r in recs)
tot_all = sum(r["nuclei_total"] for r in recs)
plate_fi = 100.0 * tot_in / tot_all

fig = plt.figure(figsize=(19, 13))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.25, wspace=0.05)

for i, r in enumerate(recs):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.imshow(np.asarray(Image.open(os.path.join(D, f"{r['well']}_fusion.png"))))
    ax.set_title(f"{r['well']}\n{r['nuclei_inside']:,}/{r['nuclei_total']:,} inside "
                 f"= {r['fusion_index_pct']:.1f}%", fontsize=12)
    ax.axis("off")

axb = fig.add_subplot(gs[2, :])
wells = [r["well"].split("_", 1)[1] for r in recs]
fis = [r["fusion_index_pct"] for r in recs]
bars = axb.bar(wells, fis, color="#db2777", edgecolor="#831843")
for b, r in zip(bars, recs):
    axb.text(b.get_x() + b.get_width() / 2, r["fusion_index_pct"] + 0.15,
             f"{r['fusion_index_pct']:.1f}%\n{r['nuclei_inside']:,}/{r['nuclei_total']:,}",
             ha="center", va="bottom", fontsize=9.5)
axb.axhline(plate_fi, ls="--", color="#374151",
            label=f"plate fusion index = {plate_fi:.1f}%")
axb.set_ylabel("fusion index  (% nuclei inside myotubes)")
axb.set_title("Per-well fusion index  (nuclei inside Desmin+ myotubes / total nuclei)")
axb.grid(axis="y", alpha=0.3); axb.legend(loc="upper right")
axb.set_ylim(0, max(fis) * 1.28)
plt.setp(axb.get_xticklabels(), rotation=12, ha="right", fontsize=10)

fig.suptitle(f"PLATE 23  —  fusion index {plate_fi:.1f}%  "
             f"({tot_in:,} of {tot_all:,} nuclei inside myotubes)",
             fontsize=20, fontweight="bold", y=0.98)
out = os.path.join(D, "plate23_fusion_summary.png")
fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
print(f"PLATE_FUSION {plate_fi:.2f}%  inside={tot_in} total={tot_all}")
for r in recs:
    print(f"  {r['well']:26s} {r['nuclei_inside']:>6,}/{r['nuclei_total']:<6,} "
          f"= {r['fusion_index_pct']:5.1f}%   myo_cov {r['myotube_coverage_pct']:.1f}%")
print("saved ->", out)
