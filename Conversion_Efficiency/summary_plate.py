"""Plate-level summary: montage of all 6 well labeled-overlays + a per-well
count bar chart + the plate total. Reads plate_results.jsonl and the per-well
*_labeled.png thumbnails already produced by count_well.py."""
import os, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "plate23_nuclei"
recs = [json.loads(l) for l in open(os.path.join(D, "plate_results.jsonl"))]
recs.sort(key=lambda r: r["well"])
total = sum(r["nuclei"] for r in recs)

fig = plt.figure(figsize=(19, 13))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.22, wspace=0.05)

# --- 6 labeled overlays (2 rows x 3 cols) ---
for i, r in enumerate(recs):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    img = Image.open(os.path.join(D, f"{r['well']}_labeled.png"))
    ax.imshow(np.asarray(img))
    ax.set_title(f"{r['well']}\n{r['nuclei']:,} nuclei  (cp={r['operating_cellprob']:+.0f})",
                 fontsize=12)
    ax.axis("off")

# --- bar chart spanning the bottom row ---
axb = fig.add_subplot(gs[2, :])
wells = [r["well"].split("_", 1)[1] for r in recs]   # drop the leading index
counts = [r["nuclei"] for r in recs]
bars = axb.bar(wells, counts, color="#3b82f6", edgecolor="#1e3a8a")
for b, c in zip(bars, counts):
    axb.text(b.get_x() + b.get_width() / 2, c + 80, f"{c:,}",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
axb.set_ylabel("nuclei (Cellpose-SAM, plateau cp)")
axb.set_title("Per-well nucleus count")
axb.grid(axis="y", alpha=0.3)
axb.set_ylim(0, max(counts) * 1.15)
plt.setp(axb.get_xticklabels(), rotation=12, ha="right", fontsize=10)

fig.suptitle(f"PLATE 23  —  {total:,} nuclei total  (6 wells, Cellpose-SAM)",
             fontsize=20, fontweight="bold", y=0.98)
out = os.path.join(D, "plate23_summary.png")
fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
print(f"PLATE_TOTAL {total}")
for r in recs:
    print(f"  {r['well']:26s} {r['nuclei']:>7,}")
print("saved ->", out)
