"""High-res zoomed crop of a well's Cellpose nuclei: raw DAPI | colored masks |
red outlines, so per-nucleus segmentation quality is visible at full pixel scale."""
import argparse, os
import numpy as np
import nd2
from PIL import Image
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--nd2", required=True)
ap.add_argument("--masks", required=True)
ap.add_argument("--nuclei-ch", type=int, default=2)
ap.add_argument("--y0", type=int, default=1500)
ap.add_argument("--x0", type=int, default=1500)
ap.add_argument("--size", type=int, default=600)
ap.add_argument("--out", required=True)
a = ap.parse_args()

with nd2.ND2File(a.nd2) as x:
    dapi = x.asarray()[a.nuclei_ch].astype(np.float32)
masks = np.load(a.masks)
y0, x0, s = a.y0, a.x0, a.size
dc = dapi[y0:y0+s, x0:x0+s]
mc = masks[y0:y0+s, x0:x0+s]

lo, hi = np.percentile(dc, 1), np.percentile(dc, 99.5)
raw = np.clip((dc - lo) / (hi - lo), 0, 1)
over = label2rgb(mc, image=raw, bg_label=0, alpha=0.5, image_alpha=1)
outl = np.stack([raw, raw, raw], -1)
outl[find_boundaries(mc, mode="outer")] = [1, 0.2, 0.2]

n = len(np.unique(mc)) - (1 if 0 in mc else 0)
fig, ax = plt.subplots(1, 3, figsize=(21, 7))
ax[0].imshow(raw, cmap="gray"); ax[0].set_title("raw DAPI", fontsize=15)
ax[1].imshow(over); ax[1].set_title(f"colored masks ({n} nuclei in crop)", fontsize=15)
ax[2].imshow(outl); ax[2].set_title("red outlines on raw", fontsize=15)
for x in ax:
    x.axis("off")
fig.suptitle(f"{os.path.basename(a.nd2)}  —  crop [{y0}:{y0+s}, {x0}:{x0+s}]",
             fontsize=13)
fig.tight_layout()
fig.savefig(a.out, dpi=110, bbox_inches="tight", facecolor="white")
print(f"crop nuclei={n}  saved -> {a.out}")
