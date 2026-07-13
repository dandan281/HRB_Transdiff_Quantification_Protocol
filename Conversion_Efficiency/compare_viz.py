"""Side-by-side: raw DAPI | old watershed (over-splits) | Cellpose-SAM (one/nucleus)."""
import nd2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import gaussian, threshold_li
from skimage.morphology import remove_small_objects
from skimage.color import label2rgb
import conversion_efficiency as ce

Y0, X0, S = 1400, 1400, 800
with nd2.ND2File("../Q_PLATES/Q_Plates/PLATE_23/32_C08_br223_igf1r.nd2") as x:
    dapi = x.asarray()[2].astype(np.float32)
crop = dapi[Y0:Y0+S, X0:X0+S]

# raw stretched
lo, hi = np.percentile(crop, 1), np.percentile(crop, 99.5)
raw = np.clip((crop-lo)/(hi-lo), 0, 1)

# watershed labels (same as pipeline)
um = 0.6493
ff, _ = ce.flatfield_correct(crop)
sm = gaussian(ff, 1.0, preserve_range=True)
M1 = remove_small_objects(sm > threshold_li(sm), min_size=int(round(15.0/(um*um))))
ws = ce.segment_nuclei(sm, M1, int(round(20.0/(um*um))), max(3, int(round(6.0/um))))
n_ws = int(ws.max())

# cellpose labels
cp = np.load("cp_c08_crop/cellpose_masks.npy")
n_cp = int(cp.max())

fig, ax = plt.subplots(1, 3, figsize=(24, 8))
ax[0].imshow(raw, cmap="gray"); ax[0].set_title("raw DAPI crop", fontsize=16)
ax[1].imshow(label2rgb(ws, image=raw, bg_label=0, alpha=0.45, image_alpha=1))
ax[1].set_title(f"OLD watershed  —  {n_ws} nuclei (over-splits)", fontsize=16)
ax[2].imshow(label2rgb(cp, image=raw, bg_label=0, alpha=0.45, image_alpha=1))
ax[2].set_title(f"Cellpose-SAM  —  {n_cp} nuclei (one per nucleus)", fontsize=16)
for a in ax:
    a.axis("off")
fig.tight_layout()
fig.savefig("cp_c08_crop/compare_watershed_vs_cellpose.png", dpi=110,
            bbox_inches="tight", facecolor="white")
print(f"watershed {n_ws}  vs  cellpose {n_cp}  -> saved compare figure")
