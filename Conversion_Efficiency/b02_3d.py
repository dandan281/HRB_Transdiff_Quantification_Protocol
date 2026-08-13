"""B02 (control) only: 3D joint distribution of Desmin+ nuclei
x = nuclear area (um2), y = host myotube length (um), z = number of nuclei."""
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from nuclei_3d import well_pairs, AMIN, AMAX

W = "23_B02_ctrl"
X, Y = well_pairs(W)
print(f"{W}: {X.size} Desmin+ nuclei  median area={np.median(X):.1f}um2  "
      f"median host-myotube={np.median(Y):.1f}um  max myotube={Y.max():.1f}um")

ymax = np.percentile(Y, 99)
xedges = np.linspace(AMIN, AMAX, 19)
yedges = np.linspace(0, ymax, 21)
H, xe, ye = np.histogram2d(X, Y, bins=[xedges, yedges])
xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
Xg, Yg = np.meshgrid(xc, yc, indexing="ij")
Hs = gaussian_filter(H, 0.7)

fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(1, 2, 1, projection="3d")
surf = ax.plot_surface(Xg, Yg, Hs, cmap="magma", edgecolor="none",
                       rstride=1, cstride=1, antialiased=True)
ax.set_xlabel("\nnuclear area (um2)"); ax.set_ylabel("\nmyotube length (um)")
ax.set_zlabel("number of nuclei")
ax.set_title("B02 control — area x host-myotube length x count")
ax.view_init(elev=32, azim=-125)
fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label="nuclei")

ax2 = fig.add_subplot(1, 2, 2)
im = ax2.pcolormesh(xe, ye, H.T, cmap="magma", shading="flat")
ax2.set_xlabel("nuclear area (um2)"); ax2.set_ylabel("myotube length (um)")
ax2.set_title(f"Same as 2D heatmap  (N={X.size:,} Desmin+ nuclei)")
ax2.axhline(np.median(Y), color="w", ls="--", lw=1,
            label=f"median host-myotube = {np.median(Y):.0f} um")
ax2.legend(loc="upper right")
fig.colorbar(im, ax=ax2, label="number of nuclei")

fig.suptitle("Plate 23  B02 (control) — Desmin+ nuclei joint distribution",
             fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "plate23_real_fusion/B02_nuclei_area_vs_myotube_length_3d.png"
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("saved ->", out)
