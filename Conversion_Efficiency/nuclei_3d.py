"""3D joint distribution of Desmin+ nuclei (nuclei INSIDE myotubes).

For every nucleus that sits inside a myotube we record two numbers:
  x = nuclear area (um^2)                      [area boundary 50-500 applied]
  y = length of the MYOTUBE it lives in (um)   [traced whole-fibre length]
and the surface height is
  z = number of such nuclei in each (area, length) bin.

Each inside-nucleus is assigned to its host fibre by propagating every traced
fibre's ID out to its territory (nearest-skeleton), then taking the dominant
fibre ID under the nucleus. Reuses the fibre tracer from real_fusion.py.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from real_fusion import trace_fibres, UM, UM2, NUC_DIR, MYO_DIR

AMIN, AMAX = 50.0, 500.0        # nucleus area boundary um^2
FRAC = 0.5                      # >= this fraction inside = "in a myotube"


def well_pairs(w):
    """Return (area_um2, host_fibre_length_um) for each Desmin+ nucleus in well w."""
    myo = np.load(os.path.join(MYO_DIR, f"{w}_myotube_mask.npy"))
    nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
    skel, idx, fibres = trace_fibres(myo)
    if not fibres:
        return np.array([]), np.array([])

    fib_len = np.array([0.0] + [f[0] for f in fibres])          # id 1..N
    fid_skel = np.zeros(skel.shape, np.int32)
    for fid, (_, pix) in enumerate(fibres, start=1):
        fid_skel[pix[:, 0], pix[:, 1]] = fid
    fid_map = fid_skel[idx[0], idx[1]]                           # fibre id everywhere
    terr = binary_fill_holes((myo > 0) & (fid_map > 0))

    flat = nuc.ravel()
    area_px = np.bincount(flat).astype(np.float64)
    inside_px = np.bincount(flat, weights=terr.ravel().astype(np.float64),
                            minlength=area_px.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(area_px > 0, inside_px / area_px, 0.0)
    area_um2 = area_px * UM2
    valid = (area_um2 >= AMIN) & (area_um2 <= AMAX)
    valid[0] = False
    is_in = (frac >= FRAC) & valid

    inside_pix = is_in[nuc]                                      # bool image
    nl = nuc[inside_pix]; fi = fid_map[inside_pix]
    good = fi > 0
    df = pd.DataFrame({"nl": nl[good], "fi": fi[good]})
    dom = df.groupby("nl")["fi"].agg(lambda s: np.bincount(s).argmax())  # host fibre

    labels = dom.index.to_numpy()
    xs = area_um2[labels]
    ys = fib_len[dom.to_numpy()]
    return xs, ys


def main():
    wells = sorted(f.replace("_myotube_mask.npy", "") for f in os.listdir(MYO_DIR)
                   if f.endswith("_myotube_mask.npy"))
    X, Y = [], []
    for w in wells:
        xs, ys = well_pairs(w)
        X.append(xs); Y.append(ys)
        print(f"{w:26s} Desmin+ nuclei = {xs.size:5d}  "
              f"median area={np.median(xs):5.1f}um2  "
              f"median host-myotube={np.median(ys):6.1f}um")
    X = np.concatenate(X); Y = np.concatenate(Y)
    print(f"TOTAL Desmin+ nuclei plotted = {X.size:,}")

    # 2D histogram -> surface height = nuclei count
    ymax = np.percentile(Y, 99)
    xedges = np.linspace(AMIN, AMAX, 19)          # 25 um^2 bins
    yedges = np.linspace(0, ymax, 21)
    H, xe, ye = np.histogram2d(X, Y, bins=[xedges, yedges])
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    Xg, Yg = np.meshgrid(xc, yc, indexing="ij")
    Hs = gaussian_filter(H, 0.7)                  # gentle smoothing for the surface

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax.plot_surface(Xg, Yg, Hs, cmap="viridis", edgecolor="none",
                           rstride=1, cstride=1, antialiased=True)
    ax.set_xlabel("\nnuclear area (um2)")
    ax.set_ylabel("\nmyotube length (um)")
    ax.set_zlabel("number of nuclei")
    ax.set_title("Desmin+ nuclei — area x host-myotube length x count")
    ax.view_init(elev=32, azim=-125)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label="nuclei")

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.pcolormesh(xe, ye, H.T, cmap="viridis", shading="flat")
    ax2.set_xlabel("nuclear area (um2)"); ax2.set_ylabel("myotube length (um)")
    ax2.set_title(f"Same as 2D heatmap  (N={X.size:,} Desmin+ nuclei)")
    ax2.axhline(np.median(Y), color="w", ls="--", lw=1)
    fig.colorbar(im, ax=ax2, label="number of nuclei")

    fig.suptitle("Plate 23 — joint distribution of Desmin+ nuclei "
                 "(inside myotubes)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "plate23_real_fusion/nuclei_area_vs_myotube_length_3d.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    np.savez("plate23_real_fusion/nuclei_3d_data.npz", area_um2=X, myotube_um=Y)
    print("saved ->", out)


if __name__ == "__main__":
    main()
