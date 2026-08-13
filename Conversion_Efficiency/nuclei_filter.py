"""Nucleus AREA boundary: keep only Cellpose nuclei with 50 <= area <= 500 um^2.

Removes sub-nuclear debris / bleed-through (too small) and merged doublets or
giant artefacts (too large). Reports the area distribution + how many nuclei each
bound removes, saves an area histogram and a kept/removed overlay, and writes the
per-well filtered totals.  pixel = 0.6493 um  ->  1 px = 0.4216 um^2.
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UM2 = 0.6493 ** 2           # um^2 per pixel
NUC_DIR = "plate23_nuclei"


def areas_um2(nuc):
    a = np.bincount(nuc.ravel()).astype(np.float64)
    a[0] = 0                                    # background
    return a * UM2                              # index = label id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amin", type=float, default=50.0)
    ap.add_argument("--amax", type=float, default=500.0)
    ap.add_argument("--outdir", default="plate23_nuclei")
    a = ap.parse_args()

    wells = sorted(f.replace("_masks.npy", "") for f in os.listdir(NUC_DIR)
                   if f.endswith("_masks.npy"))
    pooled, rows, totals = [], {}, {"raw": 0, "keep": 0}
    for w in wells:
        nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
        ar = areas_um2(nuc)[1:]                 # drop background
        ar = ar[ar > 0]
        keep = (ar >= a.amin) & (ar <= a.amax)
        small = ar < a.amin
        large = ar > a.amax
        rows[w] = {"raw": int(ar.size), "keep": int(keep.sum()),
                   "too_small": int(small.sum()), "too_large": int(large.sum()),
                   "median_um2": round(float(np.median(ar)), 1)}
        totals["raw"] += ar.size; totals["keep"] += int(keep.sum())
        pooled.append(ar)
        print(f"{w:26s} raw={ar.size:5d}  keep={int(keep.sum()):5d}  "
              f"small(<{a.amin:.0f})={int(small.sum()):4d}  "
              f"large(>{a.amax:.0f})={int(large.sum()):3d}  "
              f"median={np.median(ar):5.1f}um2")
    allA = np.concatenate(pooled)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].hist(allA, bins=np.linspace(0, 800, 81), color="#c7d2fe",
               edgecolor="#6366f1")
    ax[0].axvspan(a.amin, a.amax, color="#16a34a", alpha=0.12)
    ax[0].axvline(a.amin, color="#16a34a", ls="--", label=f"{a.amin:.0f} um2")
    ax[0].axvline(a.amax, color="#ef4444", ls="--", label=f"{a.amax:.0f} um2")
    ax[0].set_xlabel("nucleus area (um2)"); ax[0].set_ylabel("number of nuclei")
    ax[0].set_title(f"Pooled nucleus area  (N={allA.size:,})")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    kept = ((allA >= a.amin) & (allA <= a.amax)).sum()
    ax[1].bar(["raw", f"kept\n[{a.amin:.0f},{a.amax:.0f}]",
               f"< {a.amin:.0f}", f"> {a.amax:.0f}"],
              [allA.size, kept, (allA < a.amin).sum(), (allA > a.amax).sum()],
              color=["#6366f1", "#16a34a", "#f59e0b", "#ef4444"])
    for i, v in enumerate([allA.size, kept, int((allA < a.amin).sum()),
                           int((allA > a.amax).sum())]):
        ax[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
    ax[1].set_ylabel("nuclei"); ax[1].set_title("Area-filter impact (plate)")
    ax[1].grid(axis="y", alpha=0.3)
    fig.suptitle(f"Nucleus area boundary {a.amin:.0f}-{a.amax:.0f} um2 — Plate 23",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(a.outdir, "nuclei_area_filter.png"), dpi=120,
                bbox_inches="tight", facecolor="white")

    with open(os.path.join(a.outdir, "nuclei_area_filter.json"), "w") as fh:
        json.dump({"amin": a.amin, "amax": a.amax, "per_well": rows,
                   "plate": totals}, fh, indent=2)
    print("-" * 60)
    print(f"PLATE raw={totals['raw']:,}  kept[{a.amin:.0f},{a.amax:.0f}]="
          f"{totals['keep']:,}  removed={totals['raw']-totals['keep']:,} "
          f"({100*(totals['raw']-totals['keep'])/totals['raw']:.1f}%)")


if __name__ == "__main__":
    main()
