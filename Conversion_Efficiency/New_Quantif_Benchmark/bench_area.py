"""Nucleus area distribution for the benchmark set — sets the size filter.

Pools Cellpose nucleus areas over all 25 images, prints the distribution stats
(px^2 and um^2 under the MEASURED 0.521 um/px — 50 um scale bar = 96 px),
plots the histogram with the
chosen band, and reports how many nuclei each bound removes. The band is chosen
from THIS pooled histogram (data-driven), then applied identically everywhere.

Run:  cpenv/Scripts/python.exe bench_area.py [--amin-um2 50 --amax-um2 500]
"""
from __future__ import annotations
import os, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NUC_DIR = os.path.join(HERE, "nuclei")
UM = 0.521                     # measured from the burned-in 50um scale bar
UM2 = UM * UM

INK = "#1f2937"; MUT = "#6b7280"; GRID = "#e5e7eb"
BAR = "#93c5fd"; EDGE = "#3b82f6"; OK = "#16a34a"; BAD = "#ef4444"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amin-um2", type=float, default=50.0)
    ap.add_argument("--amax-um2", type=float, default=500.0)
    a = ap.parse_args()

    pooled, per_image = [], {}
    for f in sorted(glob.glob(os.path.join(NUC_DIR, "*_masks.npy")),
                    key=lambda p: int(os.path.basename(p).split("_")[0])):
        stem = os.path.basename(f).replace("_masks.npy", "")
        nuc = np.load(f)
        ar = np.bincount(nuc.ravel()).astype(np.float64)[1:]
        ar = ar[ar > 0] * UM2
        keep = (ar >= a.amin_um2) & (ar <= a.amax_um2)
        per_image[stem] = {"raw": int(ar.size), "keep": int(keep.sum()),
                           "small": int((ar < a.amin_um2).sum()),
                           "large": int((ar > a.amax_um2).sum()),
                           "median_um2": round(float(np.median(ar)), 1)}
        pooled.append(ar)
    allA = np.concatenate(pooled)
    kept = (allA >= a.amin_um2) & (allA <= a.amax_um2)

    q = np.percentile(allA, [1, 5, 25, 50, 75, 95, 99])
    print(f"pooled nuclei = {allA.size}")
    print("area um2 percentiles [1,5,25,50,75,95,99]:",
          np.round(q, 1).tolist())
    print(f"median = {np.median(allA):.1f} um2 = {np.median(allA)/UM2:.0f} px")
    print(f"band [{a.amin_um2:.0f},{a.amax_um2:.0f}] um2 keeps {kept.sum()} "
          f"({100*kept.mean():.1f}%)  small={int((allA<a.amin_um2).sum())} "
          f"large={int((allA>a.amax_um2).sum())}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(allA, bins=np.linspace(0, 800, 81), color=BAR, edgecolor=EDGE,
            linewidth=0.4)
    ax.axvspan(a.amin_um2, a.amax_um2, color=OK, alpha=0.08)
    ax.axvline(a.amin_um2, color=OK, ls="--", lw=1.2,
               label=f"keep [{a.amin_um2:.0f}, {a.amax_um2:.0f}] µm²")
    ax.axvline(a.amax_um2, color=BAD, ls="--", lw=1.2)
    ax.set_xlabel("nucleus area (µm², at measured 0.521 µm/px)", color=INK)
    ax.set_ylabel("nuclei", color=INK)
    ax.set_title(f"Benchmark pooled nucleus area — N={allA.size:,}, "
                 f"median {np.median(allA):.0f} µm²", color=INK)
    ax.legend(frameon=False)
    ax.grid(alpha=0.35, color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(NUC_DIR, "nuclei_area_hist.png"), dpi=120,
                facecolor="white")

    with open(os.path.join(NUC_DIR, "nuclei_area_filter.json"), "w") as fh:
        json.dump({"um_per_px_assumed": UM,
                   "amin_um2": a.amin_um2, "amax_um2": a.amax_um2,
                   "pooled": {"n": int(allA.size),
                              "keep": int(kept.sum()),
                              "median_um2": round(float(np.median(allA)), 1),
                              "pctiles_um2": dict(zip(
                                  ["p1", "p5", "p25", "p50", "p75", "p95", "p99"],
                                  np.round(q, 1)))},
                   "per_image": per_image}, fh, indent=2)
    print("AREA_DONE")


if __name__ == "__main__":
    main()
