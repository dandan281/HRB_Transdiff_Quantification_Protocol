"""Visualise the TRUE per-cell conversion result so it can be eyeballed.

Operating point is PLATE-GLOBAL and DATA-DRIVEN: one threshold (Otsu on the pooled
per-cell Desmin distribution) applied identically to every well. No per-well tuning
and no calibration to any expected value -- within a plate the artifacts are assumed
constant, so a single shared threshold is valid and between-well differences are
biology. (An earlier version anchored the threshold so the control read exactly 20%;
that made the control an input rather than a measurement and is NOT used here.)

Two outputs:
  1. per-well SPATIAL overlay -- every nucleus painted by its Desmin call
     (magenta = Desmin+, cyan = negative) over the dim Desmin channel. Shows WHERE
     the positive cells are: magenta should sit on/along Desmin, cyan in bare areas.
  2. plate summary bar chart -- conversion% per well with fold labels.

Reuses cached top-hat images, so it is fast.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe NEW_Quantif/visualize_final.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from percell_desmin import ring_intensity, CACHE, NUC_DIR, AMIN, AMAX, CTRL  # noqa
from real_fusion import UM, UM2  # noqa: E402

RING_UM = 10.0
ORDER = ["23_B02_ctrl", "33_C09_br223_trka", "29_C05_br223_egfrc",
         "19_B06_act104_trka", "32_C08_br223_igf1r", "22_B03_act104_egfrc"]


def classify(nuc, dbs, ring_px, thr):
    """Return (pos_label_bool, valid_label_bool) indexed by nucleus label."""
    area = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2
    valid = (area >= AMIN) & (area <= AMAX)
    valid[0] = False
    mean, cnt = ring_intensity(nuc, dbs, ring_px)
    pos = (mean > thr) & valid[:mean.size] & (cnt > 0)
    return pos, valid[:mean.size]


def main():
    ring_px = max(1, int(round(RING_UM / UM)))
    # PLATE-GLOBAL data-driven threshold: Otsu on the pooled per-cell distribution
    # (log scale). Same value for every well; no anchor, no per-well tuning.
    cache_vals = os.path.join(HERE, f"percell_values_r{RING_UM}.npz")
    z = np.load(cache_vals)
    pooled = np.concatenate([z[w] for w in ORDER])
    thr = float(10 ** threshold_otsu(np.log10(np.maximum(pooled, 1.0))))
    print(f"ring {RING_UM} um ({ring_px} px), POOLED-Otsu threshold "
          f"{thr:.1f} raw units (data-driven, uniform across wells)\n")

    summary = {}
    for w in ORDER:
        dbs = np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
        nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
        pos, valid = classify(nuc, dbs, ring_px, thr)
        npos, nval = int(pos.sum()), int(valid.sum())
        summary[w] = {"desmin_pos": npos, "valid": nval,
                      "conversion_pct": round(100 * npos / nval, 2),
                      "fold": None}

        pos_pix = pos[nuc]                      # per-pixel via label indexing
        neg_pix = valid[nuc] & (nuc > 0) & ~pos_pix
        rgb = np.zeros((*nuc.shape, 3), np.float32)
        d = dbs / (np.percentile(dbs, 99.5) + 1e-6)
        rgb[..., 1] = np.clip(0.6 * d, 0, 0.6)      # Desmin channel, dim green
        rgb[pos_pix] = [1.0, 0.1, 0.9]              # magenta = Desmin+
        rgb[neg_pix, 2] = 1.0; rgb[neg_pix, 0] = 0.1  # cyan = negative
        im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        im.thumbnail((1700, 1700))
        im.save(os.path.join(HERE, f"{w}_percell_classified.png"))
        print(f"  {w:<24} Desmin+={npos:5,}/{nval:5,} = "
              f"{summary[w]['conversion_pct']:5.1f}%  -> overlay saved")

    base = summary[CTRL]["conversion_pct"]
    for w in ORDER:
        summary[w]["fold"] = round(summary[w]["conversion_pct"] / base, 2)

    # ---- summary bar chart
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [w.split("_", 1)[1] for w in ORDER]
    convs = [summary[w]["conversion_pct"] for w in ORDER]
    colors = ["#64748b"] + ["#db2777"] * (len(ORDER) - 1)
    bars = ax.bar(labels, convs, color=colors, edgecolor="#1e293b")
    for b, w in zip(bars, ORDER):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{summary[w]['conversion_pct']:.1f}%\n{summary[w]['fold']:.2f}x",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(base, color="#64748b", ls="--", lw=1,
               label=f"control (data-driven) = {base:.1f}%")
    ax.set_ylabel("conversion efficiency (% nuclei Desmin+)")
    ax.set_title("PLATE_23 conversion efficiency — per-cell Desmin+ "
                 f"(ring {RING_UM:.0f} um, pooled-Otsu threshold {thr:.0f}, "
                 "uniform across wells)")
    ax.set_ylim(0, max(convs) * 1.25)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "conversion_summary_bar.png"), dpi=120,
                bbox_inches="tight", facecolor="white")

    with open(os.path.join(HERE, "visualize_final.json"), "w") as fh:
        json.dump({"ring_um": RING_UM, "threshold_raw": thr,
                   "threshold_method": "pooled_otsu_log_uniform",
                   "per_well": summary}, fh, indent=2)
    print("\n-> 6 spatial overlays + conversion_summary_bar.png")


if __name__ == "__main__":
    main()
