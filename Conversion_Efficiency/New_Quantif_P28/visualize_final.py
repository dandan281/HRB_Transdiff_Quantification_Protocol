"""Visualise the TRUE per-cell conversion result for PLATE_28 so it can be eyeballed.
Identical code/parameters to New_Quantif_P26/visualize_final.py; only the well list
(ORDER) and plate label differ. Load-or-compute fallback builds the ring-value cache
if absent.

Operating point PLATE-GLOBAL, DATA-DRIVEN: one Otsu threshold on the pooled per-cell
distribution, applied identically to every well. No per-well tuning, no calibration.

Outputs: per-well SPATIAL overlays (magenta = Desmin+, cyan = negative, over green
Desmin) + plate summary bar chart.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P28/visualize_final.py
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
# control first; treated ordered ascending by conversion after the first run
ORDER = ["23_B02_ctrl", "17_B08_BMPR2_HER2mb", "21_B04_br223_egfrc",
         "56_E08_br223_egfrc", "58_E10_br223_igf1r"]


def classify(nuc, dbs, ring_px, thr):
    """Return (pos_label_bool, valid_label_bool) indexed by nucleus label."""
    area = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2
    valid = (area >= AMIN) & (area <= AMAX)
    valid[0] = False
    mean, cnt = ring_intensity(nuc, dbs, ring_px)
    pos = (mean > thr) & valid[:mean.size] & (cnt > 0)
    return pos, valid[:mean.size]


def ring_values_for(w, ring_px):
    """Valid per-cell ring intensities for one well (for the pooled threshold)."""
    dbs = np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
    nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
    area = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2
    valid = (area >= AMIN) & (area <= AMAX); valid[0] = False
    mean, cnt = ring_intensity(nuc, dbs, ring_px)
    return mean[valid[:mean.size] & (cnt > 0)]


def main():
    ring_px = max(1, int(round(RING_UM / UM)))
    cache_vals = os.path.join(HERE, f"percell_values_r{RING_UM}.npz")
    if os.path.exists(cache_vals):
        z = np.load(cache_vals)
        valdict = {w: z[w] for w in ORDER}
    else:
        valdict = {w: ring_values_for(w, ring_px) for w in ORDER}
        np.savez_compressed(cache_vals, **valdict)
    pooled = np.concatenate([valdict[w] for w in ORDER])
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

    # ---- summary bar chart (control first, treated ascending by conversion)
    treated = sorted([w for w in ORDER if w != CTRL],
                     key=lambda w: summary[w]["conversion_pct"])
    plot_order = [CTRL] + treated
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [w.split("_", 1)[1] for w in plot_order]
    convs = [summary[w]["conversion_pct"] for w in plot_order]
    colors = ["#64748b"] + ["#db2777"] * (len(plot_order) - 1)
    bars = ax.bar(labels, convs, color=colors, edgecolor="#1e293b")
    for b, w in zip(bars, plot_order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{summary[w]['conversion_pct']:.1f}%\n{summary[w]['fold']:.2f}x",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(base, color="#64748b", ls="--", lw=1,
               label=f"control (data-driven) = {base:.1f}%")
    ax.set_ylabel("conversion efficiency (% nuclei Desmin+)")
    ax.set_title("PLATE_28 conversion efficiency — per-cell Desmin+ "
                 f"(ring {RING_UM:.0f} um, pooled-Otsu threshold {thr:.0f}, "
                 "uniform across wells)")
    ax.set_ylim(0, max(convs) * 1.25)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "conversion_summary_bar.png"), dpi=120,
                bbox_inches="tight", facecolor="white")

    with open(os.path.join(HERE, "visualize_final.json"), "w") as fh:
        json.dump({"plate": "PLATE_28", "ring_um": RING_UM, "threshold_raw": thr,
                   "threshold_method": "pooled_otsu_log_uniform",
                   "per_well": summary}, fh, indent=2)
    print(f"\n-> {len(ORDER)} spatial overlays + conversion_summary_bar.png")


if __name__ == "__main__":
    main()
