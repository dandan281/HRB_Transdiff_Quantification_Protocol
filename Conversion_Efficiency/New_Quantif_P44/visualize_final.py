"""Visualise the TRUE per-cell conversion result for PLATE 44.

Same artifacts and same method as `New_Quantif_P32/visualize_final.py` -- per-well
spatial overlays, a conversion summary bar chart, a per-cell value cache and a
JSON table -- with two differences forced by this plate:

* geometry comes from `p44_layout` (1.724571 um/px, DAPI=ch0), never from
  `real_fusion.UM`, which is the PLATE_2x constant;
* PLATE 44 has REPLICATE wells (n=3, n=2 for the TNFalpha panel), so the summary
  bar is per CONDITION with SEM error bars and individual wells shown. P32 had
  one well per condition, so its bar chart could be per-well.

Operating point PLATE-GLOBAL, DATA-DRIVEN: one Otsu threshold on the pooled
per-cell distribution, applied identically to every well.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/visualize_final.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    AMAX_UM2 as AMAX, AMIN_UM2 as AMIN, CONDITION_ORDER, CONTROL_CONDITION,
    RING_UM, TECHNICAL_FAILURES, UM, UM2, condition_of, well_id, wells)
from percell_desmin import CACHE, NUC_DIR, ring_intensity  # noqa: E402

VALUES_CACHE = os.path.join(HERE, f"percell_values_r{RING_UM}.npz")


def classify(nuc, dbs, ring_px, thr):
    area = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2
    valid = (area >= AMIN) & (area <= AMAX)
    valid[0] = False
    mean, cnt = ring_intensity(nuc, dbs, ring_px)
    pos = (mean > thr) & valid[:mean.size] & (cnt > 0)
    return pos, valid[:mean.size] & (cnt > 0)


def ring_values_for(w, ring_px):
    dbs = np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
    nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
    area = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2
    valid = (area >= AMIN) & (area <= AMAX)
    valid[0] = False
    mean, cnt = ring_intensity(nuc, dbs, ring_px)
    return mean[valid[:mean.size] & (cnt > 0)]


def main() -> int:
    ring_px = max(1, int(round(RING_UM / UM)))
    order = wells()

    if os.path.exists(VALUES_CACHE):
        z = np.load(VALUES_CACHE)
        valdict = {w: z[w] for w in order}
        print(f"loaded per-cell values from {os.path.basename(VALUES_CACHE)}")
    else:
        valdict = {w: ring_values_for(w, ring_px) for w in order}
        np.savez_compressed(VALUES_CACHE, **valdict)
        print(f"wrote {os.path.basename(VALUES_CACHE)}")

    pooled = np.concatenate([valdict[w] for w in order])
    thr = float(10 ** threshold_otsu(np.log10(np.maximum(pooled, 1.0))))
    print(f"ring {RING_UM} um ({ring_px} px at {UM} um/px), POOLED-Otsu "
          f"threshold {thr:.1f} raw units (data-driven, uniform across "
          f"{len(order)} wells)\n")

    summary = {}
    for i, w in enumerate(order, 1):
        dbs = np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
        nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
        pos, valid = classify(nuc, dbs, ring_px, thr)
        npos, nval = int(pos.sum()), int(valid.sum())
        summary[w] = {"well_id": well_id(w), "condition": condition_of(w),
                      "desmin_pos": npos, "valid": nval,
                      "conversion_pct": round(100 * npos / nval, 2) if nval else 0.0,
                      "technical_failure": well_id(w) in TECHNICAL_FAILURES}

        pos_pix = pos[nuc]
        neg_pix = valid[nuc] & (nuc > 0) & ~pos_pix
        rgb = np.zeros((*nuc.shape, 3), np.float32)
        d = dbs / (np.percentile(dbs, 99.5) + 1e-6)
        rgb[..., 1] = np.clip(0.6 * d, 0, 0.6)          # green = Desmin
        rgb[pos_pix] = [1.0, 0.1, 0.9]                  # magenta = Desmin+ nucleus
        rgb[neg_pix, 2] = 1.0                           # blue = Desmin- nucleus
        rgb[neg_pix, 0] = 0.1
        im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        im.thumbnail((1700, 1700))
        im.save(os.path.join(HERE, f"{w}_percell_classified.png"))
        print(f"[{i:>2}/{len(order)}] {w:<10} {summary[w]['condition']:<14} "
              f"Desmin+={npos:6,}/{nval:6,} = "
              f"{summary[w]['conversion_pct']:5.1f}%  -> overlay saved")

    # ---- condition means (well = replicate unit; failures excluded) ---------
    by_cond = {c: [] for c in CONDITION_ORDER}
    for w in order:
        if summary[w]["technical_failure"]:
            continue
        by_cond[summary[w]["condition"]].append(summary[w]["conversion_pct"])
    cond = {}
    for c in CONDITION_ORDER:
        v = np.array(by_cond[c], dtype=float)
        cond[c] = {"n": int(v.size), "mean_pct": round(float(v.mean()), 2),
                   "sem_pct": round(float(v.std(ddof=1) / np.sqrt(v.size)), 2)
                              if v.size > 1 else None,
                   "values_pct": [round(x, 2) for x in v]}
    base = cond[CONTROL_CONDITION]["mean_pct"]
    for c in CONDITION_ORDER:
        cond[c]["fold"] = round(cond[c]["mean_pct"] / base, 2)
    for w in order:
        summary[w]["fold_vs_control_mean"] = round(
            summary[w]["conversion_pct"] / base, 2)

    # ---- summary bar chart: control first, treated ascending (P32 form) -----
    treated = sorted([c for c in CONDITION_ORDER if c != CONTROL_CONDITION],
                     key=lambda c: cond[c]["mean_pct"])
    plot_order = [CONTROL_CONDITION] + treated
    means = [cond[c]["mean_pct"] for c in plot_order]
    sems = [cond[c]["sem_pct"] or 0.0 for c in plot_order]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#64748b"] + ["#db2777"] * (len(plot_order) - 1)
    bars = ax.bar(plot_order, means, color=colors, edgecolor="#1e293b",
                  zorder=2)
    ax.errorbar(range(len(plot_order)), means, yerr=sems, fmt="none",
                ecolor="#1e293b", elinewidth=1.3, capsize=4, zorder=3)
    rng = np.random.default_rng(20260813)
    for i, c in enumerate(plot_order):
        v = np.array(cond[c]["values_pct"], dtype=float)
        ax.scatter(i + rng.uniform(-0.12, 0.12, v.size), v, s=26, zorder=4,
                   color="#0f172a", alpha=0.7, edgecolor="white", linewidth=0.9)
    # Label above whichever is higher, the SEM cap or the topmost replicate dot.
    # P32's fixed +0.4 offset assumed neither existed and collides with both.
    tops = []
    for b, c in zip(bars, plot_order):
        top = max(b.get_height() + (cond[c]["sem_pct"] or 0.0),
                  max(cond[c]["values_pct"]))
        tops.append(top)
        ax.text(b.get_x() + b.get_width() / 2, top + 0.45,
                f"{cond[c]['mean_pct']:.1f}%\n{cond[c]['fold']:.2f}x",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.axhline(base, color="#64748b", ls="--", lw=1,
               label=f"control ({CONTROL_CONDITION}, data-driven) = {base:.1f}%")
    ax.set_ylabel("conversion efficiency (% nuclei Desmin+)")
    ax.set_title(f"PLATE 44 conversion efficiency — per-cell Desmin+ "
                 f"(ring {RING_UM:.0f} um, pooled-Otsu threshold {thr:.0f}, "
                 f"uniform across {len(order)} wells)\n"
                 f"condition means ± SEM over replicate wells; dots = wells")
    ax.set_ylim(0, max(tops) * 1.30)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "conversion_summary_bar.png"), dpi=120,
                bbox_inches="tight", facecolor="white")

    with open(os.path.join(HERE, "visualize_final.json"), "w") as fh:
        json.dump({"plate": "PLATE_44", "pixel_um": UM, "ring_um": RING_UM,
                   "ring_px": ring_px, "threshold_raw": thr,
                   "threshold_method": "pooled_otsu_log_uniform",
                   "control_condition": CONTROL_CONDITION,
                   "replicate_unit": "well",
                   "technical_failures_excluded_from_condition_means":
                       sorted(TECHNICAL_FAILURES),
                   "per_well": summary, "per_condition": cond}, fh, indent=2)
    print(f"\n-> {len(order)} spatial overlays + conversion_summary_bar.png "
          f"+ visualize_final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
