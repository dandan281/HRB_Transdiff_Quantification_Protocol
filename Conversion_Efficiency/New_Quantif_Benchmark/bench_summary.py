"""Benchmark summary: final figures + density-confound + area-band sensitivity.

Outputs (in summary/):
  conversion_per_image.png   per-image conversion at both overlap fractions
  sweep_stability.png        the three plateau sweeps (cellprob, k, k_low)
  density_confound.png       conversion vs nucleus count per image + r values
  SUMMARY.md                 all numbers and parameters in one place

Run:  cpenv/Scripts/python.exe bench_summary.py
"""
from __future__ import annotations
import os, json, glob
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "summary")
UM = 0.521
UM2 = UM * UM

INK = "#1f2937"; MUT = "#6b7280"; GRID = "#e5e7eb"
C25 = "#f59e0b"          # >=25% overlap (amber)  — validated pair
C50 = "#c026d3"          # >=50% overlap (magenta)
CPT = "#3b82f6"          # sweep-curve blue
STAR = "#ef4444"


def style(ax):
    ax.grid(alpha=0.35, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def nuclei_inside(nuc, territory, amin_um2, amax_um2, frac):
    flat = nuc.ravel()
    area = np.bincount(flat).astype(np.float64)
    inside = np.bincount(flat, weights=territory.ravel().astype(np.float64),
                         minlength=area.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = np.where(area > 0, inside / area, 0.0)
    area_um2 = area * UM2
    valid = (area_um2 >= amin_um2) & (area_um2 <= amax_um2)
    valid[0] = False
    return int(((fr >= frac) & valid).sum()), int(valid.sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    fus = json.load(open(os.path.join(HERE, "fusion", "fusion_results.json")))
    nuc_sw = json.load(open(os.path.join(HERE, "nuclei", "nuclei_sweep.json")))
    myo = json.load(open(os.path.join(HERE, "myotube2", "myotube2_results.json")))

    stems = sorted(fus["per_image"], key=int)
    g0 = {s: fus["per_image"][s]["gate0um"] for s in stems}
    tot = np.array([g0[s]["total_valid"] for s in stems])
    p25 = np.array([g0[s]["overlap_25pct"]["pct"] for s in stems])
    p50 = np.array([g0[s]["overlap_50pct"]["pct"] for s in stems])
    pooled = fus["pooled"]["gate0um"]

    # ---- density confound ----
    r25 = pearsonr(tot, p25); s25 = spearmanr(tot, p25)
    print(f"density confound (ov25 vs nuclei/image): Pearson r={r25[0]:+.2f} "
          f"(p={r25[1]:.3f})  Spearman rho={s25[0]:+.2f}")

    # ---- area-band sensitivity [90,500] at gate 0 ----
    sens = {}
    for band in ((50.0, 500.0), (90.0, 500.0)):
        acc = {0.25: 0, 0.5: 0, "tot": 0}
        for s in stems:
            nuc = np.load(os.path.join(HERE, "nuclei", f"{s}_masks.npy"))
            terr = binary_fill_holes(
                np.load(os.path.join(HERE, "myotube2", f"{s}_myotube_mask.npy")))
            for f in (0.25, 0.5):
                nin, ntot = nuclei_inside(nuc, terr, band[0], band[1], f)
                acc[f] += nin
                if f == 0.25:
                    acc["tot"] += ntot
        sens[band] = acc
        print(f"area band {band}: total={acc['tot']}  "
              f"ov25={acc[0.25]} ({100*acc[0.25]/acc['tot']:.2f}%)  "
              f"ov50={acc[0.5]} ({100*acc[0.5]/acc['tot']:.2f}%)")

    # ---- fig 1: per-image conversion ----
    x = np.arange(len(stems))
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(x - 0.21, p25, width=0.42, color=C25, label="≥25% overlap")
    ax.bar(x + 0.21, p50, width=0.42, color=C50, label="≥50% overlap")
    ax.axhline(pooled["overlap_25pct"]["pct"], color=C25, lw=1.2, ls="--")
    ax.axhline(pooled["overlap_50pct"]["pct"], color=C50, lw=1.2, ls="--")
    ax.text(len(stems) - 0.4, pooled["overlap_25pct"]["pct"] + 0.7,
            f"pooled {pooled['overlap_25pct']['pct']:.1f}%", color=C25,
            ha="right", fontsize=9, fontweight="bold")
    ax.text(len(stems) - 0.4, pooled["overlap_50pct"]["pct"] - 2.2,
            f"pooled {pooled['overlap_50pct']['pct']:.1f}%", color=C50,
            ha="right", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(stems)
    ax.set_xlabel("image", color=INK)
    ax.set_ylabel("nuclei inside Desmin+ territory (%)", color=INK)
    ax.set_title("Benchmark conversion efficiency per image "
                 "(Desmin+ nuclei / valid nuclei)", color=INK)
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "conversion_per_image.png"), dpi=120,
                facecolor="white")
    plt.close(fig)

    # ---- fig 2: the three plateau sweeps ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    cps = [float(c) for c in nuc_sw["pooled_totals"]]
    cnt = list(nuc_sw["pooled_totals"].values())
    axes[0].plot(cps, cnt, "-o", color=CPT)
    op = nuc_sw["operating_cellprob"]
    axes[0].plot(op, nuc_sw["pooled_totals"][str(op)], "*", ms=18, color=STAR)
    axes[0].set_xlabel("cellprob_threshold"); axes[0].set_ylabel("total nuclei")
    axes[0].set_title(f"Nuclei sweep — plateau cp={op:+.0f}")

    ks = [float(k) for k in myo["pooled_totals"]]
    kc = list(myo["pooled_totals"].values())
    axes[1].plot(ks, kc, "-o", color=CPT)
    axes[1].plot(myo["operating_k"],
                 myo["pooled_totals"][f"{myo['operating_k']:g}"], "*",
                 ms=18, color=STAR)
    axes[1].set_xlabel("seed threshold k (×σ)")
    axes[1].set_ylabel("total Desmin+ objects")
    axes[1].set_title(f"Myotube seed sweep — plateau k={myo['operating_k']:g}")

    kls = [float(k) for k in myo["hyst_totals"]]
    klc = list(myo["hyst_totals"].values())
    axes[2].plot(kls, klc, "-o", color=CPT)
    axes[2].plot(myo["operating_k_low"],
                 myo["hyst_totals"][f"{myo['operating_k_low']:g}"], "*",
                 ms=18, color=STAR)
    axes[2].set_xlabel("hysteresis low k (×σ)")
    axes[2].set_ylabel("total Desmin+ objects")
    axes[2].set_title(f"Hysteresis sweep — plateau k_low={myo['operating_k_low']:g}")
    for ax in axes:
        style(ax)
    fig.suptitle("Threshold plateau sweeps (one shared parameter set for all "
                 "25 images)", color=INK, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "sweep_stability.png"), dpi=120,
                facecolor="white")
    plt.close(fig)

    # ---- fig 3: density confound ----
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(tot, p25, s=42, color=C25, label="≥25% overlap", zorder=3)
    ax.scatter(tot, p50, s=42, color=C50, label="≥50% overlap", zorder=3)
    for xi, yi, s in zip(tot, p25, stems):
        ax.annotate(s, (xi, yi), textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color=MUT)
    ax.set_xlabel("valid nuclei per image (density)", color=INK)
    ax.set_ylabel("nuclei inside Desmin+ territory (%)", color=INK)
    ax.set_title(f"Density-confound check\nov25: Pearson r={r25[0]:+.2f}, "
                 f"Spearman ρ={s25[0]:+.2f}", color=INK)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "density_confound.png"), dpi=120,
                facecolor="white")
    plt.close(fig)

    # ---- SUMMARY.md ----
    g50 = fus["pooled"]["gate50um"]
    b50 = sens[(50.0, 500.0)]; b90 = sens[(90.0, 500.0)]
    lines = [
        "# Benchmark conversion efficiency — 25-image set",
        "",
        f"- Input: `Benchmark/1..25.tif`, 8-bit RGB ImageJ exports, 1040x1392;"
        f" green = Desmin, blue = DAPI, red empty.",
        f"- Pixel size **measured from the burned-in scale bar**: 50 um = 96 px"
        f" -> **{UM} um/px**.",
        "",
        "## Pipeline (one shared, data-driven parameter set for all images)",
        f"- Nuclei: Cellpose-SAM, cellprob plateau **cp="
        f"{nuc_sw['operating_cellprob']:+.0f}** (pooled sweep"
        f" {min(cnt)}-{max(cnt)} over cp -2..+2 — flat).",
        f"- Nucleus area filter: [50, 500] um^2 from the pooled histogram"
        f" (median 208 um^2; keeps 5,856/6,317 = 92.7%).",
        f"- Myotube territory: background-surface subtraction (8x8 masked-tile"
        f" median), shared sigma = {myo['sigma_shared']:.2f}, hysteresis"
        f" threshold {myo['threshold_abs_low']} -> {myo['threshold_abs_high']}"
        f" 8-bit units (plateaus k={myo['operating_k']:g},"
        f" k_low={myo['operating_k_low']:g}); min object 180 px; hole-filled.",
        "",
        "## Results (pooled over 25 images)",
        "",
        "| readout | >=25% overlap | >=50% overlap |",
        "|---|---|---|",
        f"| **All Desmin+ territory (primary)** | **{pooled['overlap_25pct']['inside']}"
        f" / {pooled['total_valid_nuclei']} = {pooled['overlap_25pct']['pct']:.1f}%**"
        f" | **{pooled['overlap_50pct']['inside']} / {pooled['total_valid_nuclei']}"
        f" = {pooled['overlap_50pct']['pct']:.1f}%** |",
        f"| Traced fibres >=50 um only | {g50['overlap_25pct']['inside']} ="
        f" {g50['overlap_25pct']['pct']:.1f}% | {g50['overlap_50pct']['inside']} ="
        f" {g50['overlap_50pct']['pct']:.1f}% |",
        f"| Area band [90,500] um^2 (sensitivity) | {b90[0.25]} /"
        f" {b90['tot']} = {100*b90[0.25]/b90['tot']:.1f}% | {b90[0.5]} ="
        f" {100*b90[0.5]/b90['tot']:.1f}% |",
        "",
        f"- Density confound (ov25 vs nuclei/image): Pearson r={r25[0]:+.2f}"
        f" (p={r25[1]:.3f}), Spearman rho={s25[0]:+.2f}.",
        f"- Per-image range: {p25.min():.1f}-{p25.max():.1f}% (ov25);"
        f" densest image {tot.max()} nuclei, sparsest {tot.min()}.",
        "",
        "## Files",
        "- `nuclei/` masks + sweep; `myotube2/` masks + labeled overlays;",
        "- `fusion/` per-image 3-class overlays + fusion_results.json;",
        "- `summary/` figures + this file.",
    ]
    with open(os.path.join(OUT, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("SUMMARY_DONE")


if __name__ == "__main__":
    main()
