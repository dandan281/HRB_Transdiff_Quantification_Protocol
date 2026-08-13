"""Human-vs-machine comparison on the 25-image benchmark.

Sources: Investigator 1, Investigator 2 (manual counts), the lab's reference
Program (columns from the same spreadsheet), and this pipeline (Claude,
fusion/fusion_results.json — >=25% overlap primary, >=50% secondary).
NFI = myotube nuclei / total nuclei x 100 per image.

Outputs (summary/): nfi_comparison.png, totals_comparison.png,
agreement_scatter.png, comparison_stats.json, COMPARISON.md.

Run:  cpenv/Scripts/python.exe bench_compare.py
"""
from __future__ import annotations
import os, json, csv
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "summary")

INK = "#1f2937"; MUT = "#6b7280"; GRID = "#e5e7eb"
C = {"inv1": "#3b82f6", "inv2": "#0d9488", "prog": "#f59e0b",
     "cl25": "#c026d3"}
LBL = {"inv1": "Investigator 1", "inv2": "Investigator 2",
       "prog": "Program (lab)", "cl25": "Claude (≥25% ov)"}


def style(ax):
    ax.grid(alpha=0.35, color=GRID, axis="y")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(HERE, "human_data.csv"))))
    fus = json.load(open(os.path.join(HERE, "fusion", "fusion_results.json")))

    imgs = [r["image"] for r in rows]
    d = {}
    for k in ("inv1", "inv2", "prog"):
        d[f"{k}_tot"] = np.array([float(r[f"{k}_total"]) for r in rows])
        d[f"{k}_myo"] = np.array([float(r[f"{k}_myo"]) for r in rows])
        d[f"{k}_nfi"] = 100 * d[f"{k}_myo"] / d[f"{k}_tot"]
    g0 = {s: fus["per_image"][s]["gate0um"] for s in imgs}
    d["cl_tot"] = np.array([float(g0[s]["total_valid"]) for s in imgs])
    d["cl25_myo"] = np.array([float(g0[s]["overlap_25pct"]["inside"]) for s in imgs])
    d["cl50_myo"] = np.array([float(g0[s]["overlap_50pct"]["inside"]) for s in imgs])
    d["cl25_nfi"] = 100 * d["cl25_myo"] / d["cl_tot"]
    d["cl50_nfi"] = 100 * d["cl50_myo"] / d["cl_tot"]
    inv_mean = (d["inv1_nfi"] + d["inv2_nfi"]) / 2

    # ---- stats ----
    nfi = {"inv1": d["inv1_nfi"], "inv2": d["inv2_nfi"],
           "prog": d["prog_nfi"], "cl25": d["cl25_nfi"]}
    pairs = [("inv1", "inv2"), ("inv1", "prog"), ("inv2", "prog"),
             ("inv1", "cl25"), ("inv2", "cl25"), ("prog", "cl25")]
    pair_stats = {}
    for a, b in pairs:
        r = pearsonr(nfi[a], nfi[b]); s = spearmanr(nfi[a], nfi[b])
        bias = float(np.mean(nfi[b] - nfi[a]))
        pair_stats[f"{a}-{b}"] = {"pearson_r": round(r[0], 3),
                                  "spearman_rho": round(s[0], 3),
                                  "mean_bias_pp": round(bias, 1)}
        print(f"{LBL[a]:>18s} vs {LBL[b]:<18s} r={r[0]:.2f} rho={s[0]:.2f} "
              f"bias={bias:+.1f}pp")

    means = {k: float(np.mean(v)) for k, v in nfi.items()}
    means["cl50"] = float(np.mean(d["cl50_nfi"]))
    print("mean NFI:", {k: round(v, 1) for k, v in means.items()})

    dev = d["cl25_nfi"] - inv_mean
    order = np.argsort(dev)
    print("largest Claude deficits vs investigator mean:")
    flagged = []
    for i in order[:6]:
        print(f"  img {imgs[i]:>3s}: Claude {d['cl25_nfi'][i]:.1f}% vs "
              f"inv-mean {inv_mean[i]:.1f}%  ({dev[i]:+.1f}pp)")
        if dev[i] < -12:
            flagged.append(imgs[i])

    # ---- fig 1: NFI grouped bars ----
    x = np.arange(len(imgs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(16, 5.4))
    for i, k in enumerate(("inv1", "inv2", "prog", "cl25")):
        ax.bar(x + (i - 1.5) * w, nfi[k], width=w * 0.92, color=C[k],
               label=LBL[k])
    ax.set_xticks(x); ax.set_xticklabels(imgs)
    ax.set_xlabel("image", color=INK)
    ax.set_ylabel("NFI — nuclei in myotubes (%)", color=INK)
    ax.set_title("Benchmark NFI per image: two human investigators, lab "
                 "program, and this pipeline", color=INK)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nfi_comparison.png"), dpi=120,
                facecolor="white")
    plt.close(fig)

    # ---- fig 2: total nuclei grouped bars ----
    tot = {"inv1": d["inv1_tot"], "inv2": d["inv2_tot"],
           "prog": d["prog_tot"], "cl25": d["cl_tot"]}
    fig, ax = plt.subplots(figsize=(16, 5.4))
    for i, k in enumerate(("inv1", "inv2", "prog", "cl25")):
        lbl = "Claude (valid nuclei)" if k == "cl25" else LBL[k]
        ax.bar(x + (i - 1.5) * w, tot[k], width=w * 0.92, color=C[k], label=lbl)
    ax.set_xticks(x); ax.set_xticklabels(imgs)
    ax.set_xlabel("image", color=INK)
    ax.set_ylabel("total nuclei counted", color=INK)
    ax.set_title("Total nuclei per image by source", color=INK)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "totals_comparison.png"), dpi=120,
                facecolor="white")
    plt.close(fig)

    # ---- fig 3: agreement scatter vs investigator mean ----
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), sharex=True,
                             sharey=True)
    for ax, k in zip(axes, ("prog", "cl25")):
        r = pearsonr(inv_mean, nfi[k])
        bias = float(np.mean(nfi[k] - inv_mean))
        ax.plot([0, 75], [0, 75], color=MUT, lw=1, ls="--", zorder=1)
        ax.scatter(inv_mean, nfi[k], s=46, color=C[k], zorder=3)
        for xi, yi, s in zip(inv_mean, nfi[k], imgs):
            if abs(yi - xi) > 12:
                ax.annotate(s, (xi, yi), textcoords="offset points",
                            xytext=(5, -3), fontsize=8, color=INK,
                            fontweight="bold")
        ax.set_xlabel("investigator mean NFI (%)", color=INK)
        ax.set_title(f"{LBL[k]}\nr={r[0]:.2f}, mean bias {bias:+.1f}pp",
                     color=INK)
        ax.grid(alpha=0.35, color=GRID)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("automated NFI (%)", color=INK)
    fig.suptitle("Agreement with the human consensus (identity line = perfect "
                 "agreement)", color=INK, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "agreement_scatter.png"), dpi=120,
                facecolor="white")
    plt.close(fig)

    with open(os.path.join(OUT, "comparison_stats.json"), "w") as fh:
        json.dump({"mean_nfi": {k: round(v, 2) for k, v in means.items()},
                   "pooled_nfi": {
                       "inv1": round(100 * d["inv1_myo"].sum() / d["inv1_tot"].sum(), 2),
                       "inv2": round(100 * d["inv2_myo"].sum() / d["inv2_tot"].sum(), 2),
                       "prog": round(100 * d["prog_myo"].sum() / d["prog_tot"].sum(), 2),
                       "cl25": round(100 * d["cl25_myo"].sum() / d["cl_tot"].sum(), 2),
                       "cl50": round(100 * d["cl50_myo"].sum() / d["cl_tot"].sum(), 2)},
                   "pair_stats": pair_stats,
                   "claude_deficit_images": flagged,
                   "total_nuclei_r": {
                       "inv2_vs_claude": round(pearsonr(d["inv2_tot"], d["cl_tot"])[0], 3),
                       "prog_vs_claude": round(pearsonr(d["prog_tot"], d["cl_tot"])[0], 3),
                       "inv1_vs_inv2": round(pearsonr(d["inv1_tot"], d["inv2_tot"])[0], 3)},
                   }, fh, indent=2)

    # ---- COMPARISON.md table ----
    lines = ["# Human vs machine — 25-image benchmark", "",
             "| img | Inv1 NFI | Inv2 NFI | Program NFI | Claude ov25 | Claude ov50 |",
             "|---|---|---|---|---|---|"]
    for i, s in enumerate(imgs):
        lines.append(f"| {s} | {d['inv1_nfi'][i]:.1f} | {d['inv2_nfi'][i]:.1f} "
                     f"| {d['prog_nfi'][i]:.1f} | {d['cl25_nfi'][i]:.1f} "
                     f"| {d['cl50_nfi'][i]:.1f} |")
    lines += ["",
              f"Mean NFI: Inv1 {means['inv1']:.1f}, Inv2 {means['inv2']:.1f}, "
              f"Program {means['prog']:.1f}, Claude ov25 {means['cl25']:.1f}, "
              f"Claude ov50 {means['cl50']:.1f}", ""]
    with open(os.path.join(OUT, "COMPARISON.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("COMPARE_DONE")


if __name__ == "__main__":
    main()
