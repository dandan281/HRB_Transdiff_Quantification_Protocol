"""PLATE 44 — the treatment bar graph AFTER per-well Desmin gain normalisation.

Companion to `treatment_summary.py` (absolute threshold). Same layout, same
replicate handling, same statistics; the only change is that each well's ring
Desmin is divided by that well's own p99 before the pooled threshold is derived,
which removes the staining-gain artifact diagnosed in `diagnose_staining.py`.

**This method is NOT adopted as the plate's result, and the figure says so.**
It passes three of four acceptance criteria and fails the fourth:

  1. staining artifact removed        r = +0.917 -> -0.06   PASS
  2. replicate agreement              SD 2.03 -> 0.77 pp    PASS
  3. condition ranking stable         see figure            PASS
  4. null well stays separable        B11 4.5% -> 98%       **FAIL**

Criterion 4 is the disqualifier: B11 has almost no Desmin, so its p99 reflects
noise rather than fibre brightness, and dividing by it inflates every cell. The
consequence is that the normalised readout has **no absolute scale** -- it
measures "what fraction of this well's nuclei sit in this well's brightest
Desmin", which is meaningful only if the well has real Desmin to begin with.
B11 is the single degenerate well (p99 = 329 against a next-lowest of 1,066) and
is excluded from every condition mean here, as it is everywhere else.

Also reports the C6-vs-C2 backbone contrast, which the plate's factorial design
supports and which has far more power than 13 individual comparisons.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/treatment_summary_normalised.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    CONDITION_ORDER, CONTROL_CONDITION, TECHNICAL_FAILURES, condition_of,
    well_id, wells)

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_TREAT, C_CTRL, C_FAIL = "#1baf7a", "#52514e", "#d03b3b"
NULL_WELL = "14_B11"


def holm(p):
    m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def main() -> int:
    gn = json.load(open(os.path.join(HERE, "gain_normalised_test.json")))
    per_well = gn["per_well_pct"]
    absol = json.load(open(os.path.join(HERE, "percell_desmin.json")))["per_well"]

    ok = [w for w in wells() if well_id(w) not in TECHNICAL_FAILURES]
    groups = {c: sorted((well_id(w), per_well[w]) for w in ok
                        if condition_of(w) == c) for c in CONDITION_ORDER}
    ctrl = np.array([v for _, v in groups[CONTROL_CONDITION]])
    ctrl_mean, ctrl_sd = float(ctrl.mean()), float(ctrl.std(ddof=1))

    rows, praw = [], []
    for c in CONDITION_ORDER:
        v = np.array([x for _, x in groups[c]])
        p = (float("nan") if c == CONTROL_CONDITION
             else float(stats.ttest_ind(v, ctrl, equal_var=False).pvalue))
        if c != CONTROL_CONDITION:
            praw.append(p)
        rows.append({"condition": c, "n": int(v.size),
                     "wells": [w for w, _ in groups[c]],
                     "values_pct": [round(x, 2) for x in v],
                     "mean_pct": round(float(v.mean()), 2),
                     "sd_pct": round(float(v.std(ddof=1)), 2) if v.size > 1 else None,
                     "sem_pct": round(float(v.std(ddof=1) / np.sqrt(v.size)), 2)
                                if v.size > 1 else None,
                     "fold": round(float(v.mean() / ctrl_mean), 3),
                     "p_welch": None if c == CONTROL_CONDITION else round(p, 4)})
    adj = holm(praw); it = iter(adj)
    for r in rows:
        r["p_holm"] = None if r["condition"] == CONTROL_CONDITION \
            else round(float(next(it)), 4)

    print("PLATE 44 — GAIN-NORMALISED treatment summary")
    print(f"control {CONTROL_CONDITION!r}: {ctrl_mean:.2f}% (SD {ctrl_sd:.2f}, "
          f"n={ctrl.size})\n")
    hdr = f"{'condition':<14}{'n':>3}{'mean':>8}{'SD':>7}{'fold':>7}{'p(Holm)':>10}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        ph = "" if r["p_holm"] is None else f"{r['p_holm']:.3f}"
        mark = "  <- control" if r["condition"] == CONTROL_CONDITION else ""
        print(f"{r['condition']:<14}{r['n']:>3}{r['mean_pct']:>7.1f}%"
              f"{(r['sd_pct'] or float('nan')):>7.2f}{r['fold']:>6.2f}x{ph:>10}{mark}")

    # --- backbone contrast: the design's own factorial structure -------------
    c6 = np.array([per_well[w] for w in ok if condition_of(w).startswith("C6")])
    c2 = np.array([per_well[w] for w in ok if condition_of(w).startswith("C2")])
    tt = stats.ttest_ind(c6, c2, equal_var=False)
    print(f"\nBACKBONE CONTRAST (uses the plate's factorial design, "
          f"far better powered than 13 pairwise tests)")
    print(f"  C6-containing  n={c6.size} wells  mean {c6.mean():.2f}% "
          f"(SD {c6.std(ddof=1):.2f})")
    print(f"  C2-containing  n={c2.size} wells  mean {c2.mean():.2f}% "
          f"(SD {c2.std(ddof=1):.2f})")
    print(f"  Welch t = {tt.statistic:+.2f}, p = {tt.pvalue:.4f}, "
          f"difference {c6.mean()-c2.mean():+.2f} pp "
          f"({c6.mean()/c2.mean():.2f}x)")
    # same contrast on the ABSOLUTE numbers, for comparison
    a6 = np.array([absol[w]["conversion_pct"] for w in ok
                   if condition_of(w).startswith("C6")])
    a2 = np.array([absol[w]["conversion_pct"] for w in ok
                   if condition_of(w).startswith("C2")])
    ta = stats.ttest_ind(a6, a2, equal_var=False)
    print(f"  (absolute threshold, same contrast: {a6.mean():.2f}% vs "
          f"{a2.mean():.2f}%, p = {ta.pvalue:.4f})")

    # ------------------------------- figure ---------------------------------
    fig = plt.figure(figsize=(15.5, 7.0), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 5, wspace=0.32)
    ax = fig.add_subplot(gs[0, :4]); axn = fig.add_subplot(gs[0, 4])
    for a in (ax, axn):
        a.set_facecolor(SURFACE)

    x = np.arange(len(rows))
    means = np.array([r["mean_pct"] for r in rows])
    sems = np.array([r["sem_pct"] or 0.0 for r in rows])
    is_ctrl = [r["condition"] == CONTROL_CONDITION for r in rows]

    ax.axhspan(ctrl_mean - ctrl_sd, ctrl_mean + ctrl_sd, color=C_CTRL,
               alpha=0.10, zorder=0)
    ax.axhline(ctrl_mean, color=C_CTRL, lw=1.4, ls="--", zorder=1)
    ax.bar(x, means, width=0.66, zorder=2,
           color=[C_CTRL if c else C_TREAT for c in is_ctrl],
           edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor=INK, elinewidth=1.4,
                capsize=5, capthick=1.4, zorder=3)
    rng = np.random.default_rng(20260813)
    for i, r in enumerate(rows):
        v = np.array(r["values_pct"], dtype=float)
        ax.scatter(np.full(v.size, i) + rng.uniform(-0.13, 0.13, v.size), v,
                   s=34, zorder=4, color=INK, alpha=0.75, edgecolor=SURFACE,
                   linewidth=1.2)
        ax.annotate(f"n={r['n']}", (i, 0), ha="center", va="bottom",
                    fontsize=7.5, color=MUTED, xytext=(0, 3),
                    textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels([r["condition"] for r in rows], rotation=35, ha="right",
                       fontsize=10, color=INK)
    ax.set_ylabel("Desmin+ nuclei (%), gain-normalised", fontsize=11, color=INK_2)
    ax.set_ylim(0, max(means + sems) * 1.28)
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.yaxis.grid(True, color=GRID, lw=0.9); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASELINE); ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lab in ax.get_xticklabels():
        lab.set_color(INK)
    ax.legend(handles=[
        Patch(facecolor=C_CTRL, label=f"control ({CONTROL_CONDITION}) — {ctrl_mean:.1f}%"),
        Patch(facecolor=C_TREAT, label="treatment"),
        Line2D([], [], marker="o", ls="none", color=INK, alpha=0.75, ms=6,
               markeredgecolor=SURFACE, label="individual well"),
        Line2D([], [], color=INK, lw=1.4, label="mean ± SEM"),
        Patch(facecolor=C_CTRL, alpha=0.10, label=f"control ± 1 SD ({ctrl_sd:.1f} pp)"),
    ], frameon=False, fontsize=8.5, ncol=3, loc="upper left", labelcolor=INK_2)
    ax.set_title(
        "PLATE 44 — conversion efficiency after per-well Desmin gain normalisation\n"
        f"replicate SD improves 2.03 → 0.77 pp and the staining artifact is gone "
        f"(r +0.92 → −0.06), but 0 of 13 still clear Holm correction",
        fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=14)

    # --- the disqualifying check, drawn beside it ---------------------------
    axn.bar([0, 1], [absol[NULL_WELL]["conversion_pct"], gn["per_well_pct"][NULL_WELL]],
            color=[MUTED, C_FAIL], edgecolor=SURFACE, linewidth=2)
    axn.axhline(ctrl_mean, color=C_CTRL, ls="--", lw=1.4)
    axn.annotate(f"control {ctrl_mean:.0f}%", (-0.45, ctrl_mean), ha="left",
                 va="bottom", fontsize=8, color=C_CTRL, xytext=(0, 3),
                 textcoords="offset points")
    for i, v in enumerate([absol[NULL_WELL]["conversion_pct"],
                           gn["per_well_pct"][NULL_WELL]]):
        axn.annotate(f"{v:.0f}%", (i, v), ha="center", va="bottom", fontsize=10,
                     fontweight="bold", color=INK, xytext=(0, 3),
                     textcoords="offset points")
    axn.set_xticks([0, 1])
    axn.set_xticklabels(["absolute", "normalised"], fontsize=9, color=INK)
    axn.set_ylim(0, 108)
    axn.set_ylabel("B11 null well (%)", fontsize=10, color=INK_2)
    axn.yaxis.grid(True, color=GRID, lw=0.9); axn.set_axisbelow(True)
    for s in ("top", "right"):
        axn.spines[s].set_visible(False)
    axn.spines["left"].set_color(BASELINE); axn.spines["bottom"].set_color(BASELINE)
    axn.tick_params(colors=MUTED, labelsize=9)
    # Inside the axes, not a title -- a title here collides with the main one.
    axn.annotate("NOT adopted:\nB11 has no\nDesmin, yet\nscores 98%",
                 (0.04, 0.66), xycoords="axes fraction", fontsize=8.5,
                 color=C_FAIL, fontweight="bold", va="top", linespacing=1.45)

    fig.text(0.012, 0.005,
             "METHOD NOT ADOPTED — normalisation removes the staining artifact but leaves the readout with no absolute scale "
             "(a well with no Desmin scores 98%). Shown for comparison only; the plate's result remains 'not interpretable — re-stain'.",
             fontsize=8.5, color=C_FAIL, style="italic")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.20)
    fig.savefig(os.path.join(HERE, "treatment_summary_normalised.png"), dpi=150,
                facecolor=SURFACE)

    with open(os.path.join(HERE, "treatment_summary_normalised.json"), "w") as fh:
        json.dump({"plate": "PLATE_44", "method": "ring Desmin / per-well p99",
                   "status": "NOT ADOPTED — fails the null-well criterion",
                   "control_mean_pct": round(ctrl_mean, 2),
                   "control_sd_pct": round(ctrl_sd, 2),
                   "null_well_absolute_pct": absol[NULL_WELL]["conversion_pct"],
                   "null_well_normalised_pct": gn["per_well_pct"][NULL_WELL],
                   "backbone_contrast": {
                       "c6_mean": round(float(c6.mean()), 2), "c6_n": int(c6.size),
                       "c2_mean": round(float(c2.mean()), 2), "c2_n": int(c2.size),
                       "difference_pp": round(float(c6.mean() - c2.mean()), 2),
                       "fold": round(float(c6.mean() / c2.mean()), 3),
                       "p_welch": round(float(tt.pvalue), 5),
                       "p_welch_absolute_threshold": round(float(ta.pvalue), 5)},
                   "per_condition": rows}, fh, indent=2)
    print("\n-> treatment_summary_normalised.png / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
