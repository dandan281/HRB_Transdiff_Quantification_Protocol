"""PLATE 44 treatment summary: average replicate wells, SEM error bars, bar graph.

The well is the replicate unit (n=3 per condition, n=2 for the two TNFalpha
conditions), so every mean and interval below is across WELLS, never across
cells -- pooling 18,000 cells per well would give absurdly tight intervals that
describe segmentation noise rather than biology.

Conditions are drawn in the layout sheet's own order, NOT sorted by result, so
the figure does not manufacture a ranking. Significance is Welch's t vs the
`No mb` control with Holm-Bonferroni correction across the 13 comparisons; with
n=3 this is low-powered and is labelled as such.

`B11` is excluded from the Alk1 mean by default -- its Desmin channel is
effectively empty (dbs p99 = 329 vs 1,066-2,331 plate-wide), a technical failure
identified from the image before the plate map arrived. Both values are always
reported.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/treatment_summary.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    CONDITION_ORDER, CONTROL_CONDITION, SHEET_TYPO_NOTE, TECHNICAL_FAILURES,
    condition_of, well_id)

# dataviz palette: two-category identity (reference vs treatment). Not 14 hues --
# this is one measure across conditions, so hue carries role, never rank.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_TREAT, C_CTRL, C_FAIL = "#2a78d6", "#52514e", "#d03b3b"


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-failures", action="store_true",
                    help="include technically failed wells in condition means")
    a = ap.parse_args()

    pc = json.load(open(os.path.join(HERE, "percell_desmin.json")))["per_well"]

    groups: dict[str, list[tuple[str, float]]] = {c: [] for c in CONDITION_ORDER}
    dropped: dict[str, list[str]] = {}
    for stem, rec in pc.items():
        wid, cond = well_id(stem), condition_of(stem)
        if wid in TECHNICAL_FAILURES and not a.include_failures:
            dropped.setdefault(cond, []).append(wid)
            continue
        groups[cond].append((wid, rec["conversion_pct"]))
    for c in groups:
        groups[c].sort()

    ctrl_vals = np.array([v for _, v in groups[CONTROL_CONDITION]])
    ctrl_mean = float(ctrl_vals.mean())
    ctrl_sd = float(ctrl_vals.std(ddof=1))

    # --- per-condition statistics -------------------------------------------
    rows, raw_p = [], []
    for cond in CONDITION_ORDER:
        vals = np.array([v for _, v in groups[cond]])
        n = vals.size
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else float("nan")
        sem = sd / np.sqrt(n) if n > 1 else float("nan")
        if cond == CONTROL_CONDITION:
            p = float("nan")
        else:
            p = float(stats.ttest_ind(vals, ctrl_vals, equal_var=False).pvalue)
            raw_p.append(p)
        # Complete separation: do ALL wells of this condition sit outside the
        # full range of control wells? At n=2-3 this is more informative than a
        # p-value -- it uses no variance estimate and cannot be rescued or
        # destroyed by one noisy replicate the way Welch's t can.
        if cond == CONTROL_CONDITION:
            separation = None
        elif vals.min() > ctrl_vals.max():
            separation = "above_all_controls"
        elif vals.max() < ctrl_vals.min():
            separation = "below_all_controls"
        else:
            separation = "overlaps_controls"
        rows.append({"condition": cond, "n": int(n),
                     "wells": [w for w, _ in groups[cond]],
                     "values_pct": [round(v, 2) for v in vals],
                     "mean_pct": round(mean, 2),
                     "sd_pct": round(sd, 2) if n > 1 else None,
                     "sem_pct": round(sem, 2) if n > 1 else None,
                     "fold_vs_control": round(mean / ctrl_mean, 3),
                     "delta_in_control_sd": round((mean - ctrl_mean) / ctrl_sd, 1),
                     "separation_vs_controls": separation,
                     "p_welch_vs_control": None if cond == CONTROL_CONDITION
                                           else round(p, 4),
                     "excluded_wells": dropped.get(cond, [])})
    adj = holm(raw_p)
    it = iter(adj)
    for r in rows:
        r["p_holm"] = None if r["condition"] == CONTROL_CONDITION \
            else round(next(it), 4)

    # --- console table ------------------------------------------------------
    print(f"PLATE 44 — replicate-averaged conversion efficiency")
    print(f"control = {CONTROL_CONDITION!r}: {ctrl_mean:.2f}% "
          f"(SD {ctrl_sd:.2f}, n={ctrl_vals.size}, wells "
          f"{', '.join(w for w, _ in groups[CONTROL_CONDITION])})")
    if dropped:
        for c, ws in dropped.items():
            print(f"excluded from {c!r}: {', '.join(ws)} "
                  f"({TECHNICAL_FAILURES[ws[0]]})")
    sep_mark = {"above_all_controls": "^ above all ctrl",
                "below_all_controls": "v below all ctrl",
                "overlaps_controls": "", None: ""}
    hdr = (f"{'condition':<14}{'n':>3}{'mean':>8}{'SD':>7}{'SEM':>7}"
           f"{'fold':>7}{'d/ctrlSD':>10}{'p(Holm)':>9}  separation")
    print("\n" + hdr); print("-" * (len(hdr) + 4))
    for r in rows:
        ph = "" if r["p_holm"] is None else f"{r['p_holm']:.3f}{stars(r['p_holm'])}"
        mark = "  <- control" if r["condition"] == CONTROL_CONDITION else ""
        print(f"{r['condition']:<14}{r['n']:>3}{r['mean_pct']:>7.1f}%"
              f"{(r['sd_pct'] if r['sd_pct'] is not None else float('nan')):>7.2f}"
              f"{(r['sem_pct'] if r['sem_pct'] is not None else float('nan')):>7.2f}"
              f"{r['fold_vs_control']:>6.2f}x"
              f"{r['delta_in_control_sd']:>+10.1f}{ph:>9}  "
              f"{sep_mark[r['separation_vs_controls']]}{mark}")

    sig = [r for r in rows if r["p_holm"] is not None and r["p_holm"] < 0.05]
    seps = [r for r in rows if r["separation_vs_controls"] in
            ("above_all_controls", "below_all_controls")]
    print(f"\n{len(sig)} of {len(raw_p)} conditions differ from control after "
          f"Holm correction" + (f": {', '.join(r['condition'] for r in sig)}"
                                if sig else " -- none"))
    print(f"control well-to-well SD = {ctrl_sd:.2f} pp (CV "
          f"{100*ctrl_sd/ctrl_mean:.1f}%), tight; treatment SDs run up to "
          f"{max(r['sd_pct'] for r in rows if r['sd_pct'] is not None):.2f} pp, "
          f"which is what the Welch test is reacting to.")
    if seps:
        print("completely separated from the control range (no p-value needed, "
              "but n is small):")
        for r in seps:
            print(f"  {r['condition']:<14} {r['mean_pct']:.1f}% "
                  f"({r['fold_vs_control']:.2f}x, {r['delta_in_control_sd']:+.1f} "
                  f"control SD, n={r['n']}) wells "
                  f"{', '.join(f'{v:.1f}' for v in r['values_pct'])} vs controls "
                  f"{', '.join(f'{v:.1f}' for v in np.sort(ctrl_vals)[::-1])}")
    print("NOTE: n=2-3 per condition with 13 comparisons is low-powered; "
          "absence of significance is not evidence of absence.")

    # --- figure -------------------------------------------------------------
    x = np.arange(len(rows))
    means = np.array([r["mean_pct"] for r in rows])
    sems = np.array([r["sem_pct"] if r["sem_pct"] is not None else 0.0
                     for r in rows])
    is_ctrl = [r["condition"] == CONTROL_CONDITION for r in rows]

    fig, ax = plt.subplots(figsize=(13, 6.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # control noise band first, so bars sit on top of it
    ax.axhspan(ctrl_mean - ctrl_sd, ctrl_mean + ctrl_sd, color=C_CTRL,
               alpha=0.10, zorder=0)
    ax.axhline(ctrl_mean, color=C_CTRL, lw=1.4, ls="--", zorder=1)

    bars = ax.bar(x, means, width=0.66, zorder=2,
                  color=[C_CTRL if c else C_TREAT for c in is_ctrl],
                  edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor=INK, elinewidth=1.4,
                capsize=5, capthick=1.4, zorder=3)

    rng = np.random.default_rng(20260813)          # fixed jitter, reproducible
    for i, r in enumerate(rows):
        vals = np.array(r["values_pct"], dtype=float)
        jitter = rng.uniform(-0.13, 0.13, size=vals.size)
        ax.scatter(np.full(vals.size, i) + jitter, vals, s=34, zorder=4,
                   color=INK, alpha=0.75, edgecolor=SURFACE, linewidth=1.2)
        star = stars(r["p_holm"]) if r["p_holm"] is not None else ""
        if star:
            ax.annotate(star, (i, means[i] + sems[i]), ha="center", va="bottom",
                        fontsize=13, color=INK, xytext=(0, 4),
                        textcoords="offset points")

    # Excluded well drawn at its real value, so the exclusion is visible rather
    # than only asserted in the legend.
    for wid, reason in TECHNICAL_FAILURES.items():
        if not a.include_failures and wid in [w for ws in dropped.values()
                                              for w in ws]:
            stem = next(s for s in pc if well_id(s) == wid)
            idx = CONDITION_ORDER.index(condition_of(stem))
            ax.scatter([idx], [pc[stem]["conversion_pct"]], marker="x", s=70,
                       color=C_FAIL, linewidth=2.2, zorder=6)
            ax.annotate(f"{wid} {pc[stem]['conversion_pct']:.1f}%",
                        (idx, pc[stem]["conversion_pct"]), ha="left", va="center",
                        fontsize=7.5, color=C_FAIL, zorder=6,
                        xytext=(9, 0), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor=SURFACE,
                                  edgecolor="none", alpha=0.92))

    # No floating control annotation: it collided with the bars at every
    # placement tried. The value rides in the legend entry instead, which cannot
    # collide -- and the reader needs it in exactly one place, not two.

    ax.set_xticks(x)
    ax.set_xticklabels([r["condition"] for r in rows], rotation=35, ha="right",
                       fontsize=10, color=INK)
    ax.set_ylabel("Desmin+ nuclei  (conversion efficiency, %)", fontsize=11,
                  color=INK_2)
    for i, r in enumerate(rows):
        ax.annotate(f"n={r['n']}", (i, 0), ha="center", va="bottom", fontsize=7.5,
                    color=MUTED, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, max(means + sems) * 1.22)
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.yaxis.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lab in ax.get_xticklabels():
        lab.set_color(INK)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_CTRL,
              label=f"control ({CONTROL_CONDITION}) — {ctrl_mean:.1f}%"),
        Patch(facecolor=C_TREAT, label="treatment"),
        Line2D([], [], marker="o", ls="none", color=INK, alpha=0.75, ms=6,
               markeredgecolor=SURFACE, label="individual well"),
        Line2D([], [], color=INK, lw=1.4, label="mean ± SEM"),
        Patch(facecolor=C_CTRL, alpha=0.10,
              label=f"control ± 1 SD ({ctrl_sd:.1f} pp)"),
    ]
    if dropped:
        handles.append(Line2D([], [], marker="x", ls="none", color=C_FAIL, ms=7,
                              mew=2.2,
                              label="excluded: Desmin staining failure"))
    ax.legend(handles=handles, frameon=False, fontsize=8.5, ncol=3,
              loc="upper left", labelcolor=INK_2)

    verdict = (f"{len(sig)} condition(s) differ after Holm correction"
               if sig else "no condition differs from control after correction")
    ax.set_title(
        "PLATE 44 — conversion efficiency by condition\n"
        f"well-level replicates (n=3; n=2 for the TNFalpha panel) · mean ± SEM · "
        f"Welch t vs {CONTROL_CONDITION} control, Holm-corrected over "
        f"{len(raw_p)} comparisons · {verdict}",
        fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "treatment_summary.png"), dpi=150,
                bbox_inches="tight", facecolor=SURFACE)

    out = {
        "plate": "PLATE_44",
        "replicate_unit": "well",
        "control_condition": CONTROL_CONDITION,
        "control_mean_pct": round(ctrl_mean, 2),
        "control_sd_pct": round(ctrl_sd, 2),
        "control_wells": [w for w, _ in groups[CONTROL_CONDITION]],
        "technical_failures_excluded": {k: TECHNICAL_FAILURES[k]
                                        for ws in dropped.values() for k in ws},
        "include_failures": a.include_failures,
        "sheet_typo_note": SHEET_TYPO_NOTE,
        "test": f"Welch t vs control, Holm-Bonferroni over {len(raw_p)} comparisons",
        "n_significant_after_holm": len(sig),
        "completely_separated_from_control_range":
            [r["condition"] for r in seps],
        "power_note": ("n=2-3 wells per condition with 13 comparisons is "
                       "low-powered; absence of significance is not evidence "
                       "of absence"),
        "per_condition": rows,
    }
    with open(os.path.join(HERE, "treatment_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n-> New_Quantif_P44/treatment_summary.png / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
