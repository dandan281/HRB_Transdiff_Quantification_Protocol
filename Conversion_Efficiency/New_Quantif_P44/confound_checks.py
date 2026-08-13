"""PLATE 44 QC. Three checks that decide whether the conversion ranking means
anything, run BEFORE any well is called a responder.

1. **Density confound** -- is conversion just tracking nucleus density? This is
   the failure that invalidated PLATE_28 (r = 0.99). P23 was 0.61 (real effect),
   P32 0.34, Plate 9 ~0.
2. **Plate-position trend** -- does conversion drift with row or column? A
   gradient across the plate is an acquisition/edge artifact, not biology. Plate 9
   had exactly this and it is part of why its conversion readout was withdrawn.
3. **Effect size vs plausible well-to-well noise** -- the observed spread is
   compared against the only empirical estimate of control variability this
   project has: Plate 9's SIX induced-control wells spanned 4.1x (11.3-46.7 %
   Desmin) with no treatment difference between them. PLATE 44 has no replicate
   structure declared (no layout sheet), so this is the honest yardstick.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/confound_checks.py
"""
from __future__ import annotations
import json
import os
import re
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import CTRL  # noqa: E402

# Plate 9's six induced-only control wells, the project's only measured estimate
# of control-to-control spread on a single plate (README, "Headline").
P9_CONTROL_SPREAD = (11.3, 46.7)


def main() -> int:
    pc = json.load(open(os.path.join(HERE, "percell_desmin.json")))["per_well"]
    wells = sorted(pc, key=lambda s: int(s.split("_")[0]))
    conv = np.array([pc[w]["conversion_pct"] for w in wells])
    dens = np.array([pc[w]["cells"] for w in wells], dtype=float)
    ids = [pc[w]["well_id"] for w in wells]
    rows = np.array([ord(re.match(r"([A-H])", i).group(1)) for i in ids])
    cols = np.array([int(re.match(r"[A-H](\d+)", i).group(1)) for i in ids])

    rp_d, p_d = pearsonr(dens, conv)
    rs_d = spearmanr(dens, conv)[0]
    rp_c = pearsonr(cols, conv)[0]
    rp_r = pearsonr(rows, conv)[0]

    print("PLATE 44 confound checks\n" + "=" * 46)
    print(f"1. density   Pearson r = {rp_d:+.3f} (p={p_d:.3g})  "
          f"Spearman = {rs_d:+.3f}")
    print("   reference: P28 0.99 (artifact) | P23 0.61 | P32 0.34 | P9 ~0.00")
    print(f"2. column    Pearson r = {rp_c:+.3f}")
    print(f"   row       Pearson r = {rp_r:+.3f}")

    lo, hi = conv.min(), conv.max()
    ctrl_eff = pc[CTRL]["conversion_pct"]
    rank = int((conv < ctrl_eff).sum()) + 1
    print(f"3. spread    {lo:.1f}% - {hi:.1f}%  = {hi/max(lo,1e-9):.1f}x "
          f"(excl. min: {np.sort(conv)[1]:.1f}%-{hi:.1f}% = "
          f"{hi/np.sort(conv)[1]:.1f}x)")
    print(f"   assumed control {CTRL} = {ctrl_eff:.1f}%  -> rank {rank}/40 "
          f"({100*(rank-1)/39:.0f}th percentile)")
    print(f"   Plate 9's six CONTROL wells alone spanned "
          f"{P9_CONTROL_SPREAD[0]}%-{P9_CONTROL_SPREAD[1]}% = "
          f"{P9_CONTROL_SPREAD[1]/P9_CONTROL_SPREAD[0]:.1f}x with no treatment "
          f"difference")

    verdict = []
    if abs(rp_d) >= 0.7:
        verdict.append("DENSITY ARTIFACT: ranking tracks nucleus density")
    elif abs(rp_d) >= 0.4:
        verdict.append("partial density association; interpret with care")
    else:
        verdict.append("density is not driving the ranking")
    if max(abs(rp_c), abs(rp_r)) >= 0.4:
        verdict.append("plate-position trend present (acquisition artifact risk)")
    if hi / max(np.sort(conv)[1], 1e-9) <= P9_CONTROL_SPREAD[1] / P9_CONTROL_SPREAD[0]:
        verdict.append("observed spread is WITHIN the control-only spread measured "
                       "on Plate 9 -- not separable from well-to-well noise")
    print("\nverdict: " + "; ".join(verdict))

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))
    is_ctrl = [w == CTRL for w in wells]
    cols_c = ["#db2777" if c else "#64748b" for c in is_ctrl]
    ax[0].scatter(dens, conv, c=cols_c, s=90, edgecolor="#1e293b", zorder=3)
    for x, y, l in zip(dens, conv, ids):
        ax[0].annotate(l, (x, y), fontsize=6.5, xytext=(4, 3),
                       textcoords="offset points")
    m, b = np.polyfit(dens, conv, 1)
    xs = np.array([dens.min(), dens.max()])
    ax[0].plot(xs, m * xs + b, "--", color="#94a3b8", zorder=1)
    ax[0].set_xlabel("valid nuclei (density; field area constant)")
    ax[0].set_ylabel("conversion efficiency (%)")
    ax[0].set_title(f"1. density confound\nPearson r = {rp_d:+.2f}, "
                    f"Spearman {rs_d:+.2f}  (pink = assumed ctrl)")
    ax[0].grid(alpha=0.3)

    for r in sorted(set(rows)):
        sel = rows == r
        order = np.argsort(cols[sel])
        ax[1].plot(cols[sel][order], conv[sel][order], "-o", ms=5,
                   label=chr(r))
    ax[1].axhline(ctrl_eff, color="#db2777", ls="--", lw=1.4,
                  label=f"assumed ctrl {ctrl_eff:.1f}%")
    ax[1].set_xlabel("plate column")
    ax[1].set_ylabel("conversion efficiency (%)")
    ax[1].set_title(f"2. plate position\ncolumn r = {rp_c:+.2f}, "
                    f"row r = {rp_r:+.2f}")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    order = np.argsort(conv)
    ax[2].barh(range(len(conv)), conv[order],
               color=["#db2777" if is_ctrl[i] else "#94a3b8" for i in order])
    ax[2].set_yticks(range(len(conv)))
    ax[2].set_yticklabels([ids[i] for i in order], fontsize=6)
    ax[2].axvspan(conv[order][0], conv[order][-1], color="#fbbf24", alpha=0.12)
    ax[2].set_xlabel("conversion efficiency (%)")
    ax[2].set_title(f"3. spread {lo:.1f}-{hi:.1f}%\n"
                    f"vs Plate 9 control-only spread {P9_CONTROL_SPREAD[0]}-"
                    f"{P9_CONTROL_SPREAD[1]}%")
    ax[2].grid(axis="x", alpha=0.3)

    fig.suptitle("PLATE 44 — QC before interpretation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "confound_checks.png"), dpi=120,
                bbox_inches="tight", facecolor="white")

    with open(os.path.join(HERE, "confound_checks.json"), "w") as fh:
        json.dump({"plate": "PLATE_44",
                   "density_pearson_r": round(float(rp_d), 3),
                   "density_pearson_p": float(p_d),
                   "density_spearman_r": round(float(rs_d), 3),
                   "column_pearson_r": round(float(rp_c), 3),
                   "row_pearson_r": round(float(rp_r), 3),
                   "conversion_min_pct": float(lo), "conversion_max_pct": float(hi),
                   "assumed_control": CTRL,
                   "assumed_control_pct": ctrl_eff,
                   "assumed_control_rank_of_40": rank,
                   "plate9_control_only_spread_pct": list(P9_CONTROL_SPREAD),
                   "verdict": verdict}, fh, indent=2)
    print("-> New_Quantif_P44/confound_checks.png / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
