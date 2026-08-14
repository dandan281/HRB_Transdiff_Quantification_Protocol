"""PLATE 44 troubleshooting, step 5: does gain-normalisation fix the readout?

Diagnosis so far: Desmin staining/exposure gain varies up to 1.60x between
biologically IDENTICAL replicate wells, and a fixed absolute threshold turns that
gain straight into apparent conversion (partial r = +0.83 with myotube coverage
held constant).

The candidate fix is to threshold each well relative to its OWN Desmin
brightness scale instead of an absolute raw value.

**This must be tested, not assumed.** Per-image normalisation is exactly what
caused a previous flat-1.0x-fold-change bug in this project, by normalising away
real differences in Desmin abundance. The distinction that matters:

  * dividing by a GAIN term (p99 = peak brightness) removes a multiplicative
    exposure/staining factor and preserves how MUCH of the field is myotube;
  * a full min-max or per-image contrast stretch also removes abundance, which
    is the biology. That is the bug, and it is not what this does.

Conversion efficiency asks "is this nucleus inside a myotube", which is an AREA
question, so a gain-invariant area measure is the right target. Whether it
actually works is decided by four criteria, reported honestly either way:

  1. does the brightness correlation vanish?          (must, or it did nothing)
  2. do replicate wells agree better?                 (within-condition SD)
  3. is the B11 null still separated from controls?   (must not flatten)
  4. do condition differences survive or vanish?      (the flattening test)

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/gain_normalised_test.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    AMAX_UM2, AMIN_UM2, CONDITION_ORDER, CONTROL_CONDITION, RING_PX,
    TECHNICAL_FAILURES, UM2, condition_of, well_id, wells)
from percell_desmin import ring_intensity  # noqa: E402

CACHE = os.path.join(HERE, "dbs_cache")
NUC_DIR = os.path.join(HERE, "nuclei")
NULL_WELL = "14_B11"


def holm(p):
    m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def main() -> int:
    order = wells()
    gains, ringvals = {}, {}
    for w in order:
        dbs = np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
        nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
        g = float(np.percentile(dbs, 99))            # the gain term
        gains[w] = g
        area = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2
        valid = (area >= AMIN_UM2) & (area <= AMAX_UM2); valid[0] = False
        mean, cnt = ring_intensity(nuc, dbs / g, RING_PX)   # NORMALISED image
        ringvals[w] = mean[valid[:mean.size] & (cnt > 0)]

    pooled = np.concatenate([ringvals[w] for w in order])
    lv = np.log10(np.maximum(pooled, 1e-4))
    thr = float(10 ** threshold_otsu(lv))
    conv = {w: 100 * float((ringvals[w] > thr).mean()) for w in order}
    print(f"gain-normalised pooled-Otsu threshold = {thr:.4f} "
          f"(fraction of each well's own p99)\n")

    base = json.load(open(os.path.join(HERE, "percell_desmin.json")))["per_well"]
    old = {w: base[w]["conversion_pct"] for w in order}
    ok = [w for w in order if well_id(w) not in TECHNICAL_FAILURES]

    # 1. brightness correlation, within condition
    def centred(d):
        out = {}
        for c in CONDITION_ORDER:
            ws = [w for w in ok if condition_of(w) == c]
            if len(ws) > 1:
                mu = np.mean([d[w] for w in ws])
                for w in ws:
                    out[w] = d[w] - mu
        return out
    cg, cn, co = centred(gains), centred(conv), centred(old)
    ks = sorted(cg)
    r_new = pearsonr([cg[w] for w in ks], [cn[w] for w in ks])
    r_old = pearsonr([cg[w] for w in ks], [co[w] for w in ks])
    print("1. BRIGHTNESS CORRELATION (within condition, biology held constant)")
    print(f"   absolute threshold  r = {r_old[0]:+.3f} (p={r_old[1]:.2g})")
    print(f"   gain-normalised     r = {r_new[0]:+.3f} (p={r_new[1]:.2g})")
    print(f"   -> {'REMOVED' if abs(r_new[0]) < 0.4 else 'still present'}\n")

    # 2. replicate agreement
    def sds(d):
        v = []
        for c in CONDITION_ORDER:
            ws = [w for w in ok if condition_of(w) == c]
            if len(ws) > 1:
                v.append(float(np.std([d[w] for w in ws], ddof=1)))
        return float(np.mean(v))
    print("2. REPLICATE AGREEMENT (mean within-condition SD, pp)")
    print(f"   absolute threshold  {sds(old):.2f}")
    print(f"   gain-normalised     {sds(conv):.2f}   "
          f"({100*(sds(conv)-sds(old))/sds(old):+.0f}%)\n")

    # 3. null well still separated?
    ctrl = [w for w in ok if condition_of(w) == CONTROL_CONDITION]
    cm_new = float(np.mean([conv[w] for w in ctrl]))
    cm_old = float(np.mean([old[w] for w in ctrl]))
    print("3. NULL WELL SEPARATION (must not be flattened away)")
    print(f"   absolute:       B11 {old[NULL_WELL]:5.2f}%  control {cm_old:5.2f}%"
          f"  ratio {cm_old/old[NULL_WELL]:.2f}x")
    print(f"   gain-normalised B11 {conv[NULL_WELL]:5.2f}%  control {cm_new:5.2f}%"
          f"  ratio {cm_new/max(conv[NULL_WELL],1e-9):.2f}x\n")

    # 4. condition differences
    rows, praw = [], []
    cvals = np.array([conv[w] for w in ctrl])
    for c in CONDITION_ORDER:
        ws = [w for w in ok if condition_of(w) == c]
        v = np.array([conv[w] for w in ws])
        p = (float("nan") if c == CONTROL_CONDITION
             else float(stats.ttest_ind(v, cvals, equal_var=False).pvalue))
        if c != CONTROL_CONDITION:
            praw.append(p)
        rows.append({"condition": c, "n": v.size, "mean": float(v.mean()),
                     "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                     "fold": float(v.mean() / cm_new), "p": p})
    adj = holm(praw); it = iter(adj)
    for r in rows:
        r["p_holm"] = None if r["condition"] == CONTROL_CONDITION else float(next(it))
    sig = [r for r in rows if r["p_holm"] is not None and r["p_holm"] < 0.05]

    print("4. CONDITION DIFFERENCES (gain-normalised)")
    hdr = f"{'condition':<14}{'n':>3}{'mean':>8}{'SD':>7}{'fold':>7}{'p(Holm)':>10}"
    print("   " + hdr); print("   " + "-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["mean"]):
        ph = "" if r["p_holm"] is None else f"{r['p_holm']:.3f}"
        mark = "  <- control" if r["condition"] == CONTROL_CONDITION else ""
        print(f"   {r['condition']:<14}{r['n']:>3}{r['mean']:>7.1f}%"
              f"{r['sd']:>7.2f}{r['fold']:>6.2f}x{ph:>10}{mark}")
    print(f"\n   {len(sig)} of {len(praw)} conditions significant after Holm "
          f"(was 0 with the absolute threshold)")
    if sig:
        for r in sig:
            print(f"     {r['condition']}: {r['mean']:.1f}% "
                  f"({r['fold']:.2f}x, p={r['p_holm']:.4f})")

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    ax[0].scatter([cg[w] for w in ks], [co[w] for w in ks], s=70,
                  color="#d03b3b", label=f"absolute  r={r_old[0]:+.2f}")
    ax[0].scatter([cg[w] for w in ks], [cn[w] for w in ks], s=70,
                  color="#1baf7a", label=f"normalised r={r_new[0]:+.2f}")
    ax[0].axhline(0, color="#c3c2b7"); ax[0].axvline(0, color="#c3c2b7")
    ax[0].set_xlabel("Desmin brightness deviation from condition mean")
    ax[0].set_ylabel("conversion deviation (pp)")
    ax[0].set_title("1. does the staining artifact go away?")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    x = np.arange(len(CONDITION_ORDER))
    mo = [np.mean([old[w] for w in ok if condition_of(w) == c])
          for c in CONDITION_ORDER]
    mn = [np.mean([conv[w] for w in ok if condition_of(w) == c])
          for c in CONDITION_ORDER]
    ax[1].bar(x - 0.2, mo, 0.4, label="absolute", color="#d03b3b")
    ax[1].bar(x + 0.2, mn, 0.4, label="gain-normalised", color="#1baf7a")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(CONDITION_ORDER, rotation=40, ha="right", fontsize=7.5)
    ax[1].set_ylabel("conversion (%)")
    ax[1].set_title("2. do the conditions change rank?")
    ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=0.3)

    for i, c in enumerate(CONDITION_ORDER):
        v = [conv[w] for w in ok if condition_of(w) == c]
        ax[2].scatter([i] * len(v), v, s=45, color="#0b0b0b", alpha=0.75,
                      zorder=3)
        ax[2].scatter([i], [np.mean(v)], s=130, marker="_", color="#1baf7a",
                      linewidth=3, zorder=4)
    ax[2].axhline(cm_new, color="#52514e", ls="--", lw=1.3)
    ax[2].scatter([CONDITION_ORDER.index("Alk1")], [conv[NULL_WELL]], s=80,
                  marker="x", color="#d03b3b", zorder=5, label="B11 null")
    ax[2].set_xticks(range(len(CONDITION_ORDER)))
    ax[2].set_xticklabels(CONDITION_ORDER, rotation=40, ha="right", fontsize=7.5)
    ax[2].set_ylabel("conversion (%)")
    ax[2].set_title("3. gain-normalised, per well")
    ax[2].legend(fontsize=8); ax[2].grid(axis="y", alpha=0.3)

    fig.suptitle("PLATE 44 — does gain-normalisation repair the readout?",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "gain_normalised_test.png"), dpi=130,
                bbox_inches="tight", facecolor="white")
    with open(os.path.join(HERE, "gain_normalised_test.json"), "w") as fh:
        json.dump({"plate": "PLATE_44", "method": "ring Desmin / per-well p99",
                   "threshold_fraction_of_p99": thr,
                   "r_brightness_absolute": round(float(r_old[0]), 3),
                   "r_brightness_normalised": round(float(r_new[0]), 3),
                   "within_condition_sd_absolute": round(sds(old), 3),
                   "within_condition_sd_normalised": round(sds(conv), 3),
                   "null_pct": round(conv[NULL_WELL], 2),
                   "control_pct": round(cm_new, 2),
                   "n_significant_holm": len(sig),
                   "per_condition": rows,
                   "per_well_pct": {w: round(conv[w], 2) for w in order}},
                  fh, indent=2, default=float)
    print("\n-> gain_normalised_test.png / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
