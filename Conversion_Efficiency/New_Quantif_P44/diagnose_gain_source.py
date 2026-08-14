"""PLATE 44 troubleshooting, step 6: WHERE does the Desmin gain variation come from?

Step 5 showed the artifact is real but that normalising by the Desmin channel's
own brightness is not the cure -- it scores the empty B11 null well at 98 %,
because a well with no myotube has a low p99 and dividing by it inflates
everything. Gain and abundance are not separable from one channel alone.

So the fix depends on the SOURCE of the gain variation, and there are two, with
opposite remedies:

  OPTICAL / EXPOSURE  the whole acquisition scaled -- illumination drift, focus,
                      exposure. Then EVERY channel scales together, and DAPI
                      (which is biologically flat -- nuclei are nuclei) is a
                      valid loading control. Fixable in software.

  ANTIBODY / STAINING the Desmin stain itself varied per well. DAPI would not
                      track it. Not fixable in software; needs re-staining with
                      a matched master mix, or matched-exposure re-imaging.

The test: does DAPI brightness track Desmin brightness across wells? Also checks
whether either tracks plate position, which would point at an edge/handling
gradient.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/diagnose_gain_source.py
"""
from __future__ import annotations
import json
import os
import re
import sys

import numpy as np
import nd2
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    DAPI_CH, RECEPTOR_CH, TECHNICAL_FAILURES, condition_of, nd2_path, well_id,
    wells)

CACHE = os.path.join(HERE, "dbs_cache")


def main() -> int:
    order = wells()
    rows = []
    for i, w in enumerate(order, 1):
        with nd2.ND2File(nd2_path(w)) as f:
            a = f.asarray()
        dapi = a[DAPI_CH].astype(np.float32)
        rec = a[RECEPTOR_CH].astype(np.float32)
        dbs = np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
        wid = well_id(w)
        rows.append({
            "well": w, "well_id": wid, "condition": condition_of(w),
            "desmin_p99": float(np.percentile(dbs, 99)),
            "dapi_p99": float(np.percentile(dapi, 99)),
            "dapi_median": float(np.median(dapi)),
            "receptor_p99": float(np.percentile(rec, 99)),
            "row": ord(re.match(r"([A-H])", wid).group(1)),
            "col": int(re.match(r"[A-H](\d+)", wid).group(1)),
            "ok": wid not in TECHNICAL_FAILURES})
        if i % 10 == 0:
            print(f"  read {i}/{len(order)}", flush=True)

    ok = np.array([r["ok"] for r in rows])
    des = np.array([r["desmin_p99"] for r in rows])
    dap = np.array([r["dapi_p99"] for r in rows])
    dapm = np.array([r["dapi_median"] for r in rows])
    rec = np.array([r["receptor_p99"] for r in rows])
    col = np.array([r["col"] for r in rows], dtype=float)
    rw = np.array([r["row"] for r in rows], dtype=float)

    print("\nPLATE 44 — source of the Desmin gain variation\n" + "=" * 50)
    print(f"Desmin p99   spans {des[ok].min():.0f}-{des[ok].max():.0f} "
          f"({des[ok].max()/des[ok].min():.2f}x)")
    print(f"DAPI   p99   spans {dap[ok].min():.0f}-{dap[ok].max():.0f} "
          f"({dap[ok].max()/dap[ok].min():.2f}x)")
    print(f"receptor p99 spans {rec[ok].min():.0f}-{rec[ok].max():.0f} "
          f"({rec[ok].max()/rec[ok].min():.2f}x)")

    r_dd, p_dd = pearsonr(dap[ok], des[ok])
    r_dm = pearsonr(dapm[ok], des[ok])[0]
    print(f"\nr(DAPI p99, Desmin p99)    = {r_dd:+.3f} (p={p_dd:.2g}, n={ok.sum()})")
    print(f"r(DAPI median, Desmin p99) = {r_dm:+.3f}")
    print(f"r(Desmin p99, column)      = {pearsonr(des[ok], col[ok])[0]:+.3f}")
    print(f"r(Desmin p99, row)         = {pearsonr(des[ok], rw[ok])[0]:+.3f}")
    print(f"r(DAPI p99, column)        = {pearsonr(dap[ok], col[ok])[0]:+.3f}")

    if abs(r_dd) > 0.6:
        verdict = ("OPTICAL/EXPOSURE: DAPI and Desmin brightness move together, "
                   "so the whole acquisition scaled. DAPI is usable as a loading "
                   "control and the plate is repairable in software.")
    elif abs(r_dd) > 0.3:
        verdict = ("MIXED: DAPI partly tracks Desmin. Some of the gain is "
                   "optical and correctable; some is Desmin-specific.")
    else:
        verdict = ("ANTIBODY/STAINING-SPECIFIC: DAPI does NOT track Desmin, so "
                   "the variation is in the Desmin stain itself. No software "
                   "normalisation can recover it -- this needs re-staining with "
                   "a matched master mix, or re-imaging at matched exposure.")
    print(f"\nVERDICT: {verdict}")

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    ax[0].scatter(dap[ok], des[ok], s=80, color="#2a78d6", edgecolor="#0b0b0b",
                  zorder=3)
    ax[0].scatter(dap[~ok], des[~ok], s=90, color="#d03b3b", marker="x",
                  zorder=3, label="B11")
    for r in rows:
        ax[0].annotate(r["well_id"], (r["dapi_p99"], r["desmin_p99"]),
                       fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    xs = np.array([dap[ok].min(), dap[ok].max()])
    ax[0].plot(xs, np.polyval(np.polyfit(dap[ok], des[ok], 1), xs), "--",
               color="#898781")
    ax[0].set_xlabel("DAPI p99 (loading-control proxy)")
    ax[0].set_ylabel("Desmin p99 (gain term)")
    ax[0].set_title(f"do the channels scale together?\nr = {r_dd:+.2f}")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    for lab, v, c in (("Desmin p99", des, "#eb6834"), ("DAPI p99", dap, "#2a78d6")):
        vn = (v - v[ok].mean()) / v[ok].std()
        ax[1].scatter(col[ok], vn[ok], s=60, color=c, label=lab, alpha=0.85)
    ax[1].set_xlabel("plate column"); ax[1].set_ylabel("z-scored brightness")
    ax[1].set_title("positional gradient?")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    o = np.argsort(des)
    ax[2].barh(range(ok.sum()), des[o][ok[o]], color="#eb6834")
    ax[2].set_yticks(range(ok.sum()))
    ax[2].set_yticklabels([rows[i]["well_id"] for i in o if rows[i]["ok"]],
                          fontsize=6)
    ax[2].set_xlabel("Desmin p99 (raw units)")
    ax[2].set_title("per-well Desmin brightness\n(biologically identical "
                    "replicates differ up to 1.6x)")
    ax[2].grid(axis="x", alpha=0.3)

    fig.suptitle("PLATE 44 — is the gain variation optical or antibody?",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "diagnose_gain_source.png"), dpi=130,
                bbox_inches="tight", facecolor="white")
    with open(os.path.join(HERE, "diagnose_gain_source.json"), "w") as fh:
        json.dump({"plate": "PLATE_44",
                   "r_dapi_p99_vs_desmin_p99": round(float(r_dd), 3),
                   "p_dapi_vs_desmin": float(p_dd),
                   "r_dapi_median_vs_desmin_p99": round(float(r_dm), 3),
                   "desmin_p99_range": [float(des[ok].min()), float(des[ok].max())],
                   "dapi_p99_range": [float(dap[ok].min()), float(dap[ok].max())],
                   "verdict": verdict, "per_well": rows}, fh, indent=2)
    print("\n-> diagnose_gain_source.png / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
