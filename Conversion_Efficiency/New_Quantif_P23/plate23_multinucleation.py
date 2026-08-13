"""MULTINUCLEATION for ALL of PLATE_23 -- nuclei per INDIVIDUAL myotube.

Fixed parameters (chosen by operator): individual myotube = traced fibre (through
crossings) with length >= 50 um; a nucleus belongs to a myotube if >= 50% of its
pixels fall in that myotube's territory; nucleus area boundary 50-500 um^2.

Per well reports the distribution: what % of (nucleated) myotubes have 1 / 2 / 3 /
4 / 5+ nuclei, the mean nuclei per myotube, and the % multinucleated (>= 2). Reuses
the individual-myotube tracer + nucleus assignment from b02_multinucleation.py, so
it is consistent with the conversion-efficiency pipeline.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P23/plate23_multinucleation.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.ndimage import binary_fill_holes
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from real_fusion import trace_fibres, UM2, NUC_DIR, MYO_DIR  # noqa: E402
from b02_multinucleation import host_assignment, distribution, AMIN, AMAX  # noqa: E402

GATE, FRAC = 50.0, 0.5                    # individual-myotube length gate; overlap
PALETTE = {0: (0.35, 0.35, 0.35), 1: (0.20, 0.45, 1.0), 2: (0.1, 0.85, 0.3),
           3: (1.0, 0.85, 0.1), 4: (1.0, 0.5, 0.0)}          # 5+ -> red


def analyze_well(w):
    myo = np.load(os.path.join(MYO_DIR, f"{w}_myotube_mask.npy"))
    nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
    area_um2 = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2

    skel, idx, fibres = trace_fibres(myo)
    lengths = np.array([0.0] + [f[0] for f in fibres])
    fid_skel = np.zeros(skel.shape, np.int32)
    for fid, (_, pix) in enumerate(fibres, start=1):
        fid_skel[pix[:, 0], pix[:, 1]] = fid
    fid_map = fid_skel[idx[0], idx[1]]
    fid_map = np.where(binary_fill_holes(myo > 0), fid_map, 0)

    host = host_assignment(nuc, fid_map, area_um2, FRAC)       # nucleus -> fibre id
    per_fibre = np.zeros(len(fibres) + 1, dtype=int)
    for fibre_id in host.values():
        per_fibre[fibre_id] += 1
    keep = np.zeros(len(fibres) + 1, dtype=bool)
    keep[np.flatnonzero(lengths >= GATE)] = True
    keep[0] = False
    counts = per_fibre[np.flatnonzero(keep)]
    dist = distribution(counts)

    # per-well overlay: each individual myotube coloured by nucleus count
    fid_kept = np.where(keep[fid_map], fid_map, 0)
    rgb = np.zeros((*nuc.shape, 3), np.float32)
    for fid in np.flatnonzero(keep):
        c = per_fibre[fid]
        rgb[fid_kept == fid] = PALETTE.get(c, (1.0, 0.0, 0.0)) if c >= 1 else PALETTE[0]
    rgb[np.isin(nuc, list(host.keys())) & (nuc > 0)] = (1.0, 1.0, 1.0)
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    im.thumbnail((1500, 1500))
    im.save(os.path.join(HERE, f"{w}_multinuc_overlay.png"))
    return dist


def main():
    wells = sorted(f.replace("_myotube_mask.npy", "") for f in os.listdir(MYO_DIR)
                   if f.endswith("_myotube_mask.npy"))
    order = ["1", "2", "3", "4", "5plus"]
    res = {}
    print(f"individual myotube >= {GATE:.0f} um, nucleus overlap >= {100*FRAC:.0f}%, "
          f"nucleus area {AMIN:.0f}-{AMAX:.0f} um2\n")
    hdr = (f"{'well':<24}{'#tubes':>7}{'#nucl':>7}"
           f"{'1':>7}{'2':>7}{'3':>7}{'4':>7}{'5+':>7}{'mean':>6}{'%>=2':>6}")
    print(hdr); print("-" * len(hdr))
    for w in wells:
        d = analyze_well(w)
        res[w] = d
        pn = d["pct_of_nucleated"]
        print(f"{w:<24}{d['n_myotubes']:>7}{d['n_myotubes_with_nuclei']:>7}"
              f"{pn['1']:>7.1f}{pn['2']:>7.1f}{pn['3']:>7.1f}{pn['4']:>7.1f}"
              f"{pn['5plus']:>7.1f}{d['mean_nuclei_per_nucleated']:>6.2f}"
              f"{d['pct_ge2_of_nucleated']:>6.1f}")

    # ---- plate-pooled distribution ----
    tot = {b: sum(res[w]["counts_bins"][b] for w in wells) for b in ["0"] + order}
    n_nuc = sum(res[w]["n_myotubes_with_nuclei"] for w in wells)
    n_tubes = sum(res[w]["n_myotubes"] for w in wells)
    pooled_pct = {b: (round(100 * tot[b] / n_nuc, 1) if n_nuc else 0.0) for b in order}
    print("-" * len(hdr))
    print(f"{'PLATE 23 pooled':<24}{n_tubes:>7}{n_nuc:>7}"
          + "".join(f"{pooled_pct[b]:>7.1f}" for b in order))

    with open(os.path.join(HERE, "plate23_multinucleation.json"), "w") as fh:
        json.dump({"plate": "PLATE_23", "length_gate_um": GATE, "overlap_frac": FRAC,
                   "nucleus_area_um2": [AMIN, AMAX], "per_well": res,
                   "pooled": {"n_myotubes": n_tubes, "n_nucleated": n_nuc,
                              "counts_bins": tot, "pct_of_nucleated": pooled_pct}},
                  fh, indent=2)

    # ---- figure: stacked composition per well (+ pooled) ----
    labels = [w.split("_", 1)[1] for w in wells] + ["PLATE pooled"]
    cats = order
    catcols = ["#3b82f6", "#22c55e", "#eab308", "#f97316", "#ef4444"]
    catnames = ["1", "2", "3", "4", "5+"]
    data = np.array([[res[w]["pct_of_nucleated"][b] for b in cats] for w in wells]
                    + [[pooled_pct[b] for b in cats]])
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5),
                           gridspec_kw={"width_ratios": [2.3, 1]})
    bottom = np.zeros(len(labels))
    x = np.arange(len(labels))
    for k, (c, nm) in enumerate(zip(catcols, catnames)):
        ax[0].bar(x, data[:, k], bottom=bottom, color=c, label=f"{nm} nuclei",
                  edgecolor="white", linewidth=0.6)
        bottom += data[:, k]
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax[0].set_ylabel("% of nucleated myotubes"); ax[0].set_ylim(0, 100)
    ax[0].set_title(f"Nuclei-per-myotube composition — PLATE_23 "
                    f"(individual myotubes >= {GATE:.0f} um, overlap >= {100*FRAC:.0f}%)")
    ax[0].legend(title="nuclei / myotube", ncol=5, loc="upper center",
                 bbox_to_anchor=(0.5, -0.13))
    for xi, w in enumerate(wells):
        ax[0].text(xi, 101, f"n={res[w]['n_myotubes_with_nuclei']}", ha="center",
                   fontsize=7.5, color="#334155")

    means = [res[w]["mean_nuclei_per_nucleated"] for w in wells]
    ge2 = [res[w]["pct_ge2_of_nucleated"] for w in wells]
    xw = np.arange(len(wells))
    ax[1].bar(xw - 0.2, means, 0.4, color="#7c3aed", label="mean nuclei/myotube")
    ax2 = ax[1].twinx()
    ax2.bar(xw + 0.2, ge2, 0.4, color="#db2777", label="% multinucleated (>=2)")
    ax[1].set_xticks(xw)
    ax[1].set_xticklabels([w.split("_", 1)[1] for w in wells], rotation=30,
                          ha="right", fontsize=7.5)
    ax[1].set_ylabel("mean nuclei per nucleated myotube", color="#7c3aed")
    ax2.set_ylabel("% multinucleated (>=2 nuclei)", color="#db2777")
    ax[1].set_title("Maturity per well")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "plate23_multinucleation_summary.png"), dpi=120,
                bbox_inches="tight", facecolor="white")
    print("\n-> plate23_multinucleation_summary.png + 6 overlays + .json")


if __name__ == "__main__":
    main()
