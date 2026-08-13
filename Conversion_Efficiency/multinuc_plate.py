"""MULTINUCLEATION for any plate -- nuclei per INDIVIDUAL myotube.

Same method and FIXED parameters as New_Quantif_P23/plate23_multinucleation.py:
  individual myotube = traced fibre through crossings, length >= 50 um;
  a nucleus belongs to a myotube if >= 50% of its pixels fall in that myotube's
  territory; nucleus area boundary 50-500 um^2.

Reads plate{N}_myotube/{well}_myotube_mask.npy + plate{N}_nuclei/{well}_masks.npy,
writes the table, a per-well+pooled composition figure, per-well overlays, and JSON
into New_Quantif_P{N}/. Self-contained (host-assignment + distribution inlined) so it
does not depend on the per-plate package scripts.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe multinuc_plate.py --plate 26
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from real_fusion import trace_fibres, UM2  # noqa: E402

GATE, FRAC = 50.0, 0.5
AMIN, AMAX = 50.0, 500.0
PALETTE = {0: (0.35, 0.35, 0.35), 1: (0.20, 0.45, 1.0), 2: (0.1, 0.85, 0.3),
           3: (1.0, 0.85, 0.1), 4: (1.0, 0.5, 0.0)}          # 5+ -> red


def host_assignment(nuc, fid_map, area_um2, frac):
    flat = nuc.ravel()
    area_px = np.bincount(flat).astype(np.float64)
    in_terr = (fid_map > 0).ravel().astype(np.float64)
    inside_px = np.bincount(flat, weights=in_terr, minlength=area_px.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        ov = np.where(area_px > 0, inside_px / area_px, 0.0)
    valid = (area_um2 >= AMIN) & (area_um2 <= AMAX); valid[0] = False
    is_in = valid & (ov >= frac)
    inside_pix = is_in[nuc]
    nl = nuc[inside_pix]; fi = fid_map[inside_pix]
    good = fi > 0
    if good.sum() == 0:
        return {}
    df = pd.DataFrame({"nl": nl[good], "fi": fi[good]})
    dom = df.groupby("nl")["fi"].agg(lambda s: np.bincount(s).argmax())
    return {int(k): int(v) for k, v in dom.items()}


def distribution(counts):
    n = counts.size
    bins = {"0": int((counts == 0).sum()), "1": int((counts == 1).sum()),
            "2": int((counts == 2).sum()), "3": int((counts == 3).sum()),
            "4": int((counts == 4).sum()), "5plus": int((counts >= 5).sum())}
    nucd = counts[counts >= 1]
    return {"n_myotubes": int(n), "n_myotubes_with_nuclei": int(nucd.size),
            "counts_bins": bins,
            "pct_of_nucleated": {k: (round(100 * v / nucd.size, 1) if nucd.size else 0.0)
                                 for k, v in bins.items() if k != "0"},
            "mean_nuclei_per_nucleated": round(float(nucd.mean()), 2) if nucd.size else 0.0,
            "pct_ge2_of_nucleated": round(100 * float((nucd >= 2).mean()), 1) if nucd.size else 0.0}


def analyze_well(w, myo_dir, nuc_dir, outdir):
    myo = np.load(os.path.join(myo_dir, f"{w}_myotube_mask.npy"))
    nuc = np.load(os.path.join(nuc_dir, f"{w}_masks.npy"))
    area_um2 = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2
    skel, idx, fibres = trace_fibres(myo)
    lengths = np.array([0.0] + [f[0] for f in fibres])
    fid_skel = np.zeros(skel.shape, np.int32)
    for fid, (_, pix) in enumerate(fibres, start=1):
        fid_skel[pix[:, 0], pix[:, 1]] = fid
    fid_map = fid_skel[idx[0], idx[1]]
    fid_map = np.where(binary_fill_holes(myo > 0), fid_map, 0)

    host = host_assignment(nuc, fid_map, area_um2, FRAC)
    per_fibre = np.zeros(len(fibres) + 1, dtype=int)
    for fibre_id in host.values():
        per_fibre[fibre_id] += 1
    keep = np.zeros(len(fibres) + 1, dtype=bool)
    keep[np.flatnonzero(lengths >= GATE)] = True; keep[0] = False
    dist = distribution(per_fibre[np.flatnonzero(keep)])

    fid_kept = np.where(keep[fid_map], fid_map, 0)
    rgb = np.zeros((*nuc.shape, 3), np.float32)
    for fid in np.flatnonzero(keep):
        c = per_fibre[fid]
        rgb[fid_kept == fid] = PALETTE.get(c, (1.0, 0.0, 0.0)) if c >= 1 else PALETTE[0]
    rgb[np.isin(nuc, list(host.keys())) & (nuc > 0)] = (1.0, 1.0, 1.0)
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    im.thumbnail((1500, 1500))
    im.save(os.path.join(outdir, f"{w}_multinuc_overlay.png"))
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True)
    a = ap.parse_args()
    myo_dir = f"plate{a.plate}_myotube"
    nuc_dir = f"plate{a.plate}_nuclei"
    outdir = f"New_Quantif_P{a.plate}"
    os.makedirs(outdir, exist_ok=True)

    wells = sorted(f.replace("_myotube_mask.npy", "") for f in os.listdir(myo_dir)
                   if f.endswith("_myotube_mask.npy"))
    order = ["1", "2", "3", "4", "5plus"]
    res = {}
    print(f"PLATE_{a.plate}: individual myotube >= {GATE:.0f} um, overlap >= "
          f"{100*FRAC:.0f}%, nucleus area {AMIN:.0f}-{AMAX:.0f} um2\n")
    hdr = (f"{'well':<24}{'#tubes':>7}{'#nucl':>7}"
           f"{'1':>7}{'2':>7}{'3':>7}{'4':>7}{'5+':>7}{'mean':>6}{'%>=2':>6}")
    print(hdr); print("-" * len(hdr))
    for w in wells:
        d = analyze_well(w, myo_dir, nuc_dir, outdir)
        res[w] = d
        pn = d["pct_of_nucleated"]
        print(f"{w:<24}{d['n_myotubes']:>7}{d['n_myotubes_with_nuclei']:>7}"
              f"{pn['1']:>7.1f}{pn['2']:>7.1f}{pn['3']:>7.1f}{pn['4']:>7.1f}"
              f"{pn['5plus']:>7.1f}{d['mean_nuclei_per_nucleated']:>6.2f}"
              f"{d['pct_ge2_of_nucleated']:>6.1f}")

    tot = {b: sum(res[w]["counts_bins"][b] for w in wells) for b in ["0"] + order}
    n_nuc = sum(res[w]["n_myotubes_with_nuclei"] for w in wells)
    n_tubes = sum(res[w]["n_myotubes"] for w in wells)
    pooled = {b: (round(100 * tot[b] / n_nuc, 1) if n_nuc else 0.0) for b in order}
    print("-" * len(hdr))
    print(f"{'PLATE pooled':<24}{n_tubes:>7}{n_nuc:>7}"
          + "".join(f"{pooled[b]:>7.1f}" for b in order))

    with open(os.path.join(outdir, "multinucleation.json"), "w") as fh:
        json.dump({"plate": f"PLATE_{a.plate}", "length_gate_um": GATE,
                   "overlap_frac": FRAC, "nucleus_area_um2": [AMIN, AMAX],
                   "per_well": res, "pooled": {"n_myotubes": n_tubes,
                   "n_nucleated": n_nuc, "counts_bins": tot,
                   "pct_of_nucleated": pooled}}, fh, indent=2)

    # composition figure (stacked) + maturity
    ctrl = [w for w in wells if "ctrl" in w]
    treated = sorted([w for w in wells if "ctrl" not in w],
                     key=lambda w: res[w]["pct_ge2_of_nucleated"])
    show = ctrl + treated
    labels = [w.split("_", 1)[1] for w in show] + ["POOLED"]
    catcols = ["#3b82f6", "#22c55e", "#eab308", "#f97316", "#ef4444"]
    catnames = ["1", "2", "3", "4", "5+"]
    data = np.array([[res[w]["pct_of_nucleated"][b] for b in order] for w in show]
                    + [[pooled[b] for b in order]])
    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(labels)), 6.5))
    bottom = np.zeros(len(labels)); x = np.arange(len(labels))
    for k, (c, nm) in enumerate(zip(catcols, catnames)):
        ax.bar(x, data[:, k], bottom=bottom, color=c, label=f"{nm} nuclei",
               edgecolor="white", linewidth=0.6)
        bottom += data[:, k]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("% of nucleated myotubes"); ax.set_ylim(0, 100)
    ax.set_title(f"PLATE_{a.plate} nuclei-per-myotube composition "
                 f"(individual myotubes >= {GATE:.0f} um, overlap >= {100*FRAC:.0f}%)")
    for xi, w in enumerate(show):
        ax.text(xi, 101, f"n={res[w]['n_myotubes_with_nuclei']}", ha="center",
                fontsize=7, color="#334155")
    ax.legend(title="nuclei / myotube", ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "multinucleation_summary.png"), dpi=120,
                bbox_inches="tight", facecolor="white")
    print(f"\n-> {outdir}/multinucleation_summary.png + {len(wells)} overlays + .json")


if __name__ == "__main__":
    main()
