"""MULTINUCLEATION -- nuclei per INDIVIDUAL myotube, PLATE_23 well B02.

Distinguishing individual myotubes is the crux: connected components fuse touching
/crossing fibres into one mesh, so instead we TRACE whole fibres through crossings
(straightest-through pairing at junctions -- the lab-validated method, reused from
real_fusion.trace_fibres). Each traced fibre = one individual myotube.

Then, exactly as nuclei_3d.py: propagate every fibre's ID to its territory
(nearest-skeleton), and assign each valid nucleus (area 50-500 um^2, >= overlap
fraction of its pixels inside a myotube) to its DOMINANT host fibre. Count nuclei
per myotube and report the distribution:
    what % of myotubes have 1 / 2 / 3 / 4 / 5+ nuclei.

A "myotube" here is a traced fibre >= a length gate (fragments removed). Because B02
is a fragment-dominated control, the gate matters a lot, so the distribution is
reported across gates AND both 25%/50% overlap conventions; the headline uses the
established real-myotube primary (>= 50 um, 50% overlap).

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P23/b02_multinucleation.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from real_fusion import trace_fibres, UM, UM2, NUC_DIR, MYO_DIR  # noqa: E402

WELL = "23_B02_ctrl"
AMIN, AMAX = 50.0, 500.0                 # nucleus area boundary, um^2
GATES = [0.0, 30.0, 50.0, 100.0]         # myotube min length (fragment removal)
FRACS = [0.25, 0.5]                      # nucleus-in-myotube overlap convention
PRIMARY_GATE, PRIMARY_FRAC = 50.0, 0.5


def host_assignment(nuc, fid_map, area_um2, frac):
    """host fibre id per nucleus label (dominant territory), for valid nuclei whose
    territory overlap >= frac. Returns dict {nucleus_label: fibre_id}."""
    flat = nuc.ravel()
    area_px = np.bincount(flat).astype(np.float64)
    in_terr = (fid_map > 0).ravel().astype(np.float64)
    inside_px = np.bincount(flat, weights=in_terr, minlength=area_px.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        ov = np.where(area_px > 0, inside_px / area_px, 0.0)
    valid = (area_um2 >= AMIN) & (area_um2 <= AMAX)
    valid[0] = False
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
    """counts = array of nuclei-per-myotube over the myotube set. Return dist dict."""
    n = counts.size
    bins = {"0": int((counts == 0).sum()), "1": int((counts == 1).sum()),
            "2": int((counts == 2).sum()), "3": int((counts == 3).sum()),
            "4": int((counts == 4).sum()), "5plus": int((counts >= 5).sum())}
    nucleated = counts[counts >= 1]
    return {
        "n_myotubes": int(n),
        "n_myotubes_with_nuclei": int(nucleated.size),
        "counts_bins": bins,
        # % of ALL myotubes (>= gate) in each nuclei bin
        "pct_of_all": {k: (round(100 * v / n, 1) if n else 0.0) for k, v in bins.items()},
        # % of NUCLEATED myotubes (>=1 nucleus) in each bin -- biological convention
        "pct_of_nucleated": {k: (round(100 * v / nucleated.size, 1) if nucleated.size else 0.0)
                             for k, v in bins.items() if k != "0"},
        "mean_nuclei_per_myotube": round(float(counts.mean()), 2) if n else 0.0,
        "mean_nuclei_per_nucleated": round(float(nucleated.mean()), 2) if nucleated.size else 0.0,
        "pct_ge2_of_nucleated": round(100 * float((nucleated >= 2).mean()), 1) if nucleated.size else 0.0,
    }


def main():
    myo = np.load(os.path.join(MYO_DIR, f"{WELL}_myotube_mask.npy"))
    nuc = np.load(os.path.join(NUC_DIR, f"{WELL}_masks.npy"))
    area_um2 = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2

    skel, idx, fibres = trace_fibres(myo)
    lengths = np.array([0.0] + [f[0] for f in fibres])       # fibre id 1..N
    fid_skel = np.zeros(skel.shape, np.int32)
    for fid, (_, pix) in enumerate(fibres, start=1):
        fid_skel[pix[:, 0], pix[:, 1]] = fid
    fid_map_full = fid_skel[idx[0], idx[1]]                   # nearest fibre everywhere
    fid_map_full = np.where(binary_fill_holes(myo > 0), fid_map_full, 0)
    print(f"{WELL}: {len(fibres)} individual myotubes traced "
          f"(through crossings), {int((area_um2[1:] >= AMIN).sum())} area-valid nuclei\n")

    results = {}
    for frac in FRACS:
        host = host_assignment(nuc, fid_map_full, area_um2, frac)  # nucleus -> fibre id
        per_fibre = np.zeros(len(fibres) + 1, dtype=int)
        for fibre_id in host.values():
            per_fibre[fibre_id] += 1
        for g in GATES:
            keep_ids = np.flatnonzero(lengths >= g)
            keep_ids = keep_ids[keep_ids >= 1]                 # drop id 0
            counts = per_fibre[keep_ids]
            results[f"frac{int(100*frac)}_gate{int(g)}"] = {
                "overlap_frac": frac, "length_gate_um": g,
                **distribution(counts)}

    # ---- console: headline + sensitivity ----
    key = f"frac{int(100*PRIMARY_FRAC)}_gate{int(PRIMARY_GATE)}"
    P = results[key]
    print(f"=== PRIMARY: individual myotubes >= {PRIMARY_GATE:.0f} um, "
          f"nucleus overlap >= {100*PRIMARY_FRAC:.0f}% ===")
    print(f"  individual myotubes (>= {PRIMARY_GATE:.0f} um): {P['n_myotubes']}")
    print(f"  of those, with >= 1 nucleus:            {P['n_myotubes_with_nuclei']}")
    print(f"  mean nuclei per myotube:  {P['mean_nuclei_per_myotube']}  "
          f"(per nucleated: {P['mean_nuclei_per_nucleated']})")
    print(f"  nuclei-per-myotube distribution (% of NUCLEATED myotubes):")
    for b in ["1", "2", "3", "4", "5plus"]:
        lab = "5+" if b == "5plus" else b
        print(f"      {lab:>3} nuclei: {P['pct_of_nucleated'][b]:5.1f}%  "
              f"({P['counts_bins'][b]} myotubes)")
    print(f"  % multinucleated (>= 2 nuclei) of nucleated: {P['pct_ge2_of_nucleated']:.1f}%\n")

    print("=== sensitivity: % of NUCLEATED myotubes with 1 / 2 / 3 / 4 / 5+ nuclei ===")
    print(f"{'gate':>5}{'ovlp':>6}{'#tubes':>8}{'#nucl':>7}"
          f"{'1':>7}{'2':>7}{'3':>7}{'4':>7}{'5+':>7}{'mean':>7}")
    for frac in FRACS:
        for g in GATES:
            r = results[f"frac{int(100*frac)}_gate{int(g)}"]
            pn = r["pct_of_nucleated"]
            print(f"{int(g):>5}{int(100*frac):>5}%{r['n_myotubes']:>8}"
                  f"{r['n_myotubes_with_nuclei']:>7}"
                  f"{pn['1']:>6.1f}{pn['2']:>7.1f}{pn['3']:>7.1f}"
                  f"{pn['4']:>7.1f}{pn['5plus']:>7.1f}{r['mean_nuclei_per_nucleated']:>7.2f}")

    with open(os.path.join(HERE, "b02_multinucleation.json"), "w") as fh:
        json.dump({"well": WELL, "n_individual_myotubes_traced": len(fibres),
                   "nucleus_area_um2": [AMIN, AMAX],
                   "primary": {"length_gate_um": PRIMARY_GATE, "overlap_frac": PRIMARY_FRAC},
                   "results": results}, fh, indent=2)

    # ---- figure 1: histogram of nuclei-per-myotube (primary) ----
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["1", "2", "3", "4", "5+"]
    vals = [P["pct_of_nucleated"][b] for b in ["1", "2", "3", "4", "5plus"]]
    ncts = [P["counts_bins"][b] for b in ["1", "2", "3", "4", "5plus"]]
    bars = ax.bar(labels, vals, color="#7c3aed", edgecolor="#3b0764")
    for b, v, n in zip(bars, vals, ncts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%\n(n={n})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("nuclei per individual myotube")
    ax.set_ylabel("% of nucleated myotubes")
    ax.set_title(f"{WELL} multinucleation — individual myotubes >= {PRIMARY_GATE:.0f} um, "
                 f"overlap >= {100*PRIMARY_FRAC:.0f}%\n"
                 f"{P['n_myotubes_with_nuclei']} nucleated myotubes; "
                 f"mean {P['mean_nuclei_per_nucleated']} nuclei; "
                 f"{P['pct_ge2_of_nucleated']:.0f}% multinucleated")
    ax.set_ylim(0, max(vals) * 1.2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "b02_multinucleation_hist.png"), dpi=120,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- figure 2: overlay -- each individual myotube coloured by nucleus count ----
    host = host_assignment(nuc, fid_map_full, area_um2, PRIMARY_FRAC)
    per_fibre = np.zeros(len(fibres) + 1, dtype=int)
    for fibre_id in host.values():
        per_fibre[fibre_id] += 1
    keep = np.zeros(len(fibres) + 1, dtype=bool)
    keep[np.flatnonzero(lengths >= PRIMARY_GATE)] = True
    keep[0] = False
    # colour code per myotube territory by its nucleus count
    palette = {0: (0.35, 0.35, 0.35), 1: (0.20, 0.45, 1.0), 2: (0.1, 0.85, 0.3),
               3: (1.0, 0.85, 0.1), 4: (1.0, 0.5, 0.0)}          # 5+ -> red
    rgb = np.zeros((*nuc.shape, 3), np.float32)
    fid_kept = np.where(keep[fid_map_full], fid_map_full, 0)
    for fid in np.flatnonzero(keep):
        c = per_fibre[fid]
        col = palette.get(c, (1.0, 0.0, 0.0)) if c >= 1 else palette[0]
        rgb[fid_kept == fid] = col
    # mark assigned nuclei white
    assigned_mask = np.isin(nuc, list(host.keys())) & (nuc > 0)
    rgb[assigned_mask] = (1.0, 1.0, 1.0)
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    im.thumbnail((1700, 1700))
    im.save(os.path.join(HERE, "b02_multinucleation_overlay.png"))
    print("\n-> b02_multinucleation_hist.png / _overlay.png / .json")


if __name__ == "__main__":
    main()
