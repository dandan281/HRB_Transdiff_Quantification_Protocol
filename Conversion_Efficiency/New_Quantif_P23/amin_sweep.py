"""Sweep the LOWER nucleus-area bound (artifact cut) and show what it does to the
counts and the fusion index, at a fixed fibre-length gate.

The expensive work (fibre tracing + territory rebuild) is done ONCE per well;
only the cheap area filter is swept -- same plateau logic used elsewhere in the
pipeline. Both overlap fractions (25% / 50%) are reported per convention.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe NEW_Quantif/amin_sweep.py [--gate 30]
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from real_fusion import (trace_fibres, real_territory, nuclei_inside,  # noqa: E402
                         UM2, NUC_DIR, MYO_DIR)

AMINS = [50.0, 75.0, 100.0, 125.0]
AMAX = 500.0
FRACS = [0.25, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=float, default=30.0)
    ap.add_argument("--amins", default=",".join(str(v) for v in AMINS))
    a = ap.parse_args()
    amins = [float(v) for v in a.amins.split(",")]

    wells = sorted(f.replace("_myotube_mask.npy", "") for f in os.listdir(MYO_DIR)
                   if f.endswith("_myotube_mask.npy"))

    per_well, areas_all = {}, []
    for w in wells:
        myo = np.load(os.path.join(MYO_DIR, f"{w}_myotube_mask.npy"))
        nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
        skel, idx, fibres = trace_fibres(myo)
        terr, _ = real_territory(myo, skel, idx, fibres, a.gate)   # once per well

        areas = np.bincount(nuc.ravel())[1:] * UM2                 # um^2 per nucleus
        areas_all.append(areas)
        rec = {"n_raw": int(nuc.max())}
        for amin in amins:                                         # cheap sweep
            e = {}
            for f in FRACS:
                is_in, ntot, _ = nuclei_inside(nuc, terr, amin, AMAX, frac=f)
                nin = int(is_in.sum())
                e[f"overlap_{int(100*f)}pct"] = {
                    "inside": nin, "total": ntot,
                    "fusion_pct": round(100 * nin / ntot, 2) if ntot else 0.0}
            rec[str(int(amin))] = e
        per_well[w] = rec
        print(f"  traced {w}")

    areas = np.concatenate(areas_all)
    pct = {str(p): round(float(np.percentile(areas, p)), 1)
           for p in (1, 5, 10, 25, 50)}
    print(f"\nnucleus area percentiles (um^2), all wells: {pct}")
    print(f"raw nuclei = {areas.size:,}\n")

    hdr = f"{'amin um2':>9}{'valid':>9}{'removed':>9}{'%rm':>7}" \
          f"{'in25':>8}{'idx25':>8}{'in50':>8}{'idx50':>8}"
    print(hdr); print("-" * len(hdr))
    out = {}
    for amin in amins:
        k = str(int(amin))
        tot = sum(per_well[w][k]["overlap_25pct"]["total"] for w in wells)
        i25 = sum(per_well[w][k]["overlap_25pct"]["inside"] for w in wells)
        i50 = sum(per_well[w][k]["overlap_50pct"]["inside"] for w in wells)
        raw = sum(per_well[w]["n_raw"] for w in wells)
        out[k] = {"valid": tot, "removed": raw - tot,
                  "inside_25": i25, "fusion_25_pct": round(100 * i25 / tot, 2),
                  "inside_50": i50, "fusion_50_pct": round(100 * i50 / tot, 2)}
        print(f"{amin:>9.0f}{tot:>9,}{raw-tot:>9,}{100*(raw-tot)/raw:>6.1f}%"
              f"{i25:>8,}{100*i25/tot:>7.2f}%{i50:>8,}{100*i50/tot:>7.2f}%")

    with open(os.path.join(HERE, f"amin_sweep_g{int(a.gate)}.json"), "w") as fh:
        json.dump({"gate_um": a.gate, "amax_um2": AMAX, "amins": amins,
                   "area_percentiles_um2": pct, "plate": out,
                   "per_well": per_well}, fh, indent=2)


if __name__ == "__main__":
    main()
