"""NEW_Quantif: nuclei inside Desmin+ myotubes for PLATE_23 well B02 (23_B02_ctrl).

Thin driver over the existing pipeline -- no new segmentation logic. It consumes
the masks produced here by `count_well.py` (Cellpose-SAM nuclei) and
`myotube_sweep.py` (ridge/plateau Desmin mask), then reuses `real_fusion.py`'s
traced-fibre functions to build the "real myotube" territory and count nuclei
inside it, with the 50-500 um^2 nucleus area boundary.

A nucleus counts as INSIDE if >= OVERLAP_FRAC of its pixels fall in the myotube
territory. The lab call is 25%; the pipeline default was 50%. Both are reported
so the sensitivity to that choice is explicit.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe NEW_Quantif/b02_new_quantif.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # import the pipeline modules
from real_fusion import trace_fibres, real_territory, nuclei_inside, UM2  # noqa: E402

WELL = "23_B02_ctrl"
NUC = os.path.join(HERE, "nuclei", f"{WELL}_masks.npy")
MYO = os.path.join(HERE, "myotube", f"{WELL}_myotube_mask.npy")
GATES = [0.0, 30.0, 50.0, 100.0]
PRIMARY = 50.0
AMIN, AMAX = 50.0, 500.0                            # nucleus area boundary, um^2
OVERLAP_FRACS = [0.25, 0.5]                         # nucleus-in-myotube overlap rule
OVERLAP_FRAC = 0.25                                 # headline (lab call)


def main():
    nuc = np.load(NUC)
    myo = np.load(MYO)
    n_raw = int(nuc.max())                          # Cellpose objects, unfiltered

    skel, idx, fibres = trace_fibres(myo)
    print(f"{WELL}: {n_raw} raw nuclei, {len(fibres)} traced fibres, "
          f"Desmin coverage {100*(myo>0).mean():.2f}%")

    per_gate, terr_p, is_in_p, valid_p = {}, None, None, None
    for g in GATES:
        terr, n_real = real_territory(myo, skel, idx, fibres, g)
        rec = {"n_real_fibres": n_real,
               "myotube_coverage_pct": round(100 * float(terr.mean()), 2)}
        for f in OVERLAP_FRACS:
            is_in, n_total, valid = nuclei_inside(nuc, terr, AMIN, AMAX, frac=f)
            n_in = int(is_in.sum())
            rec[f"overlap_{int(100*f)}pct"] = {
                "nuclei_inside_myotube": n_in,
                "nuclei_total_valid": n_total,
                "fusion_index_pct": round(100 * n_in / n_total, 2) if n_total else 0.0,
            }
            if g == PRIMARY and f == OVERLAP_FRAC:
                terr_p, is_in_p, valid_p = terr, is_in, valid
        per_gate[str(int(g))] = rec
        cells = "  ".join(f"ov{int(100*f)}={rec[f'overlap_{int(100*f)}pct']['nuclei_inside_myotube']:5d}"
                          for f in OVERLAP_FRACS)
        print(f"  gate>={int(g):3d}um  fibres={n_real:5d}  cov={100*terr.mean():5.2f}%  {cells}")

    # ---- overlay: magenta = nucleus inside a real myotube, blue = outside,
    #      dim green = real-myotube territory at the primary gate
    inside_pix = is_in_p[nuc]
    outside_pix = valid_p[nuc] & (nuc > 0) & ~inside_pix
    rgb = np.zeros((*nuc.shape, 3), np.float32)
    rgb[..., 1] = 0.30 * terr_p
    rgb[inside_pix] = [1.0, 0.1, 0.9]
    rgb[outside_pix, 2] = 1.0
    rgb[outside_pix, 0] = 0.1
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    im.thumbnail((1700, 1700))
    im.save(os.path.join(HERE,
                         f"{WELL}_fusion_g{int(PRIMARY)}_ov{int(100*OVERLAP_FRAC)}.png"))

    area_um2 = np.bincount(nuc.ravel())[1:] * UM2
    prim = per_gate[str(int(PRIMARY))][f"overlap_{int(100*OVERLAP_FRAC)}pct"]
    out = {
        "well": WELL,
        "plate": "PLATE_23",
        "source_nd2": "../Q_PLATES/Q_Plates/PLATE_23/23_B02_ctrl.nd2",
        "nuclei_total_raw": n_raw,
        "nuclei_total_valid": prim["nuclei_total_valid"],
        "nucleus_area_boundary_um2": [AMIN, AMAX],
        "nucleus_area_um2_median": round(float(np.median(area_um2)), 1),
        "primary_gate_um": PRIMARY,
        "primary_overlap_frac": OVERLAP_FRAC,
        "nuclei_inside_myotube": prim["nuclei_inside_myotube"],
        "fusion_index_pct": prim["fusion_index_pct"],
        "by_fibre_length_gate_um": per_gate,
    }
    with open(os.path.join(HERE, "b02_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("-" * 62)
    print(f"total nuclei (raw Cellpose)          : {n_raw}")
    print(f"total nuclei (area {AMIN:.0f}-{AMAX:.0f} um2)     : {out['nuclei_total_valid']}")
    print(f"nuclei INSIDE Desmin+ myotubes       : {prim['nuclei_inside_myotube']}"
          f"   (fibre gate >= {PRIMARY:.0f} um, overlap >= {100*OVERLAP_FRAC:.0f}%)")
    print(f"fusion index                         : {prim['fusion_index_pct']:.2f}%")


if __name__ == "__main__":
    main()
