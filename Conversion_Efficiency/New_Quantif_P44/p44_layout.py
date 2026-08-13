"""PLATE 44 (Tdiffs) acquisition constants and well list.

**This plate is NOT the Q_Plates PLATE_2x format.** Verified read-only from OME
metadata on all 40 nd2 files (`nd2 0.11.3`); every well agrees exactly:

| | PLATE_23/26/28/32 | **PLATE 44** |
|---|---|---|
| channel 0 | 561 (receptor) | **DAPI, 429 nm (nuclei)** |
| channel 1 | 488 Desmin | 488 Desmin *(same)* |
| channel 2 | 405 DAPI (nuclei) | **AF546, 571 nm (receptor)** |
| frame | 3636 x 3636 | **1818 x 1818** |
| pixel | 0.650017 um | **1.724571 um** |
| field | 2.36 mm (5.59 mm2) | **3.14 mm (9.83 mm2)** |
| depth | 16-bit | **12-bit (max 4095)** |

Two of those are silent-corruption traps, so they are stated here once and
imported everywhere rather than re-typed per script:

1. **Channels 0 and 2 are swapped** relative to PLATE_2x. Desmin is channel 1 on
   both, so a run with the old `--nuclei-ch 2` default produces a *plausible*
   myotube mask while segmenting the receptor channel as nuclei. Nothing in the
   output would look wrong. (Plate 9 has this same DAPI-first order; see
   `Plate9_C6C2_QTFCs/plate9_layout.py`.)
2. **Pixel size differs by 2.65x.** Every parameter expressed in pixels
   (top-hat radius, ring width) must be re-derived from its physical size, and
   every area in um2 must use this plate's own UM2. A 50-500 um2 nucleus is
   118-1185 px on PLATE_2x but only 17-168 px here.

The larger field at coarser pixels is a lower-magnification objective, not
binning: PLATE 44 covers 1.76x the area of a PLATE_2x field.

Raw camera units are NOT comparable across plates (different depth and
acquisition), which is why the Desmin threshold is always re-derived from this
plate's own pooled distribution. That is the standing rule, not a P44 exception.
"""
from __future__ import annotations
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ND2_DIR = os.path.join(REPO, "Plate_44 _Tdiffs")

# --- this plate's own geometry, from nd2 metadata (NOT real_fusion.UM) ---
UM = 1.724571                      # um per pixel
UM2 = UM * UM                      # 2.974 um2 per pixel
FRAME_PX = 1818

# --- channel map, verified on all 40 files ---
DAPI_CH, DESMIN_CH, RECEPTOR_CH = 0, 1, 2

# --- physical parameters, identical to every other plate ---
AMIN_UM2, AMAX_UM2 = 50.0, 500.0   # nucleus area boundary
RING_UM = 10.0                     # cytoplasmic ring for Desmin sampling
TOPHAT_UM = 26.0                   # background scale: disk(40 px) at 0.650017 um

# Pixel equivalents, derived — never hardcoded.
RING_PX = max(1, int(round(RING_UM / UM)))        # 6 px  (15 px on PLATE_2x)
TOPHAT_PX = max(1, int(round(TOPHAT_UM / UM)))    # 15 px (40 px on PLATE_2x)

# Control well. On PLATE_23/26/28/32 the well at position 23 (B02) is always the
# control, and P44 has a 23_B02. That is a POSITION CONVENTION, not a label read
# off this plate's sheet -- P44 filenames carry no treatment token at all.
# Fold-change is reported against it and flagged as assumed; every absolute
# per-well number below is independent of this choice.
CTRL = "23_B02"
CTRL_IS_ASSUMED = True

# No layout sheet for this plate has been provided, so no condition is invented
# for any well (the Plate 9 rule for G09/G10). Wells are reported individually by
# id; treatment grouping is deliberately absent until the sheet arrives.
UNLABELED = "unlabeled"

_STEM = re.compile(r"^(\d+)_([A-H])(\d+)$")


def wells() -> list[str]:
    """Every imaged well stem, ordered by acquisition position."""
    stems = [os.path.splitext(os.path.basename(p))[0]
             for p in glob.glob(os.path.join(ND2_DIR, "*.nd2"))]
    return sorted(stems, key=lambda s: int(s.split("_")[0]))


def nd2_path(stem: str) -> str:
    return os.path.join(ND2_DIR, f"{stem}.nd2")


def well_id(stem: str) -> str:
    """'23_B02' -> 'B02'."""
    m = _STEM.match(stem)
    if not m:
        raise ValueError(f"unexpected P44 stem {stem!r}")
    return f"{m.group(2)}{int(m.group(3)):02d}"


def condition_of(stem: str) -> str:
    """Always `unlabeled` on this plate -- filenames carry no treatment token.

    Deliberately not inferred. Inventing a condition is the one thing the Plate 9
    layout module refuses to do, and the same rule applies here.
    """
    return UNLABELED


if __name__ == "__main__":
    ws = wells()
    print(f"PLATE 44: {len(ws)} wells, {UM} um/px, DAPI=ch{DAPI_CH}, "
          f"Desmin=ch{DESMIN_CH}")
    print(f"  ring {RING_UM} um = {RING_PX} px | tophat {TOPHAT_UM} um = "
          f"{TOPHAT_PX} px | nucleus {AMIN_UM2}-{AMAX_UM2} um2 = "
          f"{AMIN_UM2/UM2:.0f}-{AMAX_UM2/UM2:.0f} px")
    print("  " + ", ".join(well_id(w) for w in ws))
