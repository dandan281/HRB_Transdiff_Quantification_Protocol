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

# --- plate map, transcribed from the operator's layout sheet 2026-08-13 ---
#
# Rows B-E x columns 02-11. Conditions run in READING ORDER in triplicate and
# WRAP ACROSS ROWS -- `Alk1` is B11+C02+C03 and `C2+Alk1` is C10+C11+D02, so a
# row-wise reading of the sheet would mis-assign six wells. The last two
# conditions are duplicates, not triplicates.
#
# 12 conditions x3 + 2 x2 = 40 wells, matching the 40 imaged files exactly.
#
# `No mb` (no membrane construct) is the CONTROL, n=3. This supersedes the
# position-convention guess used before the sheet arrived -- which happened to be
# right: 23_B02 is one of the three control wells.
CONDITION_BY_WELL: dict[str, str] = {
    "B02": "No mb",     "B03": "No mb",     "B04": "No mb",
    "B05": "C6",        "B06": "C6",        "B07": "C6",
    "B08": "C2",        "B09": "C2",        "B10": "C2",
    "B11": "Alk1",      "C02": "Alk1",      "C03": "Alk1",
    "C04": "TGFb",      "C05": "TGFb",      "C06": "TGFb",
    "C07": "C6+Alk1",   "C08": "C6+Alk1",   "C09": "C6+Alk1",
    "C10": "C2+Alk1",   "C11": "C2+Alk1",   "D02": "C2+Alk1",
    "D03": "C6+TGFb",   "D04": "C6+TGFb",   "D05": "C6+TGFb",
    "D06": "C2+TGFb",   "D07": "C2+TGFb",   "D08": "C2+TGFb",
    "D09": "Alk1+TGFb", "D10": "Alk1+TGFb", "D11": "Alk1+TGFb",
    "E02": "C6 full",   "E03": "C6 full",   "E04": "C6 full",
    "E05": "C2 full",   "E06": "C2 full",   "E07": "C2 full",
    # Sheet writes "C6+TNFalpa" at E08 and "C6+TNFalpha" at E09. Transcribed as
    # one condition -- a one-letter typo, not two treatments. Flagged here rather
    # than silently merged.
    "E08": "C6+TNFalpha", "E09": "C6+TNFalpha",
    "E10": "TNFalpha",  "E11": "TNFalpha",
}
SHEET_TYPO_NOTE = ('sheet writes "C6+TNFalpa" at E08 and "C6+TNFalpha" at E09; '
                   "read as one condition")

CONTROL_CONDITION = "No mb"

# Display order: control first, then singles, pairs, "full", TNFalpha panel --
# the sheet's own grouping, so the figure is not ordered by result.
CONDITION_ORDER = [
    "No mb",
    "C6", "C2", "Alk1", "TGFb",
    "C6+Alk1", "C2+Alk1", "C6+TGFb", "C2+TGFb", "Alk1+TGFb",
    "C6 full", "C2 full",
    "C6+TNFalpha", "TNFalpha",
]

# Technical failure, identified from the Desmin channel BEFORE the plate map
# arrived (dbs p99 = 329 vs 1,066-2,331 plate-wide; nuclei and background
# normal). Excluded by default from condition means as a staining/acquisition
# failure, and always reported alongside the included value.
TECHNICAL_FAILURES = {"B11": "Desmin channel effectively empty (dbs p99=329)"}

# Legacy single-well handle kept so older artifacts stay readable. The control is
# now a GROUP (`CONTROL_CONDITION`), not this well.
CTRL = "23_B02"
CTRL_IS_ASSUMED = False

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
    """Condition for a well stem, from the transcribed layout sheet."""
    wid = well_id(stem)
    try:
        return CONDITION_BY_WELL[wid]
    except KeyError:
        raise KeyError(f"{stem} ({wid}) is not on the PLATE 44 layout sheet "
                       "-- no condition is invented for unmapped wells") from None


def is_technical_failure(stem: str) -> bool:
    return well_id(stem) in TECHNICAL_FAILURES


def _selfcheck() -> None:
    """The map must cover the imaged wells exactly -- no gaps, no extras."""
    imaged = {well_id(w) for w in wells()}
    mapped = set(CONDITION_BY_WELL)
    assert imaged == mapped, (f"map/image mismatch: "
                              f"unmapped={sorted(imaged - mapped)} "
                              f"unimaged={sorted(mapped - imaged)}")
    assert set(CONDITION_ORDER) == set(CONDITION_BY_WELL.values()), \
        "CONDITION_ORDER does not match the map's conditions"
    assert CONTROL_CONDITION in CONDITION_ORDER


if __name__ == "__main__":
    ws = wells()
    _selfcheck()
    print(f"PLATE 44: {len(ws)} wells, {UM} um/px, DAPI=ch{DAPI_CH}, "
          f"Desmin=ch{DESMIN_CH}")
    print(f"  ring {RING_UM} um = {RING_PX} px | tophat {TOPHAT_UM} um = "
          f"{TOPHAT_PX} px | nucleus {AMIN_UM2}-{AMAX_UM2} um2 = "
          f"{AMIN_UM2/UM2:.0f}-{AMAX_UM2/UM2:.0f} px")
    print(f"  layout self-check PASSED: {len(CONDITION_BY_WELL)} wells mapped, "
          f"{len(CONDITION_ORDER)} conditions")
    from collections import Counter
    n = Counter(CONDITION_BY_WELL.values())
    for c in CONDITION_ORDER:
        ids = sorted(w for w, v in CONDITION_BY_WELL.items() if v == c)
        flag = "  <- CONTROL" if c == CONTROL_CONDITION else ""
        fail = [w for w in ids if w in TECHNICAL_FAILURES]
        print(f"    {c:<14} n={n[c]}  {', '.join(ids)}"
              f"{'  [tech-fail: ' + ','.join(fail) + ']' if fail else ''}{flag}")
