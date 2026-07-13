"""Constants, matching parameters, and the Q_Plates well registry.

The registry is the single source of truth mapping each benchmark well to its ground-truth files
(under Q_Plates) and its pipeline run dir (under runs/). Canonical Results.csv variant per well
follows myotube_length_analysis.R; where the R table names a variant that is absent on disk, the
_N variant actually present is used (noted inline).
"""
from pathlib import Path

# --- paths
QPLATES = Path("c:/Users/liqig/Desktop/Q_Plates")
PIPE = Path(__file__).resolve().parent.parent          # MyotubePipeline/
RUNS = PIPE / "runs"
OUT = Path(__file__).resolve().parent / "out"

# --- calibration / science
PIXEL_UM = 0.6492666666666667      # = 2360.74/3636; single constant used on BOTH GT and pred
THRESH_UM = 300.0                  # scientific split point: % below/above 300 um
IMAGE_SHAPE = (3636, 3636)         # (H, W) full-res; sanity clamp only (masks are bbox-local)

# --- Tier-1 matching parameters (see design rationale)
DILATE_RADIUS_PX = 5.0     # lateral match tolerance (~3.25 um); absorbs vertex-density/subpixel jitter
MIN_OVERLAP_FRAC = 0.5     # edge in the overlap graph if one fibre covers >=50% of the other -> split/merge
FRAG_COV_FRAC = 0.2        # a pred covering >=20% of a GT counts as a piece of it -> fragmentation metric
IOU_THRESH = 0.2           # acceptance gate for a 1:1 detection match (length-tolerant on purpose)
TOO_SHORT_RATIO = 0.8      # matched pred < 0.8x GT length -> too-short error class
TOO_LONG_RATIO = 1.25      # matched pred > 1.25x GT length -> over-extended (reported, not a named class)
BOUNDARY_SIGMA_UM = 100.0  # Gaussian weight width around 300 um for boundary-weighted length MAE

# --- well registry ----------------------------------------------------------------------------
# Fields: well_id, plate, group, roi_zip, results_csv, run_stem (None if no pipeline run),
#         split, gt_roi_ok (False = geometry unusable -> Tier-2 only), gt_geom_partial (ROI zip
#         is an incomplete re-trace vs Results.csv -> Tier-1 unreliable).
# Excluded entirely: PLATE_23/B02_Ctrl (no ROI zip at all).
def _w(well_id, plate, group, roi_zip, results_csv, run_stem=None, split="heldout",
       gt_roi_ok=True, gt_geom_partial=False):
    return dict(well_id=well_id, plate=plate, group=group, roi_zip=roi_zip,
                results_csv=results_csv, run_stem=run_stem, split=split,
                gt_roi_ok=gt_roi_ok, gt_geom_partial=gt_geom_partial)

WELLS = [
    # Ctrl
    _w("P26_B02_Ctrl", "PLATE_26", "Ctrl", "B02_Ctrl_ROIs.zip", "B02_Ctrl_Results.csv",
       run_stem="P26_23_B02_ctrl", split="test_p26"),
    _w("P28_B02_Ctrl", "PLATE_28", "Ctrl", "B02_Ctrl_ROIs.zip", "B02_Ctrl_Results.csv"),
    _w("P32_B02_Ctrl", "PLATE_32", "Ctrl", "B02_Ctrl_ROIs.zip", "B02_Ctrl_Results.csv",
       run_stem="P32_23_B02_ctrl", split="dev"),
    # BR223_EGFR
    _w("P23_C05_BR223_EGFR", "PLATE_23", "BR223_EGFR", "C05_BR223_EGFR_ROIs.zip",
       "C05_BR223_EGFR_Results.csv", run_stem="29_C05_br223_egfrc", split="test", gt_roi_ok=False),
    _w("P28_B04_BR223_EGFR", "PLATE_28", "BR223_EGFR", "B04_BR223_EGFR_ROIs.zip", "B04_BR223_EGFR_Results.csv"),
    _w("P28_E08_BR223_EGFR", "PLATE_28", "BR223_EGFR", "E08_BR223_EGFR_ROIs.zip", "E08_BR223_EGFR_Results.csv"),
    # BR223_IGF1R
    _w("P23_C08_BR223_IGF1R", "PLATE_23", "BR223_IGF1R", "C08_BR223_IGF1R_ROIs_N.zip",
       "C08_BR223_IGF1R_Results_N.csv", run_stem="32_C08_br223_igf1r", split="test"),
    _w("P26_C08_BR223_IGF1R", "PLATE_26", "BR223_IGF1R", "C08_BR223_IGF1R_ROIs.zip",
       "C08_BR223_IGF1R_Results.csv", run_stem="P26_32_C08_br223_igf1r", split="test_p26"),
    _w("P28_E10_BR223_IGF1R", "PLATE_28", "BR223_IGF1R", "E10_BR223_IGF1R_ROIs.zip", "E10_BR223_IGF1R_Results.csv"),
    _w("P32_D08_BR223_IGF1R", "PLATE_32", "BR223_IGF1R", "D08_IGF1R_BR223_ROIs_N.zip",
       "D08_IGF1R_BR223_Results_N.csv"),  # R names a _TBC variant that is absent; _N is canonical
    _w("P32_C11_BR223_IGF1R", "PLATE_32", "BR223_IGF1R", "C11_IGF1R_BMPR2-11m2_ROIs.zip", "C11_IGF1R_BMPR2-11m2_Results.csv"),
    # BR223_TrkA
    _w("P23_C09_BR223_TrkA", "PLATE_23", "BR223_TrkA", "C09_BR223_TrkA_ROIs_N.zip",
       "C09_BR223_TrkA_Results.csv", run_stem="33_C09_br223_trka", split="test", gt_geom_partial=True),
    _w("P32_D09_BR223_TrkA", "PLATE_32", "BR223_TrkA", "D09_TrkA_BR223_ROIs_N.zip", "D09_TrkA_BR223_Results.csv"),
    _w("P32_D02_BR223_TrkA", "PLATE_32", "BR223_TrkA", "D02_TrkA_BMPR2-11m2_ROIs.zip", "D02_TrkA_BMPR2-11m2_Results.csv"),
    # BR223_HER2 (BMPR2 == BR223; m2 mutant merged)
    _w("P28_B08_BR223_HER2", "PLATE_28", "BR223_HER2", "B08_BMPR2_HER2_ROIs.zip", "B08_BMPR2_HER2_Results.csv"),
    _w("P32_D11_BR223_HER2", "PLATE_32", "BR223_HER2", "D11_HER2_BR223_ROIs_N.zip", "D11_HER2_BR223_Results_N.csv"),
    _w("P32_D04_BR223_HER2", "PLATE_32", "BR223_HER2", "D04_HER2_BR223-m2_ROIs_N.zip", "D04_HER2_BR223-m2_Results.csv"),
    # ACT104_EGFR
    _w("P23_B03_ACT104_EGFR", "PLATE_23", "ACT104_EGFR", "B03_ACT104_EGFR_ROIs.zip",
       "B03_ACT104_EGFR_Results.csv", run_stem="22_B03_act104_egfrc", split="test"),
    _w("P32_C03_ACT104_EGFR", "PLATE_32", "ACT104_EGFR", "C03_EGFR_ACT104_ROIs.zip",
       "C03_EGFR_ACT104_Results.csv", run_stem="P32_27_C03_egfrc_act104", split="dev"),
    # ACT104_TrkA
    _w("P23_B06_ACT104_TrkA", "PLATE_23", "ACT104_TrkA", "B06_ACT104_TrkA_ROIs_N.zip",
       "B06_ACT104_TrkA_Results.csv", run_stem="19_B06_act104_trka", split="test", gt_geom_partial=True),
    _w("P26_B06_ACT104_TrkA", "PLATE_26", "ACT104_TrkA", "B06_ACT104_TrkA_ROIs.zip",
       "B06_ACT104_TrkA_Results.csv", run_stem="P26_19_B06_actv104_trka", split="test_p26"),
    _w("P32_C05_ACT104_TrkA", "PLATE_32", "ACT104_TrkA", "C05_TrkA_ACT104_ROIs.zip", "C05_TrkA_ACT104_Results.csv"),
    # ACT104_FGFR
    _w("P32_C02_ACT104_FGFR", "PLATE_32", "ACT104_FGFR", "C02_FGFR_ACT104_ROIs.zip",
       "C02_FGFR_ACT104_Results.csv", run_stem="P32_26_C02_fgfr_act104", split="dev"),
]

GROUP_ORDER = ["Ctrl", "BR223_EGFR", "BR223_IGF1R", "BR223_TrkA", "BR223_HER2",
               "ACT104_EGFR", "ACT104_TrkA", "ACT104_FGFR"]


def well_by_id(well_id):
    for w in WELLS:
        if w["well_id"] == well_id:
            return w
    raise KeyError(well_id)


def pred_paths(run_stem):
    """Canonical prediction artifacts for a run. Returns (final_traces.txt, final_results.csv)."""
    rd = RUNS / run_stem
    return rd / "stage4_qc" / "final_traces.txt", rd / "stage5_measure" / "final_results.csv"


def has_prediction(w):
    if not w["run_stem"]:
        return False
    ft, fr = pred_paths(w["run_stem"])
    return ft.exists() and fr.exists()
