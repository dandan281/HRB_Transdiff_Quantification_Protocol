"""Linker training config: which wells to train on, candidate-generation + labeling params.

Reuses benchmark.config for file paths and the well registry. Only wells with COMPLETE, trustworthy
GT are used — partial re-traces (C09, B06) and the long-skewed 32_C08 GT are excluded, because a
fragment lying on an un-traced fibre would be mislabeled "don't-join" and poison the classifier.
"""
from pathlib import Path

from benchmark import config as BC

# candidate generation (mirrors common/merge.py neighbour search; angle left to the classifier)
GAP_MAX_PX = 170.0     # only endpoint pairs within this distance are join candidates
DIRPTS = 6             # vertices used to estimate a fragment end's outward heading

# auto-labeling
FRAG_TO_GT_COV = 0.5      # a fragment maps to a GT fibre if >=50% of its dilated pixels lie on it
FRAG_RADIUS_PX = BC.DILATE_RADIUS_PX      # 5 px: the fragment's own footprint
GT_ASSOC_RADIUS_PX = 12.0                 # ~fibre half-width band around the GT centerline for
                                          # ASSOCIATION (a fragment anywhere on the fibre body maps);
                                          # wider than the benchmark's detection tolerance on purpose
BOTH_MAPPED_ONLY = True   # train only on pairs where BOTH fragments map to a GT fibre -> clean
                          # "same fibre vs different fibre?" signal, no noise-fragment label guessing
INCLUDE_DIM = True        # pool stage-3 dim_traces.txt with raw bright fragments, so the linker
                          # learns to bridge dim gaps between bright segments of the same fibre

# training wells (complete GT + a completed stage-2 run with bright_segments.txt)
TRAIN_WELL_IDS = [
    "P23_B03_ACT104_EGFR",   # PLATE_23
    "P32_B02_Ctrl",          # PLATE_32
    "P32_C02_ACT104_FGFR",   # PLATE_32
    "P32_C03_ACT104_EGFR",   # PLATE_32
]
# excluded from training (documented): P23_C08 (partial/long-skewed GT), P23_C09 & P23_B06 (partial
# re-trace ROI zips), P23_B02 (no GT zip), P23_C05 (corrupt GT zip).

DATA = Path(__file__).resolve().parent / "data"
MODELS = Path(__file__).resolve().parent / "models"


def fragments_path(run_stem):
    return BC.RUNS / run_stem / "stage2_bright" / "bright_segments.txt"


def train_wells():
    return [BC.well_by_id(wid) for wid in TRAIN_WELL_IDS]
