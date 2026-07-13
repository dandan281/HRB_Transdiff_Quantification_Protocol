"""Render my automated tracing (full learned stack) beside the manual ground-truth ROIs on the
actual fibre image, for visual comparison.

  python -m linker overlay                     # default PLATE_23 wells with GT
  python -m linker overlay P23_C08_BR223_IGF1R # one well
"""
from __future__ import annotations

import os
import sys

import numpy as np

from benchmark import config as BC
from benchmark import io_load as io
from . import chain as CH
from . import config as LC
from . import dataset as D
from . import extend as EX
from . import tracefilter as TF

sys.path.insert(0, os.path.join(BC.PIPE, "common"))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlays")
MINE = "#39FF14"      # green = my automated traces
MANUAL = "#FF3B3B"    # red = manual ground truth


def run_stack(wid, link_thr=0.5, keep_thr=0.5, extend_thr=50.0):
    """Full learned pipeline: raw fragments -> chain -> keep/drop filter -> endpoint-extend."""
    import joblib
    from signalmap import load_signal
    w = BC.well_by_id(wid)
    frags = D.load_fragments(w["run_stem"])
    signal = load_signal(str(BC.RUNS / w["run_stem"] / "stage1_threshold" / "signal.png"))
    link_pipe, link_feats = CH._load_model(LC.MODELS / "link.joblib")
    fm = joblib.load(LC.MODELS / "trace_filter.joblib")
    clf, ff = fm["clf"], fm["features"]
    chained = CH.chain_fragments(frags, link_pipe, link_feats, link_thr)
    Xf = np.array([[TF.trace_features(t, signal)[k] for k in ff] for t in chained], float)
    filt = [t for t, keep in zip(chained, clf.predict_proba(Xf)[:, 1] >= keep_thr) if keep]
    return EX.extend_all(filt, signal, thr=extend_thr), w


def render_well(wid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile

    mine, w = run_stack(wid)
    gt = io.read_imagej_zip(BC.QPLATES / w["plate"] / w["roi_zip"])
    img = tifffile.imread(str(BC.RUNS / w["run_stem"] / "stage1_threshold" / "ch1_adjusted8.tif"))

    fig, ax = plt.subplots(1, 3, figsize=(33, 11.6))
    for a in ax:
        a.imshow(img, cmap="gray", interpolation="nearest")
        a.axis("off")
    for t in mine:
        t = np.asarray(t); ax[0].plot(t[:, 0], t[:, 1], "-", color=MINE, lw=0.7)
    for t in gt:
        t = np.asarray(t); ax[1].plot(t[:, 0], t[:, 1], "-", color=MANUAL, lw=0.7)
    for t in mine:
        t = np.asarray(t); ax[2].plot(t[:, 0], t[:, 1], "-", color=MINE, lw=0.7, alpha=0.85)
    for t in gt:
        t = np.asarray(t); ax[2].plot(t[:, 0], t[:, 1], "-", color=MANUAL, lw=0.7, alpha=0.6)
    ax[0].set_title(f"MINE — automated learned stack: {len(mine)} traces", color=MINE, fontsize=15)
    ax[1].set_title(f"MANUAL — ground truth: {len(gt)} ROIs", color=MANUAL, fontsize=15)
    ax[2].set_title("OVERLAID (green=mine, red=manual)", fontsize=15)
    fig.suptitle(f"{wid}   ({w['plate']}, {w['group']})", fontsize=17)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{wid}_overlay.png")
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  {wid:24s} mine={len(mine):4d}  manual={len(gt):4d}  -> {p}")
    return p


def main(wids=None):
    wids = wids or ["P23_C08_BR223_IGF1R", "P23_B03_ACT104_EGFR",
                    "P23_C09_BR223_TrkA", "P23_B06_ACT104_TrkA"]
    for wid in wids:
        render_well(wid)
    print(f"\noverlays in {OUT}")
