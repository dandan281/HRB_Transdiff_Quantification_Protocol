"""Stage 3 -- Dim / Short tracer.

Selects the fibers Stage 2 left behind: the short and the dim. Starts from the full merged set
(Stage 2's all_traces.txt) minus the bright traces, optionally augmented with a dim-boost
detection pass (faint fibers the standard pass missed). Traces that lie on top of a bright fiber
(within a SMALL dilation of the bright mask) are NOT discarded silently -- they are written to
`excluded_by_brightmask.txt` so Stage 4 can review what the mask suppressed.

Usage:
  python select_dim.py --stage1 <dir> --stage2 <dir> --out <stage3 dir>
        [--dim-segments <dim_segments.txt>] [--no-recover-mask]
        [--dilate PX] [--min-len PX] [--min-bright V]
"""
from __future__ import annotations
import os
import sys
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from iohelpers import read_traces, write_traces  # noqa: E402
from geometry import polylen, midpoint, spatial_order  # noqa: E402
from merge import merge  # noqa: E402
from recovery import recover_signal_traces  # noqa: E402
from signalmap import load_signal, brightness_mean, rasterize_traces, overlap_fraction  # noqa: E402

BRIGHTMASK_OVL = 0.60     # >60% of a trace on the bright mask => it's a bright fragment, not dim
NEW_FIBER_OVL = 0.40      # dim-boost trace overlapping <40% of existing traces => genuinely new


def sig(t):
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim-segments", default=None)
    ap.add_argument("--dilate", type=int, default=8)        # SMALL radius (user feedback)
    ap.add_argument("--min-len", type=float, default=60.0)
    # >= FIBER_T (15): Stage 4's dark-gap detector treats <15 as "no fiber", so a kept dim fibre
    # must clear that bar, else it reads as one long dark gap and gets proposed for splitting.
    ap.add_argument("--min-bright", type=float, default=15.0)
    ap.add_argument("--no-recover-mask", action="store_true",
                    help="disable residual-signal skeleton recovery for missed Desmin-positive fibers")
    ap.add_argument("--recover-threshold", type=int, default=25)
    ap.add_argument("--recover-min-len", type=float, default=80.0)
    ap.add_argument("--recover-min-bright", type=float, default=18.0)
    a = ap.parse_args()

    signal = load_signal(os.path.join(a.stage1, "signal.png"))
    shape = signal.shape

    all_traces = read_traces(os.path.join(a.stage2, "all_traces.txt"))
    bright_traces = read_traces(os.path.join(a.stage2, "bright_traces.txt"))
    bright_sigs = {sig(t) for t in bright_traces}

    # dim candidates = merged traces NOT already taken as bright
    candidates = [t for t in all_traces if sig(t) not in bright_sigs]

    # dim-boost: recover faint fibers a standard detection missed (genuinely new geometry only)
    if a.dim_segments and os.path.exists(a.dim_segments):
        boost = merge(read_traces(a.dim_segments), signal)
        existing_mask = rasterize_traces(all_traces, shape, dilate=a.dilate)
        new_boost = [t for t in boost if overlap_fraction(existing_mask, t) < NEW_FIBER_OVL]
        candidates += new_boost
        print(f"dim-boost: {len(boost)} merged, {len(new_boost)} genuinely new")

    # Mask/skeleton recovery: Ridge Detection misses some broad or low-contrast Desmin-positive
    # fibers. Recover long, bright residual centerlines as additional dim candidates.
    recovered = []
    if not a.no_recover_mask:
        recovered = recover_signal_traces(
            signal, all_traces,
            threshold=a.recover_threshold,
            existing_dilate=a.dilate,
            min_len_px=a.recover_min_len,
            min_brightness=a.recover_min_bright,
        )
        candidates += recovered

    bright_mask = rasterize_traces(bright_traces, shape, dilate=a.dilate)

    kept, excluded = [], []
    for t in candidates:
        L = polylen(t)
        b = brightness_mean(signal, t)
        if L < a.min_len or b < a.min_bright:
            continue                                   # below noise floor: drop entirely
        if overlap_fraction(bright_mask, t) > BRIGHTMASK_OVL:
            excluded.append(t)                         # belongs to a bright fiber -> review layer
        else:
            kept.append(t)

    order = spatial_order(kept)
    kept = [kept[i] for i in order]
    write_traces(os.path.join(a.out, "dim_traces.txt"), kept)
    write_traces(os.path.join(a.out, "excluded_by_brightmask.txt"), excluded)

    with open(os.path.join(a.out, "dim_table.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "length_px", "brightness", "mid_x", "mid_y"])
        for newid, t in enumerate(kept, 1):
            mx, my = midpoint(t)
            w.writerow([newid, round(polylen(t), 1), round(brightness_mean(signal, t), 1),
                        round(mx, 1), round(my, 1)])

    print(f"candidates={len(candidates)} recovered={len(recovered)} dim_kept={len(kept)} "
          f"excluded_by_brightmask={len(excluded)} (dilate={a.dilate})")


if __name__ == "__main__":
    main()
