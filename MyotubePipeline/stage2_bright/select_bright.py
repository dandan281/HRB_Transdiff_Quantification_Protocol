"""Stage 2 -- Bright / Long tracer.

Merges the raw ridge segments into traces, then selects the OBVIOUS ones: long AND bright.
Writes the full merged set (`all_traces.txt`, exposed for Stage 3) and the selected
`bright_traces.txt` (spatially ordered). No Fiji here -- length is geometric, brightness is
sampled from the Stage-1 signal map.

Usage:
  python select_bright.py --stage1 <dir> --segments <bright_segments.txt> --out <stage2 dir>
                          [--l-long PX] [--bright-floor V]
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
from signalmap import load_signal, brightness_mean  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--l-long", type=float, default=None)
    ap.add_argument("--bright-floor", type=float, default=None)
    a = ap.parse_args()

    signal = load_signal(os.path.join(a.stage1, "signal.png"))
    segs = read_traces(a.segments)
    traces = merge(segs, signal)
    write_traces(os.path.join(a.out, "all_traces.txt"), traces)

    lengths = np.array([polylen(t) for t in traces]) if traces else np.array([])
    bright = np.array([brightness_mean(signal, t) for t in traces]) if traces else np.array([])

    # adaptive defaults: the longer/brighter ~half are "obvious". Tunable per calibration.
    l_long = a.l_long if a.l_long is not None else (
        float(np.clip(np.percentile(lengths, 50), 180, 400)) if lengths.size else 180.0)
    bfloor = a.bright_floor if a.bright_floor is not None else (
        float(np.percentile(bright, 45)) if bright.size else 0.0)

    keep = [i for i in range(len(traces)) if lengths[i] >= l_long and bright[i] >= bfloor]
    keep = [keep[k] for k in spatial_order([traces[i] for i in keep])]  # spatial order
    bright_traces = [traces[i] for i in keep]
    write_traces(os.path.join(a.out, "bright_traces.txt"), bright_traces)

    with open(os.path.join(a.out, "bright_table.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "src_index", "length_px", "brightness", "mid_x", "mid_y"])
        for newid, i in enumerate(keep, 1):
            mx, my = midpoint(traces[i])
            w.writerow([newid, i, round(float(lengths[i]), 1), round(float(bright[i]), 1),
                        round(mx, 1), round(my, 1)])

    print(f"merged={len(traces)} bright_kept={len(keep)} "
          f"(l_long={l_long:.0f}, bright_floor={bfloor:.1f})")


if __name__ == "__main__":
    main()
