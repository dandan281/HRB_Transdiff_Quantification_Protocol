"""traces.txt <-> polylines  (pure Python, no napari/cellpose import).

`traces.txt` is the pipeline's canonical polyline interchange (see ../../conventions.md):
one trace per line, a flat CSV of vertices ``x0,y0,x1,y1,...`` in full-image PIXEL coords
(origin top-left, x = column, y = row). Keeping the reader/writer here means the napari
plugin round-trips byte-compatibly with the existing Fiji / Python stages.
"""
from __future__ import annotations

import numpy as np


def read_traces(path) -> list[np.ndarray]:
    """Return a list of (N, 2) float arrays in (x, y) image coords.

    Lines with fewer than 2 vertices (< 4 numbers) are skipped, matching the pipeline's
    "a valid trace has >= 2 vertices" rule.
    """
    out: list[np.ndarray] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            nums = [float(v) for v in line.split(",") if v != ""]
            if len(nums) < 4:
                continue
            out.append(np.asarray(nums, dtype=float).reshape(-1, 2))
    return out


def write_traces(path, polylines) -> int:
    """Write a list of (N, 2) (x, y) arrays as traces.txt (2-decimal, pipeline format).

    Returns the number of traces written.
    """
    lines = []
    for p in polylines:
        p = np.asarray(p, dtype=float).reshape(-1, 2)
        if len(p) < 2:
            continue
        lines.append(",".join(f"{v:.2f}" for v in p.reshape(-1)))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")
    return len(lines)
