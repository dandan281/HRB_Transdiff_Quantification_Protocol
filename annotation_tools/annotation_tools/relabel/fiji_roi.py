"""Read ImageJ/Fiji `.roi` files and ROI-Manager `.zip` sets.

Written rather than depending on `roifile`, because the pm-annotate environment
is pinned and this needs ~80 lines of struct parsing for the one ROI class that
matters here: hand-traced lines down a myotube.

The operator's Fiji annotation is **centrelines** -- freehand/polyline ROIs, one
per fibre. `Results.csv` confirms it: a B02 row reads Area 317 um2 against Length
469 um, i.e. a width of 0.68 um, which is one pixel. So the human supplied the
spine and not the boundary.

That is exactly the primitive `raster.snap_mask` consumes, which means an
existing Fiji session converts into training masks with no re-annotation: the
traced path fixes identity and length, and the snap recovers width from the
image instead of inventing it.

Format reference: ImageJ `ij.io.RoiDecoder`.
"""
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np

MAGIC = b"Iout"
# ij.io.RoiDecoder offsets
VERSION, TYPE, TOP, LEFT, BOTTOM, RIGHT = 4, 6, 8, 10, 12, 14
N_COORDINATES, STROKE_WIDTH, OPTIONS, COORDINATES = 16, 34, 50, 64
SUB_PIXEL_RESOLUTION = 128

TYPES = {0: "polygon", 1: "rect", 2: "oval", 3: "line", 4: "freeline",
         5: "polyline", 6: "noRoi", 7: "freehand", 8: "traced", 9: "angle",
         10: "point"}
# Everything a traced myotube can plausibly be saved as.
LINE_TYPES = {"line", "freeline", "polyline", "polygon", "freehand", "traced"}


def _s16(b, o):
    return struct.unpack_from(">h", b, o)[0]


def _u16(b, o):
    return struct.unpack_from(">H", b, o)[0]


def decode_roi(raw: bytes, name: str = "") -> dict:
    """One `.roi` blob -> ``{name, type, points:[(row, col), ...], ...}``."""
    if raw[:4] != MAGIC:
        raise ValueError(f"{name}: not an ImageJ ROI (magic {raw[:4]!r})")
    version = _u16(raw, VERSION)
    rtype = TYPES.get(raw[TYPE], f"unknown({raw[TYPE]})")
    top, left = _s16(raw, TOP), _s16(raw, LEFT)
    n = _u16(raw, N_COORDINATES)
    options = _u16(raw, OPTIONS)

    if rtype == "line":
        # A straight line stores its endpoints as floats, not a coord array.
        x1, y1, x2, y2 = struct.unpack_from(">ffff", raw, 18)
        pts = [(float(y1), float(x1)), (float(y2), float(x2))]
        return {"name": name, "type": rtype, "version": version,
                "points": pts, "n_points": 2,
                "stroke_width": _s16(raw, STROKE_WIDTH)}

    if n == 0:
        return {"name": name, "type": rtype, "version": version,
                "points": [], "n_points": 0, "stroke_width": 0}

    sub = bool(options & SUB_PIXEL_RESOLUTION) and version >= 222
    if sub:
        # Float coordinates are absolute and sit after the integer pair arrays.
        base = COORDINATES + 4 * n
        xs = struct.unpack_from(f">{n}f", raw, base)
        ys = struct.unpack_from(f">{n}f", raw, base + 4 * n)
        pts = [(float(y), float(x)) for x, y in zip(xs, ys)]
    else:
        xs = struct.unpack_from(f">{n}h", raw, COORDINATES)
        ys = struct.unpack_from(f">{n}h", raw, COORDINATES + 2 * n)
        pts = [(float(top + y), float(left + x)) for x, y in zip(xs, ys)]

    return {"name": name, "type": rtype, "version": version,
            "points": pts, "n_points": n,
            "stroke_width": _s16(raw, STROKE_WIDTH),
            "subpixel": sub}


def read_roi_set(path: str | Path) -> list[dict]:
    """A ROI-Manager `.zip`, or a bare `.roi` saved with either extension."""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:4] == MAGIC:                       # single ROI, whatever it's named
        return [decode_roi(raw, path.stem)]
    out = []
    with zipfile.ZipFile(path) as z:
        for nm in sorted(z.namelist()):
            if not nm.lower().endswith(".roi"):
                continue
            out.append(decode_roi(z.read(nm), Path(nm).stem))
    return out


def rois_to_traces(rois: list[dict], *, width_px: float, mode: str = "snap",
                   reviewer: str | None = None, source: str = "") -> list[dict]:
    """Fiji ROIs -> relabel trace records, so both paths share one rasteriser."""
    traces = []
    for i, r in enumerate(rois, start=1):
        if r["type"] not in LINE_TYPES or len(r["points"]) < 2:
            continue
        traces.append({
            "trace_id": f"fiji-{i:04d}",
            "kind": "add",
            "points": [[p[0], p[1]] for p in r["points"]],
            "width_px": width_px,
            "mode": mode,
            "reviewer": reviewer,
            "origin": "fiji_roi",
            "source": source,
            "roi_name": r["name"],
            "roi_type": r["type"],
        })
    return traces


def summarise(rois: list[dict]) -> dict:
    kinds: dict[str, int] = {}
    for r in rois:
        kinds[r["type"]] = kinds.get(r["type"], 0) + 1
    usable = [r for r in rois if r["type"] in LINE_TYPES and len(r["points"]) >= 2]
    npts = np.array([r["n_points"] for r in usable]) if usable else np.array([0])
    return {"n_rois": len(rois), "by_type": kinds, "n_usable": len(usable),
            "points_per_roi": {"median": float(np.median(npts)),
                               "min": int(npts.min()), "max": int(npts.max())}}
