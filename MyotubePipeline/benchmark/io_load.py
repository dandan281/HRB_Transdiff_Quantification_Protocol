"""Load and normalize ground truth + predictions into on-disk (x,y) pixel coordinates.

All readers return coordinates in the pipeline's canonical (x, y) = (column, row) order, origin
top-left. NO napari (y,x) swap is applied anywhere in the harness.
"""
from __future__ import annotations

import csv
import math
import re
import zipfile

import numpy as np


# ---- ground-truth ROI geometry --------------------------------------------------------------
def read_imagej_zip(path) -> list[np.ndarray]:
    """Read an ImageJ ROI .zip (or single .roi) into a list of (N,2) float (x,y) polylines.

    Inlined (no napari-plugin dependency) via roifile. Skips ROIs with < 2 vertices.
    """
    import roifile

    rois = roifile.roiread(str(path))
    if not isinstance(rois, (list, tuple)):
        rois = [rois]
    out = []
    for r in rois:
        coords = r.coordinates()          # (N,2) in (x,y); subpixel when present
        pts = np.asarray(coords, dtype=float)
        if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] == 2:
            out.append(pts)
    return out


def gt_zip_ok(path) -> bool:
    """True if the path is a real, non-empty ImageJ ROI zip (or a single .roi)."""
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                return any(n.endswith(".roi") for n in zf.namelist())
        # single-.roi files start with the ImageJ magic 'Iout'
        with open(path, "rb") as f:
            return f.read(4) == b"Iout"
    except OSError:
        return False


# ---- ground-truth lengths (scientific endpoint) ---------------------------------------------
def extract_lengths(path) -> np.ndarray:
    """Port of myotube_length_analysis.R extract_lengths: Length is the LAST field of each data
    row, delimiter [,\\t]; keep finite values > 0. Handles both CSV schemas (with/without a Label
    column) and the tab-delimited D08 file. Header/zero-length rows drop out naturally.
    """
    vals = []
    with open(path, encoding="utf-8-sig") as f:
        for ln in f:
            parts = re.split(r"[,\t]", ln.strip())
            if len(parts) < 5:
                continue
            try:
                v = float(parts[-1].strip())
            except ValueError:
                continue
            if math.isfinite(v) and v > 0:
                vals.append(v)
    return np.asarray(vals, dtype=float)


# ---- predictions ----------------------------------------------------------------------------
def read_traces(path) -> list[np.ndarray]:
    """Read a pipeline *_traces.txt into a list of (N,2) float (x,y) polylines.

    One trace per line, flat comma-separated x0,y0,x1,y1,...; lines with < 4 numbers skipped.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            nums = [float(t) for t in ln.split(",") if t.strip() != ""]
            if len(nums) < 4:
                continue
            out.append(np.asarray(nums, dtype=float).reshape(-1, 2))
    return out


def read_pred_lengths(path) -> np.ndarray:
    """Read length_um column from a stage5 *_results.csv (utf-8-sig, header row)."""
    vals = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        col = "length_um" if reader.fieldnames and "length_um" in reader.fieldnames else None
        for row in reader:
            try:
                v = float(row[col]) if col else float(list(row.values())[4])
            except (ValueError, TypeError, IndexError, KeyError):
                continue
            if math.isfinite(v) and v > 0:
                vals.append(v)
    return np.asarray(vals, dtype=float)
