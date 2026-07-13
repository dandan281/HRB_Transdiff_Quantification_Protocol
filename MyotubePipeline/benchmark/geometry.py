"""Length and rasterization primitives.

Length is the sum of Euclidean segment lengths over ordered polyline vertices — byte-for-byte the
definition Fiji's getValue("Length") uses on an open polyline (common/geometry.polylen). NEVER a
skeleton pixel walk.

Matching primitive: each centerline is rasterized INDIVIDUALLY (line + dilation) and stored as the
sorted set of packed global pixel indices (y*IMAGE_W + x) of its foreground. Per-fibre, never a
shared label image (so crossing fibres don't overwrite each other); packed-index storage keeps
memory bounded by true-pixel count, not by a long diagonal fibre's huge bounding box.
"""
from __future__ import annotations

import numpy as np


def polylen(pts: np.ndarray) -> float:
    """Euclidean segment-sum length in pixels."""
    d = np.diff(np.asarray(pts, dtype=float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def length_um(pts: np.ndarray, pixel_um: float) -> float:
    return polylen(pts) * pixel_um


class Mask:
    """One dilated centerline as sorted packed global pixel indices + its bbox."""
    __slots__ = ("idx", "bbox", "area")

    def __init__(self, idx, bbox, area):
        self.idx = idx            # sorted unique int64 array of (y*IMAGE_W + x)
        self.bbox = bbox          # (y0, x0, y1, x1), y1/x1 exclusive
        self.area = area


def rasterize(pts: np.ndarray, radius: float, image_w: int) -> Mask:
    """Rasterize an (x,y) polyline into a dilated Mask of packed global pixel indices."""
    from skimage.draw import line as sk_line
    from skimage.morphology import binary_dilation, disk

    pts = np.asarray(pts, dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]
    r = int(round(radius))
    pad = r + 1
    gx0 = int(np.floor(xs.min())) - pad
    gy0 = int(np.floor(ys.min())) - pad
    W = int(np.ceil(xs.max())) - gx0 + pad + 1
    H = int(np.ceil(ys.max())) - gy0 + pad + 1
    m = np.zeros((H, W), dtype=bool)
    ix = np.clip(np.round(xs).astype(int) - gx0, 0, W - 1)
    iy = np.clip(np.round(ys).astype(int) - gy0, 0, H - 1)
    for k in range(len(ix) - 1):
        rr, cc = sk_line(int(iy[k]), int(ix[k]), int(iy[k + 1]), int(ix[k + 1]))
        m[rr, cc] = True
    if r > 0:
        m = binary_dilation(m, disk(r))
    ly, lx = np.nonzero(m)                      # local rows/cols
    gy = ly + gy0
    gx = lx + gx0
    idx = np.sort((gy.astype(np.int64) * image_w + gx.astype(np.int64)))
    bbox = (gy0, gx0, gy0 + H, gx0 + W)
    return Mask(idx, bbox, int(idx.size))


def intersection_area(a: Mask, b: Mask) -> int:
    """Overlap area = size of the intersection of two packed-index sets (both sorted, unique)."""
    return int(np.intersect1d(a.idx, b.idx, assume_unique=True).size)


def bbox_overlap_pairs(masks_a: list[Mask], masks_b: list[Mask]) -> np.ndarray:
    """Vectorized bbox-overlap prefilter -> (K,2) int array of (i,j) candidate pairs to score exactly."""
    if not masks_a or not masks_b:
        return np.empty((0, 2), dtype=int)
    A = np.array([m.bbox for m in masks_a], dtype=np.int64)   # (Na,4) y0,x0,y1,x1 (y1,x1 exclusive)
    B = np.array([m.bbox for m in masks_b], dtype=np.int64)
    ay0, ax0, ay1, ax1 = (A[:, 0][:, None], A[:, 1][:, None], A[:, 2][:, None], A[:, 3][:, None])
    by0, bx0, by1, bx1 = (B[:, 0][None, :], B[:, 1][None, :], B[:, 2][None, :], B[:, 3][None, :])
    ov = (ay0 < by1) & (by0 < ay1) & (ax0 < bx1) & (bx0 < ax1)
    ii, jj = np.nonzero(ov)
    return np.stack([ii, jj], axis=1)
