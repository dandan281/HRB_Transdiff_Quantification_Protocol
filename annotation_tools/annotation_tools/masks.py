"""Overlap-safe, memory-proportional instance masks.

Each independent myotube is stored as its own boolean crop plus an origin
``(r0, c0)`` inside the full field. Two crossing myotubes therefore keep two
complete masks that may share projected pixels -- the flat-label-image
representation used by ``from_label_image`` cannot express that, which is the
whole reason the annotation lane writes ``InstanceSet`` JSON directly.

Storing tight crops keeps memory proportional to total object area rather than
(field area x instance count), so a full 3636x3636 plate with hundreds of
instances stays small.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ._schema_bridge import encode_sparse_positions


class SparseMask:
    """A boolean object mask stored as its tight bounding box.

    ``origin`` is ``(r0, c0)`` of ``crop`` within an ``image_shape`` field.
    ``crop`` is a 2-D boolean array that always has at least one True pixel and
    is always minimally tight (no all-False border rows/columns).
    """

    __slots__ = ("origin", "crop", "image_shape")

    def __init__(self, origin: tuple[int, int], crop: np.ndarray,
                 image_shape: tuple[int, int]):
        crop = np.ascontiguousarray(np.asarray(crop, dtype=bool))
        if crop.ndim != 2:
            raise ValueError("crop must be 2-D")
        rows, cols = np.nonzero(crop)
        if not rows.size:
            raise ValueError("mask is empty; empty instances are not allowed")
        # Re-tighten so the bbox is canonical regardless of the caller's padding.
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        self.origin = (int(origin[0]) + r0, int(origin[1]) + c0)
        self.crop = np.ascontiguousarray(crop[r0:r1, c0:c1])
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        br0, bc0, br1, bc1 = self.bbox
        h, w = self.image_shape
        if br0 < 0 or bc0 < 0 or br1 > h or bc1 > w:
            raise ValueError("mask falls outside the image bounds")

    @classmethod
    def from_full(cls, mask: np.ndarray,
                  image_shape: tuple[int, int] | None = None) -> "SparseMask":
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("mask must be 2-D")
        shape = tuple(image_shape) if image_shape is not None else tuple(mask.shape)
        if tuple(mask.shape) != shape:
            raise ValueError("mask shape must equal image_shape")
        return cls((0, 0), mask, shape)

    @property
    def area(self) -> int:
        return int(self.crop.sum())

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        r0, c0 = self.origin
        return (r0, c0, r0 + self.crop.shape[0], c0 + self.crop.shape[1])

    def full(self) -> np.ndarray:
        """Materialise the full-field boolean mask (use only for small fields/tests)."""
        out = np.zeros(self.image_shape, dtype=bool)
        r0, c0, r1, c1 = self.bbox
        out[r0:r1, c0:c1] = self.crop
        return out

    def fortran_positions(self) -> np.ndarray:
        """Foreground indices in Fortran-flat (column-major) order for the full field."""
        rows_local, cols_local = np.nonzero(self.crop)
        r0, c0 = self.origin
        rows = rows_local.astype(np.int64) + r0
        cols = cols_local.astype(np.int64) + c0
        return rows + cols * self.image_shape[0]

    def to_rle(self) -> dict:
        """Encode as canonical uncompressed COCO RLE without a full-field scan."""
        return encode_sparse_positions(self.image_shape, self.fortran_positions())

    def touches_border(self) -> bool:
        r0, c0, r1, c1 = self.bbox
        h, w = self.image_shape
        return r0 == 0 or c0 == 0 or r1 == h or c1 == w

    def content_hash(self) -> str:
        """Stable hash of geometry only (origin + packed crop) for tamper checks."""
        h = hashlib.sha256()
        h.update(np.asarray(self.origin, dtype=np.int64).tobytes())
        h.update(np.asarray(self.image_shape, dtype=np.int64).tobytes())
        h.update(np.asarray(self.crop.shape, dtype=np.int64).tobytes())
        h.update(np.packbits(self.crop.ravel()).tobytes())
        return h.hexdigest()


def _align(a: SparseMask, b: SparseMask) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    """Return a shared origin and the two crops padded into a common bbox."""
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    r0, c0 = min(ar0, br0), min(ac0, bc0)
    r1, c1 = max(ar1, br1), max(ac1, bc1)
    aa = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    bb = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    aa[ar0 - r0:ar1 - r0, ac0 - c0:ac1 - c0] = a.crop
    bb[br0 - r0:br1 - r0, bc0 - c0:bc1 - c0] = b.crop
    return (r0, c0), aa, bb


def union(a: SparseMask, b: SparseMask) -> SparseMask:
    (r0, c0), aa, bb = _align(a, b)
    return SparseMask((r0, c0), aa | bb, a.image_shape)


def subtract(a: SparseMask, b: SparseMask) -> SparseMask:
    (r0, c0), aa, bb = _align(a, b)
    result = aa & ~bb
    if not result.any():
        raise ValueError("subtraction removed the entire instance; delete it instead")
    return SparseMask((r0, c0), result, a.image_shape)


def iou(a: SparseMask, b: SparseMask) -> float:
    (_, _), aa, bb = _align(a, b)
    inter = int((aa & bb).sum())
    uni = int((aa | bb).sum())
    return inter / uni if uni else 0.0


def shared_pixels(a: SparseMask, b: SparseMask) -> int:
    (_, _), aa, bb = _align(a, b)
    return int((aa & bb).sum())
