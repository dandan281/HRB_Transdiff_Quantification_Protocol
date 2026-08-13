"""Instance annotation schema with overlap-safe, uncompressed COCO RLE masks."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi

VALID_STATUSES = {"complete", "border_truncated", "occluded", "ambiguous"}
AUTHORITATIVE_STATUSES = {"complete"}


def encode_rle(mask: np.ndarray) -> dict:
    """Encode a 2-D mask as uncompressed COCO RLE (column-major order)."""
    a = np.asarray(mask, dtype=np.uint8)
    if a.ndim != 2:
        raise ValueError("RLE masks must be 2-D")
    flat = a.ravel(order="F")
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [flat.size]))).astype(int).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"size": [int(a.shape[0]), int(a.shape[1])], "counts": counts}


def encode_sparse_positions(shape: tuple[int, int], positions_f: np.ndarray) -> dict:
    """Encode sorted foreground indices in Fortran-flat coordinates without a full mask scan."""
    positions = np.unique(np.asarray(positions_f, dtype=np.int64))
    total = int(np.prod(shape))
    if not positions.size:
        return {"size": list(shape), "counts": [total]}
    if positions[0] < 0 or positions[-1] >= total:
        raise ValueError("foreground position outside image")
    breaks = np.flatnonzero(np.diff(positions) > 1)
    starts = positions[np.concatenate(([0], breaks + 1))]
    ends = positions[np.concatenate((breaks, [positions.size - 1]))]
    counts: list[int] = []
    cursor = 0
    for start, end in zip(starts, ends):
        counts.extend((int(start - cursor), int(end - start + 1)))
        cursor = int(end + 1)
    counts.append(total - cursor)
    return {"size": [int(shape[0]), int(shape[1])], "counts": counts}


def decode_rle(rle: dict) -> np.ndarray:
    size = tuple(int(v) for v in rle["size"])
    total = int(np.prod(size))
    out = np.empty(total, dtype=bool)
    pos = 0
    value = False
    for raw_count in rle["counts"]:
        count = int(raw_count)
        if count < 0 or pos + count > total:
            raise ValueError("invalid RLE run")
        out[pos:pos + count] = value
        pos += count
        value = not value
    if pos != total:
        raise ValueError(f"invalid RLE length: decoded {pos}, expected {total}")
    return out.reshape(size, order="F")


def rle_foreground_positions(rle: dict) -> np.ndarray:
    """Return foreground Fortran-flat indices, proportional to object area rather than image area."""
    chunks = []
    cursor = 0
    foreground = False
    for raw_count in rle["counts"]:
        count = int(raw_count)
        if foreground and count:
            chunks.append(np.arange(cursor, cursor + count, dtype=np.int64))
        cursor += count
        foreground = not foreground
    expected = int(np.prod(rle["size"]))
    if cursor != expected:
        raise ValueError(f"invalid RLE length: decoded {cursor}, expected {expected}")
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def decode_rle_cropped(rle: dict) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Decode only the tight object bounding box as ``(r0,c0,r1,c1), mask``."""
    shape = tuple(int(v) for v in rle["size"])
    positions = rle_foreground_positions(rle)
    if not positions.size:
        raise ValueError("empty RLE mask")
    rows = positions % shape[0]
    cols = positions // shape[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    crop = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    crop[rows - r0, cols - c0] = True
    return (r0, c0, r1, c1), crop


@dataclass
class InstanceRecord:
    id: str
    status: str
    rle: dict
    source: str = "manual"
    confidence: float | None = None
    reviewed: bool = False
    notes: str = ""

    def is_authoritative(self, effective_status: str | None = None) -> bool:
        """Authority requires both expert review and a complete effective status."""
        return bool(self.reviewed and (effective_status or self.status) in AUTHORITATIVE_STATUSES)

    def validate(self, image_shape: tuple[int, int]) -> None:
        if not self.id:
            raise ValueError("instance id cannot be empty")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"{self.id}: invalid status {self.status!r}")
        if tuple(self.rle.get("size", ())) != tuple(image_shape):
            raise ValueError(f"{self.id}: mask shape does not match image")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"{self.id}: confidence must be in [0,1]")
        if not rle_foreground_positions(self.rle).size:
            raise ValueError(f"{self.id}: empty mask")


@dataclass
class InstanceSet:
    image_shape: tuple[int, int]
    image_id: str
    instances: list[InstanceRecord]
    schema_version: str = "1.0"
    annotation_policy: str = "precision-first-v1"
    provenance: dict = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.image_shape) != 2 or min(self.image_shape) <= 0:
            raise ValueError("image_shape must be positive (height,width)")
        ids = [x.id for x in self.instances]
        if len(ids) != len(set(ids)):
            raise ValueError("instance ids must be unique")
        for record in self.instances:
            record.validate(tuple(self.image_shape))

    def masks(self) -> Iterable[tuple[InstanceRecord, np.ndarray]]:
        for record in self.instances:
            yield record, decode_rle(record.rle)

    def cropped_masks(self) -> Iterable[tuple[InstanceRecord, tuple[int, int, int, int], np.ndarray]]:
        for record in self.instances:
            bbox, mask = decode_rle_cropped(record.rle)
            yield record, bbox, mask

    def save(self, path: str | Path) -> None:
        self.validate()
        payload = asdict(self)
        payload["image_shape"] = list(self.image_shape)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InstanceSet":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records = [InstanceRecord(**x) for x in payload.pop("instances")]
        payload["image_shape"] = tuple(payload["image_shape"])
        result = cls(instances=records, **payload)
        result.validate()
        return result


def from_label_image(labels: np.ndarray, image_id: str, *, source: str = "imported",
                     reviewed: bool = False, default_status: str = "ambiguous") -> InstanceSet:
    """Convert a mutually exclusive label image to an overlap-safe instance set.

    Imported labels default to ambiguous so an automated proposal cannot silently become
    authoritative. Set ``default_status='complete', reviewed=True`` only for curated labels.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("label image must be 2-D")
    records = []
    objects = ndi.find_objects(labels)
    for label_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        local_rows, local_cols = np.nonzero(labels[slices] == label_id)
        rows = local_rows + int(slices[0].start)
        cols = local_cols + int(slices[1].start)
        positions = rows.astype(np.int64) + cols.astype(np.int64) * labels.shape[0]
        records.append(InstanceRecord(
            id=f"myotube_{int(label_id):04d}", status=default_status,
            source=source, reviewed=reviewed,
            rle=encode_sparse_positions(tuple(labels.shape), positions),
        ))
    result = InstanceSet(tuple(labels.shape), image_id, records)
    result.validate()
    return result
