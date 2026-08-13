"""Loss-masking for learned candidates: what is a target, what must be ignored.

Why this exists
---------------
`bootstrap_v1/<well>/ignore.tif` marks **only pixels where two reviewed instances
overlap** (see `model_labs/freeze_bootstrap.py`). It is not a statement about
review coverage. So a candidate that trains directly on
``labels.tif`` + ``ignore.tif`` implicitly treats every non-`complete` pixel as
background -- including the 839 `ambiguous` and 31 `border_truncated` proposals.

Measured across the six bootstrap wells:

===================  =======================
class                area (% of all 6 fields)
===================  =======================
`complete`           1.415
`ambiguous`          1.613
`border_truncated`   0.136
===================  =======================

More real-myotube pixels would be labelled background than foreground. A model
trained that way is explicitly taught to suppress genuine myotubes, which is the
opposite of the T02 goal and would not be visible in the loss curve.

The plan's rule -- "`border_truncated`, `ambiguous`, `occluded`, rejected ... are
not complete training targets" -- is about *targets*. This module supplies the
missing half: they must also not be *background*.

Semantics
---------
=====================  ==========================================================
decision               training role
=====================  ==========================================================
`complete` (reviewed)  foreground instance target
`ambiguous`            **ignore** - the operator declined to assert an identity,
                       so the pixel must contribute no loss in either direction
`border_truncated`     **ignore** - a real fibre that leaves the field; not a
                       measurable target, but asserting "background" would be false
instance overlap       **ignore** - a flat raster cannot hold two identities
rejected               **background** - the operator asserted "not a myotube",
                       which is informative negative signal and is kept
unproposed territory   **background** - the default
=====================  ==========================================================

`complete` always wins: a pixel that is both a reviewed target and inside some
ambiguous proposal stays a target, so ignore never erodes the 375 masks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IGNORE_STATUSES = ("ambiguous", "border_truncated", "occluded")


def build_ignore_mask(source_instances: str | Path, image_shape: tuple[int, int],
                      labels: np.ndarray | None = None,
                      overlap_ignore: np.ndarray | None = None,
                      ignore_statuses: tuple[str, ...] = IGNORE_STATUSES,
                      excluded_ids: tuple[str, ...] = (),
                      ) -> tuple[np.ndarray, dict]:
    """Return ``(ignore, stats)`` for one well.

    ``source_instances`` is the well's reviewed `*.qc.instances.json`. Records
    whose status is in ``ignore_statuses`` contribute their pixels to the ignore
    mask regardless of ``reviewed`` -- an unreviewed `ambiguous` proposal is
    exactly the "operator did not certify this" case we must not call background.

    ``excluded_ids`` are the binding `training_exclude.json` ids. They still read
    as `complete`/`reviewed` in the source file, so without this they would fall
    through to *background* -- yet they were excluded precisely because the
    operator re-reviewed them as ambiguous. They must be ignored, not asserted as
    empty field.

    ``labels`` (the bootstrap `labels.tif`) protects reviewed targets: any pixel
    belonging to a training instance is removed from the ignore mask.
    """
    from .schema_bridge import InstanceSet

    instance_set = InstanceSet.load(source_instances)
    if tuple(instance_set.image_shape) != tuple(image_shape):
        raise ValueError("instance set shape does not match the training image")

    excluded = set(excluded_ids)
    ignore = np.zeros(tuple(image_shape), dtype=bool)
    counts: dict[str, int] = {}
    seen_excluded: set[str] = set()
    for record, bbox, mask in instance_set.cropped_masks():
        is_excluded = record.id in excluded
        if record.status not in ignore_statuses and not is_excluded:
            continue
        r0, c0, r1, c1 = bbox
        ignore[r0:r1, c0:c1] |= mask
        if is_excluded:
            seen_excluded.add(record.id)
            counts["binding_exclusion"] = counts.get("binding_exclusion", 0) + 1
        else:
            counts[record.status] = counts.get(record.status, 0) + 1
    missing = excluded - seen_excluded
    if missing:
        raise ValueError(f"excluded ids not found in {source_instances}: {sorted(missing)}")

    if overlap_ignore is not None:
        overlap = np.asarray(overlap_ignore).astype(bool)
        ignore |= overlap
        counts["instance_overlap_px"] = int(overlap.sum())

    protected = 0
    if labels is not None:
        target = np.asarray(labels) > 0
        protected = int((ignore & target).sum())
        ignore &= ~target                      # a reviewed target is never ignored

    total = int(np.prod(image_shape))
    stats = {
        "ignored_px": int(ignore.sum()),
        "ignored_fraction": float(ignore.sum()) / total,
        "counts_by_status": counts,
        "target_px_protected_from_ignore": protected,
        "policy": "ambiguous/border_truncated/occluded/overlap -> ignore; "
                  "rejected and unproposed -> background; complete -> target",
    }
    return ignore, stats


def load_well_training_arrays(well_dir: str | Path, source_instances: str | Path,
                              excluded_ids: tuple[str, ...] = (),
                              ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load ``labels.tif`` and build the full training ignore mask for one well."""
    import tifffile

    well_dir = Path(well_dir)
    labels = tifffile.imread(well_dir / "labels.tif")
    overlap = tifffile.imread(well_dir / "ignore.tif")
    ignore, stats = build_ignore_mask(source_instances, labels.shape, labels=labels,
                                      overlap_ignore=overlap,
                                      excluded_ids=tuple(excluded_ids))
    stats["n_instances"] = int(labels.max())
    stats["well"] = well_dir.name
    return labels.astype(np.int32, copy=False), ignore, stats


def summarise(manifest_path: str | Path, root: str | Path) -> dict:
    """Per-well ignore statistics for the whole bootstrap (reporting helper)."""
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    bootstrap_dir = Path(manifest_path).parent
    out = {}
    for well, info in sorted(manifest["per_well"].items()):
        _, _, stats = load_well_training_arrays(
            bootstrap_dir / well, root / info["source_instances"],
            excluded_ids=tuple(info.get("excluded", ())))
        out[well] = stats
    return out
