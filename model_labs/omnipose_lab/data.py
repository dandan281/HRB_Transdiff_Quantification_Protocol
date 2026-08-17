"""Fold datasets for T02 candidate 2 (Omnipose): what the trainer actually sees.

Responsibilities, in order:

1. load one well's image, reviewed `complete` targets, and the ignore mask that
   `model_labs/_shared/training_masks.py` defines (`ambiguous`,
   `border_truncated`, `occluded`, instance overlap, binding exclusions);
2. optionally fold in the `retriage_round2` evidence tier (27 further promotions),
   always as a *toggle* so metrics can be reported with and without it;
3. apply the ignore policy chosen in :mod:`omnipose_lab.ignore_policy` -- paint the
   ignored pixels out to real local background, so the `background` label the
   trainer sees is true;
4. cut the well into training tiles.

Why tiles are centred on instances
----------------------------------
A fixed grid is wrong here for two independent reasons.

*Omnipose's crop sampler needs foreground.* `omnipose.core.random_crop_warp`
retries until a crop contains at least one label and raises after ~100 attempts.
Foreground is only 1.1-2.6% of a field, so a grid of mostly-empty tiles risks
that failure.

*A grid cuts objects.* The largest reviewed instance spans 965 px and 9.9% of
instances exceed 512 px, so no affordable grid stride keeps every object whole.
An object cut by a tile edge is worse than useless: the truncated remainder would
teach exactly the `fragment_too_short` boundary the operator flagged in 55% of
re-triage cases.

So there is **one tile per reviewed instance**, centred on that instance's bbox
(clamped to the field), sized to contain the largest object in the corpus. Each
instance is therefore whole in at least its own tile. Any *other* target that the
tile edge happens to cut is painted out of that tile with the same policy -- it is
never left half-labelled, and never relabelled background while still visible.
It keeps its own tile elsewhere, so nothing is lost.

Splits stay whole-well: tiles are produced per well and a fold only ever consumes
tiles from its five training wells. Nothing crosses a fold boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "PrecisionMyotube/annotation_work/bootstrap_v1"
ROUND2 = ROOT / "PrecisionMyotube/annotation_work/retriage_round2"

# Ceiling, not the usual size. Typical tiles are far smaller -- see
# `instance_tiles`. Do not shrink without re-measuring, and re-measure whenever a
# new corpus is added: `instance_tiles` raises rather than truncating, so a
# single over-long fibre stops the whole build.
#
#   bootstrap_v1 (PLATE_23, 375 instances)  max bbox extent  965 px -> 1024 held
#   PLATE_32 dense (5,233 instances)        max bbox extent 1534 px -> 1024 FAILS
#
# 1792 = 1534 + 2*96 margin, rounded up for headroom. The cost is bounded: only
# the handful of very long fibres get a large tile, because every other tile is
# still sized to its own instance.
TILE_PX = 1792
# Real context kept around each instance. Enough for the network to see that a
# fibre ends rather than leaves the crop.
MARGIN_PX = 96
# Floor, so every crop contains some background to learn from even for a short
# instance, and so a crop is not dominated by warp padding.
MIN_TILE_PX = 256
PIXEL_UM = 0.6493

# From `model_labs/omnipose/channel_config.json` (`desmin_only`).
NORM_LOW_PCT = 1.0
NORM_HIGH_PCT = 99.5


def normalize_field(image: np.ndarray, reference: np.ndarray | None = None,
                    low_pct: float = NORM_LOW_PCT,
                    high_pct: float = NORM_HIGH_PCT) -> tuple[np.ndarray, dict]:
    """Percentile-normalise using statistics of the **whole field**.

    Omnipose normalises each training image and each inference tile independently.
    On a field that is ~98% background that is actively harmful: a tile containing
    no fibre has its noise stretched across the full dynamic range and presented to
    the network as though it were signal. Normalising once per field, and passing
    ``normalize=False`` downstream, keeps one intensity scale across training tiles
    and inference tiles alike.

    ``reference`` supplies the percentiles when they must come from a different
    array than the one being scaled -- training uses the *unpainted* field, so that
    painting out ~6% of pixels cannot shift the scale away from what inference
    (which never paints) will see.
    """
    source = image if reference is None else reference
    low, high = np.percentile(source, [low_pct, high_pct])
    span = float(high) - float(low)
    if span <= 0:
        raise ValueError("degenerate intensity range; cannot normalise")
    out = (image.astype(np.float32) - float(low)) / span
    np.clip(out, 0.0, 1.0, out=out)
    return out, {"low_pct": low_pct, "high_pct": high_pct,
                 "low_value": float(low), "high_value": float(high),
                 "scope": "whole_field"}


@dataclass
class WellData:
    well: str
    image: np.ndarray            # painted per the ignore policy
    labels: np.ndarray           # int32, reviewed `complete` targets only
    ignore: np.ndarray           # bool, pre-paint (kept for auditing)
    stats: dict = field(default_factory=dict)


@dataclass
class Tile:
    well: str
    row: int
    col: int
    size: tuple[int, int]        # (rows, cols); tiles are sized to their instance
    image: np.ndarray
    labels: np.ndarray
    centred_on: int              # label id this tile was built for
    whole_ids: tuple[int, ...]   # well-level ids kept whole in this tile
    n_instances: int
    n_dropped_border: int


# ------------------------------------------------------------------------ loading


def _round2_targets(well: str, image_shape: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    """Rasterise the `retriage_round2` promotions for one well.

    Returns a boolean-per-instance stack collapsed to an int label map with local
    ids, plus the promoted source ids. Only ids the operator explicitly listed as
    `promoted` are used; the tier's `ignore`/`background` decisions are already
    covered by the base policy.
    """
    from _shared.schema_bridge import InstanceSet

    classes = json.loads((ROUND2 / f"{well}.round2_classes.json").read_text(encoding="utf-8"))
    promoted = list(classes.get("promoted", []))
    out = np.zeros(image_shape, dtype=np.int32)
    if not promoted:
        return out, promoted

    instance_set = InstanceSet.load(ROUND2 / f"{well}.round2.instances.json")
    if tuple(instance_set.image_shape) != tuple(image_shape):
        raise ValueError(f"{well}: round2 instance shape does not match the training image")
    wanted = set(promoted)
    seen: set[str] = set()
    next_id = 1
    for record, bbox, mask in instance_set.cropped_masks():
        if record.id not in wanted:
            continue
        if record.status != "complete":
            raise ValueError(f"{well}: promoted id {record.id} is not `complete`")
        r0, c0, r1, c1 = bbox
        window = out[r0:r1, c0:c1]
        window[mask & (window == 0)] = next_id
        seen.add(record.id)
        next_id += 1
    missing = wanted - seen
    if missing:
        raise ValueError(f"{well}: promoted ids missing from round2 export: {sorted(missing)}")
    return out, promoted


def load_well(well: str, *, policy: str = "paint_out", include_round2: bool = False,
              halo_px: int | None = None, seed: int = 0) -> WellData:
    """Load one well and apply the ignore policy. See module docstring."""
    import tifffile

    from _shared.training_masks import load_well_training_arrays
    from omnipose_lab.ignore_policy import DEFAULT_HALO_PX, POLICIES

    manifest = json.loads((BOOTSTRAP / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    info = manifest["per_well"][well]
    labels, ignore, mask_stats = load_well_training_arrays(
        BOOTSTRAP / well, ROOT / info["source_instances"],
        excluded_ids=tuple(info.get("excluded", ())))
    image = tifffile.imread(BOOTSTRAP / well / "image_fiber.tif")

    promoted: list[str] = []
    n_base = int(labels.max())
    if include_round2:
        extra, promoted = _round2_targets(well, labels.shape)
        # A promotion is a target now, so it must stop being ignored -- and it must
        # not overwrite an existing reviewed target where the two masks overlap.
        add = (extra > 0) & (labels == 0)
        labels = labels.copy()
        labels[add] = extra[add] + n_base
        ignore = ignore & ~(labels > 0)

    if policy not in POLICIES:
        raise ValueError(f"unknown ignore policy {policy!r}; have {sorted(POLICIES)}")
    result = POLICIES[policy](
        image, ignore, labels,
        halo_px=DEFAULT_HALO_PX if halo_px is None else halo_px, seed=seed)
    # Percentiles come from the raw field, never the painted copy, so both policy
    # arms and inference share one intensity scale.
    normalised, norm_stats = normalize_field(result.image, reference=image)

    stats = {
        "well": well,
        "training_masks": mask_stats,
        "ignore_policy": result.stats,
        "normalization": norm_stats,
        "n_targets_base": n_base,
        "n_targets_round2": len(promoted),
        "round2_promoted_ids": promoted,
        "include_round2": include_round2,
        "n_targets_total": int(labels.max()),
        "source_instances": info["source_instances"],
        "source_instances_sha256": info["source_instances_sha256"],
    }
    return WellData(well, normalised, labels.astype(np.int32, copy=False), ignore, stats)


# ------------------------------------------------------------------------- tiling


def instance_tiles(data: WellData, tile_px: int = TILE_PX, *,
                   margin_px: int = MARGIN_PX, min_tile_px: int = MIN_TILE_PX,
                   halo_px: int | None = None, seed: int = 0) -> list[Tile]:
    """One tile per reviewed instance, sized to that instance. See module docstring.

    Tiles are the instance's bounding box plus ``margin_px`` of real context, not a
    fixed square. A fixed 1024 px tile -- large enough to hold the 965 px maximum --
    makes a *median* 193 px instance occupy 3.5% of its own tile, and Omnipose's
    `random_crop_warp` then recurses looking for a `tyx` crop that contains a
    label and dies at depth 100 with "Sparse or over-dense image detected". Sizing
    each tile to its instance keeps the object dominant in every crop, and cuts
    memory by roughly an order of magnitude as a side effect.

    ``tile_px`` remains the hard ceiling (no tile may exceed it) and
    ``min_tile_px`` the floor, so a crop always has some background to learn from.
    """
    from scipy import ndimage

    from omnipose_lab.ignore_policy import DEFAULT_HALO_PX, paint_out

    height, width = data.labels.shape
    if tile_px > min(height, width):
        raise ValueError(f"tile ceiling {tile_px} larger than field {data.labels.shape}")
    halo = DEFAULT_HALO_PX if halo_px is None else halo_px

    # Full-field area per instance, computed once. Comparing a tile's count for an
    # id against this is what detects an edge-cut object; doing it by re-scanning
    # the field per (tile, instance) would be ~375x5 passes over 13M pixels.
    full_area = np.bincount(data.labels.ravel())

    boxes = ndimage.find_objects(data.labels)
    tiles: list[Tile] = []
    covered: set[int] = set()          # ids already whole inside some emitted tile
    for index, box in enumerate(boxes, start=1):
        if box is None:
            continue
        # Instances cluster, so consecutive centred tiles overlap heavily. If this
        # instance is already whole inside an emitted tile, another tile centred on
        # it adds no coverage -- only duplicate pixels to hold in memory and train
        # on. Every instance is still guaranteed whole in at least one tile.
        if index in covered:
            continue
        row, row_end = _span(box[0], margin_px, min_tile_px, tile_px, height)
        col, col_end = _span(box[1], margin_px, min_tile_px, tile_px, width)

        image = data.image[row:row_end, col:col_end].copy()
        labels = data.labels[row:row_end, col:col_end].copy()

        # Any target the tile edge cuts is painted out of *this* tile: leaving it
        # would teach a truncated boundary, and simply zeroing its label would
        # assert background over a visible fibre. It keeps its own tile elsewhere.
        cut = _border_cut_ids(labels, full_area)
        # The instance this tile exists for is whole by construction (its own bbox
        # plus margin); assert that rather than silently dropping it.
        assert index not in cut, (
            f"{data.well}: instance {index} does not fit in its own tile "
            f"({row}:{row_end}, {col}:{col_end}) -- TILE_PX ceiling too small")
        if cut:
            drop = np.isin(labels, list(cut))
            keep = (labels > 0) & ~drop
            painted = paint_out(image, drop, keep.astype(np.int32),
                                halo_px=halo, seed=seed + index)
            image = painted.image
            labels[drop] = 0

        remaining = sorted(set(np.unique(labels).tolist()) - {0})
        if not remaining:
            continue
        # Contiguous 1..N per tile: Omnipose formats labels anyway, and this keeps
        # ids local so cross-well id collisions can never leak in (proposal ids do
        # repeat across wells).
        remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        for new_id, old_id in enumerate(remaining, start=1):
            remap[old_id] = new_id
        covered.update(remaining)
        tiles.append(Tile(
            well=data.well, row=row, col=col, size=(row_end - row, col_end - col),
            image=image, labels=remap[labels], centred_on=index,
            whole_ids=tuple(remaining), n_instances=len(remaining),
            n_dropped_border=len(cut)))

    missed = set(range(1, int(data.labels.max()) + 1)) - covered
    missed = {m for m in missed if boxes[m - 1] is not None}
    if missed:
        raise AssertionError(f"{data.well}: instances not whole in any tile: {sorted(missed)}")
    return tiles


def _span(box: slice, margin: int, minimum: int, maximum: int,
          limit: int) -> tuple[int, int]:
    """Instance extent + ``margin``, grown to ``minimum``, capped at ``maximum``.

    The window is then shifted (not shrunk) to sit inside ``[0, limit)``, so an
    instance near the field edge keeps its full size rather than being clipped.
    """
    start = box.start - margin
    end = box.stop + margin
    size = min(max(end - start, minimum), maximum, limit)
    centre = (box.start + box.stop) // 2
    start = int(np.clip(centre - size // 2, 0, limit - size))
    return start, start + size


def _border_cut_ids(tile_labels: np.ndarray, full_area: np.ndarray) -> set[int]:
    """Ids whose pixels extend beyond this tile (i.e. the tile edge cuts them).

    ``full_area`` is the well-wide ``np.bincount`` of the label map, so this is a
    single pass over the tile rather than a scan of the field per instance.
    """
    counts = np.bincount(tile_labels.ravel(), minlength=full_area.size)
    ids = np.nonzero(counts[:full_area.size])[0]
    return {int(v) for v in ids if v != 0 and counts[v] < full_area[v]}


# -------------------------------------------------------------------- fold builder


def local_label_id(tile: Tile) -> int | None:
    """The id `tile.centred_on` carries *inside* `tile.labels`.

    `instance_tiles` remaps each tile to contiguous 1..N local ids so proposal ids
    cannot collide across wells, while `centred_on` stays a well-level id. Any
    caller that needs the centred instance's pixels has to translate, and the
    failure mode when it does not is an empty mask rather than an error.

    Returns None when the centred instance was dropped as border-cut.
    """
    try:
        return tile.whole_ids.index(tile.centred_on) + 1
    except ValueError:
        return None


def build_fold(train_wells: list[str], held_out: str, *, policy: str = "paint_out",
               include_round2: bool = False, tile_px: int = TILE_PX,
               seed: int = 0, augment_gaps: bool = False,
               gap_probability: float = 0.5) -> dict:
    """Training tiles for one leave-one-well-out fold.

    The held-out well is loaded **only** to record that it was excluded; none of
    its pixels enter the returned arrays. Assertions make the leak impossible to
    introduce silently later.

    ``augment_gaps`` **appends** a synthetically gapped copy of a tile rather than
    replacing it, so the clean example survives and the gapped one is added -- see
    `gap_augment`. Its gap distribution is measured from ``train_wells`` only, for
    the same reason the held-out well's pixels never enter the arrays. Off by
    default: it is an ablation arm, and an arm that is always on measures nothing.
    """
    if held_out in train_wells:
        raise ValueError(f"held-out well {held_out} is in the training set")

    distribution = []
    if augment_gaps:
        from .gap_augment import apply_synthetic_gap, corpus_gap_distribution
        distribution = corpus_gap_distribution(train_wells)

    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    provenance: list[dict] = []
    per_well: dict[str, dict] = {}
    for well_index, well in enumerate(train_wells):
        data = load_well(well, policy=policy, include_round2=include_round2, seed=seed)
        tiles = instance_tiles(data, tile_px, seed=seed)
        for tile in tiles:
            images.append(tile.image)
            labels.append(tile.labels)
            provenance.append({"well": tile.well, "row": tile.row, "col": tile.col,
                               "size": tile.size, "centred_on": tile.centred_on,
                               "n_instances": tile.n_instances,
                               "n_dropped_border": tile.n_dropped_border,
                               "synthetic_gap": None})
        n_augmented = 0
        if augment_gaps and distribution:
            background = float(np.median(
                data.image[(data.labels == 0) & ~data.ignore.astype(bool)]))
            rng = np.random.default_rng([seed, well_index])
            for tile in tiles:
                if rng.random() >= gap_probability:
                    continue
                local = local_label_id(tile)
                if local is None:
                    continue
                mask = tile.labels == local
                # `centred_on` is a WELL-level id while `tile.labels` is remapped to
                # contiguous local ids, so comparing them directly yields an empty
                # mask and augments nothing at all -- silently. This assertion is
                # here because that is exactly what happened.
                assert mask.any(), (
                    f"empty mask for tile centred on {tile.centred_on} "
                    f"(local {local}); the tile label id space has changed")
                gapped = np.array(tile.image, copy=True)
                record = apply_synthetic_gap(
                    gapped, mask, background=background,
                    distribution=distribution, rng=rng, pixel_um=PIXEL_UM)
                if record is None:
                    continue
                images.append(gapped)
                labels.append(tile.labels)
                provenance.append({"well": tile.well, "row": tile.row, "col": tile.col,
                                   "size": tile.size, "centred_on": tile.centred_on,
                                   "n_instances": tile.n_instances,
                                   "n_dropped_border": tile.n_dropped_border,
                                   "synthetic_gap": record})
                n_augmented += 1
        per_well[well] = {**data.stats, "n_tiles": len(tiles),
                          "n_synthetic_gap_tiles": n_augmented}
        del data

    assert all(p["well"] != held_out for p in provenance), "held-out well leaked into tiles"
    assert len(images) == len(labels) == len(provenance)
    config = {"policy": policy, "include_round2": include_round2,
              "tile_px": tile_px, "seed": seed, "pixel_um": PIXEL_UM}
    if augment_gaps:
        # Only written when enabled, so an un-augmented fold hashes exactly as it
        # did before this option existed.
        config["augment_gaps"] = True
        config["gap_probability"] = gap_probability
        config["gap_distribution_n"] = len(distribution)
        config["gap_distribution_wells"] = list(train_wells)
    return {
        "held_out": held_out, "train_wells": list(train_wells),
        "images": images, "labels": labels, "tiles": provenance,
        "per_well": per_well,
        "config": config,
        "n_tiles": len(images),
        "n_synthetic_gap_tiles": sum(1 for p in provenance if p["synthetic_gap"]),
        "n_instances": sum(p["n_instances"] for p in provenance),
    }


def dataset_hash(fold: dict) -> str:
    """Content hash of a fold's training data, for the run manifest."""
    digest = hashlib.sha256()
    digest.update(json.dumps(fold["config"], sort_keys=True).encode("utf-8"))
    for image, labels, meta in zip(fold["images"], fold["labels"], fold["tiles"]):
        digest.update(f"{meta['well']}:{meta['row']}:{meta['col']}".encode("utf-8"))
        digest.update(np.ascontiguousarray(image).tobytes())
        digest.update(np.ascontiguousarray(labels).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    import argparse
    import sys

    for path in (ROOT / "PrecisionMyotube", ROOT / "annotation_tools", ROOT / "model_labs"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    parser = argparse.ArgumentParser(description="Inspect T02 Omnipose fold datasets")
    parser.add_argument("--policy", default="paint_out")
    parser.add_argument("--include-round2", action="store_true")
    parser.add_argument("--tile-px", type=int, default=TILE_PX)
    parser.add_argument("--held-out", default=None, help="build just this fold")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    manifest = json.loads((BOOTSTRAP / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    wells = sorted(manifest["per_well"])
    held = [args.held_out] if args.held_out else wells

    report = {}
    for held_out in held:
        train = [w for w in wells if w != held_out]
        fold = build_fold(train, held_out, policy=args.policy,
                          include_round2=args.include_round2, tile_px=args.tile_px)
        dropped = sum(t["n_dropped_border"] for t in fold["tiles"])
        print(f"hold-out {held_out:24s} tiles={fold['n_tiles']:4d} "
              f"instance-slots={fold['n_instances']:5d} border-painted={dropped:4d} "
              f"hash={dataset_hash(fold)[:12]}")
        report[held_out] = {
            "n_tiles": fold["n_tiles"], "n_instances": fold["n_instances"],
            "n_border_painted": dropped, "dataset_sha256": dataset_hash(fold),
            "train_wells": fold["train_wells"], "config": fold["config"],
            "per_well": {w: {k: v for k, v in s.items() if k != "round2_promoted_ids"}
                         for w, s in fold["per_well"].items()},
        }
        del fold
    if args.out:
        out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
