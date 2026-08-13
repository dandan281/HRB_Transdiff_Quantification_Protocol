"""How the T02 Omnipose candidate handles pixels the operator never certified.

The problem
-----------
`model_labs/_shared/training_masks.py` establishes which pixels must contribute
**no** training signal: `ambiguous`, `border_truncated`, `occluded`, inter-instance
overlap, and the two binding `training_exclude.json` ids. Across the six bootstrap
wells those cover 0.73-2.33% of area, and `ambiguous` alone (1.613%) exceeds the
trainable `complete` area (1.415%). Training them as background teaches the model
to suppress genuine myotubes, invisibly in the loss.

But `cellpose_omni.models.CellposeModel.train(...)` accepts **no per-pixel loss
mask**. Three options were considered. This module records the decision and the
evidence for it, because the choice is scientifically consequential.

Option 1 - exclude contaminated tiles. **REJECTED ON MEASUREMENT.**
    :func:`tile_exclusion_report` counts, per tile size, how many tiles holding a
    `complete` pixel are entirely ignore-free. Pooled over the six wells:

    ====  ================  ==========  ==================================
    tile  ignore-free tiles  area kept  instances wholly in a clean tile
    ====  ================  ==========  ==================================
     256  213/579 (0.368)       0.363   33/375 (8.8%)
     384   62/332 (0.187)       0.174   16/375 (4.3%)
     512   19/239 (0.079)       0.090   11/375 (2.9%)
     768     2/89 (0.022)       0.021    1/375 (0.3%)
    1024     1/53 (0.019)       0.002    0/375 (0.0%)
    ====  ================  ==========  ==================================

    At the planned 512 px tile this discards 97% of the training targets, and two
    wells (`29_C05`, `32_C08`) yield **zero** clean tiles. `complete` and
    `ambiguous` proposals are spatially interleaved, so they cannot be separated
    by cropping. Smaller tiles do not rescue it: 256 px is ~166 um at 0.6493
    um/px, short relative to the objects, so it would also bias the model toward
    the short-fibre error class T02 exists to fix.

Option 2 - paint the ignored regions out of the image. **CHOSEN.**
    Replace ignored *image* pixels with real, locally-sampled background, so the
    `background` label becomes **true** for the modified image rather than a false
    assertion. Upstream Omnipose then runs completely unmodified; the intervention
    is ordinary, hashable, testable preprocessing.

Option 3 - subclass `_train_step` and mask the loss. **REJECTED ON API READING.**
    Omnipose recomputes flows per-crop inside the dataloader
    (`cellpose_omni/models.py`, "No precomputing flows with Omnipose"), so a
    precomputed ignore channel cannot survive the warp; `_train_step(x, lbl)` is
    the only seam where a warped mask exists. And `omnipose.core.loss` applies its
    per-pixel weight ``w = lbl[:,4]`` to only **5 of its 9 terms** -- `dist_loss`,
    `flow_mse`, `bd_loss`, `SSL`, `norm_loss`. `AffinityLoss` (`lossA/E/B`) and
    `lossDC` are unweighted and `raw_loss` averages all nine equally, so zeroing
    the weight would leave ~44% of the loss still supervised on ignored pixels.
    Masking properly means forking the Omnipose loss -- which would make the T03
    comparison a statement about our fork rather than about Omnipose.

The halo matters
----------------
Painting a fibre out while leaving its intensity penumbra would create a faint
elongated structure *labelled background* -- reintroducing the original failure in
a subtler form. :func:`measure_halo` measures, from the data, the radius at which
median intensity outside an ignored mask returns to the local background level;
:func:`paint_out` dilates by that radius before filling.

Both arms are implemented so the choice can be reported as a measured number
rather than an assertion: :data:`POLICIES` exposes ``paint_out`` and the naive
``ambiguous_as_background`` control that the ablation scores against.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

# `measure_halo` over the six bootstrap wells returns 3 px (2 wells) and 4 px
# (4 wells) -- the intensity profile outside an ignored mask is back within 5% of
# the local background by 4 px. 6 px is that measured maximum with a 1.5x safety
# margin, because under-dilating leaves a residual ridge labelled background
# (the failure this module exists to prevent) while over-dilating only costs a
# little extra background area. See `_runs/policy_evidence.json`.
DEFAULT_HALO_PX = 6
DEFAULT_JITTER_PX = 4


# ------------------------------------------------------------------ option 1 evidence


def tile_exclusion_report(labels: np.ndarray, ignore: np.ndarray, tile: int) -> dict:
    """Yield of the rejected 'exclude contaminated tiles' policy, for one well.

    Uses a fixed non-overlapping grid: that is what a tile sampler actually sees,
    and a random-offset sampler cannot fix an interleaving problem. An instance
    counts as usable only if it lies *wholly* inside one ignore-free tile -- a
    partially cropped instance would teach a truncated boundary, which is the very
    error class (`fragment_too_short`) the operator flagged most often.
    """
    height, width = labels.shape
    all_ids = set(np.unique(labels).tolist()) - {0}
    area_total = int((labels > 0).sum())
    n_candidate = n_free = area_free = 0
    free_ids: set[int] = set()
    for row in range(0, height - tile + 1, tile):
        for col in range(0, width - tile + 1, tile):
            block = labels[row:row + tile, col:col + tile]
            if not block.any():
                continue
            n_candidate += 1
            if ignore[row:row + tile, col:col + tile].any():
                continue
            n_free += 1
            area_free += int((block > 0).sum())
            free_ids |= set(np.unique(block).tolist()) - {0}

    whole = 0
    for value in sorted(free_ids):
        rows, cols = np.nonzero(labels == value)
        if (rows.min() // tile == rows.max() // tile
                and cols.min() // tile == cols.max() // tile):
            whole += 1
    return {
        "tile": tile,
        "candidate_tiles": n_candidate, "ignore_free_tiles": n_free,
        "tile_yield": round(n_free / n_candidate, 3) if n_candidate else None,
        "complete_area_total": area_total, "complete_area_in_free_tiles": area_free,
        "area_yield": round(area_free / area_total, 3) if area_total else None,
        "n_instances": len(all_ids), "n_instances_whole_in_free_tile": whole,
    }


# ------------------------------------------------------------------- halo measurement


def measure_halo(image: np.ndarray, ignore: np.ndarray, targets: np.ndarray,
                 max_radius: int = 24) -> dict:
    """Radial intensity profile just outside the ignored regions.

    Returns the median intensity at each euclidean distance 1..``max_radius`` from
    an ignored pixel, restricted to pixels that are neither ignored nor a training
    target, plus the background reference level. The halo radius is the smallest
    distance at which the profile has fallen to within 5% of that reference --
    i.e. where a painted-out fibre stops leaving a visible ridge.
    """
    from scipy import ndimage

    distance = ndimage.distance_transform_edt(~ignore)
    outside = ~ignore & (targets == 0)
    reference_zone = outside & (distance > max_radius)
    if not reference_zone.any():
        raise ValueError("no far-field background available to reference")
    reference = float(np.median(image[reference_zone]))

    profile = {}
    for radius in range(1, max_radius + 1):
        shell = outside & (distance > radius - 1) & (distance <= radius)
        profile[radius] = float(np.median(image[shell])) if shell.any() else None

    chosen = max_radius
    for radius in range(1, max_radius + 1):
        value = profile[radius]
        if value is not None and value <= reference * 1.05:
            chosen = radius
            break
    return {"background_reference": reference, "profile": profile,
            "halo_px": chosen, "max_radius": max_radius}


# --------------------------------------------------------------------- option 2 (chosen)


@dataclass
class PaintResult:
    image: np.ndarray            # painted copy; same dtype as the input
    painted: np.ndarray          # bool, exactly which pixels were replaced
    stats: dict


def paint_out(image: np.ndarray, ignore: np.ndarray, targets: np.ndarray,
              halo_px: int = DEFAULT_HALO_PX, jitter_px: int = DEFAULT_JITTER_PX,
              seed: int = 0) -> PaintResult:
    """Replace ignored image pixels with real, locally-sampled background.

    Fill strategy: every painted pixel takes the value of its **nearest true
    background pixel**, with the source coordinate jittered by a seeded random
    offset. Copying real neighbouring pixels rather than synthesising a value
    preserves the local intensity distribution *and* its spatial texture, which a
    smooth or noise-filled patch would not -- a network could otherwise use the
    artificial texture as a shortcut instead of learning to suppress real
    background. The jitter breaks the streaking that a pure nearest-neighbour
    copy produces along the medial axis of a long thin region.

    ``targets`` (reviewed `complete` masks) are inviolable: they are excluded from
    the dilated region, so painting can never erode a training target. A fibre
    abutting a target is therefore painted only up to that boundary; the count is
    reported as ``target_adjacent_px`` rather than silently resolved.
    """
    from scipy import ndimage

    if image.shape != ignore.shape or image.shape != targets.shape:
        raise ValueError("image, ignore and targets must share a shape")
    ignore = np.asarray(ignore, dtype=bool)
    is_target = np.asarray(targets) > 0

    grown = ndimage.binary_dilation(
        ignore, ndimage.generate_binary_structure(2, 2), iterations=int(halo_px)
    ) if halo_px > 0 else ignore.copy()
    # A reviewed target is never painted, even if a halo reaches it.
    painted = grown & ~is_target
    target_adjacent = int((grown & is_target).sum())

    if not painted.any():
        return PaintResult(image.copy(), painted,
                           {"painted_px": 0, "halo_px": halo_px, "seed": seed,
                            "policy": "paint_out", "target_adjacent_px": target_adjacent})

    # Source pool: true background only -- never another ignored region, never a
    # target. Otherwise the fill would re-inject the structure we are removing.
    background = ~painted & ~is_target
    if not background.any():
        raise ValueError("no background pixels available to sample from")

    _, indices = ndimage.distance_transform_edt(~background, return_indices=True)
    src_rows = indices[0][painted]
    src_cols = indices[1][painted]

    if jitter_px > 0:
        rng = np.random.default_rng(seed)
        n = src_rows.size
        d_row = rng.integers(-jitter_px, jitter_px + 1, size=n)
        d_col = rng.integers(-jitter_px, jitter_px + 1, size=n)
        alt_rows = np.clip(src_rows + d_row, 0, image.shape[0] - 1)
        alt_cols = np.clip(src_cols + d_col, 0, image.shape[1] - 1)
        # Only accept a jittered source that is still genuine background.
        ok = background[alt_rows, alt_cols]
        src_rows = np.where(ok, alt_rows, src_rows)
        src_cols = np.where(ok, alt_cols, src_cols)
        jitter_accepted = int(ok.sum())
    else:
        jitter_accepted = 0

    out = image.copy()
    out[painted] = image[src_rows, src_cols]

    stats = {
        "policy": "paint_out",
        "halo_px": halo_px, "jitter_px": jitter_px, "seed": seed,
        "ignore_px": int(ignore.sum()),
        "painted_px": int(painted.sum()),
        "painted_fraction": float(painted.sum()) / image.size,
        "target_adjacent_px": target_adjacent,
        "jitter_accepted_px": jitter_accepted,
        "background_pool_px": int(background.sum()),
        "note": ("Ignored pixels replaced by nearest true-background pixels "
                 "(seeded jitter). The 'background' label is therefore true for "
                 "the painted image; inference always runs on unpainted fields."),
    }
    return PaintResult(out, painted, stats)


def ambiguous_as_background(image: np.ndarray, ignore: np.ndarray,
                            targets: np.ndarray, **_) -> PaintResult:
    """Ablation control: the naive treatment this module exists to argue against.

    The image is untouched, so every `ambiguous` / `border_truncated` pixel trains
    as background. This is not a straw man -- it is exactly what a candidate that
    consumes `labels.tif` + `ignore.tif` at face value would do, and it is the arm
    the chosen policy must beat on held-out wells to justify itself.
    """
    return PaintResult(image.copy(), np.zeros(image.shape, dtype=bool),
                       {"policy": "ambiguous_as_background", "painted_px": 0,
                        "ignore_px": int(np.asarray(ignore, bool).sum()),
                        "note": "Control arm: ignored pixels train as background."})


POLICIES = {
    "paint_out": paint_out,
    "ambiguous_as_background": ambiguous_as_background,
}


# ---------------------------------------------------------------------------- CLI


def _bootstrap_wells(root: Path):
    from _shared.training_masks import load_well_training_arrays

    bootstrap = root / "PrecisionMyotube/annotation_work/bootstrap_v1"
    manifest = json.loads((bootstrap / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    for well, info in sorted(manifest["per_well"].items()):
        labels, ignore, stats = load_well_training_arrays(
            bootstrap / well, root / info["source_instances"],
            excluded_ids=tuple(info.get("excluded", ())))
        yield well, bootstrap / well, labels, ignore, stats


def main(argv=None) -> int:
    import argparse
    import sys

    import tifffile

    root = Path(__file__).resolve().parents[2]
    for path in (root / "PrecisionMyotube", root / "annotation_tools", root / "model_labs"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    parser = argparse.ArgumentParser(description="Ignore-policy evidence for T02 candidate 2")
    parser.add_argument("--out", default="model_labs/omnipose_lab/_runs/policy_evidence.json")
    parser.add_argument("--tiles", type=int, nargs="+", default=[256, 384, 512, 768, 1024])
    parser.add_argument("--skip-halo", action="store_true")
    args = parser.parse_args(argv)

    report: dict = {"tile_exclusion": {}, "halo": {}, "wells": {}}
    pooled = {t: {"candidate_tiles": 0, "ignore_free_tiles": 0, "complete_area_total": 0,
                  "complete_area_in_free_tiles": 0, "n_instances": 0,
                  "n_instances_whole_in_free_tile": 0} for t in args.tiles}

    for well, well_dir, labels, ignore, stats in _bootstrap_wells(root):
        print(f"{well:24s} ignored={stats['ignored_fraction']*100:.2f}% "
              f"complete={float((labels>0).mean())*100:.2f}%")
        rows = [tile_exclusion_report(labels, ignore, t) for t in args.tiles]
        for row in rows:
            for key in pooled[row["tile"]]:
                pooled[row["tile"]][key] += row[key]
        report["wells"][well] = {"training_masks": stats, "tiles": rows}
        if not args.skip_halo:
            image = tifffile.imread(well_dir / "image_fiber.tif")
            halo = measure_halo(image, ignore, labels)
            report["halo"][well] = halo
            print(f"   halo_px={halo['halo_px']} (background {halo['background_reference']:.0f})")

    for tile in args.tiles:
        p = pooled[tile]
        p["tile_yield"] = round(p["ignore_free_tiles"] / max(1, p["candidate_tiles"]), 3)
        p["area_yield"] = round(p["complete_area_in_free_tiles"]
                                / max(1, p["complete_area_total"]), 3)
        p["instance_yield"] = round(p["n_instances_whole_in_free_tile"]
                                    / max(1, p["n_instances"]), 3)
        print(f"pooled tile {tile:4d}: tiles {p['tile_yield']} area {p['area_yield']} "
              f"instances {p['n_instances_whole_in_free_tile']}/{p['n_instances']}")
    report["tile_exclusion"] = {str(t): pooled[t] for t in args.tiles}
    report["decision"] = {
        "option_1_tile_exclusion": "REJECTED - see tile_exclusion yields above",
        "option_2_paint_out": "CHOSEN - see module docstring",
        "option_3_loss_mask": "REJECTED - weight reaches only 5 of 9 omnipose loss terms",
        "ablation": "both arms in POLICIES are run and reported",
    }
    if report["halo"]:
        report["halo_px_selected"] = max(h["halo_px"] for h in report["halo"].values())

    out = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
