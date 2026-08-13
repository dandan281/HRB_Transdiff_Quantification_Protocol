"""Teach gap-bridging from masks the operator already certified.

Why this exists
---------------
The dominant error in this project is `fragment_too_short`: genuine **signal-gap
fragmentation** in Desmin. `ignore_policy` paints `ambiguous` regions out of the
image and replaces them with real background, which is the right call for what it
was solving -- but the gap-bridging cases are exactly the ones the operator marked
`ambiguous`, so they are erased. Omnipose was parked in part on that reasoning:
it never sees the failure it would need to fix.

The reasoning was too pessimistic, and measurement says so. Walking the skeleton
of each reviewed-complete mask geodesically and reading Desmin along it:

* **173 of 374 instances (46%) contain at least one internal gap** -- a stretch
  where the fibre's own signal collapses toward background *inside* a mask the
  operator certified as one myotube. (374, not the 375 trainable masks:
  `corpus_gap_distribution` skips masks under 50 px and masks whose skeleton
  profile fails, so its denominator is the measurement population after those
  guards. The two counts describe different populations; neither is an error.)
* 157 such gaps are at least 5 um long, 56 at least 10 um, 18 at least 20 um, and
  the longest is **79.2 um** -- which is the scale the fragment linker's 80 um
  candidate window was built for;
* when signal dips it dips deep: median depth 0.163 of the way from the fibre's
  own median down to background.

So the supervision is not absent. It is thin, and it is the tail of a
noise-dominated distribution. This module widens it the only way that does not
invent data: take a certified-complete mask, attenuate a band of its Desmin
signal to look like a real gap, and **leave the mask untouched**. The mask is the
operator's; only the image is synthetic, and the lesson -- *identity continues
through a stretch that looks like background* -- is one they already asserted by
drawing that mask.

`DEVELOPMENT_PLAN.md` §4 permits synthetic data for augmentation and forbids it
from replacing real held-out evaluation. Nothing here touches evaluation.

Two properties this module has to keep
--------------------------------------
**The gap distribution is fold-local.** :func:`corpus_gap_distribution` takes the
wells to measure and callers pass the *training* wells only. Sampling gap lengths
from a corpus that includes the held-out well would leak its image statistics
into training -- a small leak, but the project splits by whole well and this is a
whole-well question.

**Only the target instance is modified.** The attenuation is masked by the
instance itself, so a neighbouring fibre crossing the same band keeps its signal.
Otherwise the augmentation would quietly teach that *two* objects vanish there.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .data import BOOTSTRAP, PIXEL_UM

# Below one pixel (0.65 um) a "gap" is noise. 5 um is the floor at which a dip is
# a feature of the fibre rather than of the detector, and it is the bottom of the
# range that actually fragments objects; the linker's window tops the range out at
# 80 um. Measured gaps at or above this floor: 157, spanning 5.0-79.2 um.
MIN_GAP_UM = 5.0
DEPTH_FRACTION = 0.25      # "collapsed toward background" for gap detection
SMOOTH_PX = 5
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass(frozen=True)
class Gap:
    """One measured internal gap: how long it ran and how far the signal fell."""
    length_um: float
    depth: float               # 0 = down to background, 1 = no dip at all


def geodesic_order(skeleton: np.ndarray) -> np.ndarray | None:
    """Skeleton pixel coordinates ordered along the fibre, tip to tip.

    Double sweep: BFS from an arbitrary pixel to reach a true extremity, then BFS
    from there and sort by that distance. Projecting onto a principal axis is
    cheaper and wrong -- it scrambles any fibre with a bend in it, and myotubes
    bend.
    """
    points = [tuple(p) for p in np.argwhere(skeleton)]
    if len(points) < 3:
        return None
    present = set(points)

    def sweep(start):
        distance = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for dr, dc in NEIGHBOURS:
                step = (current[0] + dr, current[1] + dc)
                if step in present and step not in distance:
                    distance[step] = distance[current] + 1
                    queue.append(step)
        return distance

    tip = max(sweep(points[0]).items(), key=lambda kv: kv[1])[0]
    distance = sweep(tip)
    if len(distance) < len(points) * 0.6:
        return None            # skeleton is not one connected run; do not guess
    return np.array([p for p, _ in sorted(distance.items(), key=lambda kv: kv[1])])


def skeleton_profile(mask: np.ndarray, image: np.ndarray) -> tuple | None:
    """(ordered skeleton coordinates, intensity along them) for one instance."""
    from skimage.morphology import skeletonize

    ordered = geodesic_order(skeletonize(np.asarray(mask, dtype=bool)))
    if ordered is None:
        return None
    return ordered, image[ordered[:, 0], ordered[:, 1]].astype(float)


def measure_gaps(profile: np.ndarray, background: float, *,
                 depth_fraction: float = DEPTH_FRACTION,
                 smooth_px: int = SMOOTH_PX,
                 pixel_um: float = PIXEL_UM) -> list[Gap]:
    """Internal stretches where the fibre's signal collapses toward background.

    Runs touching either tip are excluded: a fibre tapering out at its end is not
    a gap, and counting it would inflate both the rate and the lengths.
    """
    if len(profile) < 3 * smooth_px:
        return []
    smoothed = ndi.median_filter(profile.astype(float), size=smooth_px, mode="nearest")
    median = float(np.median(smoothed))
    if median <= background:
        return []
    normalised = (smoothed - background) / (median - background)
    low = normalised < depth_fraction

    gaps, start = [], None
    for index, flag in enumerate(low):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            gaps.append((start, index))
            start = None
    if start is not None:
        gaps.append((start, len(low)))

    out = []
    for first, last in gaps:
        if first <= 2 or last >= len(low) - 2:
            continue
        out.append(Gap(length_um=(last - first) * pixel_um,
                       depth=float(normalised[first:last].min())))
    return out


def corpus_gap_distribution(wells: list[str], *, min_gap_um: float = MIN_GAP_UM,
                            root=BOOTSTRAP) -> list[Gap]:
    """Empirical internal-gap distribution over the given wells.

    Callers pass **training wells only**. There is no default well list on
    purpose: a convenient "all six" default is how a held-out well ends up in a
    training statistic without anyone deciding that it should.
    """
    import tifffile

    found: list[Gap] = []
    for well in wells:
        image = tifffile.imread(root / well / "image_fiber.tif").astype(float)
        labels = tifffile.imread(root / well / "labels.tif")
        ignore_path = root / well / "ignore.tif"
        ignore = (tifffile.imread(ignore_path) if ignore_path.is_file()
                  else np.zeros_like(labels))
        background = float(np.median(image[(labels == 0) & (ignore == 0)]))
        for label_id, box in enumerate(ndi.find_objects(labels), start=1):
            if box is None:
                continue
            mask = labels[box] == label_id
            if mask.sum() < 50:
                continue
            got = skeleton_profile(mask, image[box])
            if got is None:
                continue
            found.extend(g for g in measure_gaps(got[1], background)
                         if g.length_um >= min_gap_um)
    return found


def apply_synthetic_gap(image: np.ndarray, mask: np.ndarray, *, background: float,
                        distribution: list[Gap], rng: np.random.Generator,
                        pixel_um: float = PIXEL_UM,
                        margin_fraction: float = 0.15) -> dict | None:
    """Attenuate one band of a fibre's signal in place. The mask is not touched.

    The gap is drawn from `distribution` rather than from a fitted curve, so the
    lengths and depths are ones this corpus actually produces. It is placed away
    from both tips -- a gap at the end is a shorter fibre, not a bridged one --
    and applied only inside `mask`, so a neighbour crossing the band survives.

    Returns a record of what was done, or None when the instance is too short to
    carry the sampled gap.
    """
    if not distribution:
        return None
    got = skeleton_profile(mask, image)
    if got is None:
        return None
    ordered, _ = got

    gap = distribution[int(rng.integers(len(distribution)))]
    gap_px = max(1, int(round(gap.length_um / pixel_um)))
    margin = int(len(ordered) * margin_fraction)
    usable = len(ordered) - 2 * margin - gap_px
    if usable <= 0:
        return None                       # fibre cannot hold this gap internally

    start = margin + int(rng.integers(usable + 1))
    band = ordered[start:start + gap_px]

    # Widen the centreline band to the fibre's full cross-section by assigning
    # every mask pixel to its nearest centreline pixel and keeping those whose
    # nearest point falls in the band.
    #
    # Dilating the band isotropically instead is the obvious shortcut and it is
    # wrong: it grows the gap ALONG the fibre by the fibre's own half-width at
    # each end, so a sampled 7 um gap re-measures at about 16 um and the
    # augmentation no longer matches the empirical distribution it claims to draw
    # from. Round-tripping the measurement is what caught that.
    solid = np.asarray(mask, dtype=bool)
    centreline = np.zeros(mask.shape, dtype=bool)
    centreline[ordered[:, 0], ordered[:, 1]] = True
    _, indices = ndi.distance_transform_edt(~centreline, return_indices=True)
    in_band = np.zeros(mask.shape, dtype=bool)
    in_band[band[:, 0], band[:, 1]] = True
    nearest_is_band = in_band[indices[0], indices[1]]
    region = nearest_is_band & solid

    image[region] = background + (image[region] - background) * gap.depth
    return {"gap_length_um": round(gap.length_um, 2),
            "gap_depth": round(gap.depth, 3),
            "gap_px": gap_px,
            "pixels_attenuated": int(region.sum()),
            "skeleton_px": int(len(ordered)),
            "start_index": int(start)}


def augment_tile(image: np.ndarray, labels: np.ndarray, *, background: float,
                 distribution: list[Gap], rng: np.random.Generator,
                 probability: float = 0.5,
                 pixel_um: float = PIXEL_UM) -> tuple[np.ndarray, list[dict]]:
    """Apply synthetic gaps to a copy of one training tile.

    `probability` is per instance, not per tile: a tile holding several fibres
    should not have all of them gapped at once, or the model learns that gaps are
    a property of fields rather than of fibres.
    """
    out = np.array(image, copy=True)
    records = []
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        if rng.random() >= probability:
            continue
        mask = labels == label_id
        if mask.sum() < 50:
            continue
        record = apply_synthetic_gap(out, mask, background=background,
                                     distribution=distribution, rng=rng,
                                     pixel_um=pixel_um)
        if record is not None:
            records.append({"label": int(label_id), **record})
    return out, records
