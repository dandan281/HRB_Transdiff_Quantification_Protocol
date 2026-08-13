"""Features for the fragment linker: does the stain actually bridge this gap?

Geometry alone (gap length, collinearity) cannot tell a real join from two
unrelated fibres that happen to point at each other -- the round-1 analysis found
its best geometric feature, ``min_cos``, reaches only AUC 0.66. What the operator
actually uses when they press ``L`` to hide the outlines and look at the bare
Desmin is **whether stain runs across the gap**. That intuition is captured here
by sampling the image along the straight segment between the two endpoints.

Features (all computed from the endpoint pair + the field image):

``gap_um``            endpoint separation in microns (shorter joins are likelier)
``min_cos``          min of the two outward-direction cosines (collinearity)
``bridge_over_bg``   the **dimmest** stain along the gap segment, over the field
                     background. A real fibre keeps some stain across the gap, so
                     its minimum stays well above background; a false join crosses
                     dark pixels and drops toward 1.0. Strongest single feature in
                     round 1 (AUC 0.82).
``bridge_mean_over_bg`` mean stain along the gap over background -- a softer
                     companion to the min.
``territory_frac``   fraction of the gap segment lying inside the semantic
                     territory mask (AUC 0.72 in round 1).
``axis_cos``         |cos| between the two fragments' principal axes (PCA of each
                     object's pixels). 1 = the fibres are parallel; low = they
                     point in different directions and are unlikely one fibre.
``displacement_along_axis`` |cos| between the centroid-to-centroid vector and the
                     fragment's principal axis. 1 = the partner sits **end-to-end**
                     along the fibre (a real break); ~0 = it sits **side-by-side**
                     (a parallel neighbour, not the same fibre).

`bridge_over_bg` is deliberately a *minimum*: a single dark pixel mid-gap is
strong evidence the fibre truly stops there, and a mean would wash it out. The
segment is sampled with a short perpendicular tolerance so a slightly bowed fibre
is not punished for the straight-line assumption.

`axis_cos` and `displacement_along_axis` encode an operator heuristic (2026-07-23):
two fragments are the same broken fibre only if their long axes are parallel *and*
the offset between them runs *along* that axis, not across it. They are the robust,
whole-object version of `min_cos`, whose endpoint tangents are estimated from a
12-px local patch and so are noisy on a frayed end -- which lets a side-by-side
pair slip through as "collinear". Both use ``abs``, so they are invariant to the
arbitrary sign of a PCA eigenvector and to flipping the coordinate frame.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# Order is fixed and part of the model contract; see `link_model.FEATURE_SETS`.
FEATURE_KEYS = ("gap_um", "min_cos", "bridge_over_bg", "bridge_mean_over_bg",
                "territory_frac", "axis_cos", "displacement_along_axis")

# Background reference for the stain: a low percentile of the whole field, so a
# few bright fibres do not pull it up. Cached per field by the caller.
BG_PERCENTILE = 50.0
PERP_TOLERANCE_PX = 2          # half-width of the sampling band around the segment


@dataclass
class LinkFeatures:
    gap_um: float
    min_cos: float
    bridge_over_bg: float
    bridge_mean_over_bg: float
    territory_frac: float
    axis_cos: float = 0.0
    displacement_along_axis: float = 0.0

    def vector(self, keys=FEATURE_KEYS) -> list[float]:
        d = asdict(self)
        return [float(d[k]) for k in keys]


@dataclass
class ObjectGeom:
    """Centroid and principal-axis unit vector (row, col) of one object's pixels."""
    centroid: tuple[float, float]
    axis: tuple[float, float]
    n_px: int


def object_geometry(mask: np.ndarray, origin: tuple[int, int] = (0, 0)) -> ObjectGeom:
    """PCA of a mask's pixel coordinates -> centroid + long-axis unit vector.

    ``origin`` is added so a mask cropped to a bounding box reports full-field
    coordinates. The long axis is the eigenvector of the larger eigenvalue of the
    coordinate covariance; its sign is arbitrary and callers must use ``abs`` on any
    cosine with it.
    """
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size < 2:
        return ObjectGeom((float(origin[0]), float(origin[1])), (1.0, 0.0), int(rows.size))
    pts = np.stack([rows + origin[0], cols + origin[1]], axis=1).astype(float)
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    if not np.all(np.isfinite(cov)):
        return ObjectGeom(tuple(centroid), (1.0, 0.0), int(rows.size))
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    norm = float(np.linalg.norm(axis))
    axis = (axis / norm) if norm > 1e-9 else np.array([1.0, 0.0])
    return ObjectGeom((float(centroid[0]), float(centroid[1])),
                      (float(axis[0]), float(axis[1])), int(rows.size))


def geometry_cache(labels: np.ndarray, label_ids) -> dict[int, ObjectGeom]:
    """PCA geometry for each requested label id, computed within its bbox.

    ``scipy.ndimage.find_objects`` is called once for the whole field, so this is a
    single pass regardless of how many ids are asked for.
    """
    from scipy import ndimage as ndi

    slices = ndi.find_objects(labels)
    out: dict[int, ObjectGeom] = {}
    for lid in sorted({int(i) for i in label_ids}):
        if lid < 1 or lid > len(slices) or slices[lid - 1] is None:
            continue
        sl = slices[lid - 1]
        sub = labels[sl] == lid
        out[lid] = object_geometry(sub, origin=(sl[0].start, sl[1].start))
    return out


def axis_features(fragment: ObjectGeom, candidate: ObjectGeom) -> tuple[float, float]:
    """(axis_cos, displacement_along_axis) from the operator's direction heuristic.

    Both are ``abs`` cosines in [0, 1], so they do not depend on the arbitrary sign
    of a PCA axis or on which axis (x or y) the fibre happens to run along.
    """
    fa = np.array(fragment.axis, dtype=float)
    ca = np.array(candidate.axis, dtype=float)
    axis_cos = abs(float(fa @ ca))

    disp = np.array(candidate.centroid, dtype=float) - np.array(fragment.centroid, dtype=float)
    dn = float(np.linalg.norm(disp))
    if dn < 1e-9:
        along = 0.0                     # coincident centroids: undefined, treat as worst
    else:
        along = abs(float((disp / dn) @ fa))
    return round(axis_cos, 4), round(along, 4)


def field_background(fiber: np.ndarray, percentile: float = BG_PERCENTILE) -> float:
    bg = float(np.percentile(fiber, percentile))
    return bg if bg > 1e-6 else 1.0


def _segment_band(a: tuple[int, int], b: tuple[int, int], shape: tuple[int, int],
                  tol: int = PERP_TOLERANCE_PX) -> tuple[np.ndarray, np.ndarray]:
    """Integer pixels within ``tol`` of the segment a->b (rows, cols), clipped."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    n = max(2, int(round(length)) + 1)
    ts = np.linspace(0.0, 1.0, n)
    line = a[None, :] + ts[:, None] * (b - a)[None, :]         # (n, 2) subpixel points
    if tol > 0:
        direction = b - a
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            perp = np.array([-direction[1], direction[0]]) / norm
            offsets = np.arange(-tol, tol + 1)
            pts = (line[:, None, :] + offsets[None, :, None] * perp[None, None, :]
                   ).reshape(-1, 2)
        else:
            pts = line
    else:
        pts = line
    rows = np.clip(np.round(pts[:, 0]).astype(int), 0, shape[0] - 1)
    cols = np.clip(np.round(pts[:, 1]).astype(int), 0, shape[1] - 1)
    return rows, cols


def compute_features(fiber: np.ndarray, territory: np.ndarray | None,
                     fragment_endpoint: tuple[int, int],
                     candidate_endpoint: tuple[int, int],
                     gap_um: float, min_cos: float, pixel_um: float,
                     background: float | None = None,
                     fragment_geom: "ObjectGeom | None" = None,
                     candidate_geom: "ObjectGeom | None" = None) -> LinkFeatures:
    """Intensity/territory/axis features for one candidate pair. See module docstring.

    ``fragment_geom`` / ``candidate_geom`` are the two objects' PCA geometries. When
    both are given, the axis features are computed; when either is missing they stay
    0.0, so a caller that only has endpoints still gets the intensity features.
    """
    bg = field_background(fiber) if background is None else background
    rows, cols = _segment_band(fragment_endpoint, candidate_endpoint, fiber.shape)
    samples = fiber[rows, cols].astype(np.float64)

    # For the "dimmest across the gap" statistic, reduce the perpendicular band to
    # a per-step *max* first: a fibre that is one pixel off the straight line still
    # counts as bridged, but a genuinely dark step stays dark.
    tol = PERP_TOLERANCE_PX
    band = 2 * tol + 1 if tol > 0 else 1
    if samples.size % band == 0 and band > 1:
        per_step = samples.reshape(-1, band).max(axis=1)
    else:
        per_step = samples
    bridge_min = float(per_step.min())
    bridge_mean = float(samples.mean())

    if territory is not None:
        terr = territory[rows, cols]
        territory_frac = float((terr > 0).mean())
    else:
        territory_frac = 0.0

    if fragment_geom is not None and candidate_geom is not None:
        axis_cos, along = axis_features(fragment_geom, candidate_geom)
    else:
        axis_cos, along = 0.0, 0.0

    return LinkFeatures(
        gap_um=float(gap_um), min_cos=float(min_cos),
        bridge_over_bg=round(bridge_min / bg, 4),
        bridge_mean_over_bg=round(bridge_mean / bg, 4),
        territory_frac=round(territory_frac, 4),
        axis_cos=axis_cos, displacement_along_axis=along)
