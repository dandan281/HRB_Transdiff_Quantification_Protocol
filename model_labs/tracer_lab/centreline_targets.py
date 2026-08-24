"""Turn operator polylines into the four fields a tracer trains on.

The input is what the operator actually drew: freehand/polyline centrelines in
Fiji, one per fibre, read by `annotation_tools.relabel.fiji_roi.read_roi_set`.
Nothing here re-skeletonises a rasterised ribbon -- a skeleton of a ribbon that
was itself grown from a centreline would put two lossy transforms between the
human and the target, and would reintroduce the crossing ambiguity that this
whole candidate exists to avoid.

Four outputs, on the full field:

``centre``
    Soft centreline, ``exp(-d^2 / 2 sigma^2)`` in the distance to the nearest
    traced point. Soft rather than binary because a one-pixel target is
    unlearnable at the boundary: a prediction one pixel off scores the same as
    one fifty pixels off.

``orient``
    Two channels, ``(cos 2t, sin 2t)`` for tangent angle ``t``. **Angle-doubled
    on purpose.** A fibre traced left-to-right and the same fibre traced
    right-to-left are the same object, so the target must be invariant to
    direction reversal; ``t`` and ``t + pi`` map to the same point after
    doubling, and plain ``(cos t, sin t)`` would teach the model to predict the
    arbitrary order in which the operator happened to click.

``crossing``
    Where two or more *distinct* traces pass within ``cross_radius``. This is
    the label a segmentation corpus has to throw away. Here it is supervised
    directly, because knowing a junction is a cross-through rather than a branch
    is what lets the tracer carry an identity across it.

``orient_valid``
    False at crossings. Orientation is genuinely two-valued there and one
    angle-doubled vector cannot hold both, so the orientation loss is masked
    rather than trained on an average of two directions -- which would be a
    direction no fibre has.

The distinction between a **crossing** (two fibres, X) and a **branch** (one
fibre splitting, Y) matters and is not cosmetic: published tracers in this
family are built for vasculature and neurites, which branch. Myotubes do not
branch at ~90 degrees, so an X is always two objects, and that constraint is a
label here rather than a post-hoc rule.

Pure numpy plus `scipy.ndimage`. No torch, so it runs in `pm-annotate` and is
testable without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "annotation_tools", ROOT / "PrecisionMyotube", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@dataclass(frozen=True)
class TargetConfig:
    """Every number that shapes the target, in one place so a run can record it.

    ``sigma_px`` is the softness of the centreline. ``cross_radius_px`` is how
    close two traces must come to count as crossing -- it should be about a
    fibre half-width, since two centrelines closer than that share pixels once
    the fibres have width. ``tangent_window_px`` is the span the tangent is
    estimated over: a three-point difference on a hand-drawn polyline measures
    the operator's hand tremor, and this project has already paid ~0.2 AUC twice
    for direction features estimated on too short a window.
    """

    sigma_px: float = 2.0
    cross_radius_px: float = 5.0
    tangent_window_px: float = 15.0
    resample_px: float = 1.0
    min_points: int = 2
    # Radius of the band in which the offset-to-centreline target is defined.
    # Beyond ~1.5 fibre widths the nearest centreline is not the fibre the
    # pixel belongs to, and supervising there teaches noise.
    offset_band_px: float = 12.0


def resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """Arc-length resample to roughly `spacing` px, so tangents are uniform.

    Fiji polylines have wildly uneven vertex spacing -- a freehand drag emits a
    point per mouse event, a clicked polyline emits one per click. Tangents and
    crossing counts both weight by vertex, so leaving that unequalised would
    weight the target by how fast the operator moved the mouse.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = seg > 1e-9
    if not keep.any():
        return pts[:1]
    pts = np.vstack([pts[0], pts[1:][keep]])
    seg = seg[keep]
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(s[-1] / spacing)) + 1, 2)
    want = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(want, s, pts[:, k]) for k in range(2)])


def polyline_tangents(pts: np.ndarray, window_px: float) -> np.ndarray:
    """Unit tangent per point, from a centred difference over `window_px`.

    Returns ``(n, 2)`` of ``(d_row, d_col)``. Sign is arbitrary and irrelevant --
    the caller doubles the angle, which discards it.
    """
    n = len(pts)
    if n < 2:
        return np.zeros((n, 2))
    # points are ~1 px apart after resampling, so the window is a point count
    h = max(int(round(window_px / 2.0)), 1)
    lo = np.clip(np.arange(n) - h, 0, n - 1)
    hi = np.clip(np.arange(n) + h, 0, n - 1)
    d = pts[hi] - pts[lo]
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return np.divide(d, norm, out=np.zeros_like(d), where=norm > 1e-9)


def _round_to_grid(pts: np.ndarray, shape: tuple[int, int]):
    r = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, shape[0] - 1)
    c = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, shape[1] - 1)
    return r, c


def build_targets(shape: tuple[int, int], polylines, *,
                  config: TargetConfig | None = None) -> dict:
    """Full-field target fields from a list of ``(n, 2)`` (row, col) polylines.

    Returns ``centre``, ``orient`` (2, H, W), ``crossing``, ``orient_valid``,
    ``instance`` (nearest-trace id within `cross_radius`, 0 elsewhere) and the
    per-trace resampled points, so a caller can rebuild any of it.
    """
    from scipy import ndimage

    cfg = config or TargetConfig()
    H, W = shape

    traces, tangents = [], []
    for p in polylines:
        p = np.asarray(p, dtype=np.float64)
        if len(p) < cfg.min_points:
            continue
        rs = resample_polyline(p, cfg.resample_px)
        if len(rs) < cfg.min_points:
            continue
        traces.append(rs)
        tangents.append(polyline_tangents(rs, cfg.tangent_window_px))

    # `seed_id` carries the 1-based trace index at each traced pixel; the EDT
    # then propagates both the distance and the owning trace outward in one
    # pass, which is what makes "nearest trace" and "how far" the same query.
    seed_id = np.zeros((H, W), dtype=np.int32)
    seed_cos = np.zeros((H, W), dtype=np.float32)
    seed_sin = np.zeros((H, W), dtype=np.float32)
    for i, (pts, tan) in enumerate(zip(traces, tangents), start=1):
        r, c = _round_to_grid(pts, shape)
        theta = np.arctan2(tan[:, 0], tan[:, 1])
        seed_id[r, c] = i
        seed_cos[r, c] = np.cos(2 * theta)
        seed_sin[r, c] = np.sin(2 * theta)

    if not traces:
        # `distance_transform_edt` measures distance to the nearest zero, so an
        # all-background field has no zero to find and returns 0 everywhere --
        # which would make `centre` exp(0) = 1 on every pixel and declare the
        # whole field a centreline. Fail to empty, loudly in the record.
        return {"centre": np.zeros((H, W), np.float32),
                "offset": np.zeros((2, H, W), np.float32),
                "offset_valid": np.zeros((H, W), bool),
                "orient": np.zeros((2, H, W), np.float32),
                "crossing": np.zeros((H, W), bool),
                "orient_valid": np.zeros((H, W), bool),
                "instance": np.zeros((H, W), np.int32),
                "distance": np.full((H, W), np.inf, np.float32),
                "traces": [], "tangents": [], "n_traces": 0, "config": cfg}

    background = seed_id == 0
    dist, (ir, ic) = ndimage.distance_transform_edt(background, return_indices=True)
    nearest = seed_id[ir, ic]

    centre = np.exp(-(dist ** 2) / (2.0 * cfg.sigma_px ** 2)).astype(np.float32)

    # Displacement from each pixel to its nearest centreline point, in px.
    # Measured 2026-08-23: a network regressing `centre` directly returns a
    # 12 px-FWHM bump against this 4 px target -- MSE against a peaked target
    # under positional uncertainty is minimised by a wide flat one, and no
    # threshold on that recovers a centreline. An offset field has no peak to
    # flatten: it is smooth, locally linear, and the centreline is where it
    # vanishes, so the ridge is reconstructed sharply rather than predicted.
    rr, cc = np.mgrid[0:H, 0:W]
    offset = np.stack([(ir - rr), (ic - cc)]).astype(np.float32)
    band = dist <= cfg.offset_band_px
    offset = np.where(band, offset, 0.0).astype(np.float32)
    orient = np.stack([seed_cos[ir, ic], seed_sin[ir, ic]]).astype(np.float32)

    # A pixel is a crossing when two DIFFERENT traces are both within reach.
    # Counting distinct ids, not points, is the whole distinction: a single
    # fibre doubling back on itself is not a crossing, and a dense parallel
    # bundle is not one either.
    crossing = _crossing_map(seed_id, traces, cfg.cross_radius_px, shape)

    # `nearest > 0` is belt-and-braces against the same class of failure: a
    # pixel is only inside a fibre if some trace actually owns it.
    within = (dist <= cfg.cross_radius_px) & (nearest > 0)
    instance = np.where(within, nearest, 0).astype(np.int32)
    orient_valid = within & ~crossing

    return {"centre": centre,
            "offset": offset,
            "offset_valid": band,
            "orient": orient,
            "crossing": crossing,
            "orient_valid": orient_valid,
            "instance": instance,
            "distance": dist.astype(np.float32),
            "traces": traces,
            "tangents": tangents,
            "n_traces": len(traces),
            "config": cfg}


def _crossing_map(seed_id, traces, radius: float, shape) -> np.ndarray:
    """True where >=2 distinct traces have a point within `radius`.

    Done per trace with a dilation rather than by pairwise distances: 5,004
    instances over a full field makes the pairwise form quadratic in traces and
    this form linear.
    """
    from scipy import ndimage

    H, W = shape
    r = int(np.ceil(radius))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disc = (yy ** 2 + xx ** 2) <= radius ** 2

    count = np.zeros((H, W), dtype=np.uint8)
    for pts in traces:
        rr, cc = _round_to_grid(pts, shape)
        # Dilate inside the trace's own bounding box. A fibre covers a few
        # thousand pixels of a 13-megapixel field, so the full-field form spends
        # ~99.9% of its work on empty space and makes this quadratic in practice.
        r0, r1 = max(rr.min() - r, 0), min(rr.max() + r + 1, H)
        c0, c1 = max(cc.min() - r, 0), min(cc.max() + r + 1, W)
        m = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        m[rr - r0, cc - c0] = True
        sub = count[r0:r1, c0:c1]
        # saturating add: three fibres meeting is still "a crossing"
        np.minimum(sub + ndimage.binary_dilation(m, disc), 255, out=sub)
    return count >= 2


def targets_from_roi_zip(roi_path, shape, *, config: TargetConfig | None = None):
    """Convenience: Fiji ROI-Manager zip -> target fields."""
    from annotation_tools.relabel.fiji_roi import LINE_TYPES, read_roi_set

    rois = read_roi_set(roi_path)
    polylines = [np.asarray(r["points"], dtype=np.float64)
                 for r in rois
                 if r["type"] in LINE_TYPES and len(r["points"]) >= 2]
    return build_targets(shape, polylines, config=config)
