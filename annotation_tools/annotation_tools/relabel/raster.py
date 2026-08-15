"""Turn a centreline trace into a myotube mask.

The operator supplies a polyline down the fibre's spine and one width. Two
rasterisations are available and both are kept, because they fail in opposite
directions:

``ribbon``
    Constant-width band around the polyline. Predictable, never leaks, but
    squares off a fibre that tapers or swells.

``snap`` (default)
    The ribbon, dilated to a generous search band, intersected with pixels the
    local signal says are fibre. The mask then follows the real edge and inherits
    the width variation the operator did not have to draw. Falls back to the
    plain ribbon wherever the intersection would empty the mask, so a faint fibre
    can never silently vanish.

`snap` is a measurement decision, not a cosmetic one: `width_median_um` is a
release-gated metric, and a constant-width ribbon would encode the operator's
slider position as if it were data.
"""
from __future__ import annotations

import numpy as np


def polyline_pixels(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Dense (row, col) samples along a polyline, ~1 px apart."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        raise ValueError("need >= 2 (row, col) points")
    rows, cols = [], []
    for (r0, c0), (r1, c1) in zip(pts[:-1], pts[1:]):
        n = max(int(np.hypot(r1 - r0, c1 - c0)), 1) + 1
        rows.append(np.linspace(r0, r1, n))
        cols.append(np.linspace(c0, c1, n))
    return np.concatenate(rows), np.concatenate(cols)


def ribbon_mask(points, shape, width_px: float) -> np.ndarray:
    """Constant-width band around the centreline."""
    from scipy import ndimage as ndi

    rows, cols = polyline_pixels(points)
    h, w = shape
    rr = np.clip(np.round(rows).astype(int), 0, h - 1)
    cc = np.clip(np.round(cols).astype(int), 0, w - 1)
    spine = np.zeros(shape, dtype=bool)
    spine[rr, cc] = True
    # Distance from the spine is exact, so the band is a true half-width.
    dist = ndi.distance_transform_edt(~spine)
    return dist <= max(width_px, 1.0) / 2.0


def snap_mask(points, image: np.ndarray, width_px: float, *,
              search_factor: float = 2.2, min_keep: float = 0.35,
              max_band_fraction: float = 0.85,
              min_contrast: float = 1e-3) -> tuple[np.ndarray, dict]:
    """Ribbon intersected with local signal, with three safety guards.

    The threshold is **Otsu over the intensities inside the search band**, not a
    fixed percentile. A percentile cannot work here: the band is deliberately
    ~2.2x wider than the fibre, so most of it is background, and any fixed
    percentile lands in the background for a thin fibre and in the fibre for a
    fat one. Otsu finds the actual bimodal split, whatever the fill fraction.

    Three ways this refuses to snap, each of which would otherwise corrupt
    `width_median_um` silently:

    * **no contrast** in the band -- nothing to hug, the operator's ribbon is the
      only real statement;
    * **the threshold kept nearly the whole band** -- it discriminated nothing,
      and accepting it would return a mask `search_factor` times too wide;
    * **the threshold erased the fibre** -- wrong threshold for this object.
    """
    from scipy import ndimage as ndi

    base = ribbon_mask(points, image.shape, width_px)
    search = ribbon_mask(points, image.shape, width_px * search_factor)
    vals = image[search].astype(np.float64)
    if vals.size == 0:
        return base, {"mode": "ribbon", "reason": "empty search band"}

    contrast = float(np.percentile(vals, 99) - np.percentile(vals, 1))
    if contrast <= min_contrast:
        return base, {"mode": "ribbon", "reason": "no contrast in search band",
                      "contrast": contrast}

    try:
        from skimage.filters import threshold_otsu
        thr = float(threshold_otsu(vals))
    except Exception:
        thr = float(vals.mean())
    hugged = search & (image > thr)

    band_fraction = float(hugged.sum() / max(search.sum(), 1))
    if band_fraction > max_band_fraction:
        return base, {"mode": "ribbon",
                      "reason": "threshold did not discriminate",
                      "threshold": thr, "band_fraction": band_fraction}

    # Keep only the component touching the spine, so a bright neighbour that
    # happens to fall inside the search band is not annexed.
    lab, n = ndi.label(hugged)
    if n:
        rows, cols = polyline_pixels(points)
        rr = np.clip(np.round(rows).astype(int), 0, image.shape[0] - 1)
        cc = np.clip(np.round(cols).astype(int), 0, image.shape[1] - 1)
        touched = set(int(v) for v in lab[rr, cc] if v)
        hugged = np.isin(lab, list(touched)) if touched else np.zeros_like(hugged)

    # A snap that erases most of the fibre means the threshold was wrong for this
    # object; the operator's ribbon is the more trustworthy statement.
    if hugged.sum() < min_keep * base.sum():
        return base, {"mode": "ribbon", "reason": "snap kept too little",
                      "threshold": thr}
    # The drawn spine is never lost, but only the spine itself -- re-adding the
    # whole ribbon outside the search band would defeat the snap.
    spine = ribbon_mask(points, image.shape, 1.0)
    merged = hugged | spine
    return merged, {"mode": "snap", "threshold": thr,
                    "band_fraction": band_fraction,
                    "kept_fraction": float(hugged.sum() / max(base.sum(), 1))}


def rasterize_trace(trace: dict, image: np.ndarray) -> tuple[np.ndarray, dict]:
    """One stored trace -> boolean mask."""
    pts = [(float(p[0]), float(p[1])) for p in trace["points"]]
    width = float(trace.get("width_px", 8.0))
    if trace.get("mode", "snap") == "ribbon":
        return ribbon_mask(pts, image.shape, width), {"mode": "ribbon"}
    return snap_mask(pts, image, width)


def unlabelled_fibre_ignore(territory: np.ndarray, labels: np.ndarray, *,
                            halo_px: float = 6.0) -> tuple[np.ndarray, dict]:
    """Fibre-like pixels that carry no label -> ignore.

    This is the difference between dense relabelling being useful and being
    actively harmful. Omnipose regresses a distance field and a flow field, and
    the background target is not neutral -- it is "distance 0, flow 0". A real
    fibre left unlabelled therefore teaches the network to SUPPRESS fibre-like
    appearance. Partial dense labelling without this mask is worse than the
    sparse corpus it replaces, because it adds targets while leaving their
    unlabelled neighbours asserting the opposite.

    ``halo_px`` keeps a margin around every label OUT of the ignore mask. The
    transition from fibre to true background is exactly what the distance field
    has to learn; ignoring it would blur every edge the operator just drew.
    """
    from scipy import ndimage as ndi

    territory = np.asarray(territory).astype(bool)
    target = np.asarray(labels) > 0
    if not target.any():
        ignore = territory.copy()
        clear = 0
    else:
        # Distance from the nearest labelled pixel; only genuinely separate
        # fibre-like regions get ignored.
        dist = ndi.distance_transform_edt(~target)
        ignore = territory & (dist > halo_px)
        clear = int((territory & ~ignore & ~target).sum())
    return ignore, {
        "territory_px": int(territory.sum()),
        "labelled_px": int(target.sum()),
        "unlabelled_fibre_ignored_px": int(ignore.sum()),
        "halo_px": halo_px,
        "halo_kept_as_background_px": clear,
        "note": ("fibre-like territory carrying no label is ignored so it "
                 "cannot be taught as background; a halo around each label is "
                 "deliberately left as background to preserve the edge signal"),
    }


def compose_labels(base_labels: np.ndarray, traces: list[dict],
                   image: np.ndarray, *, rejected: set[int] | None = None
                   ) -> tuple[np.ndarray, list[dict]]:
    """Existing certified labels + new traces -> one label image.

    Existing instances keep their identity unless explicitly rejected. New traces
    are appended with fresh ids. Where a new trace overlaps an existing instance,
    the EXISTING one wins -- a relabelling session must not silently rewrite a
    mask a human already certified.
    """
    rejected = rejected or set()
    out = np.zeros_like(base_labels, dtype=np.int32)
    provenance: list[dict] = []

    next_id = 1
    for lid in range(1, int(base_labels.max()) + 1):
        m = base_labels == lid
        if not m.any() or lid in rejected:
            continue
        out[m] = next_id
        provenance.append({"label": next_id, "origin": "bootstrap_v1",
                           "source_label": lid})
        next_id += 1

    for t in traces:
        mask, info = rasterize_trace(t, image)
        mask &= out == 0                      # existing certified masks win
        if mask.sum() < 4:
            provenance.append({"label": None, "origin": "relabel",
                               "trace_id": t.get("trace_id"),
                               "skipped": "fully overlapped or empty"})
            continue
        out[mask] = next_id
        provenance.append({"label": next_id, "origin": "relabel",
                           "trace_id": t.get("trace_id"),
                           "reviewer": t.get("reviewer"),
                           "raster": info})
        next_id += 1

    return out, provenance
