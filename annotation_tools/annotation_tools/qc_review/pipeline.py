"""Per-proposal features + crop thumbnails, assembled into review cases.

Features are hand-crafted and interpretable (this is the whole point of the QC
model — you can SEE the learned rule). No skimage/sklearn needed here; features
come from numpy + scipy so the page can be built in any environment.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from scipy import ndimage as ndi


# Interpretable features attached to every proposal. The model uses a subset.
def proposal_features(mask_crop: np.ndarray, fiber_crop: np.ndarray,
                      territory_crop: np.ndarray | None, pixel_um: float,
                      touches_border: bool) -> dict:
    ys, xs = np.nonzero(mask_crop)
    area_px = int(mask_crop.sum())
    # PCA of the pixel coordinates -> major/minor axis lengths (length & width proxy).
    coords = np.stack([ys.astype(float), xs.astype(float)], axis=1)
    coords -= coords.mean(0)
    if len(coords) >= 2:
        cov = np.cov(coords.T)
        evals = np.clip(np.linalg.eigvalsh(cov), 0, None)
        major = 4.0 * float(np.sqrt(evals[-1]))
        minor = 4.0 * float(np.sqrt(evals[0]))
    else:
        major = minor = 1.0
    aspect = major / minor if minor > 1e-6 else float(major)

    # Solidity via convex-hull area (filled hull of the mask).
    filled = ndi.binary_fill_holes(mask_crop)
    hull_area = _convex_area(mask_crop)
    solidity = area_px / hull_area if hull_area else 1.0
    bbox_area = mask_crop.shape[0] * mask_crop.shape[1]
    extent = area_px / bbox_area if bbox_area else 0.0

    fiber_vals = fiber_crop[mask_crop]
    fiber_mean = float(fiber_vals.mean()) if fiber_vals.size else 0.0

    if territory_crop is not None:
        terr_overlap = float((mask_crop & territory_crop).sum()) / max(area_px, 1)
    else:
        terr_overlap = 0.0

    return {
        "length_um": round(major * pixel_um, 2),
        "width_um": round(minor * pixel_um, 2),
        "area_um2": round(area_px * pixel_um * pixel_um, 1),
        "aspect": round(aspect, 2),
        "solidity": round(solidity, 3),
        "extent": round(extent, 3),
        "fiber_mean": round(fiber_mean, 1),
        "territory_overlap": round(terr_overlap, 3),
        "touches_border": 1.0 if touches_border else 0.0,
    }


def _convex_area(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return float(mask.sum())
    pts = np.stack([xs, ys], axis=1).astype(float)
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(pts).volume)  # 2-D 'volume' == area
    except Exception:
        return float(mask.sum())


# Mask outline / overlay colour: magenta, to contrast the green fiber + blue nuclei.
OUTLINE_RGB = (255, 74, 170)


def _composite_native(fiber_crop, dapi_crop, lo, hi, dlo, dhi) -> np.ndarray:
    """Two-channel fluorescence composite: green Desmin fiber, blue DAPI nuclei."""
    fg = np.clip((fiber_crop.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    if dapi_crop is not None:
        dg = np.clip((dapi_crop.astype(np.float32) - dlo) / max(dhi - dlo, 1e-6), 0, 1)
    else:
        dg = np.zeros_like(fg)
    rgb = np.stack([dg * 0.30, fg, dg * 0.95], axis=-1)   # R (faint), G=fiber, B=DAPI
    return (rgb * 255).astype(np.uint8)


def _resize_rgb(rgb_native: np.ndarray, nw: int, nh: int):
    from PIL import Image
    img = Image.fromarray(rgb_native, mode="RGB")
    down = nw < rgb_native.shape[1]
    return img.resize((nw, nh), Image.LANCZOS if down else Image.BICUBIC)


def _outlined_thumb(rgb_native, mask_crop, size):
    """Small colour composite thumbnail with a crisp magenta outline (grid scan)."""
    from PIL import Image
    h, w = mask_crop.shape
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    base = _resize_rgb(rgb_native, nw, nh)
    mask_rs = np.asarray(
        Image.fromarray((mask_crop.astype(np.uint8) * 255)).resize((nw, nh), Image.NEAREST)) > 127
    outline = mask_rs & ~ndi.binary_erosion(mask_rs, iterations=max(1, round(scale)))
    arr = np.asarray(base).copy()
    arr[outline] = OUTLINE_RGB
    s = max(nw, nh)
    canvas = Image.new("RGB", (s, s), (8, 12, 11))
    canvas.paste(Image.fromarray(arr), ((s - nw) // 2, (s - nh) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _composite_jpeg(rgb_native, nw, nh, quality: int = 80):
    """Clean colour composite (no outline) for the in-browser editor background."""
    base = _resize_rgb(rgb_native, nw, nh)
    buf = io.BytesIO()
    base.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _rowmajor_rle(mask: np.ndarray) -> dict:
    """Compact row-major run length (counts start with a background run)."""
    flat = mask.ravel(order="C").astype(np.uint8)
    if not flat.size:
        return {"h": 0, "w": 0, "counts": []}
    idx = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], idx, [flat.size]))).astype(int).tolist()
    if flat[0] == 1:
        counts = [0] + counts
    return {"h": int(mask.shape[0]), "w": int(mask.shape[1]), "counts": [int(x) for x in counts]}


def _mask_resized(mask_crop, nw, nh):
    from PIL import Image
    return np.asarray(
        Image.fromarray((mask_crop.astype(np.uint8) * 255)).resize((nw, nh), Image.NEAREST)) > 127


def _orientation_field(fiber: np.ndarray, sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel local orientation as (cos2θ, sin2θ) via the structure tensor."""
    g = fiber.astype(np.float32)
    gy = ndi.sobel(g, axis=0)
    gx = ndi.sobel(g, axis=1)
    jxx = ndi.gaussian_filter(gx * gx, sigma)
    jyy = ndi.gaussian_filter(gy * gy, sigma)
    jxy = ndi.gaussian_filter(gx * gy, sigma)
    two_theta = np.arctan2(2 * jxy, jxx - jyy + 1e-9)
    return np.cos(two_theta), np.sin(two_theta)


def orientation_hypotheses(mask: np.ndarray, fiber: np.ndarray, *, ks=(2, 3),
                           min_part_px: int = 40) -> list[list[np.ndarray]]:
    """Return decomposition hypotheses for a (possibly merged) proposal.

    Hypothesis 0 is always the whole proposal as one myotube. Further hypotheses
    cluster the pixels by local orientation into K coherent parts and split any
    disconnected pieces — so a vertical fibre crossing two horizontals separates
    into vertical + two horizontals. Each hypothesis is a list of boolean part
    masks. Heuristic and imperfect by design: the reviewer cycles and corrects.
    """
    hyps: list[list[np.ndarray]] = [[mask.copy()]]
    ys, xs = np.nonzero(mask)
    if ys.size < 2 * min_part_px:
        return hyps
    c2, s2 = _orientation_field(fiber)
    feats = np.column_stack([c2[ys, xs], s2[ys, xs]])
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return hyps

    for k in ks:
        if ys.size < k * min_part_px:
            continue
        lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(feats)
        parts: list[np.ndarray] = []
        for cl in range(k):
            band = np.zeros_like(mask)
            band[ys[lab == cl], xs[lab == cl]] = True
            comp, ncomp = ndi.label(band)
            for ci in range(1, ncomp + 1):
                piece = comp == ci
                if int(piece.sum()) >= min_part_px:
                    parts.append(piece)
        if 2 <= len(parts) <= 5 and not _same_partition(parts, hyps):
            hyps.append(parts)
    return hyps


def _same_partition(parts, hyps) -> bool:
    sig = tuple(sorted(int(p.sum()) for p in parts))
    return any(tuple(sorted(int(p.sum()) for p in h)) == sig for h in hyps)


def _dir_from_end(coords_set: set, end: tuple, length: int) -> np.ndarray:
    """Unit direction pointing from ~``length`` px inside a segment toward ``end``."""
    from collections import deque
    dist = {end: 0}
    q = deque([end])
    far = end
    while q:
        p = q.popleft()
        if dist[p] >= length:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (p[0] + dy, p[1] + dx)
                if nb in coords_set and nb not in dist:
                    dist[nb] = dist[p] + 1
                    q.append(nb)
                    far = nb
    v = np.array(end, float) - np.array(far, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(2)


def _prune_spurs(skel: np.ndarray, min_len: int, passes: int = 4) -> np.ndarray:
    """Remove short dead-end branches so noise doesn't create false junctions."""
    skel = skel.copy()
    K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    struct = np.ones((3, 3))
    for _ in range(passes):
        nb = ndi.convolve(skel.astype(int), K, mode="constant") * skel
        junction = skel & (nb >= 3)
        endpoint = skel & (nb == 1)
        seg, nseg = ndi.label(skel & ~ndi.binary_dilation(junction, structure=struct), structure=struct)
        removed = False
        for i in range(1, nseg + 1):
            si = seg == i
            if int(si.sum()) < min_len and (si & endpoint).any():
                skel[si] = False
                removed = True
        if not removed:
            break
    return skel


def trace_hypotheses(mask: np.ndarray, *, min_part_px: int = 25, dir_len: int = 6,
                     antiparallel: float = -0.2, min_spur: int = 12):
    """Decompose a merged proposal by tracing fibres through crossings.

    Skeletonise; at each junction, pair the two branch-ends whose directions are
    most anti-parallel (a fibre passing straight through); union those branches
    into fibres. A smooth single fibre has no junction, so it is never split.

    Returns ``(segments, hypotheses)`` where ``segments`` is a list of atomic
    branch-territory masks (the click-reassign unit) and ``hypotheses`` is a list
    of groupings (group index per segment): whole, traced fibres, and split-at-
    every-junction. Returns ``None`` when there is nothing to split.
    """
    from skimage.morphology import skeletonize
    skel = skeletonize(mask)
    if skel.sum() < 3:
        return None
    skel = _prune_spurs(skel, min_spur)
    K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    nb = ndi.convolve(skel.astype(int), K, mode="constant") * skel
    junction = skel & (nb >= 3)
    struct = np.ones((3, 3))
    jclust, njc = ndi.label(ndi.binary_dilation(junction, structure=struct) & skel, structure=struct)
    branch_px = skel & (jclust == 0)
    seg, nseg = ndi.label(branch_px, structure=struct)
    if nseg < 2:
        return None

    parent = list(range(nseg + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seg_pixels = {i: {tuple(p) for p in np.argwhere(seg == i)} for i in range(1, nseg + 1)}
    for jc in range(1, njc + 1):
        jmask = jclust == jc
        jcen = np.argwhere(jmask).mean(0)
        adj = ndi.binary_dilation(jmask, structure=struct)
        touch = []
        for i in range(1, nseg + 1):
            si = seg == i
            hit = si & adj
            if not hit.any():
                continue
            cand = np.argwhere(hit)
            end = tuple(cand[np.argmin(((cand - jcen) ** 2).sum(1))])
            touch.append((i, _dir_from_end(seg_pixels[i], end, dir_len)))
        pairs = sorted(
            ((float(np.dot(touch[a][1], touch[b][1])), a, b)
             for a in range(len(touch)) for b in range(a + 1, len(touch))),
            key=lambda t: t[0])
        used = set()
        for dot, a, b in pairs:
            if dot > antiparallel:
                break
            if a in used or b in used:
                continue
            parent[find(touch[a][0])] = find(touch[b][0])
            used.update((a, b))

    # nearest-skeleton territory -> segment id per foreground pixel
    _, inds = ndi.distance_transform_edt(~skel, return_indices=True)
    nearest_seg = np.where(mask, seg[inds[0], inds[1]], 0)
    seg_ids = [i for i in range(1, nseg + 1) if int((nearest_seg == i).sum()) >= min_part_px]
    if len(seg_ids) < 2:
        return None
    segments = [(nearest_seg == s) for s in seg_ids]

    grp = {}
    trace_group = [grp.setdefault(find(s), len(grp)) for s in seg_ids]
    n_traced = max(trace_group) + 1
    if n_traced < 2:
        return None                          # tracer sees one fibre -> offer no split
    whole_group = [0] * len(seg_ids)
    hyps = [whole_group, trace_group]
    if len(seg_ids) != n_traced:             # a "split every crossing" fallback
        hyps.append(list(range(len(seg_ids))))
    return segments, hyps


def build_cases(label_image: np.ndarray, fiber: np.ndarray, pixel_um: float,
                territory: np.ndarray | None = None, *, dapi: np.ndarray | None = None,
                thumb_px: int = 200, edit_px: int = 384, pad_frac: float = 0.5,
                min_margin: int = 32, max_margin: int = 240,
                max_cases: int | None = None,
                only_ids: set | None = None) -> list[dict]:
    """Return review cases, one per label. Each carries:

    - ``features`` + cold-start ``prior`` (ranks likely-real proposals first);
    - ``thumb``: small outlined thumbnail for the grid;
    - ``edit_img`` / ``mask_rle`` / ``geom``: a clean fiber crop plus an editable
      mask and the geometry to map an edited mask back to full-field pixels, so
      you can ADD and REMOVE annotation in the enlarged editor.
    """
    label_image = np.asarray(label_image)
    fiber = np.asarray(fiber)
    dapi = None if dapi is None else np.asarray(dapi)
    territory = None if territory is None else np.asarray(territory).astype(bool)
    H, W = label_image.shape
    lo, hi = float(np.percentile(fiber, 1.0)), float(np.percentile(fiber, 99.5))
    dlo, dhi = (float(np.percentile(dapi, 1.0)), float(np.percentile(dapi, 99.5))) \
        if dapi is not None else (0.0, 1.0)

    slices = ndi.find_objects(label_image)
    cases: list[dict] = []
    for label_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        if only_ids is not None and label_id not in only_ids:
            continue
        r0, c0, r1, c1 = sl[0].start, sl[1].start, sl[0].stop, sl[1].stop
        touches = (r0 == 0 or c0 == 0 or r1 == H or c1 == W)

        # features from the tight mask
        sub = label_image[sl] == label_id
        terr_sub = territory[sl] if territory is not None else None
        feats = proposal_features(sub, fiber[sl], terr_sub, pixel_um, touches)

        # padded crop with generous context (room to extend a short label along
        # the fibre), clipped to the field
        mh = int(np.clip(round((r1 - r0) * pad_frac), min_margin, max_margin))
        mw = int(np.clip(round((c1 - c0) * pad_frac), min_margin, max_margin))
        pr0, pc0 = max(0, r0 - mh), max(0, c0 - mw)
        pr1, pc1 = min(H, r1 + mh), min(W, c1 + mw)
        fiber_pad = fiber[pr0:pr1, pc0:pc1]
        dapi_pad = dapi[pr0:pr1, pc0:pc1] if dapi is not None else None
        mask_pad = (label_image[pr0:pr1, pc0:pc1] == label_id)
        ph, pw = mask_pad.shape
        rgb_native = _composite_native(fiber_pad, dapi_pad, lo, hi, dlo, dhi)

        scale = edit_px / max(ph, pw)
        ew, eh = max(1, round(pw * scale)), max(1, round(ph * scale))
        mask_edit = _mask_resized(mask_pad, ew, eh)

        case = {
            "id": f"myotube_{label_id:04d}",
            "features": feats,
            "prior": round(_prior_score(feats), 3),
            "thumb": _outlined_thumb(rgb_native, mask_pad, thumb_px),
            "edit_img": _composite_jpeg(rgb_native, ew, eh),
            "mask_rle": _rowmajor_rle(mask_edit),
            "geom": {"origin": [int(pr0), int(pc0)], "src_h": int(ph), "src_w": int(pw),
                     "edit_h": int(eh), "edit_w": int(ew)},
        }
        traced = trace_hypotheses(mask_edit)
        if traced is not None:
            segs, hyps = traced
            case["segments"] = [_rowmajor_rle(s) for s in segs]
            case["hypotheses"] = hyps        # whole / traced-fibres / split-at-junctions
        cases.append(case)

    cases.sort(key=lambda c: -c["prior"])
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def _prior_score(f: dict) -> float:
    """Transparent cold-start likelihood a proposal is a real complete myotube."""
    length = f["length_um"]
    length_term = 1.0 - np.exp(-length / 50.0)          # saturates ~50 um
    elong = min(f["aspect"] / 4.0, 1.0)                 # elongated is myotube-like
    terr = f["territory_overlap"]
    border = 0.6 if f["touches_border"] else 1.0        # edge fibers are truncated
    return float(length_term * (0.4 + 0.6 * elong) * (0.3 + 0.7 * terr) * border)
