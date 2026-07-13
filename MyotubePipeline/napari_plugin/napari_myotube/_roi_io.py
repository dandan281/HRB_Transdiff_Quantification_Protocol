"""ImageJ ROI <-> polylines, and the bridge to Cellpose training data (pure Python).

Your ground-truth quantifications are ImageJ RoiManager ``.zip`` sets (magic ``Iout``) of
polyline / freehand-line ROIs -- i.e. myotube CENTERLINES. Cellpose trains on a filled label
image: every object is a region of pixels tagged with a unique integer id. So the bridge is:

    ImageJ ROI centerlines  --rasterize + dilate to fiber width-->  uint16 label mask

and, for inference, the inverse:

    Cellpose label mask  --skeletonize + order-->  centerline polylines (for length)

No napari import here so this module is unit-testable with the base anaconda env.
"""
from __future__ import annotations

import os
import zipfile

import numpy as np


# ---------------------------------------------------------------- ImageJ ROI I/O

def read_imagej_zip(path) -> list[np.ndarray]:
    """List of (N, 2) (x, y) polylines from an ImageJ ROI ``.zip``. Requires ``roifile``."""
    from roifile import roiread

    rois = roiread(path)
    rois = rois if isinstance(rois, (list, tuple)) else [rois]
    out: list[np.ndarray] = []
    for r in rois:
        c = np.asarray(r.coordinates(), dtype=float)   # (N, 2) x, y (subpixel when present)
        if c.ndim == 2 and len(c) >= 2:
            out.append(c)
    return out


def write_imagej_zip(path, polylines) -> int:
    """Write (x, y) polylines as an ImageJ polyline-ROI ``.zip`` (names 0001.roi ...).

    Lets napari-curated traces flow back into a Fiji/ImageJ workflow. Requires ``roifile``.
    """
    from roifile import ImagejRoi, ROI_TYPE

    if os.path.exists(path):
        os.remove(path)
    n = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for i, p in enumerate(polylines, 1):
            p = np.asarray(p, dtype=float).reshape(-1, 2)
            if len(p) < 2:
                continue
            roi = ImagejRoi.frompoints(p, name=f"{i:04d}")
            roi.roitype = ROI_TYPE.POLYLINE          # frompoints defaults to a closed freehand
            z.writestr(roi.name + ".roi", roi.tobytes())
            n += 1
    return n


# ------------------------------------------------------ centerlines -> label mask

def polylines_to_label_mask(polylines, shape, fiber_width_px: float = 15.0) -> np.ndarray:
    """Rasterize centerline polylines into a uint16 label image of ``shape = (H, W)``.

    Each polyline becomes one integer label (1..N): the centerline is drawn then dilated by
    ~``fiber_width_px`` / 2. Where fibers overlap, the later label wins -- a real Cellpose
    limitation for crossing myotubes (a pixel can carry only one label). Work is done in a
    per-fiber bounding box so a 3636x3636 field with hundreds of fibers stays fast.
    """
    from skimage.draw import line as skline
    from skimage.morphology import binary_dilation, disk

    H, W = int(shape[0]), int(shape[1])
    lab = np.zeros((H, W), dtype=np.uint16)
    rad = max(1, int(round(fiber_width_px / 2.0)))
    selem = disk(rad)
    for idx, p in enumerate(polylines, 1):
        p = np.asarray(p, dtype=float).reshape(-1, 2)
        if len(p) < 2:
            continue
        xs, ys = p[:, 0], p[:, 1]
        x0 = max(0, int(np.floor(xs.min())) - rad - 2); x1 = min(W, int(np.ceil(xs.max())) + rad + 2)
        y0 = max(0, int(np.floor(ys.min())) - rad - 2); y1 = min(H, int(np.ceil(ys.max())) + rad + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for (ax, ay), (bx, by) in zip(p[:-1], p[1:]):
            rr, cc = skline(int(round(ay - y0)), int(round(ax - x0)),
                            int(round(by - y0)), int(round(bx - x0)))
            rr = np.clip(rr, 0, sub.shape[0] - 1); cc = np.clip(cc, 0, sub.shape[1] - 1)
            sub[rr, cc] = True
        sub = binary_dilation(sub, selem)
        region = lab[y0:y1, x0:x1]
        region[sub] = idx
    return lab


def export_cellpose_pair(image, polylines, out_dir, stem, fiber_width_px: float = 15.0) -> int:
    """Write one Cellpose training pair: ``<stem>.tif`` + ``<stem>_masks.tif`` (uint16 labels).

    ``image`` is the fiber-channel image the model will also see at inference (keep the
    preprocessing identical between train and inference). Returns the number of labels.
    """
    import tifffile

    os.makedirs(out_dir, exist_ok=True)
    image = np.asarray(image)
    H, W = image.shape[:2]
    lab = polylines_to_label_mask(polylines, (H, W), fiber_width_px)
    tifffile.imwrite(os.path.join(out_dir, f"{stem}.tif"), image)
    tifffile.imwrite(os.path.join(out_dir, f"{stem}_masks.tif"), lab)
    return int(lab.max())


# ------------------------------------------------------ label mask -> centerlines

def label_to_centerlines(label_img) -> list[np.ndarray]:
    """Convert a Cellpose label image to ordered (x, y) centerline polylines (one per label).

    Skeletonize each label, then greedily order the skeleton pixels from one extreme end
    (nearest-neighbour walk). Good enough for length and for seeding an editable napari layer;
    branchy skeletons are walked as a single path (the reviewer fixes the rare bad one).
    """
    from skimage.morphology import skeletonize

    out: list[np.ndarray] = []
    for lbl in np.unique(label_img):
        if lbl == 0:
            continue
        mask = label_img == lbl
        skel = skeletonize(mask)
        pts = np.column_stack(np.nonzero(skel)).astype(float)   # (row=y, col=x)
        if len(pts) < 2:
            ys, xs = np.nonzero(mask)
            if len(xs) < 2:
                continue
            pts = np.column_stack([ys, xs]).astype(float)
        ordered = _order_points_nn(pts)                          # (y, x)
        out.append(ordered[:, ::-1])                             # -> (x, y)
    return out


def _order_points_nn(pts: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbour ordering of (N, 2) points, starting at the top-left extreme."""
    n = len(pts)
    remaining = list(range(n))
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    remaining.remove(start)
    order = [start]
    while remaining:
        last = pts[order[-1]]
        rem = np.asarray(remaining)
        d = np.sum((pts[rem] - last) ** 2, axis=1)
        nxt = int(rem[int(np.argmin(d))])
        order.append(nxt)
        remaining.remove(nxt)
    return pts[order]
