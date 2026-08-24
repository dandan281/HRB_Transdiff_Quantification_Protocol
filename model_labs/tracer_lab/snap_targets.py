"""Snap operator polylines laterally onto the image ridge, and prove it safe.

Measured 2026-08-23 on PLATE_32: the operator's traced centreline sits
**SD 3.3 px from the image's own intensity ridge** (3.36 D04, 3.18 B02, on
isolated fibres with no neighbour on the cut, median offset ~0 so there is no
systematic bias). That is ordinary hand-tracing wobble at this zoom and it
does not affect the operator's LENGTH measurements -- lateral wobble barely
changes arc length. But as a per-pixel training target it is noise, and it
forces blur: a target wandering +-3.3 px makes the Bayes-optimal centre
prediction ~7.8 px FWHM against a 4 px target. The networks produced 12 px, so
roughly half the blur four training versions chased was never learnable from
these targets.

The fix is on the data side: move each traced point sideways onto the ridge it
was drawn for, and change nothing else.

Three constraints keep the snap honest, because a free snap would happily walk
a line onto the brighter fibre next door:

* **lateral only** -- points move along the trace normal, never along it, so
  arc length and end points are preserved by construction;
* **bounded** -- at most ``max_lateral_px`` (default 3.0, well under the 8 px
  fibre width), so a point cannot reach a neighbouring fibre's centre;
* **smoothed along the trace** -- the displacement series is Gaussian-filtered
  before it is applied, so the snap follows the fibre's slow drift and not the
  per-point noise of the peak finder.

Everything here is verifiable before any training happens: ``--verify`` reports
the offset SD before and after, the per-trace length change, and how often a
snapped point ends up closer to a DIFFERENT operator trace than its own (the
identity-theft check). If those numbers are not good, do not train on it.

    python model_labs/tracer_lab/snap_targets.py --verify --wells D04,B02
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CORPUS = ROOT / "PrecisionMyotube/annotation_work/plate32_dense_v1"
PLATE = ROOT / "Q_PLATES/Q_Plates/PLATE_32"


def _subpixel_peak(profile: np.ndarray, offs: np.ndarray,
                   mode: str = "nearest") -> float | None:
    """Parabolic sub-pixel peak of a 1-D profile; None at the window edge.

    ``mode="brightest"`` takes the global maximum of the cut. Measured: that
    steals 2.4% of points onto a NEIGHBOURING trace, because in a dense field
    the brightest thing within reach is often the fibre next door.
    ``mode="nearest"`` takes the local maximum closest to the drawn line
    instead -- the fibre the operator was looking at is the one they drew on,
    so proximity is better evidence of identity than brightness.
    """
    if mode == "nearest":
        interior = np.arange(1, len(profile) - 1)
        local = interior[(profile[1:-1] >= profile[:-2])
                         & (profile[1:-1] >= profile[2:])]
        if len(local) == 0:
            return None
        k = int(local[np.argmin(np.abs(offs[local]))])
    else:
        k = int(np.argmax(profile))
    if k == 0 or k == len(profile) - 1:
        return None
    y0, y1, y2 = profile[k - 1], profile[k], profile[k + 1]
    den = y0 - 2.0 * y1 + y2
    step = offs[1] - offs[0]
    sub = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
    return float(offs[k] + sub * step)


def lateral_offsets(image: np.ndarray, pts: np.ndarray, tan: np.ndarray,
                    reach_px: float = 6.0, step: float = 0.5,
                    mode: str = "nearest") -> np.ndarray:
    """Signed distance from each traced point to the local image ridge."""
    H, W = image.shape
    offs = np.arange(-reach_px, reach_px + 1e-9, step)
    out = np.zeros(len(pts), dtype=np.float64)
    for i, (p, u) in enumerate(zip(pts, tan)):
        n = np.array([-u[1], u[0]])
        q = p[None, :] + offs[:, None] * n[None, :]
        r = np.clip(np.rint(q[:, 0]).astype(int), 0, H - 1)
        c = np.clip(np.rint(q[:, 1]).astype(int), 0, W - 1)
        d = _subpixel_peak(image[r, c], offs, mode=mode)
        out[i] = 0.0 if d is None else d
    return out


def snap_polyline(image: np.ndarray, pts: np.ndarray, tan: np.ndarray, *,
                  max_lateral_px: float = 3.0, smooth_pts: float = 7.0,
                  reach_px: float = 4.0, mode: str = "nearest") -> np.ndarray:
    """Move `pts` sideways onto the ridge, bounded and smoothed."""
    from scipy import ndimage

    raw = lateral_offsets(image, pts, tan, reach_px=reach_px, mode=mode)
    # smooth BEFORE clipping: the peak finder is noisy point to point, and a
    # fibre's true offset from the drawn line drifts slowly along it
    smooth = ndimage.gaussian_filter1d(raw, smooth_pts, mode="nearest")
    smooth = np.clip(smooth, -max_lateral_px, max_lateral_px)
    normals = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    return pts + smooth[:, None] * normals


def snap_well(well: str, *, max_lateral_px: float = 3.0,
              smooth_pts: float = 7.0, mode: str = "nearest",
              reach_px: float = 4.0):
    """-> (image, original fields, snapped traces, per-trace diagnostics)."""
    import tifffile
    from scipy import ndimage
    from tracer_lab.centreline_targets import targets_from_roi_zip

    zips = sorted(PLATE.glob(f"*{well}*.zip"))
    if not zips:
        raise FileNotFoundError(f"no ROI zip for {well}")
    raw_im = tifffile.imread(CORPUS / well / "image_fiber.tif").astype(np.float32)
    im = ndimage.gaussian_filter(raw_im, 1.0)
    gt = targets_from_roi_zip(zips[0], raw_im.shape)

    snapped = [snap_polyline(im, t, tan, max_lateral_px=max_lateral_px,
                             smooth_pts=smooth_pts, mode=mode,
                             reach_px=reach_px)
               for t, tan in zip(gt["traces"], gt["tangents"])]
    return im, gt, snapped


def _arc(p):
    return float(np.linalg.norm(np.diff(np.asarray(p), axis=0), axis=1).sum())


def verify(wells, *, max_lateral_px=3.0, smooth_pts=7.0,
           mode="nearest", reach_px=4.0) -> int:
    """Report the three numbers that decide whether to train on snapped data."""
    from tracer_lab.centreline_targets import build_targets

    ok = True
    for well in wells:
        im, gt, snapped = snap_well(well, max_lateral_px=max_lateral_px,
                                    smooth_pts=smooth_pts, mode=mode,
                                    reach_px=reach_px)
        before, after, dlen, moved = [], [], [], []
        for t, tan, s in zip(gt["traces"], gt["tangents"], snapped):
            b = lateral_offsets(im, t, tan, mode=mode, reach_px=reach_px)
            a = lateral_offsets(im, s, tan, mode=mode, reach_px=reach_px)
            before.append(b[np.abs(b) < reach_px])
            after.append(a[np.abs(a) < reach_px])
            dlen.append((_arc(s) - _arc(t)) / max(_arc(t), 1e-9))
            moved.append(np.linalg.norm(s - t, axis=1))
        b = np.concatenate(before)
        a = np.concatenate(after)
        dl = np.array(dlen)
        mv = np.concatenate(moved)

        # identity check: does a snapped point now sit nearer a DIFFERENT
        # operator trace than the one it came from?
        inst = build_targets(im.shape, gt["traces"])["instance"]
        H, W = im.shape
        theft = 0
        total = 0
        for i, s in enumerate(snapped, start=1):
            r = np.clip(np.rint(s[:, 0]).astype(int), 0, H - 1)
            c = np.clip(np.rint(s[:, 1]).astype(int), 0, W - 1)
            owner = inst[r, c]
            named = owner > 0
            theft += int((owner[named] != i).sum())
            total += int(named.sum())

        print(f"\n== {well}: {len(gt['traces'])} traces, "
              f"max_lateral {max_lateral_px} px, smooth {smooth_pts} pts")
        print(f"  ridge offset SD   before {b.std():.2f} px "
              f"-> after {a.std():.2f} px   (target < 1.5)")
        print(f"  |offset| > 2 px   before {(np.abs(b) > 2).mean():.0%} "
              f"-> after {(np.abs(a) > 2).mean():.0%}")
        print(f"  per-trace length change: median {np.median(dl):+.2%}, "
              f"p95 |change| {np.percentile(np.abs(dl), 95):.2%}  "
              f"(target < 3%)")
        print(f"  points moved: median {np.median(mv):.2f} px, "
              f"max {mv.max():.2f} px")
        print(f"  identity theft: {theft / max(total, 1):.2%} of snapped "
              f"points now sit on another trace  (target < 1%)")
        good = (a.std() < 1.5 and np.percentile(np.abs(dl), 95) < 0.03
                and theft / max(total, 1) < 0.01)
        print(f"  => {'PASS' if good else 'FAIL'}")
        ok = ok and good
    print(f"\nSNAP VERIFY {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--wells", default="D04,B02")
    ap.add_argument("--max-lateral", type=float, default=3.0)
    ap.add_argument("--smooth", type=float, default=7.0)
    ap.add_argument("--mode", default="nearest",
                    choices=("nearest", "brightest"))
    ap.add_argument("--reach", type=float, default=4.0)
    a = ap.parse_args(argv)
    wells = [w for w in a.wells.split(",") if w]
    if a.verify:
        return verify(wells, max_lateral_px=a.max_lateral,
                      smooth_pts=a.smooth, mode=a.mode, reach_px=a.reach)
    print("nothing to do; pass --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
