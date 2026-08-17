"""Turn a Fiji ROI set into coloured instances and into training masks.

`python -m annotation_tools.relabel import-fiji --roi <zip> --well <well>`

Two outputs, deliberately separate:

* **`<well>__fiji_instances.png`** -- every ROI in its own colour over the raw
  field. This is the picture: Fiji draws every ROI the same yellow, so a human
  cannot see where one fibre ends and the next begins, and neither can a reader
  of the figure. One colour per instance makes the identity visible.
* **traces JSONL** -- the same ROIs as relabel traces, so `apply` rasterises them
  through `snap_mask` exactly as hand-drawn ones. The Fiji annotation is
  centrelines only (Area/Length says one pixel wide), so width comes from the
  image rather than from a stroke setting.

Nothing is overwritten: traces land in the relabel store for the named well and
`apply` still has to be run to build a corpus version.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fiji_roi import read_roi_set, rois_to_traces, summarise
from .store import TraceStore

BOOTSTRAP = Path("PrecisionMyotube/annotation_work/bootstrap_v1")
PIXEL_UM = 0.650017


def colour_overlay(field: np.ndarray, traces: list[dict], *,
                   line_px: float = 3.0, seed: int = 7) -> np.ndarray:
    """Raw field in grey, one colour per ROI centreline.

    Stamps a small disc along each polyline rather than calling `ribbon_mask`,
    which would run a full-field distance transform per trace -- fine for one
    object, minutes per well at 350+ of them.
    """
    from .raster import polyline_pixels

    lo, hi = np.percentile(field, 1), np.percentile(field, 99.5)
    g = np.clip((field.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.repeat(g[:, :, None], 3, axis=2) * 0.55        # dim the background
    h, w = field.shape

    r = max(int(round(line_px / 2)), 1)
    disc = [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1)
            if dr * dr + dc * dc <= r * r]

    rng = np.random.default_rng(seed)
    for t in traces:
        pts = [(p[0], p[1]) for p in t["points"]]
        if len(pts) < 2:
            continue
        rows, cols = polyline_pixels(pts)
        rr = np.round(rows).astype(int)
        cc = np.round(cols).astype(int)
        colour = rng.integers(90, 255, size=3) / 255.0
        for dr, dc in disc:
            r2 = rr + dr
            c2 = cc + dc
            ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
            rgb[r2[ok], c2[ok]] = colour
    return np.clip(rgb, 0, 1)


def main(argv=None) -> int:
    import tifffile
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roi", required=True, help="Fiji ROI .zip (or .roi)")
    ap.add_argument("--well", required=True,
                    help="bootstrap well these ROIs were drawn on")
    ap.add_argument("--bootstrap", default=str(BOOTSTRAP))
    ap.add_argument("--traces", default="PrecisionMyotube/annotation_work/relabel")
    ap.add_argument("--out", default="model_labs/omnipose_lab/_runs/fiji_import")
    ap.add_argument("--width-px", type=float, default=8.0,
                    help="starting ribbon width before snapping; the median "
                         "certified myotube is 5.4 um = 8.3 px")
    ap.add_argument("--mode", default="snap", choices=["snap", "ribbon"])
    ap.add_argument("--reviewer", default=None)
    ap.add_argument("--commit", action="store_true",
                    help="write the traces into the relabel store. Without "
                         "this, only the picture and a report are produced.")
    args = ap.parse_args(argv)

    roi_path = Path(args.roi)
    rois = read_roi_set(roi_path)
    info = summarise(rois)
    print(f"{roi_path.name}: {info['n_rois']} ROIs  {info['by_type']}")
    print(f"  usable as centrelines: {info['n_usable']}  "
          f"points/ROI median {info['points_per_roi']['median']:.0f} "
          f"(min {info['points_per_roi']['min']}, "
          f"max {info['points_per_roi']['max']})")
    if info["n_usable"] == 0:
        raise SystemExit("no line-like ROIs found; are these area ROIs?")

    well_dir = Path(args.bootstrap) / args.well
    if not well_dir.is_dir():
        raise SystemExit(f"no well at {well_dir}")
    field = tifffile.imread(well_dir / "image_fiber.tif")

    traces = rois_to_traces(rois, width_px=args.width_px, mode=args.mode,
                            reviewer=args.reviewer, source=str(roi_path))

    # Do the ROIs actually land on this field?
    rr = np.array([p[0] for t in traces for p in t["points"]])
    cc = np.array([p[1] for t in traces for p in t["points"]])
    oob = int(((rr < 0) | (rr >= field.shape[0]) |
               (cc < 0) | (cc >= field.shape[1])).sum())
    print(f"  field {field.shape}  ROI extent rows {rr.min():.0f}-{rr.max():.0f} "
          f"cols {cc.min():.0f}-{cc.max():.0f}  out-of-bounds points: {oob}")
    if oob:
        print("  !! points fall outside the field -- are these ROIs from a "
              "different image or a scaled copy?")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = colour_overlay(field, traces)
    png = out_dir / f"{args.well}__fiji_instances.png"
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    im.thumbnail((2400, 2400))
    im.save(png)
    print(f"\n-> {png}   ({len(traces)} instances, one colour each)")

    lens_px = []
    for t in traces:
        p = np.asarray(t["points"])
        lens_px.append(float(np.hypot(*(np.diff(p, axis=0).T)).sum()))
    lens_um = np.array(lens_px) * PIXEL_UM
    print(f"   traced length um: median {np.median(lens_um):.1f}  "
          f"p10 {np.percentile(lens_um,10):.1f}  "
          f"p90 {np.percentile(lens_um,90):.1f}  max {lens_um.max():.1f}")

    base = well_dir / "labels.tif"
    if base.exists():
        n_base = int(tifffile.imread(base).max())
        print(f"   bootstrap_v1 has {n_base} instances for this well "
              f"-> {len(traces)/max(n_base,1):.1f}x more")

    report = {"roi_file": str(roi_path), "well": args.well, **info,
              "n_traces": len(traces), "width_px": args.width_px,
              "mode": args.mode, "out_of_bounds_points": oob,
              "traced_length_um": {"median": float(np.median(lens_um)),
                                   "p10": float(np.percentile(lens_um, 10)),
                                   "p90": float(np.percentile(lens_um, 90)),
                                   "max": float(lens_um.max())},
              "committed": bool(args.commit)}
    (out_dir / f"{args.well}__fiji_import.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    if args.commit:
        store = TraceStore(args.traces, args.well)
        for t in traces:
            store.append(t)
        print(f"\ncommitted {len(traces)} traces -> {store.path}")
        print("run `apply` to build a corpus version from them.")
    else:
        print("\ndry run: look at the PNG first, then re-run with --commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
