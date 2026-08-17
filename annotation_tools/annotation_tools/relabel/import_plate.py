"""Import a whole plate's Fiji ROI sets straight from the nd2 files.

`import_fiji` works against `bootstrap_v1`, which only covers PLATE_23. Most of
the operator's tracing is on plates that were never bootstrapped, so this reads
the nd2 directly and pairs each `*_ROIs*.zip` with its well by well code
(`B02_Ctrl_ROIs.zip` <-> `23_B02_ctrl.nd2`).

Per well it writes an instance-labelled TIFF, a colour overlay, and per-instance
geometry measured with the canonical `measure_mask`, so the output is directly
comparable to every other length/width number in the project.

Pixel size and channel order are read from each file's own metadata rather than
assumed. PLATE_44 taught that lesson: its channels are swapped and its pixel is
2.65x coarser, and the defaults would have produced confident nonsense.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

from .fiji_roi import read_roi_set, rois_to_traces, summarise
from .import_fiji import colour_overlay

WELL_RE = re.compile(r"([A-H])0?(\d{1,2})", re.I)


def well_code(name: str) -> str | None:
    m = WELL_RE.search(name)
    return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else None


def pair_plate(plate_dir: Path) -> list[tuple[str, Path, Path]]:
    """[(well code, roi zip, nd2), ...] paired by well code."""
    nd2s: dict[str, Path] = {}
    for p in sorted(plate_dir.glob("*.nd2")):
        # 23_B02_ctrl.nd2 -> the well code is the SECOND token, not the first
        parts = p.stem.split("_")
        code = well_code(parts[1]) if len(parts) > 1 else well_code(p.stem)
        if code:
            nd2s.setdefault(code, p)

    pairs = []
    for roi in sorted(plate_dir.glob("*ROIs*.zip")):
        code = well_code(roi.stem.split("_")[0])
        if code is None:
            print(f"  ?? cannot read a well code from {roi.name}")
            continue
        if code not in nd2s:
            print(f"  ?? {roi.name}: no nd2 for well {code}")
            continue
        pairs.append((code, roi, nd2s[code]))
    return pairs


def read_desmin(nd2_path: Path) -> tuple[np.ndarray, float, dict]:
    """Desmin channel + pixel size, both from the file's own metadata."""
    import nd2

    with nd2.ND2File(nd2_path) as f:
        arr = f.asarray()
        px = float(f.voxel_size().x)
        names = [c.channel.name for c in f.metadata.channels]
        ems = [c.channel.emissionLambdaNm for c in f.metadata.channels]

    # Desmin is the ~488/505 nm channel. Prefer emission, fall back to name,
    # and only then to the historical index -- never assume the index.
    idx = None
    for i, e in enumerate(ems):
        if e is not None and 495 <= float(e) <= 530:
            idx = i
            break
    if idx is None:
        for i, n in enumerate(names):
            if n and ("488" in str(n) or "AF488" in str(n).upper()):
                idx = i
                break
    if idx is None:
        idx = 1
    meta = {"channels": [str(n) for n in names], "emission": ems,
            "desmin_channel": idx, "pixel_um": px, "shape": list(arr.shape)}
    return arr[idx].astype(np.float32), px, meta


def process_well(code: str, roi_path: Path, nd2_path: Path, out_dir: Path,
                 *, width_px: float, mode: str, reviewer: str | None) -> dict:
    import tifffile
    from PIL import Image

    from .raster import compose_labels
    sys.path.insert(0, str(Path.cwd() / "PrecisionMyotube"))
    from precision_myotube.geometry import measure_mask  # noqa: E402

    rois = read_roi_set(roi_path)
    info = summarise(rois)
    field, px_um, meta = read_desmin(nd2_path)
    traces = rois_to_traces(rois, width_px=width_px, mode=mode,
                            reviewer=reviewer, source=str(roi_path))

    rr = np.array([p[0] for t in traces for p in t["points"]])
    cc = np.array([p[1] for t in traces for p in t["points"]])
    oob = int(((rr < 0) | (rr >= field.shape[0]) |
               (cc < 0) | (cc >= field.shape[1])).sum())

    labels, prov = compose_labels(np.zeros(field.shape, np.int32), traces, field)
    tifffile.imwrite(out_dir / f"{code}__labels.tif", labels.astype(np.uint16))

    rgb = colour_overlay(field, traces)
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    im.thumbnail((2400, 2400))
    im.save(out_dir / f"{code}__instances.png")

    from scipy import ndimage as ndi
    h, w = labels.shape
    rows, n_border = [], 0
    for lid, box in enumerate(ndi.find_objects(labels), start=1):
        if box is None:
            continue
        sub = labels[box] == lid
        if sub.sum() < 4:
            continue
        try:
            g = measure_mask(sub, px_um)
        except ValueError:
            continue
        # A fibre that leaves the field has a length that is a LOWER BOUND, not a
        # measurement, and must not become a complete training target -- the same
        # `border_truncated -> ignore` rule bootstrap_v1 already applies. Tracing
        # runs a few px past the edge on ~0.1% of points, so this is real.
        ys, xs = box
        touches = bool(ys.start <= 0 or xs.start <= 0
                       or ys.stop >= h or xs.stop >= w)
        n_border += touches
        rows.append({"instance": lid, "area_um2": round(g.area_um2, 2),
                     "length_um": round(g.length_um, 2),
                     "width_median_um": round(g.width_median_um, 3),
                     "width_area_over_length_um": round(
                         g.width_area_over_length_um, 3)
                     if np.isfinite(g.width_area_over_length_um) else None,
                     "aspect_ratio": round(g.length_um / g.width_median_um, 2)
                     if g.width_median_um > 0 else None,
                     "bbox_h": int(ys.stop - ys.start),
                     "bbox_w": int(xs.stop - xs.start),
                     "touches_border": touches})
    if rows:
        with open(out_dir / f"{code}__instances.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    L = np.array([r["length_um"] for r in rows]) if rows else np.array([0.0])
    W = np.array([r["width_median_um"] for r in rows]) if rows else np.array([0.0])
    ext = (np.array([max(r["bbox_h"], r["bbox_w"]) for r in rows]) if rows
           else np.array([0]))
    rec = {"well": code, "roi_file": roi_path.name, "nd2": nd2_path.name,
           "n_rois": info["n_rois"], "by_type": info["by_type"],
           "n_instances": len(rows), "out_of_bounds_points": oob,
           "n_touching_border": int(n_border),
           "max_bbox_extent_px": int(ext.max()),
           "acquisition": meta,
           "length_um": {"median": round(float(np.median(L)), 1),
                         "p10": round(float(np.percentile(L, 10)), 1),
                         "p90": round(float(np.percentile(L, 90)), 1),
                         "max": round(float(L.max()), 1)},
           "width_median_um": {"median": round(float(np.median(W)), 2)},
           "labelled_fraction": round(float((labels > 0).mean()), 5)}
    print(f"  {code:<5} {info['n_rois']:>4} ROIs -> {len(rows):>4} instances  "
          f"len_med={rec['length_um']['median']:>6.1f}um  "
          f"wid_med={rec['width_median_um']['median']:>5.2f}um  "
          f"fg={100*rec['labelled_fraction']:.2f}%  "
          f"border={n_border:>3}  maxbbox={int(ext.max()):>5}px", flush=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plate-dir", required=True)
    ap.add_argument("--out", default=None,
                    help="default: model_labs/omnipose_lab/_runs/fiji_<plate>")
    ap.add_argument("--width-px", type=float, default=8.0)
    ap.add_argument("--mode", default="snap", choices=["snap", "ribbon"])
    ap.add_argument("--reviewer", default=None)
    ap.add_argument("--wells", nargs="+", default=None)
    args = ap.parse_args(argv)

    plate_dir = Path(args.plate_dir)
    if not plate_dir.is_dir():
        raise SystemExit(f"no plate at {plate_dir}")
    out_dir = Path(args.out) if args.out else Path(
        "model_labs/omnipose_lab/_runs") / f"fiji_{plate_dir.name.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"plate  : {plate_dir}")
    pairs = pair_plate(plate_dir)
    if args.wells:
        want = {well_code(w) or w for w in args.wells}
        pairs = [p for p in pairs if p[0] in want]
    print(f"paired : {len(pairs)} wells\nout    : {out_dir}\n")

    recs = []
    for code, roi, nd2 in pairs:
        try:
            recs.append(process_well(code, roi, nd2, out_dir,
                                     width_px=args.width_px, mode=args.mode,
                                     reviewer=args.reviewer))
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  !! {code}: {type(exc).__name__}: {exc}")
            recs.append({"well": code, "error": f"{type(exc).__name__}: {exc}"})

    ok = [r for r in recs if "error" not in r]
    tot = sum(r["n_instances"] for r in ok)
    print(f"\n{len(ok)}/{len(recs)} wells   {tot:,} instances total "
          f"(mean {tot/max(len(ok),1):.0f}/well)")

    # Pre-flight for the corpus build. `instance_tiles` RAISES on an instance
    # that will not fit its tile ceiling, so finding it here -- with the number
    # needed -- is the difference between a one-line config change and a failed
    # build several minutes in.
    border = sum(r["n_touching_border"] for r in ok)
    print(f"{border:,} instances touch the field border "
          f"({100*border/max(tot,1):.1f}%) -- these are length LOWER BOUNDS and "
          f"must be ignored, never trained as complete targets")
    if ok:
        try:
            sys.path.insert(0, str(Path.cwd() / "model_labs"))
            from omnipose_lab.data import MARGIN_PX, TILE_PX
        except Exception:
            TILE_PX, MARGIN_PX = 1792, 96
        worst = max(r["max_bbox_extent_px"] for r in ok)
        n_over = sum(1 for r in ok if r["max_bbox_extent_px"] > TILE_PX)
        print(f"largest bbox extent {worst} px against TILE_PX={TILE_PX}"
              + (f"  -- OK" if worst <= TILE_PX else
                 f"\n  !! {n_over} well(s) exceed the ceiling; raise TILE_PX to "
                 f"{int(np.ceil((worst + 2*MARGIN_PX)/64)*64)} before building"))

    (out_dir / "plate_import.json").write_text(json.dumps(
        {"plate_dir": str(plate_dir), "width_px": args.width_px,
         "mode": args.mode, "reviewer": args.reviewer,
         "measurement": "precision_myotube.geometry.measure_mask",
         "n_wells": len(ok), "n_instances_total": tot,
         "per_well": recs}, indent=2), encoding="utf-8")
    print(f"-> {out_dir}/plate_import.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
