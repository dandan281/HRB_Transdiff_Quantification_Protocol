"""Build a training corpus from operator-traced ROIs alone.

Produces the same on-disk layout `omnipose_lab.data` already consumes
(`<well>/image_fiber.tif`, `labels.tif`, `ignore.tif` + a manifest), so nothing
downstream needs to know where the annotation came from.

Three pixel classes, and the ignore rules are the whole point of this file:

``target``   a traced fibre, not touching the field border
``ignore``   contributes NO gradient in either direction --
             * pixels where two traces overlap (a flat raster cannot hold two
               identities, and handing the crossing to whichever fibre was drawn
               first CUTS the other one, teaching a false end at every crossing);
             * instances touching the field border (their length is a lower
               bound, not a measurement);
             * fibre-like territory carrying no trace, when a territory mask is
               supplied.
``background`` everything else.

Omnipose regresses a distance+flow field where "background" means "distance 0,
flow 0". Every one of those ignore classes would otherwise assert that a visible
myotube is not a myotube.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from .fiji_roi import read_roi_set, rois_to_traces
from .import_plate import pair_plate, read_desmin, well_code
from .raster import rasterize_trace_local


def compose_with_ignore(field: np.ndarray, traces: list[dict], *,
                        halo_px: float = 6.0, territory: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Traces -> (labels, ignore, stats). Overlaps become ignore, not a winner."""
    from scipy import ndimage as ndi

    h, w = field.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cover = np.zeros((h, w), dtype=np.uint8)      # how many traces claim a pixel
    masks: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []

    for t in traces:
        sub, _info, box = rasterize_trace_local(t, field)
        if sub.size == 0 or sub.sum() < 4:
            masks.append((None, None))
            continue
        r0, c0, r1, c1 = box
        masks.append((sub, box))
        np.add(cover[r0:r1, c0:c1], sub.astype(np.uint8),
               out=cover[r0:r1, c0:c1], casting="unsafe")

    overlap = cover >= 2
    next_id, kept, dropped_small = 1, [], 0
    for t, (sub, box) in zip(traces, masks):
        if sub is None:
            continue
        r0, c0, r1, c1 = box
        own = sub & ~overlap[r0:r1, c0:c1]        # crossing pixels go to nobody
        if own.sum() < 4:
            dropped_small += 1
            continue
        view = labels[r0:r1, c0:c1]
        view[own & (view == 0)] = next_id
        kept.append({"label": next_id, "trace_id": t.get("trace_id"),
                     "roi_name": t.get("roi_name")})
        next_id += 1

    # Border-touching instances: length is a lower bound, so they are evidence
    # of a fibre but not a measurable target.
    border = np.zeros((h, w), dtype=bool)
    n_border = 0
    for lid, box in enumerate(ndi.find_objects(labels), start=1):
        if box is None:
            continue
        ys, xs = box
        if ys.start <= 0 or xs.start <= 0 or ys.stop >= h or xs.stop >= w:
            border |= labels == lid
            n_border += 1
    labels[border] = 0

    ignore = overlap | border
    n_unlabelled = 0
    if territory is not None:
        target = labels > 0
        dist = ndi.distance_transform_edt(~target) if target.any() else None
        unl = (territory.astype(bool) & (dist > halo_px)) if dist is not None \
            else territory.astype(bool)
        n_unlabelled = int(unl.sum())
        ignore |= unl
    ignore &= labels == 0                          # a target is never ignored

    # Renumber to 1..N so ids are contiguous after the border drop.
    remaining = [v for v in np.unique(labels) if v]
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for new, old in enumerate(remaining, start=1):
        remap[old] = new
    labels = remap[labels]

    stats = {
        "n_traces": len(traces),
        "n_instances": int(labels.max()),
        "n_dropped_border": n_border,
        "n_dropped_tiny_after_overlap": dropped_small,
        "overlap_px": int(overlap.sum()),
        "border_px": int(border.sum()),
        "unlabelled_fibre_px": n_unlabelled,
        "target_fraction": round(float((labels > 0).mean()), 5),
        "ignore_fraction": round(float(ignore.mean()), 5),
    }
    return labels, ignore, stats


def main(argv=None) -> int:
    import tifffile

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plate-dir", required=True)
    ap.add_argument("--out", required=True,
                    help="new corpus directory (must not exist)")
    ap.add_argument("--territory-cache", default=None,
                    help="dir of <well>.territory.npy; without it, unlabelled "
                         "fibre-like pixels are asserted as background")
    ap.add_argument("--width-px", type=float, default=8.0)
    ap.add_argument("--mode", default="snap", choices=["snap", "ribbon"])
    ap.add_argument("--halo-px", type=float, default=6.0)
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--wells", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    plate_dir = Path(args.plate_dir)
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force")
    out.mkdir(parents=True, exist_ok=True)

    pairs = pair_plate(plate_dir)
    if args.wells:
        want = {well_code(w) or w for w in args.wells}
        pairs = [p for p in pairs if p[0] in want]
    print(f"plate  : {plate_dir}\nout    : {out}\nwells  : {len(pairs)}")
    if args.territory_cache:
        print(f"terr   : {args.territory_cache}")
    else:
        print("terr   : NONE -- unlabelled fibre-like pixels will be BACKGROUND")
    print()

    hdr = (f"{'well':<6}{'inst':>6}{'border':>8}{'target%':>9}{'ignore%':>9}"
           f"{'overlap%':>10}")
    print(hdr); print("-" * len(hdr))
    per_well, started = {}, time.time()
    for code, roi, nd2 in pairs:
        rois = read_roi_set(roi)
        field, px_um, meta = read_desmin(nd2)
        traces = rois_to_traces(rois, width_px=args.width_px, mode=args.mode,
                                reviewer=args.reviewer, source=str(roi))
        terr = None
        if args.territory_cache:
            tp = Path(args.territory_cache) / f"{code}.territory.npy"
            if tp.exists():
                terr = np.load(tp).astype(bool)
        labels, ignore, stats = compose_with_ignore(
            field, traces, halo_px=args.halo_px, territory=terr)

        wd = out / code
        wd.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(wd / "image_fiber.tif", field.astype(np.uint16))
        tifffile.imwrite(wd / "labels.tif", labels.astype(np.uint16))
        tifffile.imwrite(wd / "ignore.tif", ignore.astype(np.uint8))
        stats.update({"well": code, "roi_file": roi.name, "nd2": nd2.name,
                      "pixel_um": px_um, "acquisition": meta,
                      "territory_used": terr is not None})
        (wd / "well_manifest.json").write_text(json.dumps(stats, indent=2),
                                               encoding="utf-8")
        per_well[code] = stats
        print(f"{code:<6}{stats['n_instances']:>6}{stats['n_dropped_border']:>8}"
              f"{100*stats['target_fraction']:>8.2f}%"
              f"{100*stats['ignore_fraction']:>8.2f}%"
              f"{100*stats['overlap_px']/field.size:>9.2f}%", flush=True)

    tot = sum(v["n_instances"] for v in per_well.values())
    print("-" * len(hdr))
    print(f"{'TOTAL':<6}{tot:>6}")

    manifest = {
        "corpus": out.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_plate": str(plate_dir),
        "reviewer": args.reviewer,
        "annotation_method": ("operator-traced centrelines in Fiji, rasterised "
                              "to ribbons and snapped to local signal"),
        "ignore_policy": ("trace overlap -> ignore (a crossing handed to one "
                          "fibre cuts the other); border-touching instance -> "
                          "ignore (length is a lower bound); unlabelled "
                          "fibre-like territory -> ignore when a territory mask "
                          "is supplied, else BACKGROUND"),
        "territory_cache": args.territory_cache,
        "width_px": args.width_px, "mode": args.mode, "halo_px": args.halo_px,
        "n_wells": len(per_well), "n_instances_total": tot,
        "evidence_class": "single_operator_dense_direct_trace",
        "limitations": [
            "single operator, single session, one convention",
            "NOT proposal-conditioned -- drawn directly, so not exchangeable "
            "with bootstrap_v1 instances for provenance",
            "mask edges come from the snap threshold, not a drawn boundary; "
            "length is operator-placed, width is fitted",
            "instances split by a crossing remain split -- Omnipose `links` "
            "would declare the pieces one object and is not yet wired",
        ],
        "per_well": per_well,
        "seconds": round(time.time() - started, 1),
    }
    (out / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2),
                                              encoding="utf-8")
    print(f"\n-> {out}/corpus_manifest.json")
    if not args.territory_cache:
        print("\n!! no territory mask: unlabelled fibre-like pixels are being "
              "taught as background.\n   Supply --territory-cache to ignore "
              "them instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
