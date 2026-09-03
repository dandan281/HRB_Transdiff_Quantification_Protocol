"""Quantify plate 32 with the Omnipose candidate, on the tracer's yardstick.

Companion to `model_labs/tracer_lab/quantify_plate.py`: same ten wells, same
operator annotation, same two questions -- how many fibres, and how much total
fibre length, against the human.

Two conventions are deliberate so the comparison is not rigged:

* **Length comes from `precision_myotube.geometry.measure_mask`**, the
  project's existing convention (skeletonise the instance, take the longest
  geodesic). It is what the classical pipeline and the sealed benchmark use,
  so an Omnipose mask is measured exactly as the project has always measured
  masks -- not by a rule invented for this comparison.
* **Inference runs on the unpainted field**, as `infer_fold.py` does: the
  ignore/paint policy applies to training images only, and a well is scored as
  the microscope produced it.

The tracer produces polylines and Omnipose produces masks, so the per-fibre
matching metrics are not directly comparable between the two candidates.
Fibre count and total length ARE, because both reduce to "how many objects,
how long is each" in micrometres, and that is what the table reports.

Needs a fine-tuned checkpoint; the stock pretrained weights are not a
candidate. See the operator block in the session notes for retrieving it.

    python model_labs/omnipose_lab/quantify_plate_omnipose.py \\
        --checkpoint <path-to-checkpoint>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "annotation_tools",
           ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CORPUS = ROOT / "PrecisionMyotube/annotation_work/plate32_dense_v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    import torch
    import tifffile
    from cellpose_omni import models
    from scipy import ndimage
    from precision_myotube.geometry import measure_mask

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--min-um", type=float, default=50.0)
    ap.add_argument("--min-px", type=int, default=20,
                    help="instances smaller than this are debris, dropped "
                         "before measuring")
    ap.add_argument("--wells", default="")
    ap.add_argument("--held-out", default="B02")
    ap.add_argument("--out",
                    default="model_labs/omnipose_lab/_runs/plate32_omnipose.json")
    a = ap.parse_args(argv)

    ck = Path(a.checkpoint)
    if not ck.exists():
        print(f"checkpoint not found: {ck}", file=sys.stderr)
        return 1
    print(f"checkpoint {ck.name}  sha256 {_sha256(ck)[:12]}...")

    wells = ([w for w in a.wells.split(",") if w]
             or sorted(p.name for p in CORPUS.iterdir() if p.is_dir()))

    model = models.CellposeModel(
        gpu=torch.cuda.is_available(), omni=True, dim=2, nchan=1,
        nclasses=2, diam_mean=0.0, pretrained_model=str(ck))

    hdr = (f"{'well':<6}{'human n':>9}{'human mm':>10}"
           f"{'omni n':>9}{'omni mm':>10}{'n ratio':>9}{'mm ratio':>10}")
    print(f"\nlength convention: precision_myotube.geometry.measure_mask "
          f"(skeleton longest geodesic), min {a.min_um} um\n")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for well in wells:
        t0 = time.time()
        manifest = json.loads((CORPUS / well / "well_manifest.json").read_text())
        um = manifest["pixel_um"]
        image = tifffile.imread(CORPUS / well / "image_fiber.tif").astype(np.float32)
        from omnipose_lab.data import normalize_field
        norm, _ = normalize_field(image)

        # identical to eval_on_bootstrap.py's inference path, so numbers
        # are commensurable with the T03 Omnipose row. NOTE rescale=None:
        # rescale=False is read as 0.0 and resizes the image to nothing.
        masks = np.asarray(model.eval(
            norm, channels=None, channel_axis=None, normalize=False,
            omni=True, rescale=None, diameter=None, net_avg=False,
            tile=True, bsize=224, tile_overlap=0.1, mask_threshold=0.0,
            flow_threshold=0.0, min_size=15, cluster=False, resample=True,
            compute_masks=True, verbose=False)[0], dtype=np.int32)

        lens = []
        for lab in range(1, int(masks.max()) + 1):
            m = masks == lab
            n = int(m.sum())
            if n < a.min_px:
                continue
            sl = ndimage.find_objects(m.astype(np.uint8))[0]
            try:
                g = measure_mask(m[sl], um)
            except ValueError:
                continue
            if g.length_um >= a.min_um:
                lens.append(g.length_um)
        lens = np.array(lens)

        # the operator's own traces — SMOOTHED convention (2026-08-27 §7d:
        # raw arc on freehand lines inflates 10-15% with drawing jitter)
        from tracer_lab.centreline_targets import targets_from_roi_zip
        from tracer_lab.length_classes import class_shares, smooth_polyline
        zips = sorted((ROOT / "Q_PLATES/Q_Plates/PLATE_32")
                      .glob(f"*{well}*.zip"))
        gt = targets_from_roi_zip(zips[0], image.shape)
        h = np.array([float(np.linalg.norm(
            np.diff(smooth_polyline(t), axis=0), axis=1).sum()) * um
            for t in gt["traces"]])
        h = h[h >= a.min_um]

        rec = {"well": well, "held_out": well == a.held_out,
               "human_n": int(len(h)), "human_mm": float(h.sum() / 1000),
               "omni_n": int(len(lens)), "omni_mm": float(lens.sum() / 1000),
               "omni_median_um": float(np.median(lens)) if len(lens) else 0.0,
               "human_median_um": float(np.median(h)) if len(h) else 0.0,
               "human_length_classes": class_shares(h),
               "omni_length_classes": class_shares(lens),
               "raw_instances": int(masks.max())}
        rows.append(rec)
        print(f"{well:<6}{rec['human_n']:>9}{rec['human_mm']:>10.1f}"
              f"{rec['omni_n']:>9}{rec['omni_mm']:>10.1f}"
              f"{rec['omni_n'] / max(rec['human_n'], 1):>9.2f}"
              f"{rec['omni_mm'] / max(rec['human_mm'], 1e-9):>10.2f}"
              f"{'  (HELD OUT)' if rec['held_out'] else ''}"
              f"   [{time.time() - t0:.0f} s]", flush=True)

    hn = sum(r["human_n"] for r in rows)
    hm = sum(r["human_mm"] for r in rows)
    on = sum(r["omni_n"] for r in rows)
    om = sum(r["omni_mm"] for r in rows)
    print("-" * len(hdr))
    print(f"{'PLATE':<6}{hn:>9}{hm:>10.1f}{on:>9}{om:>10.1f}"
          f"{on / max(hn, 1):>9.2f}{om / max(hm, 1e-9):>10.2f}")
    if len(rows) > 2:
        hmv = np.array([r["human_mm"] for r in rows])
        omv = np.array([r["omni_mm"] for r in rows])
        print(f"\nper-well correlation with the operator: "
              f"length r = {np.corrcoef(hmv, omv)[0, 1]:+.3f}")

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": str(ck), "min_um": a.min_um,
                               "rows": rows}, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
