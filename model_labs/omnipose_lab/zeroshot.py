"""ZERO-SHOT Omnipose on myotube fields -- no training, no fine-tuning.

Answers the question the fine-tuning run cannot answer about itself: what do the
pretrained bacterial weights do on this data BEFORE any adaptation? Without this
reference, a fine-tuned score cannot be attributed -- see the plan's step 1.

Three ways this differs from `infer_fold.py`, all deliberate:

1. **The checkpoint defines its own architecture.** `infer_fold` forces
   `nchan=1, nclasses=2` because that is what we trained. Here we let
   `CellposeModel(model_type=...)` resolve whatever the pretrained weights want
   and adapt the INPUT to it. That is why zero-shot can evaluate `bact_phase_omni`
   -- which forces `nchan=2` plus a boundary head and is therefore unusable for
   fine-tuning on 1-channel tiles -- while fine-tuning cannot. A 2-channel model
   simply gets a blank second channel, the standard Cellpose convention, and an
   unused boundary output is harmless at inference.

2. **Polarity is an explicit, declared axis.** Bacterial *phase-contrast* models
   were trained on dark objects against a light ground. Desmin fluorescence is the
   opposite. Feeding the wrong polarity makes Omnipose fail for a trivial reason
   and would look like the architecture being wrong. `--polarity both` runs each.

3. **No ground truth is read and nothing is scored here.** This writes masks,
   overlays and per-instance geometry only. Benchmarking against the sealed eval
   GT is T03's lane, deliberately not done in this script.

Report EVERY cell of the {model x polarity x diameter} grid you run. Reporting
only the best cell is tuning on the evaluation set.

Length and width come from `precision_myotube.geometry.measure_mask` -- the same
canonical measurement the benchmark and all scientific reporting use, so these
numbers are directly comparable to the classical candidate's.

Usage (from the repo root, in the pm-omnipose env)::

    python model_labs/omnipose_lab/zeroshot.py --list-models
    python model_labs/omnipose_lab/zeroshot.py --wells 23_B02_ctrl --crop 1024
    python model_labs/omnipose_lab/zeroshot.py --models bact_phase_affinity \\
        --polarity both --wells 23_B02_ctrl 32_C08_br223_igf1r
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BOOTSTRAP = ROOT / "PrecisionMyotube/annotation_work/bootstrap_v1"
PIXEL_UM = 0.650017          # PLATE_23 nd2 metadata
DEFAULT_MODELS = ["bact_phase_affinity"]

# Matches the fine-tuning inference path so the two are commensurable.
EVAL_KW = dict(net_avg=False, tile=True, bsize=224, tile_overlap=0.1,
               mask_threshold=0.0, flow_threshold=0.0, min_size=15,
               cluster=False, resample=True, compute_masks=True, verbose=False)


def list_models() -> list[str]:
    """Every pretrained model name this cellpose_omni ships, so the grid is
    chosen from what exists rather than from what we assume exists."""
    from cellpose_omni import models as m
    names = []
    for attr in ("MODEL_NAMES", "C1_MODELS", "C2_MODEL_NAMES", "BD_MODEL_NAMES"):
        names += list(getattr(m, attr, []) or [])
    return sorted(dict.fromkeys(names))


def normalize_field(img: np.ndarray, lo_pct=1.0, hi_pct=99.5) -> np.ndarray:
    lo, hi = np.percentile(img, lo_pct), np.percentile(img, hi_pct)
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def prepare_input(norm: np.ndarray, nchan: int, polarity: str) -> np.ndarray:
    """Adapt a 1-channel normalised field to whatever the checkpoint wants."""
    img = 1.0 - norm if polarity == "invert" else norm
    if nchan <= 1:
        return img
    stack = np.zeros((nchan, *img.shape), dtype=np.float32)
    stack[0] = img                       # blank remaining channels
    return stack


def measure_instances(labels: np.ndarray, pixel_um: float) -> list[dict]:
    """Per-instance length and width via the canonical measurement.

    Cropped to each bounding box for speed -- a full-field geodesic per instance
    is needlessly expensive -- but `touches_border` is recomputed against the
    FULL field, because a crop edge is not a field edge.
    """
    from scipy import ndimage as ndi
    from precision_myotube.geometry import measure_mask

    h, w = labels.shape
    rows = []
    for value, box in enumerate(ndi.find_objects(labels), start=1):
        if box is None:
            continue
        sub = labels[box] == value
        if sub.sum() < 4:
            continue
        try:
            g = measure_mask(sub, pixel_um)
        except ValueError:
            continue
        ys, xs = box
        touches = (ys.start == 0 or xs.start == 0
                   or ys.stop >= h or xs.stop >= w)
        rows.append({
            "instance_id": value,
            "area_um2": round(g.area_um2, 2),
            "length_um": round(g.length_um, 2),
            "width_median_um": round(g.width_median_um, 3),
            "width_area_over_length_um": round(g.width_area_over_length_um, 3)
            if np.isfinite(g.width_area_over_length_um) else None,
            "width_p10_um": round(g.width_p10_um, 3),
            "width_p90_um": round(g.width_p90_um, 3),
            "width_cv": round(g.width_cv, 3) if np.isfinite(g.width_cv) else None,
            "aspect_ratio": round(g.length_um / g.width_median_um, 2)
            if g.width_median_um > 0 else None,
            "branch_count": g.branch_count,
            "components": g.components,
            "touches_border": bool(touches),
        })
    return rows


def save_overlay(norm: np.ndarray, labels: np.ndarray, path: Path) -> None:
    from PIL import Image
    from skimage.color import label2rgb
    over = label2rgb(labels, image=norm, bg_label=0, alpha=0.55, image_alpha=1,
                     bg_color=(0, 0, 0))
    im = Image.fromarray((np.clip(over, 0, 1) * 255).astype(np.uint8))
    im.thumbnail((2000, 2000))
    im.save(path)


def run_cell(model, model_name: str, well: str, norm: np.ndarray, polarity: str,
             diameter, out_dir: Path, pixel_um: float) -> dict:
    import torch

    nchan = int(getattr(model, "nchan", 1) or 1)
    x = prepare_input(norm, nchan, polarity)
    channel_axis = 0 if x.ndim == 3 else None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = model.eval(x, channels=None, channel_axis=channel_axis,
                     normalize=False, omni=True, rescale=None,
                     diameter=diameter, **EVAL_KW)
    secs = time.time() - t0
    labels = np.asarray(out[0], dtype=np.int32)
    peak = (float(torch.cuda.max_memory_allocated()) / 1e9
            if torch.cuda.is_available() else 0.0)

    tag = f"{well}__{model_name}__{polarity}" + (
        f"__d{diameter:g}" if diameter else "")
    cell_dir = out_dir / tag
    cell_dir.mkdir(parents=True, exist_ok=True)
    np.save(cell_dir / "labels.npy", labels)
    save_overlay(norm, labels, cell_dir / "overlay.png")

    rows = measure_instances(labels, pixel_um)
    if rows:
        with open(cell_dir / "instances.csv", "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wtr.writeheader()
            wtr.writerows(rows)

    L = np.array([r["length_um"] for r in rows]) if rows else np.array([])
    W = np.array([r["width_median_um"] for r in rows]) if rows else np.array([])
    rec = {
        "well": well, "model": model_name, "polarity": polarity,
        "diameter": diameter, "nchan_used": nchan,
        "n_instances": len(rows),
        "inference_seconds": round(secs, 1),
        "peak_gpu_gb": round(peak, 2),
        "length_um": {
            "median": round(float(np.median(L)), 2) if L.size else None,
            "p90": round(float(np.percentile(L, 90)), 2) if L.size else None,
            "max": round(float(L.max()), 2) if L.size else None,
            "n_over_100um": int((L >= 100).sum()) if L.size else 0},
        "width_median_um": {
            "median": round(float(np.median(W)), 2) if W.size else None,
            "p90": round(float(np.percentile(W, 90)), 2) if W.size else None},
        "outputs": {"labels": str((cell_dir / "labels.npy").relative_to(ROOT)),
                    "overlay": str((cell_dir / "overlay.png").relative_to(ROOT)),
                    "instances": str((cell_dir / "instances.csv").relative_to(ROOT))
                    if rows else None},
    }
    print(f"  {tag:<58} n={len(rows):>5}  "
          f"len_med={rec['length_um']['median']}  "
          f"wid_med={rec['width_median_um']['median']}  "
          f"{secs:5.1f}s  {peak:.1f}GB", flush=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list-models", action="store_true",
                    help="print every pretrained model this install ships, then exit")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--wells", nargs="+", default=None,
                    help="default: every well in bootstrap_v1")
    ap.add_argument("--polarity", choices=["as_is", "invert", "both"],
                    default="both",
                    help="fluorescence is bright-on-dark; phase-contrast models "
                         "were trained dark-on-light. 'both' runs each and "
                         "reports both -- do not report only the better one")
    ap.add_argument("--diameters", nargs="*", type=float, default=[None],
                    help="pass one or more diameters to let Omnipose rescale; "
                         "default is no rescaling, matching the fine-tune config")
    ap.add_argument("--crop", type=int, default=0,
                    help="centre crop of N px for a fast first look (0 = full field)")
    ap.add_argument("--pixel-um", type=float, default=PIXEL_UM)
    ap.add_argument("--out", default="model_labs/omnipose_lab/_runs/zeroshot")
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args(argv)

    if args.list_models:
        for n in list_models():
            print(n)
        return 0

    import tifffile
    import torch
    from cellpose_omni import models

    wells = args.wells or sorted(p.name for p in BOOTSTRAP.iterdir()
                                 if p.is_dir())
    pols = ["as_is", "invert"] if args.polarity == "both" else [args.polarity]
    diams = args.diameters if args.diameters else [None]
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.is_available() and not args.no_gpu

    print(f"ZERO-SHOT Omnipose (no fine-tuning)  gpu={gpu}")
    print(f"  wells={wells}")
    print(f"  models={args.models}  polarity={pols}  diameters={diams}")
    print(f"  crop={args.crop or 'full field'}  pixel={args.pixel_um} um\n")

    fields = {}
    for w in wells:
        img = tifffile.imread(BOOTSTRAP / w / "image_fiber.tif")
        if args.crop:
            c = args.crop
            y = (img.shape[0] - c) // 2
            x = (img.shape[1] - c) // 2
            img = img[y:y + c, x:x + c]
        fields[w] = normalize_field(img)

    records = []
    for name in args.models:
        print(f"=== {name} ===", flush=True)
        try:
            model = models.CellposeModel(gpu=gpu, omni=True, model_type=name)
        except Exception as exc:
            print(f"  !! could not load {name}: {type(exc).__name__}: {exc}")
            records.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        print(f"  resolved nchan={getattr(model, 'nchan', '?')} "
              f"nclasses={getattr(model, 'nclasses', '?')} "
              f"diam_mean={getattr(model, 'diam_mean', '?')}")
        for w in wells:
            for pol in pols:
                for d in diams:
                    try:
                        records.append(run_cell(model, name, w, fields[w], pol,
                                                d, out_dir, args.pixel_um))
                    except Exception as exc:
                        print(f"  !! {w}/{pol}/{d}: {type(exc).__name__}: {exc}")
                        records.append({"well": w, "model": name,
                                        "polarity": pol, "diameter": d,
                                        "error": f"{type(exc).__name__}: {exc}"})

    manifest = {
        "run": "zeroshot_omnipose",
        "purpose": ("reference point for what pretrained weights do BEFORE "
                    "fine-tuning; the fine-tuned run cannot attribute its own "
                    "result without this"),
        "no_training": True, "no_ground_truth_read": True,
        "scoring_note": ("not scored here -- benchmarking against the sealed "
                         "eval GT is T03's lane"),
        "reporting_rule": ("report EVERY grid cell; reporting only the best is "
                           "tuning on the evaluation set"),
        "pixel_um": args.pixel_um, "crop": args.crop,
        "measurement": ("precision_myotube.geometry.measure_mask -- longest "
                        "geodesic length, EDT-based widths; identical to the "
                        "benchmark and to classical-candidate reporting"),
        "eval_kwargs": {k: v for k, v in EVAL_KW.items()},
        "gpu": gpu, "cells": records,
    }
    (out_dir / "zeroshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    ok = [r for r in records if "error" not in r]
    print(f"\n{len(ok)}/{len(records)} cells succeeded -> "
          f"{(out_dir / 'zeroshot_manifest.json').relative_to(ROOT)}")
    if ok:
        print("\ninspect the overlays before reading any number:")
        for r in ok[:6]:
            print(f"  {r['outputs']['overlay']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
