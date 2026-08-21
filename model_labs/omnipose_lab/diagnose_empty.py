"""Why did the trained model return zero masks?

"No cell pixels found" on every field is a pipeline failure, not a result. This
separates the three explanations without guessing:

1. **The model never learned.** Then it also produces nothing on a tile it was
   TRAINED on. That is the decisive test and it is first.
2. **Rescaling is degenerate.** We build with `diam_mean=0.0` and call
   `eval(rescale=None, diameter=None)`; Cellpose computes `rescale = diam_mean /
   diameter`, so a zero mean can yield 0 or NaN and resize the field to nothing.
   The zero-shot models had `diam_mean=30.0` and worked.
3. **The threshold is wrong for this checkpoint.** `mask_threshold=0.0` cuts the
   distance field at zero; if the trained distance output sits below zero
   everywhere, everything is background.

Reports the RAW network output range in every case, because that distinguishes
"predicted nothing" from "predicted something that was then thresholded away".

    python model_labs/omnipose_lab/diagnose_empty.py --checkpoint <path>
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BASE = dict(net_avg=False, tile=True, bsize=224, tile_overlap=0.1,
            min_size=15, cluster=False, resample=True, compute_masks=True,
            verbose=False)


def describe(tag, labels, flows):
    n = int(np.asarray(labels).max()) if labels is not None else 0
    # flows[2] is the distance field for omni models; flows[0] is the RGB flow
    # render. Guard because the tuple shape varies with options.
    dist = None
    try:
        d = flows[2]
        dist = np.asarray(d, dtype=np.float32)
    except Exception:
        pass
    if dist is not None and dist.size:
        print(f"  {tag:<34} n={n:<6} dist min={dist.min():+.3f} "
              f"max={dist.max():+.3f} mean={dist.mean():+.3f} "
              f">0: {100*(dist > 0).mean():.2f}%")
    else:
        print(f"  {tag:<34} n={n:<6} (no distance field returned)")
    return n


def main(argv=None) -> int:
    import tifffile
    import torch
    from cellpose_omni import models

    from omnipose_lab.data import normalize_field

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus",
                    default="PrecisionMyotube/annotation_work/plate32_dense_v1")
    ap.add_argument("--train-well", default="C02")
    ap.add_argument("--bootstrap",
                    default="PrecisionMyotube/annotation_work/bootstrap_v1")
    ap.add_argument("--test-well", default="23_B02_ctrl")
    ap.add_argument("--crop", type=int, default=1280)
    args = ap.parse_args(argv)

    gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=gpu, omni=True, dim=2, nchan=1,
                                 nclasses=2, diam_mean=0.0,
                                 pretrained_model=str(args.checkpoint))
    print(f"model diam_mean = {getattr(model, 'diam_mean', '?')}   "
          f"nchan={getattr(model, 'nchan', '?')} "
          f"nclasses={getattr(model, 'nclasses', '?')}\n")

    def field(path, crop):
        img = tifffile.imread(path)
        if crop and min(img.shape) > crop:
            y = (img.shape[0] - crop) // 2
            x = (img.shape[1] - crop) // 2
            img = img[y:y + crop, x:x + crop]
        n, _ = normalize_field(img)
        return n

    train_img = field(Path(args.corpus) / args.train_well / "image_fiber.tif",
                      args.crop)
    test_img = field(Path(args.bootstrap) / args.test_well / "image_fiber.tif",
                     args.crop)
    print(f"train tile {train_img.shape} range "
          f"[{train_img.min():.3f}, {train_img.max():.3f}]")
    print(f"test  tile {test_img.shape} range "
          f"[{test_img.min():.3f}, {test_img.max():.3f}]\n")

    # --- 1. TRAINING DATA. If this is empty, the model never learned. --------
    print("1. on data the model was TRAINED on (should be its best case):")
    for tag, kw in (("rescale=None (as used)", dict(rescale=None, diameter=None)),
                    ("rescale=1.0", dict(rescale=1.0, diameter=None)),
                    ("mask_threshold=-1", dict(rescale=1.0, diameter=None,
                                               mask_threshold=-1.0)),
                    ("mask_threshold=-5", dict(rescale=1.0, diameter=None,
                                               mask_threshold=-5.0))):
        k = {**BASE, "mask_threshold": 0.0, "flow_threshold": 0.0, **kw}
        try:
            out = model.eval(train_img, channels=None, channel_axis=None,
                             normalize=False, omni=True, **k)
            describe(tag, out[0], out[1])
        except Exception as exc:
            print(f"  {tag:<34} ERROR {type(exc).__name__}: {exc}")

    # --- 2. TEST DATA -------------------------------------------------------
    print("\n2. on the held-out PLATE_23 field:")
    for tag, kw in (("rescale=None (as used)", dict(rescale=None, diameter=None)),
                    ("rescale=1.0", dict(rescale=1.0, diameter=None)),
                    ("mask_threshold=-1", dict(rescale=1.0, diameter=None,
                                               mask_threshold=-1.0))):
        k = {**BASE, "mask_threshold": 0.0, "flow_threshold": 0.0, **kw}
        try:
            out = model.eval(test_img, channels=None, channel_axis=None,
                             normalize=False, omni=True, **k)
            describe(tag, out[0], out[1])
        except Exception as exc:
            print(f"  {tag:<34} ERROR {type(exc).__name__}: {exc}")

    print("\nreading this:")
    print("  all zero on TRAINING data      -> the model did not learn; the")
    print("                                    training target or loss is wrong")
    print("  works at rescale=1.0 only      -> diam_mean=0.0 broke the rescale")
    print("  works at a negative threshold  -> distance field learned but offset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
