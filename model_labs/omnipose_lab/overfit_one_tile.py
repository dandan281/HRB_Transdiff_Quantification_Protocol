"""Can the net fit a handful of tiles it is shown every single step?

This is the decisive test once the data has been cleared. A network that cannot
memorise four images has a broken training path -- wrong loss, wrong target
plumbing, weights that never load, an optimiser that never steps. A network that
CAN memorise them has a working training path, and the 300-epoch failure is about
scale, regularisation or schedule instead. The two conclusions point at opposite
halves of the code, so guessing between them is what this run replaces.

It also reports what the model predicts BEFORE any training. The candidate is
initialised from ``bact_phase_affinity`` via ``model_type=``, and Cellpose loads
state dicts with ``strict=False`` -- a key mismatch is silent. If the untrained
init already emits a constant field on our images, the fine-tune never had
anything to build on and every downstream symptom follows from that alone.

Small and cheap on purpose: a few tiles, a few hundred steps, minutes on one GPU.

    python model_labs/omnipose_lab/overfit_one_tile.py --epochs 400 --lr 0.01
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "model_labs",
           ROOT / "model_labs" / "omnipose_lab"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EVAL_KW = dict(net_avg=False, tile=False, mask_threshold=0.0, flow_threshold=0.0,
               min_size=15, cluster=False, resample=True, compute_masks=True,
               verbose=False, rescale=None, diameter=None)


def report(tag, model, imgs, labs):
    """Predicted distance field and instance count on the given tiles."""
    for i, (img, lab) in enumerate(zip(imgs, labs)):
        out = model.eval(img, channels=None, channel_axis=None,
                         normalize=False, omni=True, **EVAL_KW)
        masks, flows = out[0], out[1]
        d = np.asarray(flows[2], dtype=np.float32)
        n = int(np.asarray(masks).max())
        print(f"  {tag:<9} tile {i}: n_pred={n:<5} gt={int(lab.max()):<5} "
              f"dist [{d.min():+.3f}, {d.max():+.3f}] "
              f"spread={d.max() - d.min():.3f} >0: {100 * (d > 0).mean():5.2f}% "
              f"(gt fg {100 * (lab > 0).mean():5.2f}%)")


def main(argv=None) -> int:
    import torch
    from cellpose_omni import models

    from omnipose_lab.train_fold import DEFAULTS, build_dense_fold

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus",
                    default="PrecisionMyotube/annotation_work/plate32_dense_v1")
    ap.add_argument("--held-out", default="B02")
    ap.add_argument("--window-px", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--tyx", type=int, default=DEFAULTS["tyx"])
    ap.add_argument("--init-model", default=DEFAULTS["init_model"],
                    help="'scratch' for random weights")
    ap.add_argument("--no-links", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    fold = build_dense_fold(Path(a.corpus), a.held_out,
                            window_px=a.window_px, overlap=a.overlap, seed=0)
    k = min(a.tiles, len(fold["images"]))
    imgs = [np.asarray(t, dtype=np.float32) for t in fold["images"][:k]]
    labs = [np.asarray(t, dtype=np.int32) for t in fold["labels"][:k]]
    links = [None] * k if a.no_links else list(fold["links"][:k])

    init = None if a.init_model == "scratch" else a.init_model
    print(f"tiles {k} of {len(fold['images'])}   epochs {a.epochs}   lr {a.lr}   "
          f"init {init or 'scratch'}   links {'OFF' if a.no_links else 'ON'}")
    print(f"gpu {torch.cuda.is_available()}\n")

    model = models.CellposeModel(
        gpu=torch.cuda.is_available(), omni=True, dim=2, nchan=1,
        nclasses=DEFAULTS["nclasses"], diam_mean=0.0,
        **({"model_type": init} if init else {"pretrained_model": False}))
    print(f"arch: nchan={model.nchan} out={model.nclasses}\n")

    print("BEFORE training (the initialisation alone):")
    report("init", model, imgs, labs)

    out = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="overfit_"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"\ntraining on the SAME {k} tiles every step -> {out}\n")
    model.train(imgs, labs, train_links=links,
                channels=None, channel_axis=None, normalize=False,
                save_path=str(out), save_every=max(50, a.epochs),
                n_epochs=a.epochs, learning_rate=a.lr,
                weight_decay=DEFAULTS["weight_decay"], batch_size=a.batch_size,
                SGD=True, rescale=False, min_train_masks=1,
                tyx=(a.tyx, a.tyx), netstr="overfit")

    print("\nAFTER training on those same tiles:")
    report("overfit", model, imgs, labs)

    print("\nreading this:")
    print("  init already flat            -> the pretrained weights are not")
    print("                                  loading; fix the init, not the lr")
    print("  still flat after overfitting -> the training path is broken (loss,")
    print("                                  target plumbing, or no gradient)")
    print("  fits these but not the fold  -> the path works; the full run is a")
    print("                                  schedule/capacity problem")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
