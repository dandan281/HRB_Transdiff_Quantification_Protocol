"""Watch the real training step: what reaches the loss, and does raw loss fall?

Two questions, one run, no guessing.

**What does the loss actually receive?** `omnipose.core.loss` documents a
seven-channel label stack for dim=2 -- masks, thresholded mask, boundary, smooth
distance, weights, then the flow components last. Calling `labels_to_flows`
directly returns five. If the training path also passes five then `dist =
lbl[:,3]` is really a flow component and `boundary = lbl[:,2]` is another, and
the target is unlearnable. If it passes seven, the target is fine and that whole
line of suspicion dies. Training computes flows inside the step ("No precomputing
flows with Omnipose"), so the only honest way to find out is to look at the
tensor as it arrives.

**Is the raw loss falling?** The logged number is not the loss. Every term is
put through `scale_to_tenths`, which multiplies by a power of ten that maps it
into [0.1, 1) -- the mantissa. Nine terms so normalised sum to something in
[0.9, 9) no matter how training is going, which is why 10.40 -> 10.15 over 400
steps meant nothing either way. `loss()` returns `raw_loss` as its second value
and nothing logs it. This does.

Works by wrapping `omnipose.core.loss` for the duration of a short `model.train()`
on a few tiles, then restoring it. Nothing in the package is modified on disk.

    python model_labs/omnipose_lab/trace_loss.py --epochs 40 --lr 0.01
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "model_labs",
           ROOT / "model_labs" / "omnipose_lab"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv=None) -> int:
    import torch
    from cellpose_omni import models
    import omnipose.core as ocore

    from omnipose_lab.train_fold import DEFAULTS, build_dense_fold

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus",
                    default="PrecisionMyotube/annotation_work/plate32_dense_v1")
    ap.add_argument("--held-out", default="B02")
    ap.add_argument("--window-px", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--tyx", type=int, default=DEFAULTS["tyx"])
    ap.add_argument("--init-model", default=DEFAULTS["init_model"])
    ap.add_argument("--no-links", action="store_true")
    ap.add_argument("--no-rescale", action="store_true",
                    help="neutralise scale_to_tenths, so each loss term keeps "
                         "its own magnitude in the gradient")
    a = ap.parse_args(argv)

    print("----- cellpose_omni CellposeModel.loss_fn " + "-" * 24)
    try:
        print(inspect.getsource(models.CellposeModel.loss_fn))
    except (OSError, TypeError) as exc:
        print(f"  (unavailable: {exc})")
    print("-" * 66 + "\n")

    fold = build_dense_fold(Path(a.corpus), a.held_out,
                            window_px=a.window_px, overlap=a.overlap, seed=0)
    k = min(a.tiles, len(fold["images"]))
    imgs = [np.asarray(t, dtype=np.float32) for t in fold["images"][:k]]
    labs = [np.asarray(t, dtype=np.int32) for t in fold["labels"][:k]]
    links = [None] * k if a.no_links else list(fold["links"][:k])

    init = None if a.init_model == "scratch" else a.init_model
    model = models.CellposeModel(
        gpu=torch.cuda.is_available(), omni=True, dim=2, nchan=1,
        nclasses=DEFAULTS["nclasses"], diam_mean=0.0,
        **({"model_type": init} if init else {"pretrained_model": False}))
    print(f"arch: nchan={model.nchan} out={model.nclasses}  dim={model.dim}")
    print(f"tiles {k}  epochs {a.epochs}  lr {a.lr}  "
          f"links {'OFF' if a.no_links else 'ON'}\n")

    # ---- wrap the loss so the live label stack and raw loss are visible ----
    original = ocore.loss
    original_scale = ocore.scale_to_tenths
    seen: list[tuple[float, float]] = []
    described: list[bool] = []

    # `scale_to_tenths` is applied once per loss term, in source order, so
    # wrapping it both records every term and gives us the switch to turn the
    # dynamic rescaling off.
    TERMS = ["flow_mse", "SSL", "bd_loss", "norm_loss", "dist_loss",
             "lossA", "lossE", "lossB", "lossDC"]
    per_step: list[list[float]] = []

    def traced_scale(x, max_gain=1e12):
        try:
            per_step[-1].append(float(x.detach()))
        except (AttributeError, TypeError, RuntimeError):
            per_step[-1].append(float("nan"))
        return x if a.no_rescale else original_scale(x, max_gain=max_gain)

    def traced(self, lbl, y, ext_loss=0):
        if not described:
            described.append(True)
            print("LABEL STACK AS THE TRAINING STEP DELIVERS IT")
            print(f"  lbl {tuple(lbl.shape)}   y {tuple(y.shape)}")
            names = ["masks", "thresholded mask", "boundary field",
                     "smooth distance", "weights", "flow[0]", "flow[1]"]
            for c in range(lbl.shape[1]):
                ch = lbl[:, c].detach().float()
                nz = float((ch != 0).float().mean())
                print(f"   ch{c} {names[c] if c < len(names) else 'extra':<18}"
                      f"[{float(ch.min()):+8.3f}, {float(ch.max()):+8.3f}]  "
                      f"nonzero {100 * nz:6.2f}%")
            want = self.dim + 5
            print(f"\n  loss expects {want} channels for dim={self.dim}; "
                  f"got {lbl.shape[1]}   "
                  f"{'OK' if lbl.shape[1] == want else '<-- MISMATCH'}\n")
            print(f"{'step':>6}{'raw loss':>14}{'logged (scaled)':>18}")
            print("-" * 38)

        per_step.append([])
        out = original(self, lbl, y, ext_loss=ext_loss)
        if isinstance(out, tuple) and len(out) == 2:
            scaled, raw = out
            seen.append((float(raw), float(scaled)))
            n = len(seen)
            if n <= 5 or n % 10 == 0:
                print(f"{n:>6}{float(raw):>14.6f}{float(scaled):>18.6f}")
        return out

    ocore.loss = traced
    ocore.scale_to_tenths = traced_scale
    if a.no_rescale:
        print("scale_to_tenths NEUTRALISED: each term keeps its own magnitude\n")
    out_dir = Path(tempfile.mkdtemp(prefix="traceloss_"))
    try:
        model.train(imgs, labs, train_links=links,
                    channels=None, channel_axis=None, normalize=False,
                    save_path=str(out_dir), save_every=max(50, a.epochs),
                    n_epochs=a.epochs, learning_rate=a.lr,
                    weight_decay=DEFAULTS["weight_decay"],
                    batch_size=a.batch_size, SGD=True, rescale=False,
                    min_train_masks=1, tyx=(a.tyx, a.tyx), netstr="trace")
    finally:
        ocore.loss = original
        ocore.scale_to_tenths = original_scale

    # ---- which term is big, and which one is stuck? ------------------------
    full = [t for t in per_step if len(t) >= len(TERMS)]
    if full:
        first, last = full[0], full[-1]
        print("\nPER-TERM RAW VALUES (the nine summed into raw_loss)")
        print(f"{'term':<12}{'step 1':>14}{'last step':>14}{'change':>12}")
        print("-" * 52)
        for i, name in enumerate(TERMS):
            f, l = first[i], last[i]
            print(f"{name:<12}{f:>14.6f}{l:>14.6f}"
                  f"{(l - f) / abs(f) * 100 if f else float('nan'):>11.1f}%")
        extra = len(first) - len(TERMS)
        if extra > 0:
            print(f"({extra} further term(s) recorded: external losses)")

    if not seen:
        print("!! the loss was never called through omnipose.core.loss")
        return 1

    raw = np.array([s[0] for s in seen])
    head = float(raw[:max(len(raw) // 10, 1)].mean())
    tail = float(raw[-max(len(raw) // 10, 1):].mean())
    print("-" * 38)
    print(f"\n{len(raw)} steps   raw loss  first 10%: {head:.6f}   "
          f"last 10%: {tail:.6f}   ratio {tail / max(head, 1e-12):.3f}")

    if tail < 0.9 * head:
        print("\n=> THE RAW LOSS IS FALLING. Training does optimise; the logged")
        print("   number was always going to look flat. The failure is then")
        print("   downstream -- what the fitted target produces at inference --")
        print("   not in the fitting.")
    else:
        print("\n=> THE RAW LOSS IS NOT FALLING either. This is a real training")
        print("   failure, and the channel table above says whether the target")
        print("   is the reason.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
