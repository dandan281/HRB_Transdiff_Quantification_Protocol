"""What is the loss actually measuring, and can it ever reach zero?

The overfit test settled the argument. `bact_phase_affinity` predicts sensibly
BEFORE training (297 instances, 8.0% foreground against 7.3% true), and then 400
steps on four fixed tiles move the loss not at all: 10.40 at epoch 0, 10.15 at
epoch 185. Four images cannot defeat 6.6M parameters. A loss that will not fall
on a memorisation task is not a slow loss -- it is a loss whose minimum the
network cannot express.

That happens when the target and the output are in different conventions. The
check is direct: build the prediction FROM the target and score it. A correctly
plumbed loss returns ~0 for that; whatever it actually returns is the floor the
optimiser has been sitting on.

The convention is not documented, so this sweeps the plausible ones -- flow
scale (Omnipose trains flows at 5x), which channel carries distance, and whether
background distance is 0 or -dist_bg -- and reports the loss of each. The
combination that bottoms out names the convention. If NONE of them approaches
zero, no assembly of the target satisfies this loss and the mismatch is
structural: the head cannot emit what the loss demands.

Also dumps the loss source, since that is the ground truth for all of it.

    python model_labs/omnipose_lab/diagnose_loss.py --held-out B02
"""
from __future__ import annotations

import argparse
import inspect
import itertools
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "model_labs",
           ROOT / "model_labs" / "omnipose_lab"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv=None) -> int:
    import torch
    from cellpose_omni import models
    from omnipose import core as ocore

    from omnipose_lab.train_fold import DEFAULTS, build_dense_fold

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus",
                    default="PrecisionMyotube/annotation_work/plate32_dense_v1")
    ap.add_argument("--held-out", default="B02")
    ap.add_argument("--window-px", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--crop", type=int, default=384, help="matches tyx")
    ap.add_argument("--init-model", default=DEFAULTS["init_model"])
    ap.add_argument("--no-links", action="store_true")
    ap.add_argument("--no-source", action="store_true")
    a = ap.parse_args(argv)

    fold = build_dense_fold(Path(a.corpus), a.held_out,
                            window_px=a.window_px, overlap=a.overlap, seed=0)
    img = np.asarray(fold["images"][0], dtype=np.float32)
    lab = np.asarray(fold["labels"][0], dtype=np.int32)
    lk = None if a.no_links else fold["links"][0]

    # A crop with foreground in it, the size the trainer actually uses.
    ys, xs = np.nonzero(lab > 0)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    c = a.crop
    r0 = int(np.clip(cy - c // 2, 0, lab.shape[0] - c))
    c0 = int(np.clip(cx - c // 2, 0, lab.shape[1] - c))
    img = img[r0:r0 + c, c0:c0 + c]
    lab = lab[r0:r0 + c, c0:c0 + c]
    keep = set(int(v) for v in np.unique(lab)) - {0}
    lk = {p for p in (lk or set()) if p[0] in keep and p[1] in keep} or None
    print(f"crop ({r0},{c0}) {c}x{c}   labels {len(keep)}   "
          f"fg {100 * (lab > 0).mean():.2f}%   links {len(lk) if lk else 0}\n")

    init = None if a.init_model == "scratch" else a.init_model
    model = models.CellposeModel(
        gpu=torch.cuda.is_available(), omni=True, dim=2, nchan=1,
        nclasses=DEFAULTS["nclasses"], diam_mean=0.0,
        **({"model_type": init} if init else {"pretrained_model": False}))
    nout = model.nclasses
    print(f"arch: nchan={model.nchan} out={nout}   "
          f"dist_bg={getattr(model, 'dist_bg', '?')}\n")

    if not a.no_source:
        for name, fn in (("CellposeModel.loss_fn", type(model).loss_fn),
                         ("omnipose.core.loss", getattr(ocore, "loss", None))):
            if fn is None:
                continue
            print(f"----- {name} " + "-" * max(58 - len(name), 3))
            try:
                print(inspect.getsource(fn))
            except (OSError, TypeError) as exc:
                print(f"  (source unavailable: {exc})")
        print("-" * 66 + "\n")

    # The target exactly as model.train() builds it.
    flows = ocore.labels_to_flows([lab], links=[lk], files=None, use_gpu=False,
                                  device=None, dim=2, omni=True)
    lbl = np.asarray(flows[0], dtype=np.float32)[np.newaxis]      # (1, C, H, W)
    nch = lbl.shape[1]
    print(f"target lbl shape {lbl.shape}")
    for i in range(nch):
        ch = lbl[0, i]
        print(f"   ch{i}: [{ch.min():+8.3f}, {ch.max():+8.3f}]  "
              f"mean {ch.mean():+8.3f}  nonzero {100 * (ch != 0).mean():5.2f}%")
    print()

    dev = next(model.net.parameters()).device
    # `omnipose.core.loss` calls `lbl[:,4].detach()`, so the label stack has to
    # arrive as a device tensor. It is kept as numpy above for inspection only.
    lbl_t = torch.as_tensor(np.ascontiguousarray(lbl), dtype=torch.float32,
                            device=dev)

    def score(y_np):
        y = torch.as_tensor(np.ascontiguousarray(y_np), dtype=torch.float32,
                            device=dev)
        with torch.no_grad():
            return float(model.loss_fn(lbl_t, y))

    # --- references ---------------------------------------------------------
    x = torch.as_tensor(img[np.newaxis, np.newaxis], dtype=torch.float32,
                        device=dev)
    with torch.no_grad():
        y_net = model.net(x)[0].detach().cpu().numpy()
    print(f"y_net shape {y_net.shape}")
    print(f"{'prediction':<46}{'loss':>10}")
    print("-" * 56)
    print(f"{'the pretrained net itself':<46}{score(y_net):>10.4f}")
    print(f"{'all zeros':<46}{score(np.zeros_like(y_net)):>10.4f}")
    rng = np.random.default_rng(0)
    print(f"{'gaussian noise':<46}"
          f"{score(rng.normal(size=y_net.shape).astype(np.float32)):>10.4f}")
    print()

    # --- the target, assembled under each plausible convention --------------
    # Flow channels are the two that live in [-1, 1]; distance is a non-negative
    # channel. Identify them from the data rather than assuming an index.
    flow_ch = [i for i in range(nch)
               if lbl[0, i].min() < -0.01 and abs(lbl[0, i]).max() <= 1.5]
    dist_ch = [i for i in range(1, nch)
               if i not in flow_ch and lbl[0, i].max() > 1.0]
    print(f"flow channels {flow_ch}   candidate distance channels {dist_ch}\n")
    if len(flow_ch) != 2 or not dist_ch:
        print("!! could not identify the target channels; read the dump above")
        return 1

    print(f"{'target assembled as':<46}{'loss':>10}")
    print("-" * 56)
    best = (float("inf"), None)
    for fscale, dch, bg in itertools.product((1.0, 5.0), dist_ch, (0.0, -5.0)):
        y = np.zeros_like(y_net)
        y[0, 0] = lbl[0, flow_ch[0]] * fscale
        y[0, 1] = lbl[0, flow_ch[1]] * fscale
        d = lbl[0, dch].copy()
        if bg:
            d[lbl[0, 0] == 0] = bg
        if y.shape[1] > 2:
            y[0, 2] = d
        tag = f"flows x{fscale:g}, dist=ch{dch}, bg={bg:+g}"
        s = score(y)
        print(f"{tag:<46}{s:>10.4f}")
        best = min(best, (s, tag))
    print("-" * 56)
    print(f"best: {best[1]}  ->  {best[0]:.4f}\n")

    if best[0] < 0.5:
        print("=> THE LOSS CAN REACH ZERO. Plumbing is sound, so the optimiser")
        print(f"   is the problem: it sat at ~10 while {best[0]:.3f} was")
        print("   available. Suspect the gradient path -- autocast, a frozen")
        print("   parameter group, or an optimiser built over the wrong params.")
    else:
        print("=> NO ASSEMBLY OF THE TARGET REACHES ZERO. The loss demands")
        print("   something this 3-channel head cannot emit, so training could")
        print(f"   never do better than ~{best[0]:.1f}. That is the observed")
        print("   plateau, and it is a plumbing bug, not a schedule.")
        print("   Compare with --no-links to see whether links cause it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
