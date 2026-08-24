"""The T04 three-head U-Net: centre / orient / crossing. Small on purpose.

5,004 instances over ten wells does not justify Omnipose's 6.6 M parameters --
the base width here gives ~1.9 M. Heads are 1x1 convs off a shared decoder:

``centre``    1 ch, raw. MSE against the soft Gaussian target. Not BCE: the
              BCE of a soft label against its own logit is the label's
              entropy, which the probe measured at 0.082 on real tiles -- a
              floor that would read as a mysterious plateau in every training
              curve. MSE's floor is zero.
``orient``    2 ch, raw. Masked MSE against the unit ``(cos 2t, sin 2t)``
              target, masked by ``orient_valid`` -- at a crossing the target is
              genuinely two-valued and an average of two directions is a
              direction no fibre has, so those pixels teach nothing.
``crossing``  1 ch, logit. BCE-with-logits with ``pos_weight``: crossings are
              ~0.2% of pixels and an unweighted BCE would let the head predict
              "never" at negligible cost.

The loss is a weighted sum, but the weights are NOT trusted by construction --
`tracer_loss` returns every term separately and `grad_shares` measures each
head's actual share of the gradient. That instrumentation exists because this
project spent four diagnostic rounds discovering that 92% of Omnipose's
`raw_loss` had no gradient path at all; here the accounting comes first.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))


class TracerNet(nn.Module):
    """U-Net, depth 4, base 32: 1 -> (centre 1, orient 2, crossing 1)."""

    def __init__(self, base: int = 32):
        super().__init__()
        b = base
        self.e1 = _block(1, b)
        self.e2 = _block(b, 2 * b)
        self.e3 = _block(2 * b, 4 * b)
        self.e4 = _block(4 * b, 8 * b)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2)
        self.d3 = _block(8 * b, 4 * b)
        self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
        self.d2 = _block(4 * b, 2 * b)
        self.u1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
        self.d1 = _block(2 * b, b)
        self.head_centre = nn.Conv2d(b, 1, 1)
        self.head_orient = nn.Conv2d(b, 2, 1)
        self.head_crossing = nn.Conv2d(b, 1, 1)
        self.head_offset = nn.Conv2d(b, 2, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return {"centre": self.head_centre(d1),
                "orient": self.head_orient(d1),
                "crossing": self.head_crossing(d1),
                "offset": self.head_offset(d1)}


def tracer_loss(pred: dict, tgt: dict, *, w_centre=1.0, w_orient=1.0,
                w_crossing=1.0, w_offset=1.0, crossing_pos_weight=8.0,
                centre_ridge_weight=10.0, centre_dice_weight=1.0,
                offset_band_px=12.0):
    """Per-term losses plus their weighted total. Every term is returned;
    nothing is hidden inside a sum.

    ``centre_ridge_weight``: per-pixel weight ``1 + w * target`` on the centre
    MSE. Unweighted, the first 60-epoch run let the ~94% background dominate
    and produced a ridge the walk could not stand on -- on-ridge median 0.235
    against an off-ridge p99 of 0.347 (D04, best.pt of that run). The floor is
    still exactly 0. ``crossing_pos_weight`` was 25 in the same run and bought
    recall at hopeless precision (off-crossing p99 0.79); 8 is the retrain's
    setting, recorded here because both changed at once.
    """
    # Weighted MSE keeps the map calibrated; soft Dice is what forces it to be
    # SHARP. Measured twice (net_v3 centre head, net_v4 offset head): MSE
    # alone returns a 12 px-FWHM bump against a 4 px target with the peak in
    # the right place and the amplitude collapsed -- because under positional
    # uncertainty, hedging across the fibre width is MSE-optimal. Dice is
    # scale-sensitive: spreading the same mass over 3x the width cuts the
    # overlap ratio, so hedging stops being free. Both terms are exactly 0 for
    # a perfect prediction, so the probe still reads a true floor.
    w = 1.0 + centre_ridge_weight * tgt["centre"]
    l_mse = (w * (pred["centre"][:, 0] - tgt["centre"]) ** 2).mean()
    p = pred["centre"][:, 0].clamp(0.0, 1.0)
    t = tgt["centre"]
    inter = (p * t).sum(dim=(-2, -1))
    denom = (p * p).sum(dim=(-2, -1)) + (t * t).sum(dim=(-2, -1))
    l_dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()
    l_centre = l_mse + centre_dice_weight * l_dice

    m = tgt["orient_valid"].unsqueeze(1).float()
    n_valid = m.sum().clamp(min=1.0)
    l_orient = (((pred["orient"] - tgt["orient"]) ** 2) * m).sum() \
        / (2.0 * n_valid)

    pw = torch.as_tensor(crossing_pos_weight, device=pred["crossing"].device)
    l_crossing = F.binary_cross_entropy_with_logits(
        pred["crossing"][:, 0], tgt["crossing"].float(), pos_weight=pw)

    # Offset to the nearest centreline, in units of the band radius, masked
    # to the band. Smooth and locally linear, so positional uncertainty costs
    # a small displacement error instead of flattening a peak -- the failure
    # measured on the centre head (12 px FWHM against a 4 px target).
    mo = tgt["offset_valid"].unsqueeze(1).float()
    n_off = mo.sum().clamp(min=1.0)
    l_offset = (((pred["offset"] - tgt["offset"] / offset_band_px) ** 2)
                * mo).sum() / (2.0 * n_off)

    total = (w_centre * l_centre + w_orient * l_orient
             + w_crossing * l_crossing + w_offset * l_offset)
    return total, {"centre": l_centre, "orient": l_orient,
                   "crossing": l_crossing, "offset": l_offset}


def grad_shares(model: nn.Module, terms: dict) -> dict:
    """Each term's share of the gradient norm on the SHARED trunk.

    The Omnipose failure mode was a loss whose largest terms had no gradient
    path; this is the direct measurement that ours do, and in what proportion.
    Heads' own 1x1 convs are excluded -- a head always feels its own loss; the
    question is who steers the trunk.
    """
    trunk = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.startswith("head_")]
    norms = {}
    for name, term in terms.items():
        gs = torch.autograd.grad(term, trunk, retain_graph=True,
                                 allow_unused=True)
        norms[name] = float(torch.sqrt(sum(
            (g ** 2).sum() for g in gs if g is not None)))
    tot = sum(norms.values()) or 1.0
    return {k: v / tot for k, v in norms.items()}
