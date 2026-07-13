"""Tier-2 scientific endpoint: fibre count, % below/above 300 um, and boundary-weighted length error.

Errors near the 300 um threshold matter most (a small length error there flips a fibre's class and
corrupts the published readout), so those are surfaced explicitly.
"""
from __future__ import annotations

import numpy as np


def pct_below(lengths_um, thresh) -> float:
    lengths_um = np.asarray(lengths_um, dtype=float)
    if lengths_um.size == 0:
        return float("nan")
    return 100.0 * float(np.mean(lengths_um < thresh))


def endpoint_metrics(gt_lengths_um, pred_lengths_um, thresh) -> dict:
    gt = np.asarray(gt_lengths_um, dtype=float)
    pr = np.asarray(pred_lengths_um, dtype=float)
    gt_below, pr_below = pct_below(gt, thresh), pct_below(pr, thresh)
    return dict(
        n_gt=int(gt.size), n_pred=int(pr.size), count_delta=int(pr.size) - int(gt.size),
        gt_pct_below=gt_below, pred_pct_below=pr_below,
        pct_below_delta=(pr_below - gt_below),
        gt_pct_above=(100.0 - gt_below), pred_pct_above=(100.0 - pr_below),
        gt_median_um=float(np.median(gt)) if gt.size else float("nan"),
        pred_median_um=float(np.median(pr)) if pr.size else float("nan"),
    )


def boundary_flip_rate(matches, gt_lens_um, pred_lens_um, thresh) -> float:
    """Fraction of matched pairs where GT and prediction land on opposite sides of the threshold."""
    if not matches:
        return float("nan")
    flips = 0
    for i, j in matches:
        if (gt_lens_um[i] < thresh) != (pred_lens_um[j] < thresh):
            flips += 1
    return flips / len(matches)


def boundary_weighted_mae(matches, gt_lens_um, pred_lens_um, thresh, sigma) -> float:
    """Length MAE over matched pairs, Gaussian-weighted toward fibres near the threshold."""
    if not matches:
        return float("nan")
    num = den = 0.0
    for i, j in matches:
        w = np.exp(-((gt_lens_um[i] - thresh) / sigma) ** 2)
        num += w * abs(pred_lens_um[j] - gt_lens_um[i])
        den += w
    return num / den if den else float("nan")


def bland_altman(deltas) -> dict:
    d = np.asarray([x for x in deltas if np.isfinite(x)], dtype=float)
    if d.size == 0:
        return dict(mean_bias=float("nan"), loa_low=float("nan"), loa_high=float("nan"), n=0)
    mean, sd = float(np.mean(d)), float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    return dict(mean_bias=mean, loa_low=mean - 1.96 * sd, loa_high=mean + 1.96 * sd, n=int(d.size))
