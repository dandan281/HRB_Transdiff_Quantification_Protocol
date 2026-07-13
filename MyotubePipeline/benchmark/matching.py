"""Tier-1 detection: overlap graph -> greedy 1:1 match -> precision/recall/F1 + 3 error classes."""
from __future__ import annotations

import numpy as np

from . import geometry as g


class Overlap:
    """Sparse overlap between GT (rows i) and Pred (cols j) fibre masks."""

    def __init__(self, n_gt, n_pred):
        self.n_gt, self.n_pred = n_gt, n_pred
        self.pairs = []          # list of (i, j, area, cov_gt, cov_pred, iou)

    def add(self, i, j, area, cov_gt, cov_pred, iou):
        self.pairs.append((i, j, area, cov_gt, cov_pred, iou))


def build_overlap(gt_masks, pred_masks) -> Overlap:
    ov = Overlap(len(gt_masks), len(pred_masks))
    cand = g.bbox_overlap_pairs(gt_masks, pred_masks)
    for i, j in cand:
        gm, pm = gt_masks[i], pred_masks[j]
        area = g.intersection_area(gm, pm)
        if area <= 0:
            continue
        cov_gt = area / gm.area if gm.area else 0.0
        cov_pred = area / pm.area if pm.area else 0.0
        union = gm.area + pm.area - area
        iou = area / union if union else 0.0
        ov.add(int(i), int(j), area, cov_gt, cov_pred, iou)
    return ov


def greedy_match(ov: Overlap, iou_thresh: float) -> list[tuple[int, int]]:
    """Greedy 1:1 matching: take highest-IoU pairs (>= thresh) first, each GT/Pred used once."""
    cands = sorted((p for p in ov.pairs if p[5] >= iou_thresh), key=lambda p: -p[5])
    gt_used, pred_used, matches = set(), set(), []
    for i, j, *_ in cands:
        if i in gt_used or j in pred_used:
            continue
        gt_used.add(i)
        pred_used.add(j)
        matches.append((i, j))
    return matches


def detection_metrics(matches, n_gt, n_pred) -> dict:
    tp = len(matches)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(tp=tp, fp=n_pred - tp, fn=n_gt - tp, n_gt=n_gt, n_pred=n_pred,
                precision=precision, recall=recall, f1=f1)


def classify_errors(ov: Overlap, matches, gt_lens_px, pred_lens_px,
                    min_overlap_frac, too_short_ratio, too_long_ratio, frag_cov_frac=0.2) -> dict:
    """Read the error classes off the overlap graph.

    - false-split (strict): one GT covered (>= min_overlap_frac) by >= 2 distinct predictions.
    - fragmentation (loose): one GT with >= 2 predictions each covering >= frag_cov_frac of it —
      catches the dominant "one main trace + short stubs" breakup that the strict rule misses,
      while excluding incidental crossings (which cover little of the GT).
    - over-merge : one prediction covering (>= min_overlap_frac) >= 2 distinct GT.
    - too-short  : matched pair with pred_len < too_short_ratio * gt_len.
    """
    # thresholded overlap graph (edge if either fibre covers >= frac of the other)
    gt_to_pred, pred_to_gt = {}, {}
    gt_frag = {}      # GT -> preds covering >= frag_cov_frac of that GT
    for i, j, _area, cov_gt, cov_pred, _iou in ov.pairs:
        if max(cov_gt, cov_pred) >= min_overlap_frac:
            gt_to_pred.setdefault(i, set()).add(j)
            pred_to_gt.setdefault(j, set()).add(i)
        if cov_gt >= frag_cov_frac:
            gt_frag.setdefault(i, set()).add(j)

    false_split = {i: sorted(js) for i, js in gt_to_pred.items() if len(js) >= 2}
    over_merge = {j: sorted(is_) for j, is_ in pred_to_gt.items() if len(is_) >= 2}
    fragmented = {i: sorted(js) for i, js in gt_frag.items() if len(js) >= 2}
    extra_fragments = sum(len(js) - 1 for js in fragmented.values())  # surplus pieces to re-link

    too_short, too_long, len_ratios = [], [], []
    for i, j in matches:
        gl, pl = gt_lens_px[i], pred_lens_px[j]
        if gl <= 0:
            continue
        ratio = pl / gl
        len_ratios.append(ratio)
        if ratio < too_short_ratio:
            too_short.append((i, j, ratio))
        elif ratio > too_long_ratio:
            too_long.append((i, j, ratio))

    n_match = len(matches)
    return dict(
        false_split=false_split, over_merge=over_merge, fragmented=fragmented,
        too_short=too_short, too_long=too_long,
        false_split_count=len(false_split), over_merge_count=len(over_merge),
        fragmented_count=len(fragmented), extra_fragments=extra_fragments,
        too_short_count=len(too_short), too_long_count=len(too_long),
        too_short_rate=(len(too_short) / n_match if n_match else 0.0),
        false_split_rate=(len(false_split) / ov.n_gt if ov.n_gt else 0.0),
        fragmented_rate=(len(fragmented) / ov.n_gt if ov.n_gt else 0.0),
        over_merge_rate=(len(over_merge) / ov.n_pred if ov.n_pred else 0.0),
        median_len_ratio=(float(np.median(len_ratios)) if len_ratios else float("nan")),
    )
