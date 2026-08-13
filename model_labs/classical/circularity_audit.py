"""Quantify how much of the classical floor's score is proposal circularity.

Why this is mandatory, not optional
-----------------------------------
The reviewed ground truth is *proposal-conditioned*: the operator was shown
machine proposals and mostly accepted them. Only 40 of the 375 trainable
`complete` masks were edited at all, so **89.3% of the ground truth is a verbatim
accepted proposal**.

The classical candidate re-derives its instances from the same deterministic
recipe (`semantic_territory`) that generated those proposals. So when a predicted
instance matches a reviewed mask exactly, that is the pipeline reproducing
*itself*, not a model demonstrating segmentation skill. Measured on the v1 run,
**47.4% of all matched pairs are pixel-identical (IoU = 1.000)** and the median
matched IoU is exactly 1.000 in three of six wells.

DEVELOPMENT_PLAN.md section 9 already says the classical proof-of-loop figures
"are illustrative and circular ... must not be reported scientifically". This
module turns that warning into a number, and splits the benchmark into the part
that is contaminated and the part that is not.

The honest split
----------------
* **unedited GT** (335 masks): the operator accepted the proposal as-is. A
  classical candidate sharing the proposal generator has a structural advantage
  here that no learned candidate gets. Scores on this subset are near-ceiling by
  construction.
* **edited GT** (40 masks, the real correction pairs): the operator had to change
  the proposal, so ground truth and proposal genuinely differ. This is the
  **non-circular subset** and the only part where the floor's recall means what
  the word normally means.

Consequence for T03: the classical floor must not be compared head-to-head with a
learned candidate on the pooled metric. The learned candidate does not share the
proposal generator, so the pooled number is biased toward the floor.

Usage::

    python model_labs/classical/circularity_audit.py --run model_labs/classical/_runs/v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

for _p in (ROOT / "PrecisionMyotube", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from precision_myotube.benchmark import benchmark_instances  # noqa: E402

BOOTSTRAP = ROOT / "PrecisionMyotube/annotation_work/bootstrap_v1"
IDENTICAL = 0.9999


def edited_ids_by_well() -> dict[str, set[str]]:
    """Ids of reviewed masks the operator actually changed (the correction pairs)."""
    out: dict[str, set[str]] = {}
    for line in (BOOTSTRAP / "corrections.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        out.setdefault(record["stem"], set()).add(record["id"])
    return out


def audit(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    edited = edited_ids_by_well()

    folds = []
    pooled_iou: list[float] = []
    pooled_edited, pooled_unedited = [], []
    for fold in manifest["folds"]:
        well = fold["held_out_well"]
        gt_path = ROOT / fold["eval_gt"]["path"]
        pred_path = ROOT / fold["prediction"]["instances"]
        metrics = benchmark_instances(gt_path, pred_path)
        matched = metrics["matched_instances"]

        well_edited = edited.get(well, set())
        n_gt = metrics["n_gt"]
        n_gt_edited = len(well_edited)

        ious = np.array([m["iou"] for m in matched], dtype=float)
        edited_hits = [m for m in matched if m["ground_truth_id"] in well_edited]
        unedited_hits = [m for m in matched if m["ground_truth_id"] not in well_edited]
        pooled_iou.extend(ious.tolist())
        pooled_edited.extend(m["iou"] for m in edited_hits)
        pooled_unedited.extend(m["iou"] for m in unedited_hits)

        folds.append({
            "held_out_well": well,
            "n_gt": n_gt,
            "n_gt_edited": n_gt_edited,
            "n_gt_unedited": n_gt - n_gt_edited,
            "matched_total": len(matched),
            "identical_pairs": int((ious >= IDENTICAL).sum()) if ious.size else 0,
            "identical_fraction_of_matched": (
                round(float((ious >= IDENTICAL).mean()), 3) if ious.size else None),
            "recall_all": round(metrics["recall"], 3),
            "recall_edited_subset": (round(len(edited_hits) / n_gt_edited, 3)
                                     if n_gt_edited else None),
            "recall_unedited_subset": (round(len(unedited_hits) / (n_gt - n_gt_edited), 3)
                                       if n_gt - n_gt_edited else None),
            "median_iou_edited": (round(float(np.median([m["iou"] for m in edited_hits])), 3)
                                  if edited_hits else None),
            "median_iou_unedited": (round(float(np.median([m["iou"] for m in unedited_hits])), 3)
                                    if unedited_hits else None),
        })

    pooled = np.asarray(pooled_iou)
    total_gt = sum(f["n_gt"] for f in folds)
    total_edited = sum(f["n_gt_edited"] for f in folds)
    summary = {
        "total_gt": total_gt,
        "total_gt_edited": total_edited,
        "total_gt_unedited": total_gt - total_edited,
        "unedited_fraction_of_gt": round((total_gt - total_edited) / total_gt, 3),
        "matched_pairs": int(pooled.size),
        "pixel_identical_pairs": int((pooled >= IDENTICAL).sum()),
        "pixel_identical_fraction_of_matched": round(float((pooled >= IDENTICAL).mean()), 3),
        "median_iou_all": round(float(np.median(pooled)), 3) if pooled.size else None,
        "median_iou_edited_subset": (round(float(np.median(pooled_edited)), 3)
                                     if pooled_edited else None),
        "median_iou_unedited_subset": (round(float(np.median(pooled_unedited)), 3)
                                       if pooled_unedited else None),
        "matched_edited": len(pooled_edited),
        "matched_unedited": len(pooled_unedited),
        "recall_edited_subset": (round(len(pooled_edited) / total_edited, 3)
                                 if total_edited else None),
        "recall_unedited_subset": (round(len(pooled_unedited) / (total_gt - total_edited), 3)
                                   if total_gt - total_edited else None),
    }
    return {
        "audit": "classical_floor_proposal_circularity",
        "candidate": manifest["candidate"],
        "run": str(run_dir.relative_to(ROOT)),
        "input_manifest_sha256": manifest["input_manifest_sha256"],
        "summary": summary,
        "folds": folds,
        "interpretation": [
            "The reviewed ground truth is proposal-conditioned: "
            f"{summary['unedited_fraction_of_gt']:.1%} of it is a verbatim accepted "
            "proposal that the operator never edited.",
            "The classical candidate re-derives instances from the same deterministic "
            "recipe that produced those proposals, so "
            f"{summary['pixel_identical_fraction_of_matched']:.1%} of its matched pairs "
            "are pixel-identical to the ground truth.",
            "Pooled recall and pooled matched-IoU therefore substantially measure "
            "reproducibility of the proposal generator, not segmentation skill.",
            "The edited subset (the 40 real correction pairs) is the non-circular "
            "evaluation: there, ground truth and proposal genuinely differ.",
            "T03 must not compare this floor head-to-head with a learned candidate on "
            "the pooled metric; the learned candidate does not share the proposal "
            "generator and so does not receive the same structural advantage.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="model_labs/classical/_runs/v1")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    report = audit(run_dir)
    out = Path(args.out) if args.out else run_dir / "circularity_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = report["summary"]
    print("=== proposal-circularity audit ===")
    print(f"  GT masks                     : {s['total_gt']} "
          f"({s['total_gt_unedited']} unedited, {s['total_gt_edited']} edited)")
    print(f"  unedited fraction of GT      : {s['unedited_fraction_of_gt']:.1%}")
    print(f"  matched pairs pixel-identical: {s['pixel_identical_pairs']}/"
          f"{s['matched_pairs']} ({s['pixel_identical_fraction_of_matched']:.1%})")
    print(f"  recall  (unedited subset)    : {s['recall_unedited_subset']}   <- circular")
    print(f"  recall  (edited subset)      : {s['recall_edited_subset']}   <- meaningful")
    print(f"  medIoU  (unedited subset)    : {s['median_iou_unedited_subset']}")
    print(f"  medIoU  (edited subset)      : {s['median_iou_edited_subset']}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
