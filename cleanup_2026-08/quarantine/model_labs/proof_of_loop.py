"""Proof-of-loop: run the whole pipeline end-to-end on real QC decisions.

    decisions.json --apply--> reviewed-complete ground truth
                   --export--> training image/label pairs (ready for a real model)
    proposals      --baseline--> prediction (reality check)
                   --QC model--> filtered prediction
    both predictions --benchmark--> precision / recall / split-merge / length-width error

This proves every stage runs and produces comparable metrics BEFORE investing in
full-scale annotation. Swap the predictor for a trained Omnipose/micro-sam model
and the same harness scores it. It is deliberately honest: with ~100 pilot labels
the numbers are illustrative (and the QC filter is scored in-sample), not a
release-grade evaluation.

    python model_labs/proof_of_loop.py --package <annotation_package> \
        --decisions <decisions.json> --reviewer <name> --out <out_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "PrecisionMyotube", ROOT / "annotation_tools", ROOT / "model_labs"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from annotation_tools.qc_review.cli import _load_package, cmd_apply           # noqa: E402
from annotation_tools.qc_review.pipeline import proposal_features             # noqa: E402
from annotation_tools.qc_review import model as qcmodel                       # noqa: E402
from _shared.schema_bridge import (                                           # noqa: E402
    InstanceSet, InstanceRecord, from_label_image, encode_sparse_positions)
from _shared.train_export import export_training                              # noqa: E402
from precision_myotube.benchmark import benchmark_instances                   # noqa: E402


def build_ground_truth(package, decisions, reviewer, out):
    cmd_apply(argparse.Namespace(package=str(package), decisions=str(decisions),
                                 reviewer=reviewer, out=str(out)))
    return out


def baseline_prediction(labels, image_id, out):
    """Every raw proposal as a 'complete' prediction — the classical reality check."""
    inst = from_label_image(labels, image_id, source="proposal_baseline", default_status="complete")
    inst.save(out)
    return out, len(inst.instances)


def qc_filtered_prediction(labels, fiber, territory, pixel_um, image_id, out):
    """Only proposals the trained QC model predicts 'accept'."""
    H, W = labels.shape
    records = []
    for lid, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        sub = labels[sl] == lid
        r0, c0, r1, c1 = sl[0].start, sl[1].start, sl[0].stop, sl[1].stop
        touches = (r0 == 0 or c0 == 0 or r1 == H or c1 == W)
        terr_sub = territory[sl] if territory is not None else None
        feats = proposal_features(sub, fiber[sl], terr_sub, pixel_um, touches)
        action, p = qcmodel.predict_default(feats)
        if action != "accept":
            continue
        ys, xs = np.nonzero(sub)
        pos = (ys + r0).astype(np.int64) + (xs + c0).astype(np.int64) * H
        records.append(InstanceRecord(id=f"pred_{lid:04d}", status="complete",
                                      rle=encode_sparse_positions((H, W), pos),
                                      source="qc_model", reviewed=False, confidence=p))
    inst = InstanceSet((H, W), image_id, records)
    inst.save(out)
    return out, len(records)


def _summary(m):
    keys = ["n_gt", "n_pred", "tp", "precision", "recall", "f1",
            "false_split_rate", "over_merge_rate", "length_mdape", "width_mdape"]
    return {k: (round(m[k], 3) if isinstance(m.get(k), float) else m.get(k)) for k in keys}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--reviewer", default="pilot")
    ap.add_argument("--out", default="model_labs/_proof")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem, labels, fiber, territory, pixel_um, dapi = _load_package(Path(args.package))
    territory = None if territory is None else np.asarray(territory).astype(bool)

    gt_path = build_ground_truth(args.package, args.decisions, args.reviewer, out / "ground_truth.json")
    gt = InstanceSet.load(gt_path)
    n_gt_complete = sum(1 for r in gt.instances if r.reviewed and r.status == "complete")

    train = export_training(gt_path, fiber, out / "training", dapi=dapi)

    base_path, n_base = baseline_prediction(labels, stem, out / "pred_baseline.json")
    qc_path, n_qc = qc_filtered_prediction(labels, fiber, territory, pixel_um, stem, out / "pred_qc.json")

    m_base = benchmark_instances(gt_path, base_path)
    m_qc = benchmark_instances(gt_path, qc_path)

    report = {
        "stem": stem,
        "ground_truth_complete_masks": n_gt_complete,
        "training_export": train,
        "predictions": {"baseline_proposals": n_base, "qc_model_filtered": n_qc},
        "benchmark": {"baseline": _summary(m_base), "qc_filtered": _summary(m_qc)},
    }
    (out / "proof_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("\n=== end-to-end loop ran ===")
    print(f"ground truth: {n_gt_complete} reviewed-complete masks")
    print(f"training export: {train['n_trainable_instances']} instances, "
          f"{train['n_overlap_pixels']} overlap px -> {out/'training'}")
    print(f"baseline (all {n_base} proposals):  precision {m_base['precision']:.3f}  recall {m_base['recall']:.3f}")
    print(f"QC-model-filtered ({n_qc} proposals): precision {m_qc['precision']:.3f}  recall {m_qc['recall']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
