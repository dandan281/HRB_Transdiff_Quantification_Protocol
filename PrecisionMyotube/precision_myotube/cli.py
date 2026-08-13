"""Command-line interface for the canonical precision workflow."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analysis import analyze
from .batch import run_batch
from .benchmark import benchmark_instances, benchmark_manifest, check_release, select_model
from .dataset import audit_manifest, export_annotation_package, export_training_sample
from .environment import fingerprint_environment
from .io import prepare_run
from .inference import adapt_prediction
from .integrity import verify_run_integrity
from .linked_candidate import run_linked_candidate
from .models import candidate_commands
from .pilot import (build_pilot_candidates, build_pilot_handoff, evaluate_g1,
                    select_pilot_tasks)
from .report import create_reports
from .review import (compare_pilot_reviews, create_pilot_review_template,
                     validate_pilot_review)
from .schema import from_label_image
from .segmentation import create_component_proposals, create_nuclei, create_territory
from .statistics import analyze_statistics_manifest
from .t03 import write_t03_assessment


def _json_print(value) -> None:
    print(json.dumps(value, indent=2))


def _load_labels(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    import tifffile
    return tifffile.imread(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="precision-myotube")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="extract a 2-D ND2 and resolve channel roles")
    p.add_argument("--nd2", required=True); p.add_argument("--out", required=True)
    p.add_argument("--fiber-ch", type=int); p.add_argument("--dapi-ch", type=int)

    p = sub.add_parser("territory", help="generate semantic Desmin territory")
    p.add_argument("--run", required=True)

    p = sub.add_parser("nuclei", help="segment DAPI nuclei with Cellpose-SAM plateau sweep")
    p.add_argument("--run", required=True); p.add_argument("--model")

    p = sub.add_parser("proposals", help="create non-authoritative instance proposals")
    p.add_argument("--run", required=True); p.add_argument("--out")

    p = sub.add_parser("import-labels", help="convert an edited label TIFF/NPY to instance JSON")
    p.add_argument("--labels", required=True); p.add_argument("--image-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--properties",
                   help="annotation-package instance_properties.csv with per-label status/review")
    p.add_argument("--reviewed-complete", action="store_true",
                   help="declare these labels expert-reviewed and complete")

    p = sub.add_parser("analyze", help="measure reviewed instances, nuclei, and field fusion")
    p.add_argument("--run", required=True); p.add_argument("--instances", required=True)
    p.add_argument("--nuclei-masks"); p.add_argument("--territory")
    p.add_argument("--amin-um2", type=float, default=50.0)
    p.add_argument("--amax-um2", type=float, default=500.0)

    p = sub.add_parser("run", help="prepare + segment; analyze only if reviewed instances supplied")
    p.add_argument("--nd2", required=True); p.add_argument("--out", required=True)
    p.add_argument("--fiber-ch", type=int); p.add_argument("--dapi-ch", type=int)
    p.add_argument("--instances"); p.add_argument("--nuclei-masks")
    p.add_argument("--cellpose-model")

    p = sub.add_parser("benchmark", help="score prediction instances against reviewed ground truth")
    p.add_argument("--ground-truth", required=True); p.add_argument("--prediction", required=True)
    p.add_argument("--out", required=True); p.add_argument("--model", default="candidate")

    p = sub.add_parser("benchmark-manifest",
                       help="score one candidate across locked plate and density strata")
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)

    p = sub.add_parser("select-model", help="apply the predefined candidate selection gate")
    p.add_argument("metrics", nargs="+"); p.add_argument("--out", required=True)

    p = sub.add_parser("release-check", help="apply prospective scientific release gates")
    p.add_argument("--metrics", required=True); p.add_argument("--out", required=True)

    p = sub.add_parser(
        "t03-assess",
        help="independently assess one sealed T02 run under the binding T03 contract")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap-resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260723)

    p = sub.add_parser(
        "linked-candidate-run",
        help="build the fold-refit classical-floor + fragment-linker candidate")
    p.add_argument("--base-run", default="model_labs/classical/_runs/v1")
    p.add_argument(
        "--pairs",
        default=("PrecisionMyotube/annotation_work/links_active_r3/banked/"
                 "combined_pairs_r123.jsonl"))
    p.add_argument(
        "--out",
        default="PrecisionMyotube/runs/t02/classical_linker_constrained_v2",
    )
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--gap-um", type=float, default=80.0)
    p.add_argument("--cos-min", type=float, default=0.70)
    p.add_argument(
        "--merge-policy",
        choices=("constrained_axis", "legacy_transitive_closure"),
        default="constrained_axis",
        help=("constrained_axis for new candidates; legacy_transitive_closure only "
              "for exact sealed-v1 reproduction"),
    )

    p = sub.add_parser(
        "statistics-summary",
        help="summarize nested measurements using declared biological/technical units")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("dataset-audit", help="check annotation volume and plate-level split rules")
    p.add_argument("--manifest", required=True); p.add_argument("--out")

    p = sub.add_parser("export-training", help="export reviewed masks for model training")
    p.add_argument("--run", required=True); p.add_argument("--instances", required=True)
    p.add_argument("--out", required=True); p.add_argument("--stem")

    p = sub.add_parser("annotation-package", help="export TIFFs for napari/micro-sam correction")
    p.add_argument("--run", required=True); p.add_argument("--out", required=True)
    p.add_argument("--instances", help="optional proposals or existing annotations")

    p = sub.add_parser("model-commands", help="write reproducible candidate training commands")
    p.add_argument("--train", required=True); p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("batch", help="run a restartable field/plate manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--summary-dir")
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("adapt-prediction",
                       help="normalize model labels/polygons/RLE to unreviewed InstanceSet")
    p.add_argument("--input", required=True)
    p.add_argument("--format", required=True, choices=("labels", "json"))
    p.add_argument("--image-id", required=True)
    p.add_argument("--architecture", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--checkpoint")
    p.add_argument("--environment")
    p.add_argument("--thresholds", help="JSON object of inference thresholds")
    p.add_argument("--confidence-map")

    p = sub.add_parser("verify-run", help="verify source and latest analysis artifact hashes")
    p.add_argument("--run", required=True)

    p = sub.add_parser("fingerprint-environment",
                       help="freeze the active environment and bind it to validation evidence")
    p.add_argument("--out", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--validation-summary")
    p.add_argument("--expected-total-nuclei", type=int)

    p = sub.add_parser("pilot-select", help="select deterministic development-only H01 review tasks")
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", type=int, default=100)
    p.add_argument("--minimum-hard", type=int, default=25)
    p.add_argument("--seed", default="precision-myotube-pilot-v1")

    p = sub.add_parser("pilot-candidates",
                       help="derive H01 proposal strata from canonical run directories")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("pilot-handoff",
                       help="bind the frozen pilot to hashed Claude annotation packages")
    p.add_argument("--manifest", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("pilot-review-validate",
                       help="validate one independent pilot review and its field exports")
    p.add_argument("--manifest", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--out")

    p = sub.add_parser("pilot-review-template",
                       help="create a 100-task reviewer decision template")
    p.add_argument("--manifest", required=True)
    p.add_argument("--handoff", required=True)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("pilot-review-compare",
                       help="compare two valid independent pilot reviews for adjudication")
    p.add_argument("--manifest", required=True)
    p.add_argument("--review-a", required=True)
    p.add_argument("--review-b", required=True)
    p.add_argument("--mask-iou", type=float, default=0.8)
    p.add_argument("--out", required=True)

    p = sub.add_parser("gate-g1", help="evaluate pilot-readiness evidence without guessing")
    p.add_argument("--evidence", required=True)
    p.add_argument("--out")

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        _json_print(prepare_run(args.nd2, args.out, force_fiber=args.fiber_ch,
                                force_dapi=args.dapi_ch))
    elif args.command == "territory":
        print(create_territory(args.run))
    elif args.command == "nuclei":
        print(create_nuclei(args.run, args.model))
    elif args.command == "proposals":
        print(create_component_proposals(args.run, args.out))
    elif args.command == "import-labels":
        status = "complete" if args.reviewed_complete else "ambiguous"
        result = from_label_image(_load_labels(args.labels), args.image_id,
                                  source="expert_import" if args.reviewed_complete else "imported_proposal",
                                  reviewed=args.reviewed_complete, default_status=status)
        if args.properties:
            with Path(args.properties).open(newline="", encoding="utf-8-sig") as handle:
                properties = {int(row["label"]): row for row in csv.DictReader(handle)}
            for record in result.instances:
                label_id = int(record.id.rsplit("_", 1)[-1])
                row = properties.get(label_id)
                if not row:
                    continue
                record.id = row.get("id") or record.id
                record.status = (row.get("status") or record.status).strip()
                reviewed = (row.get("reviewed") or "").strip().lower()
                if reviewed:
                    if reviewed not in {"true", "false", "1", "0", "yes", "no"}:
                        raise ValueError(f"label {label_id}: invalid reviewed value {reviewed!r}")
                    record.reviewed = reviewed in {"true", "1", "yes"}
                record.source = row.get("source") or record.source
                record.notes = row.get("notes") or ""
        result.save(args.out); print(args.out)
    elif args.command == "analyze":
        result = analyze(args.run, args.instances, nuclei_masks_path=args.nuclei_masks,
                         territory_path=args.territory, amin_um2=args.amin_um2,
                         amax_um2=args.amax_um2)
        create_reports(args.run, result); _json_print(result["summary"])
    elif args.command == "run":
        prepare_run(args.nd2, args.out, force_fiber=args.fiber_ch, force_dapi=args.dapi_ch)
        create_territory(args.out)
        if args.nuclei_masks:
            masks = _load_labels(args.nuclei_masks).astype(np.int32)
            np.save(Path(args.out) / "nuclei_masks.npy", masks)
        else:
            create_nuclei(args.out, args.cellpose_model)
        if args.instances:
            result = analyze(args.out, args.instances)
            create_reports(args.out, result); _json_print(result["summary"])
        else:
            proposals = create_component_proposals(args.out)
            print(f"Stopped at required review gate. Proposals: {proposals}")
    elif args.command == "benchmark":
        metrics = benchmark_instances(args.ground_truth, args.prediction)
        metrics["model"] = args.model
        Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _json_print(metrics)
    elif args.command == "benchmark-manifest":
        metrics = benchmark_manifest(args.manifest)
        Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _json_print(metrics)
    elif args.command == "select-model":
        result = select_model(args.metrics)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        _json_print(result)
    elif args.command == "release-check":
        metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
        result = check_release(metrics)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        _json_print(result)
    elif args.command == "t03-assess":
        result = write_t03_assessment(
            args.run, args.out,
            bootstrap_resamples=args.bootstrap_resamples, seed=args.seed)
        _json_print({
            "out": args.out,
            "candidate": result["candidate"],
            "integrity_passed": result["integrity"]["passed"],
            "t03_complete": result["gate"]["t03_complete"],
            "disposition": result["gate"]["disposition"],
        })
    elif args.command == "linked-candidate-run":
        result = run_linked_candidate(
            args.out,
            base_run=args.base_run,
            pairs_path=args.pairs,
            threshold=args.threshold,
            gap_um=args.gap_um,
            cos_min=args.cos_min,
            merge_policy=args.merge_policy,
        )
        _json_print({
            "out": args.out,
            "candidate": result["candidate"],
            "candidate_version": result["candidate_version"],
            "micro_summary": result["summary"],
        })
    elif args.command == "statistics-summary":
        result = analyze_statistics_manifest(args.manifest)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        _json_print(result)
    elif args.command == "dataset-audit":
        result = audit_manifest(args.manifest)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        _json_print(result)
    elif args.command == "export-training":
        _json_print(export_training_sample(args.run, args.instances, args.out, args.stem))
    elif args.command == "annotation-package":
        _json_print(export_annotation_package(args.run, args.out, args.instances))
    elif args.command == "model-commands":
        _json_print(candidate_commands(args.train, args.test, args.out))
    elif args.command == "batch":
        result = run_batch(args.manifest, resume=args.resume, summary_dir=args.summary_dir)
        _json_print(result)
    elif args.command == "adapt-prediction":
        if (args.height is None) != (args.width is None):
            raise ValueError("--height and --width must be supplied together")
        expected_shape = (args.height, args.width) if args.height is not None else None
        thresholds = json.loads(args.thresholds) if args.thresholds else None
        result = adapt_prediction(
            input_path=args.input, input_format=args.format, output_path=args.out,
            image_id=args.image_id, architecture=args.architecture,
            expected_shape=expected_shape, checkpoint=args.checkpoint,
            environment=args.environment, thresholds=thresholds,
            confidence_path=args.confidence_map)
        _json_print({"out": args.out, "instances": len(result.instances),
                     "provenance": result.provenance})
    elif args.command == "verify-run":
        result = verify_run_integrity(args.run)
        _json_print(result)
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "fingerprint-environment":
        result = fingerprint_environment(
            args.out, label=args.label,
            validation_summary=args.validation_summary,
            expected_total_nuclei=args.expected_total_nuclei,
        )
        _json_print(result)
    elif args.command == "pilot-select":
        result = select_pilot_tasks(
            args.candidates, args.out, target=args.target,
            minimum_hard=args.minimum_hard, seed=args.seed)
        _json_print(result["audit"])
    elif args.command == "pilot-candidates":
        result = build_pilot_candidates(args.manifest, args.out)
        _json_print({"out": args.out, "candidates": len(result["candidates"])})
    elif args.command == "pilot-handoff":
        result = build_pilot_handoff(args.manifest, args.runs, args.out)
        _json_print({"out": args.out, "fields": result["field_count"],
                     "tasks": result["task_count"]})
    elif args.command == "pilot-review-validate":
        result = validate_pilot_review(args.manifest, args.review)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        _json_print(result)
    elif args.command == "pilot-review-template":
        result = create_pilot_review_template(
            args.manifest, args.handoff, args.reviewer, args.out)
        _json_print({"out": args.out, "reviewer": result["reviewer"],
                     "tasks": len(result["tasks"])})
    elif args.command == "pilot-review-compare":
        result = compare_pilot_reviews(
            args.manifest, args.review_a, args.review_b, mask_iou_threshold=args.mask_iou)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        _json_print({key: value for key, value in result.items()
                     if key not in {"disagreements", "validation_a", "validation_b"}})
    elif args.command == "gate-g1":
        result = evaluate_g1(args.evidence)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        _json_print(result)
if __name__ == "__main__":
    main()
