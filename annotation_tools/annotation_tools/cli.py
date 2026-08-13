"""Command line for the annotation lane.

Subcommands
-----------
launch            Open the napari/micro-sam GUI on an annotation package.
verify-roundtrip  Run the CL02 overlap round-trip proof and print a report.
validate          Validate an exported InstanceSet against the canonical schema.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annotation-tools",
                                     description="PrecisionMyotube assisted annotation lane")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("launch", help="open the GUI on an annotation package")
    p.add_argument("--package", required=True, help="annotation-package directory")
    p.add_argument("--out", help="where to save the InstanceSet JSON on export")

    p = sub.add_parser("verify-roundtrip", help="run the CL02 overlap round-trip proof")
    p.add_argument("--out", default="annotation_tools/_roundtrip", help="output directory")
    p.add_argument("--size", type=int, default=64, help="synthetic field size")

    p = sub.add_parser("validate", help="validate an InstanceSet against the frozen schema")
    p.add_argument("--instances", required=True)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "launch":
        from .napari_app import launch
        launch(args.package, export_path=args.out)
        return 0

    if args.command == "verify-roundtrip":
        from .roundtrip import run_overlap_roundtrip
        report = run_overlap_roundtrip(args.out, shape=(args.size, args.size))
        print(json.dumps({"passed": report.passed, "checks": report.checks,
                          "details": report.details}, indent=2, default=str))
        return 0 if report.passed else 1

    if args.command == "validate":
        from ._schema_bridge import InstanceSet
        instance_set = InstanceSet.load(args.instances)
        authoritative = [r.id for r in instance_set.instances
                         if r.reviewed and r.status == "complete"]
        print(json.dumps({
            "valid": True, "image_id": instance_set.image_id,
            "image_shape": list(instance_set.image_shape),
            "n_instances": len(instance_set.instances),
            "n_authoritative_complete": len(authoritative),
        }, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
