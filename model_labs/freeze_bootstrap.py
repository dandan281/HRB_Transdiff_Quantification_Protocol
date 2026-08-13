"""Build the versioned T01 bootstrap dataset from the six reviewed Plate-23 wells.

Development is intentionally not blocked on the optional G-SO1 repeatability recheck. The export
remains explicit about its proposal-conditioned, single-operator status and applies every binding
training exclusion to both real masks and their synthetic derivatives.

Run from the repository root:

    python model_labs/freeze_bootstrap.py \
      --exclude PrecisionMyotube/annotation_work/training_exclude.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile

from _shared.schema_bridge import InstanceSet


WELLS = [
    ("32_C08_smoke", "32_C08_br223_igf1r"),
    ("19_B06_act104_trka", "19_B06_act104_trka"),
    ("22_B03_act104_egfrc", "22_B03_act104_egfrc"),
    ("29_C05_br223_egfrc", "29_C05_br223_egfrc"),
    ("33_C09_br223_trka", "33_C09_br223_trka"),
    ("23_B02_ctrl", "23_B02_ctrl"),
]
ROOT = Path("PrecisionMyotube/annotation_work")
REQUIRED_PAIR_ARRAYS = {"fiber", "dapi", "proposal", "corrected"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(exclude_path: str | Path | None) -> set[tuple[str, str]]:
    if not exclude_path:
        return set()
    manifest = json.loads(Path(exclude_path).read_text(encoding="utf-8"))
    result = {(str(row["well"]), str(row["real_id"])) for row in manifest.get("exclude", [])}
    if len(result) != len(manifest.get("exclude", [])):
        raise ValueError("training exclusion manifest contains duplicates")
    return result


def _eligible_records(inst: InstanceSet, stem: str, exclusions: set[tuple[str, str]]):
    return [
        record
        for record in inst.instances
        if record.reviewed
        and record.status == "complete"
        and (stem, record.id) not in exclusions
    ]


def _plan(root: Path, wells, exclusions: set[tuple[str, str]]) -> dict:
    per_well: dict[str, dict] = {}
    total = 0
    seen_exclusions: set[tuple[str, str]] = set()
    for package, stem in wells:
        source = root / package / f"{stem}.qc.instances.json"
        inst = InstanceSet.load(source)
        keep = _eligible_records(inst, stem, exclusions)
        dropped = [
            record.id
            for record in inst.instances
            if record.status == "complete" and (stem, record.id) in exclusions
        ]
        seen_exclusions.update((stem, item) for item in dropped)
        per_well[stem] = {
            "package": package,
            "complete_kept": len(keep),
            "excluded": dropped,
            "source_instances": source.as_posix(),
            "source_instances_sha256": _sha256(source),
        }
        total += len(keep)
    missing = sorted(exclusions - seen_exclusions)
    if missing:
        raise ValueError(f"exclusions do not identify reviewed-complete records: {missing}")
    return {"trainable_complete": total, "excluded_total": len(exclusions), "per_well": per_well}


def _export_well(
    root: Path,
    package: str,
    stem: str,
    out_dir: Path,
    exclusions: set[tuple[str, str]],
) -> dict:
    package_dir = root / package
    instances_path = package_dir / f"{stem}.qc.instances.json"
    fiber_path = package_dir / "fiber_raw16.tif"
    dapi_path = package_dir / "dapi_raw16.tif"
    inst = InstanceSet.load(instances_path)
    fiber = tifffile.imread(fiber_path)
    dapi = tifffile.imread(dapi_path)
    if tuple(fiber.shape) != tuple(inst.image_shape) or tuple(dapi.shape) != tuple(inst.image_shape):
        raise ValueError(f"{stem}: image and instance shapes do not match")

    out_dir.mkdir(parents=True, exist_ok=False)
    labels = np.zeros(inst.image_shape, dtype=np.int32)
    ignore = np.zeros(inst.image_shape, dtype=bool)
    mapping = []
    train_label = 0
    for record, bbox, crop in inst.cropped_masks():
        if not (
            record.reviewed
            and record.status == "complete"
            and (stem, record.id) not in exclusions
        ):
            continue
        train_label += 1
        r0, c0, r1, c1 = bbox
        region = labels[r0:r1, c0:c1]
        ignore[r0:r1, c0:c1] |= (region > 0) & crop
        region[crop] = train_label
        mapping.append({"train_label": train_label, "source_id": record.id})

    outputs = {
        "image_fiber.tif": fiber,
        "image_dapi.tif": dapi,
        "labels.tif": labels,
        "ignore.tif": ignore.astype(np.uint8),
    }
    for name, array in outputs.items():
        tifffile.imwrite(out_dir / name, array, compression="zlib")
    mapping_path = out_dir / "instance_mapping.jsonl"
    mapping_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in mapping), encoding="utf-8"
    )

    files = {
        name: {"sha256": _sha256(out_dir / name), "bytes": (out_dir / name).stat().st_size}
        for name in [*outputs, "instance_mapping.jsonl"]
    }
    result = {
        "well": stem,
        "image_shape": list(inst.image_shape),
        "n_trainable_instances": train_label,
        "n_overlap_pixels": int(ignore.sum()),
        "excluded_ids": sorted(record_id for well, record_id in exclusions if well == stem),
        "channels": ["fiber", "dapi"],
        "source": {
            "instances": instances_path.as_posix(),
            "instances_sha256": _sha256(instances_path),
            "fiber_sha256": _sha256(fiber_path),
            "dapi_sha256": _sha256(dapi_path),
        },
        "files": files,
    }
    manifest_path = out_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["files"]["training_manifest.json"] = {
        "sha256": _sha256(manifest_path),
        "bytes": manifest_path.stat().st_size,
    }
    return result


def _is_excluded_pair(row: dict, exclusions: set[tuple[str, str]]) -> bool:
    source_ids = str(row["id"]).split("+")
    return any((str(row["stem"]), source_id) in exclusions for source_id in source_ids)


def _pair_inventory(
    source_dir: Path,
    metadata_pattern: str,
    exclusions: set[tuple[str, str]],
    out_path: Path,
) -> dict:
    records = []
    excluded = []
    for metadata_path in sorted(source_dir.glob(metadata_pattern)):
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            npz_path = source_dir / row["npz"]
            if not npz_path.is_file():
                raise FileNotFoundError(npz_path)
            with np.load(npz_path) as arrays:
                if set(arrays.files) != REQUIRED_PAIR_ARRAYS:
                    raise ValueError(f"{npz_path}: unexpected arrays {arrays.files}")
                shapes = {tuple(arrays[name].shape) for name in REQUIRED_PAIR_ARRAYS}
                if len(shapes) != 1:
                    raise ValueError(f"{npz_path}: pair arrays have inconsistent shapes")
            item = dict(row)
            item["source_npz"] = npz_path.as_posix()
            item["sha256"] = _sha256(npz_path)
            if _is_excluded_pair(row, exclusions):
                excluded.append(item)
            else:
                records.append(item)
    out_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )
    return {
        "eligible": len(records),
        "excluded": len(excluded),
        "manifest": out_path.name,
        "manifest_sha256": _sha256(out_path),
        "excluded_npz": [row["npz"] for row in excluded],
    }


def build_dataset(
    root: Path,
    wells,
    exclude_path: Path,
    out_dir: Path,
    *,
    dry_run: bool = False,
) -> dict:
    exclusions = _excluded(exclude_path)
    plan = _plan(root, wells, exclusions)
    summary = {
        **plan,
        "exclude_file": exclude_path.as_posix(),
        "exclude_sha256": _sha256(exclude_path),
        "dry_run": dry_run,
        "evidence_class": "development_bootstrap_single_operator_proposal_conditioned",
    }
    if dry_run:
        return summary
    if out_dir.exists():
        raise FileExistsError(f"output already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    well_results = {}
    for package, stem in wells:
        well_results[stem] = _export_well(root, package, stem, out_dir / stem, exclusions)
    correction_inventory = _pair_inventory(
        root / "corrections", "*.corrections.jsonl", set(), out_dir / "corrections.jsonl"
    )
    synthetic_inventory = _pair_inventory(
        root / "synth", "*.synth.jsonl", exclusions, out_dir / "synthetic.jsonl"
    )
    summary.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "wells": well_results,
            "correction_pairs": correction_inventory,
            "synthetic_pairs": synthetic_inventory,
            "split_policy": "whole-well leave-one-well-out; never split object crops across folds",
            "usage": {
                "real_complete_masks": "development training targets",
                "synthetic_pairs": "pretraining/augmentation only",
                "correction_pairs": "separate real-error refinement evaluation; not synthetic",
            },
            "limitations": [
                "single human operator",
                "labels are conditioned on ridge-mask proposals",
                "not consensus ground truth",
                "not prospective validation",
            ],
        }
    )
    manifest_path = out_dir / "bootstrap_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "manifest_sha256": _sha256(manifest_path)}, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", required=True, help="binding training exclusion JSON")
    parser.add_argument("--out", default=str(ROOT / "bootstrap_v1"))
    parser.add_argument("--dry-run", action="store_true", help="validate counts without writing")
    args = parser.parse_args(argv)
    result = build_dataset(ROOT, WELLS, Path(args.exclude), Path(args.out), dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
