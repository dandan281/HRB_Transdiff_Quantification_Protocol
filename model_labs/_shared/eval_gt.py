"""The evaluation ground truth every T02 candidate is scored against.

One definition, so T03 compares candidates on the *same* masks rather than on two
independent re-derivations that happen to agree: each well's reviewed `complete`
instances, minus the binding `training_exclude.json` ids -- 375 masks in total,
matching `bootstrap_manifest.json`.

The exclusions matter and are easy to miss: `19_B06/myotube_0377` and
`22_B03/myotube_0321` still read `reviewed`/`complete` in the source
`*.qc.instances.json`, because the operator's blind repeat re-classified them
afterwards. A scorer that simply filters on status would silently evaluate against
masks the plan forbids using.

This lives in `_shared` rather than in `classical/` because the classical
candidate is sealed at `model_labs/classical/_runs/v1/` and its source should not
be edited to serve a later candidate; importing it is also impossible from the
`pm-omnipose` environment, which has no `skan`. Equality with the sealed run is
therefore *verified* by test (`test_eval_gt_matches_sealed_classical_run`) rather
than assumed by sharing an import.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_eval_gt(manifest: dict, well: str, out_dir: Path) -> dict:
    """Reviewed-complete GT minus the binding exclusions, sealed and hashed."""
    from precision_myotube.schema import InstanceSet

    info = manifest["per_well"][well]
    source = ROOT / info["source_instances"]
    actual = sha256_file(source)
    if actual != info["source_instances_sha256"]:
        raise RuntimeError(f"{well}: source instances hash mismatch "
                           f"(expected {info['source_instances_sha256']}, got {actual})")
    excluded = set(info.get("excluded", []))
    full = InstanceSet.load(source)
    kept = [r for r in full.instances
            if r.reviewed and r.status == "complete" and r.id not in excluded]
    if len(kept) != info["complete_kept"]:
        raise RuntimeError(f"{well}: expected {info['complete_kept']} evaluation masks, "
                           f"built {len(kept)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{well}.eval_gt.instances.json"
    InstanceSet(tuple(full.image_shape), full.image_id, kept,
                provenance={
                    "derived_from": info["source_instances"],
                    "derived_from_sha256": actual,
                    "excluded_ids": sorted(excluded),
                    "filter": "reviewed AND status==complete AND id not in training_exclude",
                    "evidence_class": "single_operator_proposal_conditioned",
                }).save(path)
    return {"path": path, "n_gt": len(kept), "sha256": sha256_file(path),
            "image_id": full.image_id, "image_shape": tuple(full.image_shape),
            "excluded_ids": sorted(excluded)}


def load_bootstrap_manifest(manifest_path: str | Path | None = None) -> dict:
    path = Path(manifest_path) if manifest_path else (
        ROOT / "PrecisionMyotube/annotation_work/bootstrap_v1/bootstrap_manifest.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))
