"""Turn re-triage exports into a versioned round-2 annotation artifact.

Evidence tiering (binding)
--------------------------
Round-2 promotions are **not** merged indistinguishably into the frozen 375.
They come from a second, deliberately less conservative pass over cases the same
operator once marked `ambiguous`, so they are weaker evidence than the first-pass
set and are recorded as a separate tier:

* canonical ``status`` stays ``"complete"`` (the schema's vocabulary is
  ``complete`` / ``border_truncated`` / ``occluded`` / ``ambiguous``; inventing
  ``complete_round2`` would fail validation);
* ``source`` is ``qc_retriage_round2`` and ``notes`` carries the tier, so any
  consumer can split the tiers with a filter rather than a guess;
* set-level ``provenance`` records the tier, the reviewer, the export hash, and
  an explicit instruction to report metrics with and without this tier.

Only explicitly decided cases are used. A card the operator never confirmed
exports ``decided_at: null`` and is dropped -- it is not evidence.

Category -> role
----------------
================================  ================================================
category                          role
================================  ================================================
`complete`                        round-2 training target
`branched_one_myotube`            round-2 training target (flagged `branched`)
`fragment_too_short`              ignore (an under-traced piece of a longer fibre)
`merged_too_long`                 ignore (spans two or more fibres)
`unresolvable`                    ignore
`not_myotube`                     background -- an informative negative, kept
================================  ================================================
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

from .retriage import CATEGORIES, PROMOTES_TO_TARGET

IGNORE_CATEGORIES = ("fragment_too_short", "merged_too_long", "unresolvable")
BACKGROUND_CATEGORIES = ("not_myotube",)
TIER = "retriage_round2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_exports(paths: list[str | Path]) -> tuple[dict, list[dict]]:
    """Merge one or more batch exports, keeping only explicit decisions."""
    decisions: dict[str, dict] = {}
    sources = []
    for path in paths:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "retriage.v1":
            raise ValueError(f"{path}: not a retriage.v1 export")
        if not payload.get("reviewer"):
            raise ValueError(f"{path}: export carries no reviewer; not usable as evidence")
        kept = dropped = 0
        for uid, decision in payload["decisions"].items():
            if not decision.get("decided_at"):
                dropped += 1
                continue
            if decision["category"] not in CATEGORIES:
                raise ValueError(f"{path}: unknown category {decision['category']!r}")
            if uid in decisions and decisions[uid]["category"] != decision["category"]:
                raise ValueError(f"conflicting decisions for {uid} across exports")
            decisions[uid] = decision
            kept += 1
        sources.append({
            "file": str(path), "sha256": _sha256(path),
            "batch_id": payload.get("batch_id"), "reviewer": payload["reviewer"],
            "session_started_at": payload.get("session_started_at"),
            "exported_at": payload.get("exported_at"),
            "n_cases": payload.get("n_cases"),
            "n_explicitly_decided": kept, "n_dropped_undecided": dropped,
        })
    return decisions, sources


def apply_retriage(export_paths: list[str | Path], packages: dict[str, Path],
                   out_dir: str | Path) -> dict:
    """Write per-well round-2 instance sets and ignore-class records.

    ``packages`` maps well stem -> annotation package directory (which holds
    ``starting_labels.tif``, the proposal raster the categories refer to).
    """
    import tifffile
    from scipy import ndimage as ndi

    from .._schema_bridge import (InstanceRecord, InstanceSet,
                                  encode_sparse_positions)

    decisions, sources = load_exports(export_paths)
    if not decisions:
        raise ValueError("no explicitly decided cases in the supplied exports")
    reviewer = sources[0]["reviewer"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_well: dict[str, list[tuple[str, dict]]] = {}
    for uid, decision in decisions.items():
        by_well.setdefault(decision["well"], []).append((uid, decision))

    wells_out = {}
    totals = {c: 0 for c in CATEGORIES}
    for well in sorted(by_well):
        package = Path(packages[well])
        labels = tifffile.imread(package / "starting_labels.tif")
        height, width = labels.shape
        slices = ndi.find_objects(labels)

        promoted: list[InstanceRecord] = []
        ignore_ids: list[dict] = []
        background_ids: list[dict] = []
        for uid, decision in sorted(by_well[well]):
            category = decision["category"]
            totals[category] += 1
            label_id = int(decision["id"].split("_")[-1])
            if label_id > len(slices) or slices[label_id - 1] is None:
                raise ValueError(f"{well}: proposal {decision['id']} not in starting_labels")
            record = {"id": decision["id"], "category": category,
                      "decided_at": decision["decided_at"],
                      "machine_category": decision.get("machine_category")}
            if category in PROMOTES_TO_TARGET:
                sl = slices[label_id - 1]
                mask = labels[sl] == label_id
                rows, cols = np.nonzero(mask)
                positions = ((rows + sl[0].start).astype(np.int64)
                             + (cols + sl[1].start).astype(np.int64) * height)
                promoted.append(InstanceRecord(
                    id=decision["id"], status="complete", reviewed=True,
                    source="qc_retriage_round2",
                    notes=f"{TIER}:{category}",
                    rle=encode_sparse_positions((height, width), positions)))
            elif category in IGNORE_CATEGORIES:
                ignore_ids.append(record)
            elif category in BACKGROUND_CATEGORIES:
                background_ids.append(record)

        instance_path = out_dir / f"{well}.round2.instances.json"
        InstanceSet(
            (height, width), well, promoted,
            provenance={
                "tier": TIER,
                "evidence_class": "single_operator_second_pass_on_first_pass_ambiguous",
                "reviewer": reviewer,
                "derived_from": "first-pass `ambiguous` proposals re-triaged into six categories",
                "exports": sources,
                "WARNING": ("Weaker evidence than the frozen first-pass 375. These "
                            "cases were originally marked ambiguous under a "
                            "conservative rule and re-read in a second, less "
                            "conservative pass by the same operator. Never merge "
                            "indistinguishably with the first-pass set; report "
                            "metrics BOTH with and without this tier."),
                "filter_hint": "source == 'qc_retriage_round2'",
            }).save(instance_path)

        classes_path = out_dir / f"{well}.round2_classes.json"
        classes_path.write_text(json.dumps({
            "well": well, "tier": TIER, "reviewer": reviewer,
            "promoted": [r.id for r in promoted],
            "ignore": ignore_ids,
            "background": background_ids,
            "policy": {
                "promoted": "round-2 training target (separate tier)",
                "ignore": "must contribute no loss; not background",
                "background": "operator asserted not-a-myotube; informative negative",
            },
        }, indent=2), encoding="utf-8")

        wells_out[well] = {
            "instances": str(instance_path), "instances_sha256": _sha256(instance_path),
            "classes": str(classes_path),
            "n_promoted": len(promoted), "n_ignore": len(ignore_ids),
            "n_background": len(background_ids),
            "n_reviewed": len(by_well[well]),
        }

    manifest = {
        "schema": "retriage_round2.v1",
        "tier": TIER,
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "reviewer": reviewer,
        "sources": sources,
        "n_reviewed": len(decisions),
        "n_promoted": sum(w["n_promoted"] for w in wells_out.values()),
        "n_ignore": sum(w["n_ignore"] for w in wells_out.values()),
        "n_background": sum(w["n_background"] for w in wells_out.values()),
        "category_totals": totals,
        "wells": wells_out,
        "evidence_class": "single_operator_second_pass_on_first_pass_ambiguous",
        "limitations": [
            "second pass by the same operator on cases they originally marked ambiguous",
            "less conservative than the first-pass rule that produced the frozen 375",
            "single operator; not consensus, not inter-rater agreement",
            "report model metrics both with and without this tier",
        ],
    }
    manifest_path = out_dir / "round2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
