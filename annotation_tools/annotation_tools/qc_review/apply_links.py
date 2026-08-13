"""Bank a fragment-linking pass: merged masks plus the linker's training pairs.

Two products, and the second is the more valuable one:

1. **Linker training pairs** -- every candidate the operator was *offered*, with
   its geometry and the answer. Declined candidates are kept as negatives; a set
   of positives alone would teach a model to join everything.
2. **Merged masks** -- the union of each confirmed chain of fragments.

Why merged masks are NOT marked `complete`
------------------------------------------
The operator answered "these two pieces are the same fibre". That is **not** the
same as "the union is a whole, fully measurable myotube" -- a chain of two
fragments may still be missing further pieces beyond the ones offered. Under the
plan only reviewed `complete` full-area instances may support length or width, so
merged objects are exported as `ambiguous` (schema-valid, non-authoritative) and
must be confirmed complete in their own pass before they can train measurements.

Conflicts and silent decisions
------------------------------
Some pairs are offered from both fragments' sides. When the two answers disagree
the pair is **excluded from the positives and recorded**, never silently resolved
by whichever direction happened to be processed first.

A card that was advanced with Enter but had nothing selected is *decided* but
carries no assertion. It is neither a join nor a confirmed negative, so it is
excluded from both and reported.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

TIER = "link_round2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_link_exports(paths: list[str | Path]) -> tuple[dict, list[dict]]:
    decisions: dict[str, dict] = {}
    sources = []
    for path in paths:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "fragment_links.v1":
            raise ValueError(f"{path}: not a fragment_links.v1 export")
        if not payload.get("reviewer"):
            raise ValueError(f"{path}: export carries no reviewer; not usable as evidence")
        kept = dropped = 0
        for uid, decision in payload["decisions"].items():
            if not decision.get("decided_at"):
                dropped += 1
                continue
            decisions[uid] = decision
            kept += 1
        sources.append({
            "file": str(path), "sha256": _sha256(path),
            "batch_id": payload.get("batch_id"), "reviewer": payload["reviewer"],
            "session_started_at": payload.get("session_started_at"),
            "exported_at": payload.get("exported_at"),
            "gap_um": payload.get("gap_um"), "cos_min": payload.get("cos_min"),
            "n_cases": payload.get("n_cases"),
            "n_explicitly_decided": kept, "n_dropped_undecided": dropped,
        })
    return decisions, sources


def find_conflicts(decisions: dict[str, dict]) -> list[dict]:
    """Pairs offered from both sides where the two answers disagree."""
    by_fragment = {(v["well"], v["fragment_id"]): v for v in decisions.values()}
    conflicts, seen = [], set()
    for (well, fragment), decision in by_fragment.items():
        for offered in decision["offered"]:
            other = by_fragment.get((well, offered["id"]))
            if other is None:
                continue
            if not any(o["id"] == fragment for o in other["offered"]):
                continue                       # not a true two-sided presentation
            key = (well,) + tuple(sorted([fragment, offered["id"]]))
            if key in seen:
                continue
            seen.add(key)
            forward = offered["id"] in decision["linked_to"]
            back = fragment in other["linked_to"]
            if forward != back:
                conflicts.append({
                    "well": well, "a": fragment, "b": offered["id"],
                    "a_says_join": forward, "b_says_join": back,
                    "gap_um": offered["gap_um"],
                })
    return conflicts


def training_pairs(decisions: dict[str, dict], conflicts: list[dict]) -> list[dict]:
    """One row per offered candidate: geometry plus the operator's answer."""
    blocked = {(c["well"],) + tuple(sorted([c["a"], c["b"]])) for c in conflicts}
    rows = []
    for decision in decisions.values():
        well = decision["well"]
        fragment = decision["fragment_id"]
        silent = (not decision["linked_to"] and not decision["no_join"]
                  and not decision["unsure"])
        for offered in decision["offered"]:
            key = (well,) + tuple(sorted([fragment, offered["id"]]))
            joined = offered["id"] in decision["linked_to"]
            excluded = None
            if key in blocked:
                excluded = "two-sided answers disagree"
            elif decision["unsure"]:
                excluded = "operator marked unsure"
            elif silent:
                excluded = "decided without an explicit selection"
            rows.append({
                "well": well, "fragment_id": fragment, "candidate_id": offered["id"],
                "gap_um": offered["gap_um"],
                "cos_fragment": offered["cos_fragment"],
                "cos_candidate": offered["cos_candidate"],
                "min_cos": min(offered["cos_fragment"], offered["cos_candidate"]),
                "label": int(joined),
                "usable": excluded is None,
                "excluded_reason": excluded,
                "decided_at": decision["decided_at"],
            })
    return rows


def merge_groups(decisions: dict[str, dict], conflicts: list[dict]) -> dict[str, list[list[str]]]:
    """Union-find over confirmed links -> chains of proposal ids, per well."""
    blocked = {(c["well"],) + tuple(sorted([c["a"], c["b"]])) for c in conflicts}
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for decision in decisions.values():
        well = decision["well"]
        for candidate in decision["linked_to"]:
            key = (well,) + tuple(sorted([decision["fragment_id"], candidate]))
            if key in blocked:
                continue
            union((well, decision["fragment_id"]), (well, candidate))

    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    out: dict[str, list[list[str]]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        well = members[0][0]
        out.setdefault(well, []).append(sorted(m[1] for m in members))
    return {w: sorted(v) for w, v in out.items()}


def apply_links(export_paths: list[str | Path], packages: dict[str, Path],
                out_dir: str | Path) -> dict:
    import tifffile
    from scipy import ndimage as ndi

    from .._schema_bridge import (InstanceRecord, InstanceSet,
                                  encode_sparse_positions)

    decisions, sources = load_link_exports(export_paths)
    if not decisions:
        raise ValueError("no explicitly decided fragments in the supplied exports")
    reviewer = sources[0]["reviewer"]
    conflicts = find_conflicts(decisions)
    pairs = training_pairs(decisions, conflicts)
    groups = merge_groups(decisions, conflicts)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "link_pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in pairs),
                          encoding="utf-8")

    wells_out = {}
    for well in sorted(groups):
        package = Path(packages[well])
        labels = tifffile.imread(package / "starting_labels.tif")
        height, width = labels.shape
        slices = ndi.find_objects(labels)
        records = []
        for index, members in enumerate(groups[well], start=1):
            mask = np.zeros((height, width), dtype=bool)
            for member in members:
                label_id = int(member.split("_")[-1])
                if label_id > len(slices) or slices[label_id - 1] is None:
                    raise ValueError(f"{well}: {member} not in starting_labels")
                sl = slices[label_id - 1]
                mask[sl] |= labels[sl] == label_id
            rows, cols = np.nonzero(mask)
            positions = rows.astype(np.int64) + cols.astype(np.int64) * height
            records.append(InstanceRecord(
                id=f"linked_{index:04d}", status="ambiguous", reviewed=False,
                source="qc_link_round2",
                notes=f"{TIER}:merged_from={'+'.join(members)}",
                rle=encode_sparse_positions((height, width), positions)))
        path = out_dir / f"{well}.merged.instances.json"
        InstanceSet((height, width), well, records, provenance={
            "tier": TIER, "reviewer": reviewer,
            "evidence_class": "single_operator_confirmed_fragment_joins",
            "WARNING": ("The operator confirmed these pieces are the SAME fibre. "
                        "That is not a claim that the union is a whole, fully "
                        "measurable myotube -- further pieces may exist beyond the "
                        "offered candidates. Status is deliberately `ambiguous`; "
                        "these must pass their own completeness review before they "
                        "may support length or width."),
            "sources": sources,
        }).save(path)
        wells_out[well] = {"instances": str(path), "instances_sha256": _sha256(path),
                           "n_merged_objects": len(records),
                           "n_fragments_absorbed": sum(len(g) for g in groups[well])}

    usable = [r for r in pairs if r["usable"]]
    manifest = {
        "schema": "link_round2.v1", "tier": TIER,
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "reviewer": reviewer, "sources": sources,
        "n_fragments_decided": len(decisions),
        "n_candidate_pairs": len(pairs),
        "n_positive": sum(r["label"] for r in usable),
        "n_negative": sum(1 - r["label"] for r in usable),
        "n_excluded_pairs": len(pairs) - len(usable),
        "excluded_reasons": {
            reason: sum(1 for r in pairs if r["excluded_reason"] == reason)
            for reason in {r["excluded_reason"] for r in pairs if r["excluded_reason"]}
        },
        "conflicts": conflicts,
        "merged_objects": sum(w["n_merged_objects"] for w in wells_out.values()),
        "fragments_absorbed": sum(w["n_fragments_absorbed"] for w in wells_out.values()),
        "training_pairs": str(pairs_path), "training_pairs_sha256": _sha256(pairs_path),
        "wells": wells_out,
        "limitations": [
            "single operator; not consensus and not inter-rater agreement",
            "merged objects are NOT certified complete; status is ambiguous",
            "candidates were only offered within the gap/collinearity window, so a "
            "join beyond that window could not be expressed",
        ],
    }
    manifest_path = out_dir / "links_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
