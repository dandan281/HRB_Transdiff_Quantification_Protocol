"""Banking a linking pass: conflicts, silent decisions, tiering, negatives."""
import json

import numpy as np
import pytest
import tifffile

from annotation_tools.qc_review.apply_links import (
    TIER, apply_links, find_conflicts, load_link_exports, merge_groups, training_pairs)
from annotation_tools._schema_bridge import InstanceSet


def _package(tmp_path, well="well_a"):
    pkg = tmp_path / well
    pkg.mkdir(parents=True, exist_ok=True)
    labels = np.zeros((64, 200), dtype=np.int32)
    labels[30:34, 5:60] = 1
    labels[30:34, 70:120] = 2
    labels[30:34, 130:190] = 3
    tifffile.imwrite(pkg / "starting_labels.tif", labels)
    return pkg


def _dec(well, frag, linked, declined, offered=None, decided="2026-07-23T00:00:00Z",
         no_join=False, unsure=False):
    offered = offered or [{"id": i, "gap_um": 10.0, "cos_fragment": 0.95,
                           "cos_candidate": 0.95} for i in (linked + declined)]
    return {"well": well, "fragment_id": frag, "decided_at": decided,
            "linked_to": linked, "declined": declined, "offered": offered,
            "no_join": no_join, "unsure": unsure}


def _export(tmp_path, decisions, name="l.json", reviewer="reviewer_01"):
    payload = {"schema": "fragment_links.v1", "batch_id": "b01", "reviewer": reviewer,
               "session_started_at": "2026-07-22T00:00:00Z",
               "exported_at": "2026-07-23T00:00:00Z", "gap_um": 40, "cos_min": 0.8,
               "n_cases": len(decisions), "decisions": decisions}
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_declined_candidates_are_kept_as_negatives(tmp_path):
    """A linker trained only on positives learns to join everything."""
    decisions = {"well_a/myotube_0001":
                 _dec("well_a", "myotube_0001", ["myotube_0002"], ["myotube_0003"])}
    rows = training_pairs(decisions, [])
    labels = {r["candidate_id"]: r["label"] for r in rows}
    assert labels == {"myotube_0002": 1, "myotube_0003": 0}
    assert all(r["usable"] for r in rows)


def test_two_sided_disagreement_is_detected_and_excluded(tmp_path):
    """A pair shown from both fragments' sides must not be resolved silently."""
    offered_a = [{"id": "myotube_0002", "gap_um": 15.6, "cos_fragment": 0.99,
                  "cos_candidate": 0.99}]
    offered_b = [{"id": "myotube_0001", "gap_um": 15.6, "cos_fragment": 0.99,
                  "cos_candidate": 0.99}]
    decisions = {
        "w/myotube_0001": _dec("w", "myotube_0001", ["myotube_0002"], [], offered_a),
        "w/myotube_0002": _dec("w", "myotube_0002", [], ["myotube_0001"], offered_b,
                               no_join=True),
    }
    conflicts = find_conflicts(decisions)
    assert len(conflicts) == 1
    assert conflicts[0]["a_says_join"] != conflicts[0]["b_says_join"]

    rows = training_pairs(decisions, conflicts)
    assert all(not r["usable"] for r in rows)
    assert all(r["excluded_reason"] == "two-sided answers disagree" for r in rows)
    # and the contested pair must not be merged
    assert merge_groups(decisions, conflicts) == {}


def test_agreeing_two_sided_pair_is_not_a_conflict():
    offered_a = [{"id": "myotube_0002", "gap_um": 5.0, "cos_fragment": 0.99,
                  "cos_candidate": 0.99}]
    offered_b = [{"id": "myotube_0001", "gap_um": 5.0, "cos_fragment": 0.99,
                  "cos_candidate": 0.99}]
    decisions = {
        "w/myotube_0001": _dec("w", "myotube_0001", ["myotube_0002"], [], offered_a),
        "w/myotube_0002": _dec("w", "myotube_0002", ["myotube_0001"], [], offered_b),
    }
    assert find_conflicts(decisions) == []
    assert merge_groups(decisions, []) == {"w": [["myotube_0001", "myotube_0002"]]}


def test_silent_decision_is_neither_join_nor_negative():
    """Enter with nothing selected is a decision without an assertion."""
    decisions = {"w/myotube_0001": _dec("w", "myotube_0001", [], ["myotube_0002"])}
    decisions["w/myotube_0001"]["no_join"] = False
    rows = training_pairs(decisions, [])
    assert all(not r["usable"] for r in rows)
    assert rows[0]["excluded_reason"] == "decided without an explicit selection"


def test_explicit_no_join_is_a_usable_negative():
    decisions = {"w/myotube_0001": _dec("w", "myotube_0001", [], ["myotube_0002"],
                                        no_join=True)}
    rows = training_pairs(decisions, [])
    assert rows[0]["usable"] and rows[0]["label"] == 0


def test_unsure_is_excluded():
    decisions = {"w/myotube_0001": _dec("w", "myotube_0001", [], ["myotube_0002"],
                                        unsure=True)}
    rows = training_pairs(decisions, [])
    assert not rows[0]["usable"]


def test_chains_of_three_merge_transitively():
    decisions = {
        "w/myotube_0001": _dec("w", "myotube_0001", ["myotube_0002"], []),
        "w/myotube_0002": _dec("w", "myotube_0002", ["myotube_0003"], []),
    }
    groups = merge_groups(decisions, [])
    assert groups == {"w": [["myotube_0001", "myotube_0002", "myotube_0003"]]}


def test_merged_masks_are_ambiguous_not_complete(tmp_path):
    """Confirming two pieces are the same fibre is NOT a claim that the union is
    a whole, measurable myotube."""
    pkg = _package(tmp_path)
    exp = _export(tmp_path, {
        "well_a/myotube_0001": _dec("well_a", "myotube_0001", ["myotube_0002"], []),
    })
    manifest = apply_links([exp], {"well_a": pkg}, tmp_path / "out")
    assert manifest["merged_objects"] == 1

    instances = InstanceSet.load(tmp_path / "out" / "well_a.merged.instances.json")
    record = instances.instances[0]
    assert record.status == "ambiguous", "a merged chain is not certified complete"
    assert record.reviewed is False
    assert record.source == "qc_link_round2"
    assert record.notes.startswith(TIER)
    assert "not a claim" in instances.provenance["WARNING"]

    # the merged mask is the union of both proposals
    _, mask = next(iter(instances.masks()))
    assert mask[32, 10] and mask[32, 100], "union must cover both fragments"


def test_export_without_reviewer_is_refused(tmp_path):
    exp = _export(tmp_path, {"w/myotube_0001": _dec("w", "myotube_0001", [], [])},
                  reviewer="")
    with pytest.raises(ValueError, match="reviewer"):
        load_link_exports([exp])


def test_undecided_fragments_are_dropped(tmp_path):
    exp = _export(tmp_path, {
        "w/myotube_0001": _dec("w", "myotube_0001", ["myotube_0002"], [],
                               decided=None),
    })
    decisions, sources = load_link_exports([exp])
    assert decisions == {}
    assert sources[0]["n_dropped_undecided"] == 1
