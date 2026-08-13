"""Round-2 banking: evidence tiering, provenance, and dropping undecided cards."""
import json

import numpy as np
import pytest
import tifffile

from annotation_tools.qc_review.apply_retriage import TIER, apply_retriage, load_exports
from annotation_tools._schema_bridge import InstanceSet


def _package(tmp_path, well="well_a"):
    """A minimal annotation package: three separated proposals in a label raster."""
    pkg = tmp_path / well
    pkg.mkdir(parents=True, exist_ok=True)
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[5:9, 5:40] = 1
    labels[20:24, 5:40] = 2
    labels[40:44, 5:40] = 3
    tifffile.imwrite(pkg / "starting_labels.tif", labels)
    return pkg


def _export(tmp_path, decisions, name="b01.json", reviewer="reviewer_01"):
    payload = {"schema": "retriage.v1", "batch_id": "b01", "reviewer": reviewer,
               "session_started_at": "2026-07-22T00:00:00Z",
               "exported_at": "2026-07-22T01:00:00Z",
               "n_cases": len(decisions), "decisions": decisions}
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _d(well, pid, category, decided="2026-07-22T00:30:00Z"):
    return {"well": well, "id": pid, "category": category, "decided_at": decided,
            "machine_category": "branched_one_myotube", "first_pass": "ambiguous"}


def test_promotions_are_a_separate_tier_not_merged(tmp_path):
    """Round-2 promotions must stay identifiable, never indistinguishable from
    the frozen first-pass 375: they are a second, less conservative pass."""
    pkg = _package(tmp_path)
    exp = _export(tmp_path, {
        "well_a/myotube_0001": _d("well_a", "myotube_0001", "complete"),
        "well_a/myotube_0002": _d("well_a", "myotube_0002", "branched_one_myotube"),
        "well_a/myotube_0003": _d("well_a", "myotube_0003", "fragment_too_short"),
    })
    manifest = apply_retriage([exp], {"well_a": pkg}, tmp_path / "out")

    assert manifest["n_promoted"] == 2 and manifest["n_ignore"] == 1
    instances = InstanceSet.load(tmp_path / "out" / "well_a.round2.instances.json")
    assert len(instances.instances) == 2
    for record in instances.instances:
        assert record.status == "complete"        # canonical vocabulary only
        assert record.reviewed is True
        assert record.source == "qc_retriage_round2"
        assert record.notes.startswith(TIER)
    prov = instances.provenance
    assert prov["tier"] == TIER
    assert "WARNING" in prov and "both with and without" in prov["WARNING"].lower()
    assert prov["filter_hint"] == "source == 'qc_retriage_round2'"


def test_branched_promotion_is_flagged_distinctly(tmp_path):
    pkg = _package(tmp_path)
    exp = _export(tmp_path, {
        "well_a/myotube_0001": _d("well_a", "myotube_0001", "complete"),
        "well_a/myotube_0002": _d("well_a", "myotube_0002", "branched_one_myotube"),
    })
    apply_retriage([exp], {"well_a": pkg}, tmp_path / "out")
    notes = {r.id: r.notes for r in
             InstanceSet.load(tmp_path / "out" / "well_a.round2.instances.json").instances}
    assert notes["myotube_0001"].endswith("complete")
    assert notes["myotube_0002"].endswith("branched_one_myotube")


def test_undecided_cards_are_dropped(tmp_path):
    """decided_at=null means the operator never confirmed it; not evidence."""
    pkg = _package(tmp_path)
    exp = _export(tmp_path, {
        "well_a/myotube_0001": _d("well_a", "myotube_0001", "complete"),
        "well_a/myotube_0002": _d("well_a", "myotube_0002", "complete", decided=None),
    })
    manifest = apply_retriage([exp], {"well_a": pkg}, tmp_path / "out")
    assert manifest["n_reviewed"] == 1
    assert manifest["n_promoted"] == 1
    assert manifest["sources"][0]["n_dropped_undecided"] == 1


def test_ignore_and_background_are_separated(tmp_path):
    """`not_myotube` is an informative negative and stays background; fragments,
    merges and unresolvable cases must be ignored instead."""
    pkg = _package(tmp_path)
    exp = _export(tmp_path, {
        "well_a/myotube_0001": _d("well_a", "myotube_0001", "fragment_too_short"),
        "well_a/myotube_0002": _d("well_a", "myotube_0002", "merged_too_long"),
        "well_a/myotube_0003": _d("well_a", "myotube_0003", "not_myotube"),
    })
    manifest = apply_retriage([exp], {"well_a": pkg}, tmp_path / "out")
    assert manifest["n_ignore"] == 2 and manifest["n_background"] == 1
    classes = json.loads((tmp_path / "out" / "well_a.round2_classes.json").read_text())
    assert [c["id"] for c in classes["background"]] == ["myotube_0003"]
    assert {c["id"] for c in classes["ignore"]} == {"myotube_0001", "myotube_0002"}


def test_export_without_reviewer_is_refused(tmp_path):
    exp = _export(tmp_path, {"well_a/myotube_0001":
                             _d("well_a", "myotube_0001", "complete")}, reviewer="")
    with pytest.raises(ValueError, match="reviewer"):
        load_exports([exp])


def test_conflicting_decisions_across_exports_are_refused(tmp_path):
    a = _export(tmp_path, {"well_a/myotube_0001":
                           _d("well_a", "myotube_0001", "complete")}, name="a.json")
    b = _export(tmp_path, {"well_a/myotube_0001":
                           _d("well_a", "myotube_0001", "not_myotube")}, name="b.json")
    with pytest.raises(ValueError, match="conflicting"):
        load_exports([a, b])


def test_unknown_category_is_refused(tmp_path):
    exp = _export(tmp_path, {"well_a/myotube_0001":
                             _d("well_a", "myotube_0001", "definitely_a_myotube")})
    with pytest.raises(ValueError, match="unknown category"):
        load_exports([exp])
