"""Re-triage pre-classifier: calibration guards and queue ordering."""
from pathlib import Path

import numpy as np
import pytest

from annotation_tools.qc_review.retriage import (
    CATEGORIES, LONG_UM, PROMOTES_TO_TARGET, SHORT_UM, SkeletonEvidence, band_yield,
    batch, classify, queue_priority, select_ambiguous, skeleton_evidence)


def _f(length=150.0, aspect=12.0, solidity=0.6, width=11.0, border=0.0):
    return {"length_um": length, "aspect": aspect, "solidity": solidity,
            "width_um": width, "touches_border": border}


def test_every_category_is_declared():
    cat, _ = classify(_f(), SkeletonEvidence())
    assert cat in CATEGORIES
    assert set(PROMOTES_TO_TARGET) <= set(CATEGORIES)


def test_clean_single_fibre_is_complete():
    cat, why = classify(_f(), SkeletonEvidence(n_junctions=0, traced_fiber_count=1))
    assert cat == "complete" and "clean single fibre" in why


def test_junctions_alone_mean_branched_not_merged():
    """A branched object is one myotube; it must stay promotable."""
    cat, _ = classify(_f(length=150.0), SkeletonEvidence(n_junctions=3, traced_fiber_count=1))
    assert cat == "branched_one_myotube"
    assert cat in PROMOTES_TO_TARGET


def test_multi_traced_alone_does_not_mean_merged():
    """Regression: `traced_fiber_count >= 2` fires on 39% of KNOWN-GOOD accepted
    masks, because the crossing tracer fragments real single myotubes. A first
    version keyed merging off it alone and misfiled 39% of accepted masks. A
    normal-length, reasonably solid object must stay promotable no matter how
    many fibres the tracer thinks it sees."""
    cat, _ = classify(_f(length=150.0, solidity=0.6),
                      SkeletonEvidence(n_junctions=4, traced_fiber_count=6))
    assert cat in PROMOTES_TO_TARGET, "tracer fragmentation must not imply a merge"


def test_merged_requires_long_and_multi_traced_and_unsolid():
    long_f = _f(length=LONG_UM + 50, solidity=0.30)
    assert classify(long_f, SkeletonEvidence(traced_fiber_count=3))[0] == "merged_too_long"
    # each condition alone is not enough
    assert classify(long_f, SkeletonEvidence(traced_fiber_count=1))[0] != "merged_too_long"
    assert classify(_f(length=LONG_UM + 50, solidity=0.8),
                    SkeletonEvidence(traced_fiber_count=3))[0] != "merged_too_long"
    assert classify(_f(length=100.0, solidity=0.30),
                    SkeletonEvidence(traced_fiber_count=3))[0] != "merged_too_long"


def test_short_is_a_fragment_unless_it_touches_the_border():
    assert classify(_f(length=SHORT_UM - 10), SkeletonEvidence())[0] == "fragment_too_short"
    other, _ = classify(_f(length=SHORT_UM - 10, border=1.0), SkeletonEvidence())
    assert other != "fragment_too_short", "a border case is truncated, not a fragment"


def test_squat_wide_object_is_not_a_myotube():
    assert classify(_f(aspect=2.0, width=40.0), SkeletonEvidence())[0] == "not_myotube"


def test_unresolvable_needs_tangle_and_length():
    tangled = SkeletonEvidence(n_junctions=5, traced_fiber_count=3)
    assert classify(_f(length=LONG_UM + 100, solidity=0.2), tangled)[0] == "unresolvable"
    # a short knotty object is not written off
    assert classify(_f(length=90.0, solidity=0.2), tangled)[0] != "unresolvable"


def test_classification_always_explains_itself():
    """The operator must see why, so a wrong suggestion is easy to overrule."""
    for feats, ev in ((_f(), SkeletonEvidence()),
                      (_f(length=30.0), SkeletonEvidence()),
                      (_f(aspect=2.0, width=40.0), SkeletonEvidence())):
        _, why = classify(feats, ev)
        assert why and len(why) > 10


# ------------------------------------------------------------------ queue order


def test_band_yield_rises_with_length():
    assert band_yield(30) < band_yield(100) < band_yield(200) < band_yield(400)


def test_promotable_cases_are_queued_before_write_offs():
    feats = _f(length=200.0)
    assert queue_priority(feats, "complete") > queue_priority(feats, "not_myotube")
    assert queue_priority(feats, "branched_one_myotube") > queue_priority(feats, "unresolvable")


def test_longer_cases_are_queued_first_within_a_category():
    assert (queue_priority(_f(length=300.0), "complete")
            > queue_priority(_f(length=40.0), "complete"))


def test_border_cases_are_deprioritised():
    """A border-touching object can never become a complete target."""
    assert (queue_priority(_f(length=200.0, border=1.0), "complete")
            < queue_priority(_f(length=200.0, border=0.0), "complete"))


def test_select_ambiguous_picks_only_first_pass_ambiguous():
    decisions = {"myotube_0003": {"action": "ambiguous"},
                 "myotube_0001": {"action": "accept"},
                 "myotube_0002": {"action": "ambiguous"},
                 "myotube_0004": {"action": "reject"}}
    assert select_ambiguous(decisions) == ["myotube_0002", "myotube_0003"]


def test_batching_is_bounded_and_lossless():
    cases = [{"i": i} for i in range(250)]
    batches = batch(cases, 120)
    assert [len(b) for b in batches] == [120, 120, 10]
    assert sum(len(b) for b in batches) == len(cases)
    with pytest.raises(ValueError):
        batch(cases, 0)


# --------------------------------------------------------------- skeleton evidence


def test_skeleton_evidence_on_a_straight_fibre():
    mask = np.zeros((60, 60), dtype=bool)
    mask[28:32, 5:55] = True
    ev = skeleton_evidence(mask)
    assert ev.skeleton_px > 0
    assert ev.n_junctions == 0, "a straight fibre has no junction"


def test_skeleton_evidence_on_a_branch():
    mask = np.zeros((80, 80), dtype=bool)
    mask[38:42, 5:75] = True      # spine
    mask[10:40, 38:42] = True     # long branch (survives spur pruning)
    ev = skeleton_evidence(mask)
    assert ev.n_junctions >= 1, "a T-branch must register a junction"


def test_skeleton_evidence_handles_tiny_masks():
    ev = skeleton_evidence(np.zeros((10, 10), dtype=bool))
    assert ev.n_junctions == 0 and ev.skeleton_px == 0


# ------------------------------------------------------- page keying / provenance


def _case(well, pid, cat="complete"):
    return {"id": pid, "well": well, "uid": f"{well}/{pid}",
            "dom_id": f"{well}__{pid}", "features": {"length_um": 100.0, "aspect": 9.0,
                                                     "solidity": 0.6},
            "machine_category": cat, "machine_why": "because", "priority": 0.5,
            "thumb": "data:image/png;base64,AA", "edit_img": "data:image/jpeg;base64,AA"}


def test_page_keys_by_uid_not_id(tmp_path):
    """Regression: proposal ids repeat across wells (384 collisions in the real
    839-case queue). Keying page state or the export by bare `id` would make two
    wells' cards share one decision and silently overwrite each other."""
    from annotation_tools.qc_review.retriage_page import build_retriage_page

    cases = [_case("well_a", "myotube_0161"), _case("well_b", "myotube_0161")]
    out = build_retriage_page(cases, tmp_path / "b.html", batch_id="b01",
                              reviewer="reviewer_01",
                              session_started_at="2026-07-22T00:00:00Z")
    import json as _json

    html = Path(out).read_text(encoding="utf-8")
    start = html.index("const DATA = ") + len("const DATA = ")
    payload = _json.loads(html[start:html.index(";</script>", start)])

    uids = [c["uid"] for c in payload["cases"]]
    dom_ids = [c["dom_id"] for c in payload["cases"]]
    assert uids == ["well_a/myotube_0161", "well_b/myotube_0161"]
    assert len(set(uids)) == 2, "colliding ids must yield distinct uids"
    assert len(set(dom_ids)) == 2, "colliding ids must yield distinct DOM ids"

    # Cards are rendered at runtime, so assert the JS keys off uid, never bare id.
    assert "state[c.uid]" in html
    assert "state[c.id]" not in html
    assert "payload.decisions[c.uid]" in html, "export must be keyed by uid"
    assert "card_${c.dom_id}" in html


def test_page_requires_a_reviewer(tmp_path):
    """An export without an identified reviewer cannot be used as evidence --
    this is exactly what failed G-SO1 provenance the first time."""
    from annotation_tools.qc_review.retriage_page import build_retriage_page

    with pytest.raises(ValueError, match="reviewer"):
        build_retriage_page([_case("w", "myotube_0001")], tmp_path / "b.html",
                            batch_id="b01", reviewer="",
                            session_started_at="2026-07-22T00:00:00Z")


def test_page_carries_provenance_and_completion_guard(tmp_path):
    from annotation_tools.qc_review.retriage_page import build_retriage_page

    out = build_retriage_page([_case("w", "myotube_0001")], tmp_path / "b.html",
                              batch_id="b01", reviewer="reviewer_01",
                              session_started_at="2026-07-22T00:00:00Z")
    html = Path(out).read_text(encoding="utf-8")
    for token in ("session_started_at", "exported_at", "decided_at",
                  "n_explicitly_decided", "reviewer_01"):
        assert token in html, f"missing provenance/guard field {token}"
    # untouched cards must be reported, not silently exported as decided
    assert "still untouched" in html
