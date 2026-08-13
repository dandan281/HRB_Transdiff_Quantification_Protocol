"""Blind-repeat construction, instrument telemetry, and intra-rater agreement.

Round 1 could not distinguish "the reviewer was not discriminating" from "the
over-merge rule is blind to most merges". A blind repeat separates them, but only if
the second pass shows the *same objects* in a *different order* under *different*
uids, and only if the pairing is by identity rather than by position. Those are the
properties pinned here.
"""
from __future__ import annotations

import json

import pytest

from annotation_tools.qc_review.cli import _intra_rater, _note_discipline, _telemetry_summary
from annotation_tools.qc_review.over_merge_page import (
    INSTRUMENT_VERSION, NOTE_REQUIRED_FOR, build_over_merge_page, PANELS)


def _case(uid="om_001", **extra):
    return {"uid": uid, "n_fragments": 2, "gaps_um": [12.0],
            "panels": {p: f"data:image/jpeg;base64,{p}" for p in PANELS}, **extra}


# ------------------------------------------------------- instrument: note discipline


def test_a_different_myotubes_call_requires_a_reason():
    assert NOTE_REQUIRED_FOR == ["different_myotubes"], (
        "the call that costs the linker something is the one that must be justified")


def test_the_page_blocks_completion_until_the_reason_exists(tmp_path):
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b",
                               reviewer="reviewer_01",
                               session_started_at="2026-07-30T00:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    assert "const needsNote" in html
    assert "isComplete" in html
    assert "NEEDS A REASON" in html, "the pending state must be visible on screen"
    assert "awaiting a reason" in html, "and counted in the header tally"
    assert "note_missing: needsNote(s)" in html, "and shipped in the export"


def test_the_page_focuses_the_reason_field_at_the_moment_of_the_call(tmp_path):
    """Asking at export time is too late -- round 1 exported seven unexplained calls."""
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b",
                               reviewer="reviewer_01",
                               session_started_at="2026-07-30T00:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    assert "if(needsNote(s)) document.getElementById('note').focus();" in html


# ------------------------------------------------------- instrument: telemetry


def test_the_page_records_dwell_and_ships_it(tmp_path):
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b",
                               reviewer="reviewer_01",
                               session_started_at="2026-07-30T00:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    for token in ("panel_dwell_ms", "panel_views", "ms_on_case_at_decision",
                  "reference_panel_seen_before_decision", "function flush()",
                  "function retarget()"):
        assert token in html, token
    assert INSTRUMENT_VERSION in html


def test_away_time_is_not_counted_as_looking(tmp_path):
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b",
                               reviewer="reviewer_01",
                               session_started_at="2026-07-30T00:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    assert "addEventListener('blur', flush)" in html
    assert "addEventListener('focus'" in html


def test_telemetry_summary_reports_unavailable_rather_than_zero():
    """A round with no telemetry must say so, not imply the reviewer looked at nothing."""
    rows = [{"uid": "a", "decision": "same_myotube", "note": ""}]
    out = _telemetry_summary(rows, {"a": {"decision": "same_myotube"}})
    assert out["available"] is False
    assert "no dwell data" in out["note"]


def test_telemetry_summary_flags_decisions_made_without_the_reference_panel():
    rows = [{"uid": "a", "decision": "different_myotubes", "note": "x"},
            {"uid": "b", "decision": "same_myotube", "note": ""}]
    decisions = {
        "a": {"ms_on_case": 4000, "ms_on_case_at_decision": 4000,
              "reference_panel_seen_before_decision": False,
              "panel_dwell_ms": {"fragments": 4000}},
        "b": {"ms_on_case": 30000, "ms_on_case_at_decision": 30000,
              "reference_panel_seen_before_decision": True,
              "panel_dwell_ms": {"references": 20000, "fragments": 10000}}}
    out = _telemetry_summary(rows, decisions)
    assert out["available"] is True and out["n"] == 2
    assert out["decided_without_viewing_reference_masks"] == 1
    assert out["seconds_to_decision"]["n_under_5s"] == 1
    assert out["total_dwell_seconds_by_panel"]["references"] == 20.0


def test_note_discipline_distinguishes_enforced_from_legacy():
    rows = [{"uid": "a", "decision": "different_myotubes", "note": ""}]
    legacy = _note_discipline(rows, {"a": {"decision": "different_myotubes"}})
    assert legacy["enforced"] is False and legacy["n_missing"] == 1
    enforced = _note_discipline(rows, {"a": {"note_required": True, "note_missing": True}})
    assert enforced["enforced"] is True and enforced["missing"] == ["a"]


# ------------------------------------------------------- intra-rater agreement


def _keys(pairs):
    """pairs: [(kind, first_decision, second_decision), ...]"""
    first = {"batch_id": "r1", "key": {}}
    first_export = {"decisions": {}}
    second_key, second_dec = {}, {}
    for n, (kind, d1, d2) in enumerate(pairs, start=1):
        u1, u2 = f"r1_{n:03d}", f"r2_{n:03d}"
        first["key"][u1] = {"well": "w", "merged_label": 100 + n, "case_kind": kind}
        first_export["decisions"][u1] = {"decision": d1}
        second_key[u2] = {"well": "w", "merged_label": 100 + n, "case_kind": kind,
                          "first_pass_uid": u1}
        second_dec[u2] = {"decision": d2}
    return second_key, second_dec, first, first_export


def test_perfect_self_agreement():
    sk, sd, fk, fe = _keys([("over_merge", "different_myotubes", "different_myotubes"),
                            ("control", "same_myotube", "same_myotube"),
                            ("control", "same_myotube", "same_myotube")])
    out = _intra_rater(sk, sd, fk, fe)
    assert out["agreement"] == 1.0 and out["n_flips"] == 0
    assert out["cohens_kappa"] == 1.0


def test_agreement_at_chance_gives_kappa_near_zero():
    """The round-1 worry made quantitative: alternating verdicts on the same objects."""
    pairs = [("control", "same_myotube", "different_myotubes"),
             ("control", "different_myotubes", "same_myotube"),
             ("control", "same_myotube", "different_myotubes"),
             ("control", "different_myotubes", "same_myotube")]
    out = _intra_rater(*_keys(pairs))
    assert out["agreement"] == 0.0
    assert out["cohens_kappa"] is not None and out["cohens_kappa"] < 0


def test_flips_are_listed_with_the_object_identity():
    sk, sd, fk, fe = _keys([("over_merge", "different_myotubes", "same_myotube"),
                            ("control", "same_myotube", "same_myotube")])
    out = _intra_rater(sk, sd, fk, fe)
    assert out["n_flips"] == 1
    flip = out["flips"][0]
    assert flip["merged_label"] == 101 and flip["kind"] == "over_merge"
    assert flip["first"] == "different_myotubes" and flip["second"] == "same_myotube"


def test_pairing_is_by_identity_not_position():
    """If the repeat is reordered -- which it must be -- pairing by position would
    silently compare different objects."""
    sk, sd, fk, fe = _keys([("control", "same_myotube", "same_myotube"),
                            ("control", "different_myotubes", "different_myotubes")])
    reordered = {k: sk[k] for k in reversed(list(sk))}
    out = _intra_rater(reordered, sd, fk, fe)
    assert out["agreement"] == 1.0, "identity pairing survives reordering"


def test_unscored_cases_are_excluded_not_counted_as_agreement():
    sk, sd, fk, fe = _keys([("over_merge", "different_myotubes", None),
                            ("control", "same_myotube", "same_myotube")])
    out = _intra_rater(sk, sd, fk, fe)
    assert out["n_scored_both_passes"] == 1 and out["agreement"] == 1.0


def test_an_entry_without_a_first_pass_uid_is_reported_unpaired():
    sk, sd, fk, fe = _keys([("control", "same_myotube", "same_myotube")])
    sk["r2_999"] = {"well": "w", "merged_label": 999, "case_kind": "control"}
    sd["r2_999"] = {"decision": "same_myotube"}
    out = _intra_rater(sk, sd, fk, fe)
    assert out["unpaired"] == ["r2_999"]


# ------------------------------------------------------- the built repeat packet


REPEAT_KEY = ("PrecisionMyotube/annotation_work/over_merge_r2/over_merge_r2.key.json")
FIRST_KEY = ("PrecisionMyotube/annotation_work/over_merge_r1/over_merge_r1.key.json")


@pytest.mark.skipif(not __import__("pathlib").Path(REPEAT_KEY).is_file(),
                    reason="repeat packet not built")
def test_the_built_repeat_shows_the_same_objects_in_a_different_order():
    from pathlib import Path

    first = json.loads(Path(FIRST_KEY).read_text(encoding="utf-8"))
    second = json.loads(Path(REPEAT_KEY).read_text(encoding="utf-8"))
    assert second["repeat_of_batch"] == first["batch_id"]
    # the r1 key predates the order_seed field; before it existed the shuffle seed was
    # the order seed, so fall back rather than rewriting a reviewed artifact
    order_seed = lambda k: k.get("order_seed", k.get("shuffle_seed"))
    assert order_seed(second) != order_seed(first)
    objects = lambda k: {(v["well"], v["merged_label"]) for v in k["key"].values()}
    assert objects(first) == objects(second), "a repeat must show the identical objects"
    assert all("first_pass_uid" in v for v in second["key"].values())
    position = lambda uid: uid.rsplit("_", 1)[-1]
    first_pos = {(v["well"], v["merged_label"]): position(u)
                 for u, v in first["key"].items()}
    coincide = sum(1 for u, v in second["key"].items()
                   if first_pos[(v["well"], v["merged_label"])] == position(u))
    assert coincide < len(second["key"]), "the order must actually change"
