"""Scoring an over-merge review export against its key.

The rules this implements were fixed *before* the review was run (see
`coordination/reports/claude_over_merge_review_packet_2026-07-29.md` sec.4), so the
point of these tests is that the scorer cannot quietly drift into a friendlier
summary than the one that was pre-registered.
"""
from __future__ import annotations

import json

import pytest

from annotation_tools.qc_review.cli import build_parser


def _packet(tmp_path, kinds_and_decisions, *, threshold=0.9, batch="b1",
            started="2026-07-30T04:46:36Z"):
    """kinds_and_decisions: [(kind, decision, decided_at), ...]"""
    key = {"batch_id": batch, "threshold": threshold,
           "threshold_status": "LOCKED -- test", "key": {}}
    export = {"batch_id": batch, "reviewer": "reviewer_01", "threshold": threshold,
              "session_started_at": started, "exported_at": "2026-07-30T05:00:00Z",
              "decision_vocabulary": ["same_myotube", "different_myotubes", "ambiguous_2d"],
              "decisions": {}}
    for n, (kind, decision, when) in enumerate(kinds_and_decisions, start=1):
        uid = f"{batch}_{n:03d}"
        key["key"][uid] = {"well": "w1", "merged_label": 100 + n,
                           "case_kind": kind, "fragment_ids": [1, 2],
                           "accepted_pairs": [], "overlapping_references": []}
        export["decisions"][uid] = {"decision": decision, "decided_at": when, "note": ""}
    kp = tmp_path / "k.json"; kp.write_text(json.dumps(key), encoding="utf-8")
    dp = tmp_path / "d.json"; dp.write_text(json.dumps(export), encoding="utf-8")
    return kp, dp


def _score(tmp_path, rows, capsys, first=None, **kw):
    kp, dp = _packet(tmp_path, rows, **kw)
    argv = ["score-over-merge-review", "--key", str(kp), "--decisions", str(dp)]
    if first is not None:
        fk, fd = first
        argv += ["--first-pass-key", str(fk), "--first-pass-decisions", str(fd)]
    args = build_parser().parse_args(argv)
    args.func(args)
    return json.loads(capsys.readouterr().out)


def _first_pass(tmp_path, key_path, decisions, name="first"):
    """Build a matching first pass whose key pairs to the second by object identity."""
    second = json.loads(key_path.read_text(encoding="utf-8"))
    fk = {"batch_id": "r0", "threshold": second["threshold"], "key": {}}
    fe = {"decisions": {}}
    for n, (uid, meta) in enumerate(second["key"].items(), start=1):
        fuid = f"r0_{n:03d}"
        fk["key"][fuid] = {"well": meta["well"], "merged_label": meta["merged_label"],
                           "case_kind": meta["case_kind"]}
        fe["decisions"][fuid] = {"decision": decisions[n - 1]}
        meta["first_pass_uid"] = fuid
    key_path.write_text(json.dumps(second), encoding="utf-8")
    p1 = tmp_path / f"{name}_key.json"; p1.write_text(json.dumps(fk), encoding="utf-8")
    p2 = tmp_path / f"{name}_dec.json"; p2.write_text(json.dumps(fe), encoding="utf-8")
    return p1, p2


T = "2026-07-30T04:47:00Z"


def test_a_flagged_case_called_different_is_a_confirmed_over_merge(tmp_path, capsys):
    out = _score(tmp_path, [("over_merge", "different_myotubes", T),
                            ("control", "same_myotube", T),
                            ("control", "same_myotube", T),
                            ("control", "same_myotube", T),
                            ("control", "same_myotube", T)], capsys)
    assert out["flagged"]["different_myotubes"] == 1
    assert [c["merged_label"] for c in out["confirmed_over_merges"]] == [101]
    assert out["calibration_failed"] is False
    assert out["verdicts_track_the_flag"] is True


def test_ambiguous_is_never_pooled_with_same_myotube(tmp_path, capsys):
    out = _score(tmp_path, [("over_merge", "ambiguous_2d", T),
                            ("control", "same_myotube", T)], capsys)
    assert out["flagged"]["ambiguous_2d"] == 1
    assert out["flagged"]["same_myotube"] == 0
    assert [c["uid"] for c in out["unresolved_flagged_cases"]] == ["b1_001"]


def test_an_undecided_flagged_case_is_unresolved_not_agreement(tmp_path, capsys):
    out = _score(tmp_path, [("over_merge", None, None),
                            ("control", "same_myotube", T)], capsys)
    assert out["flagged"]["undecided"] == 1
    assert len(out["unresolved_flagged_cases"]) == 1


def test_controls_called_different_at_a_high_rate_void_the_case_verdicts(tmp_path, capsys):
    """The pre-registered calibration rule: merges the benchmark did NOT flag being
    called over-merges means the verdicts are not tracking the flag."""
    rows = [("over_merge", "same_myotube", T)] + \
           [("control", "different_myotubes", T)] * 6 + \
           [("control", "same_myotube", T)] * 6
    out = _score(tmp_path, rows, capsys)
    assert out["control_called_different_rate"] == pytest.approx(0.5)
    assert out["calibration_failed"] is True
    assert out["verdict"].startswith("UNRESOLVED")


def test_no_discrimination_when_flagged_rate_does_not_exceed_control_rate(tmp_path, capsys):
    """The cutoff-free criterion: if the flagged cases are called `different` no more
    often than the controls, the verdicts carry no information about the flag."""
    rows = [("over_merge", "different_myotubes", T), ("over_merge", "same_myotube", T),
            ("over_merge", "same_myotube", T)] + \
           [("control", "different_myotubes", T)] * 6 + \
           [("control", "same_myotube", T)] * 6
    out = _score(tmp_path, rows, capsys)
    assert out["flagged_called_different_rate"] == pytest.approx(0.3333, abs=1e-4)
    assert out["control_called_different_rate"] == pytest.approx(0.5)
    assert out["verdicts_track_the_flag"] is False


def test_a_high_control_rate_from_a_SELF_CONSISTENT_reviewer_indicts_the_flag(tmp_path, capsys):
    """The revision of 2026-07-30. Controls called `different` at a high rate was
    pre-registered as "the reviewer is guessing" -- but that was a proxy for reviewer
    noise. When a blind repeat measures noise directly and finds none, the same data
    means the benchmark's flag is missing over-merges, not that the reviewer is."""
    rows = [("over_merge", "same_myotube", T)] + \
           [("control", "different_myotubes", T)] * 6 + \
           [("control", "same_myotube", T)] * 6
    kp, dp = _packet(tmp_path, rows)
    same_again = ["same_myotube"] + ["different_myotubes"] * 6 + ["same_myotube"] * 6
    first = _first_pass(tmp_path, kp, same_again)
    args = build_parser().parse_args(
        ["score-over-merge-review", "--key", str(kp), "--decisions", str(dp),
         "--first-pass-key", str(first[0]), "--first-pass-decisions", str(first[1])])
    args.func(args)
    out = json.loads(capsys.readouterr().out)

    assert out["intra_rater"]["cohens_kappa"] == 1.0
    assert out["calibration_failed"] is False, "a consistent reviewer is not a failure"
    assert "under-detecting" in out["verdict"]
    assert "rule_revision_2026_07_30" in out


def test_a_high_control_rate_from_an_INCONSISTENT_reviewer_still_voids_the_round(tmp_path, capsys):
    """The revision must not become a way to rescue any result: if the repeat shows the
    reviewer disagreeing with themselves, the round is still void."""
    rows = [("over_merge", "same_myotube", T)] + \
           [("control", "different_myotubes", T)] * 6 + \
           [("control", "same_myotube", T)] * 6
    kp, dp = _packet(tmp_path, rows)
    flipped = ["different_myotubes"] + ["same_myotube"] * 6 + ["different_myotubes"] * 6
    first = _first_pass(tmp_path, kp, flipped)
    args = build_parser().parse_args(
        ["score-over-merge-review", "--key", str(kp), "--decisions", str(dp),
         "--first-pass-key", str(first[0]), "--first-pass-decisions", str(first[1])])
    args.func(args)
    out = json.loads(capsys.readouterr().out)

    assert out["intra_rater"]["cohens_kappa"] < 0.8
    assert out["calibration_failed"] is True
    assert "not self-consistent" in out["verdict"]


def test_without_a_repeat_the_verdict_says_to_run_one(tmp_path, capsys):
    rows = [("over_merge", "same_myotube", T)] + \
           [("control", "different_myotubes", T)] * 6 + \
           [("control", "same_myotube", T)] * 6
    out = _score(tmp_path, rows, capsys)
    assert out["calibration_failed"] is True
    assert "blind repeat" in out["verdict"]


def test_confidence_running_backwards_is_reported(tmp_path, capsys):
    """A model whose most confident merges are its wrong ones cannot be fixed by
    raising the threshold, so the direction of the relationship has to be surfaced."""
    kp, dp = _packet(tmp_path, [("control", "same_myotube", T)] * 3 +
                     [("control", "different_myotubes", T)] * 3)
    key = json.loads(kp.read_text(encoding="utf-8"))
    for n, (uid, meta) in enumerate(key["key"].items()):
        meta["accepted_pairs"] = [{"fragments": [1, 2],
                                   "probability": 0.90 if n < 3 else 1.0}]
    kp.write_text(json.dumps(key), encoding="utf-8")
    args = build_parser().parse_args(
        ["score-over-merge-review", "--key", str(kp), "--decisions", str(dp)])
    args.func(args)
    out = json.loads(capsys.readouterr().out)

    cal = out["confidence_calibration"]
    assert cal["available"] is True
    assert cal["auc_probability_predicts_same"] == 0.0, "perfectly anti-correlated"
    assert cal["n_at_probability_1"] == 3
    assert cal["n_at_probability_1_called_different"] == 3


def test_confidence_calibration_needs_both_verdicts(tmp_path, capsys):
    out = _score(tmp_path, [("control", "same_myotube", T)] * 3, capsys)
    assert out["confidence_calibration"]["available"] is False


def test_clean_controls_leave_the_case_verdicts_usable(tmp_path, capsys):
    rows = [("over_merge", "different_myotubes", T), ("over_merge", "different_myotubes", T)] + \
           [("control", "same_myotube", T)] * 8
    out = _score(tmp_path, rows, capsys)
    assert out["calibration_failed"] is False
    assert out["verdicts_track_the_flag"] is True
    assert "usable" in out["verdict"]


def test_latency_is_reported_from_the_timestamps(tmp_path, capsys):
    rows = [("over_merge", "same_myotube", "2026-07-30T04:46:41Z"),      # +5s
            ("control", "same_myotube", "2026-07-30T04:46:47Z"),          # +6s
            ("control", "same_myotube", "2026-07-30T04:47:47Z")]          # +60s
    out = _score(tmp_path, rows, capsys, started="2026-07-30T04:46:36Z")
    lat = out["decision_latency_seconds"]
    assert lat["n"] == 3 and lat["min"] == 5.0 and lat["max"] == 60.0
    assert lat["median"] == pytest.approx(6.0)
    assert lat["n_under_10s"] == 2


def test_a_threshold_mismatch_is_refused(tmp_path):
    kp, dp = _packet(tmp_path, [("over_merge", "same_myotube", T)])
    payload = json.loads(dp.read_text(encoding="utf-8"))
    payload["threshold"] = 0.7
    dp.write_text(json.dumps(payload), encoding="utf-8")
    args = build_parser().parse_args(
        ["score-over-merge-review", "--key", str(kp), "--decisions", str(dp)])
    with pytest.raises(SystemExit, match="threshold mismatch"):
        args.func(args)


def test_a_batch_mismatch_is_refused(tmp_path):
    kp, dp = _packet(tmp_path, [("over_merge", "same_myotube", T)])
    payload = json.loads(dp.read_text(encoding="utf-8"))
    payload["batch_id"] = "someone_elses_packet"
    dp.write_text(json.dumps(payload), encoding="utf-8")
    args = build_parser().parse_args(
        ["score-over-merge-review", "--key", str(kp), "--decisions", str(dp)])
    with pytest.raises(SystemExit, match="batch_id mismatch"):
        args.func(args)


def test_cases_absent_from_the_export_are_listed_not_ignored(tmp_path, capsys):
    kp, dp = _packet(tmp_path, [("over_merge", "same_myotube", T),
                                ("control", "same_myotube", T)])
    payload = json.loads(dp.read_text(encoding="utf-8"))
    del payload["decisions"]["b1_002"]
    dp.write_text(json.dumps(payload), encoding="utf-8")
    args = build_parser().parse_args(
        ["score-over-merge-review", "--key", str(kp), "--decisions", str(dp)])
    args.func(args)
    out = json.loads(capsys.readouterr().out)
    assert out["missing_from_export"] == ["b1_002"]
    assert out["n_scored"] == 1


def test_the_pre_registered_rules_travel_with_the_report(tmp_path, capsys):
    """The report has to carry the rules it applied, so a reader cannot mistake a
    post-hoc summary for the pre-registered one."""
    out = _score(tmp_path, [("over_merge", "same_myotube", T),
                            ("control", "same_myotube", T)], capsys)
    joined = " ".join(out["pre_registered_rules"]).lower()
    assert "ambiguous_2d is unresolved" in joined
    assert "no statistical power" in joined
    assert "does not promote the linker" in joined
    assert out["threshold_status"].startswith("LOCKED")
