"""Guardrails for the blind-repeat (SO01 / G-SO1) path."""
from __future__ import annotations

import json

import numpy as np
import pytest

from annotation_tools.qc_review.cli import build_parser, cmd_blind_compare
from annotation_tools.qc_review.pipeline import build_cases


def test_build_cases_only_ids_filter():
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[3:8, 3:8] = 1
    labels[12:17, 12:17] = 2
    labels[20:25, 5:10] = 3
    fiber = np.zeros((30, 30), np.uint16)
    cases = build_cases(labels, fiber, 0.65, only_ids={2})
    assert [c["id"] for c in cases] == ["myotube_0002"]     # only the requested label is built


def test_blind_page_carries_provenance(tmp_path):
    """The export must record WHO and WHEN (reviewer + timestamps) so washout is verifiable
    — the gap Codex fail-closed on in G-SO1."""
    from annotation_tools.qc_review.page import build_page
    case = {"id": "case_01", "features": {"length_um": 100.0, "touches_border": 0}, "prior": 0.0,
            "thumb": "data:,", "edit_img": "data:,", "mask_rle": {"h": 2, "w": 2, "counts": [4]},
            "geom": {"origin": [0, 0], "src_h": 2, "src_w": 2, "edit_h": 2, "edit_w": 2}}
    out = tmp_path / "b.html"
    build_page("blind_repeat", [case], 1, str(out), blind=True, reviewer="reviewer_01")
    h = out.read_text(encoding="utf-8")
    assert '"reviewer": "reviewer_01"' in h              # identity baked into DATA
    for tok in ("session_started_at", "exported_at", "decided_at", "stampDecision"):
        assert tok in h                                  # timestamp wiring present in the export path


def test_blind_compare_agreement_math(tmp_path, capsys):
    key = {"seed": 1, "n": 3, "key": {
        "case_01": {"well": "w", "real_id": "myotube_0001", "stratum": "complete",
                    "first_action": "accept", "first_touches_border": 0},
        "case_02": {"well": "w", "real_id": "myotube_0002", "stratum": "reject",
                    "first_action": "reject", "first_touches_border": 0},
        "case_03": {"well": "w", "real_id": "myotube_0003", "stratum": "ambiguous",
                    "first_action": "ambiguous", "first_touches_border": 0}}}
    second = {"stem": "blind_repeat", "decisions": {
        "case_01": {"action": "accept", "features": {"touches_border": 0}},   # agree (complete==complete)
        "case_02": {"action": "accept", "features": {"touches_border": 0}},   # DISAGREE (reject -> accept)
        "case_03": {"action": "ambiguous", "features": {"touches_border": 0}}}}   # agree
    kp = tmp_path / "k.json"; kp.write_text(json.dumps(key))
    dp = tmp_path / "d.json"; dp.write_text(json.dumps(second))

    args = build_parser().parse_args(["blind-compare", "--key", str(kp), "--decisions", str(dp)])
    args.func(args)
    out = json.loads(capsys.readouterr().out)

    assert out["n_compared"] == 3
    assert out["disposition_agreement_pct"] == pytest.approx(66.7, abs=0.1)
    assert out["meets_agreement"] is False           # 66.7% < 85% target
    assert out["complete_complete_pairs"] == 1
    assert [d["case"] for d in out["disagreements"]] == ["case_02"]
