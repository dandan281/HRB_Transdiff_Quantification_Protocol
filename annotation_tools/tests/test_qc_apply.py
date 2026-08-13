"""CL05 apply-path guardrails (regression tests for Codex review findings).

Covers the two critical bugs Codex found in `qc_review apply`:
  1. reviewer identity must be persisted (provenance + review log), not just checked.
  2. a border-touching accept must become `border_truncated`, never authoritative `complete`.
Plus: ambiguous stays reviewed=False; --reviewer is mandatory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from annotation_tools.qc_review.cli import build_parser


def _make_package(tmp: Path):
    import tifffile
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:10, 5:10] = 1        # interior fibre
    labels[0:4, 12:17] = 2        # a fibre we will mark border-touching
    labels[14:18, 2:7] = 3        # ambiguous
    tifffile.imwrite(tmp / "starting_labels.tif", labels)
    tifffile.imwrite(tmp / "fiber_raw16.tif", np.zeros((20, 20), np.uint16))
    (tmp / "README.json").write_text(json.dumps({"image_id": "testimg", "pixel_um": 0.65}))
    feats = lambda tb: {"length_um": 30.0, "width_um": 5.0, "area_um2": 25.0, "aspect": 6.0,
                        "solidity": 0.6, "extent": 0.3, "fiber_mean": 1000.0,
                        "territory_overlap": 1, "touches_border": tb}
    decisions = {"stem": "testimg", "decisions": {
        "myotube_0001": {"action": "accept", "note": "", "features": feats(0), "edited": False},
        "myotube_0002": {"action": "accept", "note": "", "features": feats(1), "edited": False},
        "myotube_0003": {"action": "ambiguous", "note": "", "features": feats(0), "edited": False},
    }}
    dpath = tmp / "testimg.decisions.json"
    dpath.write_text(json.dumps(decisions))
    return dpath


def _run_apply(pkg: Path, dpath: Path, out: Path, reviewer="tester"):
    args = build_parser().parse_args(
        ["apply", "--package", str(pkg), "--decisions", str(dpath),
         "--reviewer", reviewer, "--out", str(out)])
    args.func(args)


def test_apply_persists_reviewer_and_border_status(tmp_path):
    dpath = _make_package(tmp_path)
    out = tmp_path / "testimg.qc.instances.json"
    _run_apply(tmp_path, dpath, out)

    inst = json.loads(out.read_text())
    # 1. reviewer identity is actually saved, not merely checked
    assert inst["provenance"]["reviewer"] == "tester"
    by_id = {r["id"]: r for r in inst["instances"]}
    # 2. interior accept -> authoritative complete; border-touching accept -> border_truncated
    assert by_id["myotube_0001"]["status"] == "complete"
    assert by_id["myotube_0001"]["reviewed"] is True
    assert by_id["myotube_0002"]["status"] == "border_truncated"
    assert by_id["myotube_0002"]["status"] != "complete"     # excluded from training targets
    # ambiguous is retained but not authoritative
    assert by_id["myotube_0003"]["status"] == "ambiguous"
    assert by_id["myotube_0003"]["reviewed"] is False

    # 3. a reviewer-linked review log exists with one row per decision
    log = out.with_name(out.stem + ".review_log.jsonl")   # ...qc.instances.review_log.jsonl
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(rows) == 3
    assert all(r["reviewer"] == "tester" for r in rows)


def test_export_corrections_backfills_precapture_edit(tmp_path):
    """An edit made before the capture code (no original_rle) must still yield a pair,
    with the proposal reconstructed from starting_labels and the reason inferred."""
    import tifffile
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:10, 5:8] = 1                       # short proposal (5x3 = 15 px) in a 5x5 window
    tifffile.imwrite(tmp_path / "starting_labels.tif", labels)
    tifffile.imwrite(tmp_path / "fiber_raw16.tif", np.full((20, 20), 1000, np.uint16))
    (tmp_path / "README.json").write_text(json.dumps({"image_id": "t2", "pixel_um": 0.65}))
    dec = {"stem": "t2", "decisions": {"myotube_0001": {
        "action": "ambiguous", "note": "", "edited": True,
        "labels_rle": [{"h": 5, "w": 5, "counts": [0, 25]}],       # human extended to full 5x5
        "geom": {"origin": [5, 5], "src_h": 5, "src_w": 5, "edit_h": 5, "edit_w": 5},
        "features": {"touches_border": 0}}}}                        # NO original_rle -> backfill path
    dpath = tmp_path / "t2.decisions.json"
    dpath.write_text(json.dumps(dec))
    out = tmp_path / "corr"
    args = build_parser().parse_args(
        ["export-corrections", "--package", str(tmp_path), "--decisions", str(dpath), "--out", str(out)])
    args.func(args)

    man = [json.loads(x) for x in (out / "t2.corrections.jsonl").read_text().splitlines() if x.strip()]
    assert len(man) == 1
    r = man[0]
    assert r["backfilled"] is True              # proposal came from starting_labels, not the JSON
    assert r["reason"] == "too_short"           # machine stopped short; human extended
    assert r["added_px"] == 10 and r["removed_px"] == 0
    z = np.load(out / r["npz"])
    assert {"fiber", "proposal", "corrected"}.issubset(set(z.files))


def test_reviewer_is_mandatory(tmp_path):
    dpath = _make_package(tmp_path)
    with pytest.raises(SystemExit):     # argparse errors out when --reviewer is missing
        build_parser().parse_args(
            ["apply", "--package", str(tmp_path), "--decisions", str(dpath),
             "--out", str(tmp_path / "x.json")])
