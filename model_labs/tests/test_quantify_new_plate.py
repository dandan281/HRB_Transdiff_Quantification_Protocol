"""Contract tests for the plate runner's pure parts (no GPU, no nd2).

What is pinned: well-token discovery (the naming rule that maps an nd2 to a
well), the refusal to guess on token collisions, ROI matching, and the
pixel-size rescale trigger -- the three ways a real plate silently goes
wrong before any tracing happens.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tracer_lab import quantify_new_plate as q                  # noqa: E402


def test_well_token_finds_the_well_in_qplate_names():
    assert q.well_token("19_B06_actv104_trka") == "B06"
    assert q.well_token("23_B02_ctrl") == "B02"
    assert q.well_token("18_B07") == "B07"
    assert q.well_token("58_E10_br223_igf1r") == "E10"


def test_well_token_is_case_insensitive_and_falls_back_to_stem():
    assert q.well_token("plate9_c03_x") == "C03"
    assert q.well_token("no_well_here") == "no_well_here"


def test_discover_orders_and_filters(tmp_path):
    for n in ("23_B02_ctrl.nd2", "19_B06_x.nd2", "32_C08_y.nd2", "notes.txt"):
        (tmp_path / n).write_bytes(b"")
    got = q.discover(tmp_path)
    assert [w for w, _ in got] == ["B06", "B02", "C08"]        # sorted by file
    assert [w for w, _ in q.discover(tmp_path, ["C08"])] == ["C08"]


def test_discover_refuses_ambiguous_wells(tmp_path):
    """Two files that map to the same well must stop the run, not pick one."""
    (tmp_path / "23_B02_ctrl.nd2").write_bytes(b"")
    (tmp_path / "24_B02_repeat.nd2").write_bytes(b"")
    with pytest.raises(SystemExit, match="collision"):
        q.discover(tmp_path)


def test_discover_empty_folder_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="no .nd2"):
        q.discover(tmp_path)


def test_find_rois_matches_on_well_prefix(tmp_path):
    (tmp_path / "B02_Ctrl_ROIs.zip").write_bytes(b"")
    (tmp_path / "B06_ACT104_TrkA_ROIs.zip").write_bytes(b"")
    assert q.find_rois(tmp_path, "B02").name == "B02_Ctrl_ROIs.zip"
    assert q.find_rois(tmp_path, "b06").name == "B06_ACT104_TrkA_ROIs.zip"
    assert q.find_rois(tmp_path, "C08") is None


def test_rescale_trigger_matches_the_documented_tolerance():
    """0.6493 (P26/P28) must NOT rescale; 1.7246 (P44) must."""
    assert abs(0.649269 / q.TARGET_UM - 1) <= q.RESCALE_TOL
    assert abs(1.7245709 / q.TARGET_UM - 1) > q.RESCALE_TOL
    assert q.TARGET_UM == pytest.approx(0.650017)
