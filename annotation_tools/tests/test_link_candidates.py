"""Gap-bridging link candidates: geometry rule and page contract."""
from pathlib import Path

import numpy as np
import pytest

from annotation_tools.qc_review.link_candidates import (
    DEFAULT_COS_MIN, DEFAULT_GAP_UM, endpoint_directions, find_link_candidates)

PIXEL_UM = 0.6493


def _field():
    """Two collinear horizontal fragments with a gap, plus an unrelated fibre."""
    labels = np.zeros((120, 200), dtype=np.int32)
    labels[38:42, 10:70] = 1        # fragment
    labels[38:42, 85:150] = 2       # collinear continuation across a ~15 px gap
    labels[90:94, 10:150] = 3       # parallel but far away
    return labels


def test_collinear_neighbour_is_offered():
    found = find_link_candidates(_field(), [1], PIXEL_UM)
    ids = [c.candidate_id for c in found["myotube_0001"]]
    assert "myotube_0002" in ids, "the collinear continuation must be offered"
    assert "myotube_0003" not in ids, "a distant parallel fibre must not be offered"


def test_both_endpoint_directions_must_agree():
    """A fibre that merely passes nearby must not be offered.

    T-shaped: the vertical bar's endpoint points at the horizontal fragment, but
    the horizontal fragment's own endpoint does not point back at it.
    """
    labels = np.zeros((120, 200), dtype=np.int32)
    labels[38:42, 10:70] = 1
    labels[45:100, 100:104] = 2      # perpendicular, offset
    found = find_link_candidates(labels, [1], PIXEL_UM)
    assert found["myotube_0001"] == [], "perpendicular neighbour must be rejected"


def test_gap_limit_is_enforced():
    labels = np.zeros((120, 400), dtype=np.int32)
    labels[38:42, 10:70] = 1
    labels[38:42, 300:380] = 2       # far beyond 40 um at 0.6493 um/px (~62 px)
    assert find_link_candidates(labels, [1], PIXEL_UM, gap_um=10.0)["myotube_0001"] == []


def test_candidates_are_sorted_nearest_first():
    """The most plausible join must be offered as option A."""
    labels = np.zeros((120, 300), dtype=np.int32)
    labels[38:42, 10:70] = 1
    labels[38:42, 78:120] = 2        # near
    labels[38:42, 128:200] = 3       # farther, still collinear
    found = find_link_candidates(labels, [1], PIXEL_UM, gap_um=80.0)
    gaps = [c.gap_um for c in found["myotube_0001"]]
    assert gaps == sorted(gaps)


def test_endpoint_directions_point_outward():
    mask = np.zeros((60, 120), dtype=bool)
    mask[28:32, 10:110] = True
    ends = endpoint_directions(mask)
    assert len(ends) >= 2
    xs = [d[1] for d in ends]
    # a horizontal bar's two endpoints must point in opposing directions
    assert min(float(np.dot(a, b)) for a in xs for b in xs) < -0.8


def test_no_self_candidate():
    found = find_link_candidates(_field(), [1], PIXEL_UM)
    assert all(c.candidate_id != "myotube_0001" for c in found["myotube_0001"])


def test_tiny_proposals_are_ignored_as_partners():
    labels = np.zeros((120, 200), dtype=np.int32)
    labels[38:42, 10:70] = 1
    labels[39:41, 78:80] = 2         # a few pixels: noise, not a fibre
    assert find_link_candidates(labels, [1], PIXEL_UM)["myotube_0001"] == []


# ------------------------------------------------------------------ page contract


def _case(uid="w/myotube_0001", n_cand=2):
    from annotation_tools.qc_review.link_page import CANDIDATE_RGB
    return {"id": "myotube_0001", "well": "w", "uid": uid,
            "dom_id": uid.replace("/", "__"), "img": "data:image/jpeg;base64,AA",
            "overlay": "data:image/png;base64,AA",
            "candidates": [
                {"letter": CANDIDATE_RGB[i][0],
                 "rgb": ",".join(str(v) for v in CANDIDATE_RGB[i][1]),
                 "candidate_id": f"myotube_000{i+2}", "gap_um": 5.0 * (i + 1),
                 "cos_fragment": 0.9, "cos_candidate": 0.9}
                for i in range(n_cand)]}


def test_page_records_declined_candidates_as_negatives(tmp_path):
    """A linker trained only on positives learns to join everything."""
    from annotation_tools.qc_review.link_page import build_link_page

    out = build_link_page([_case()], tmp_path / "l.html", batch_id="b01",
                          reviewer="reviewer_01",
                          session_started_at="2026-07-22T00:00:00Z",
                          gap_um=DEFAULT_GAP_UM, cos_min=DEFAULT_COS_MIN)
    html = Path(out).read_text(encoding="utf-8")
    assert "declined:" in html and "offered:" in html
    assert "linked_to:" in html


def test_page_supports_linking_at_both_ends(tmp_path):
    """A fragment can be the middle of a fibre; single-choice would lose joins."""
    from annotation_tools.qc_review.link_page import build_link_page

    out = build_link_page([_case(n_cand=3)], tmp_path / "l.html", batch_id="b01",
                          reviewer="reviewer_01",
                          session_started_at="2026-07-22T00:00:00Z",
                          gap_um=DEFAULT_GAP_UM, cos_min=DEFAULT_COS_MIN)
    html = Path(out).read_text(encoding="utf-8")
    assert "s.picked.indexOf" in html, "candidate toggles must be multi-select"


def test_page_requires_reviewer(tmp_path):
    from annotation_tools.qc_review.link_page import build_link_page

    with pytest.raises(ValueError, match="reviewer"):
        build_link_page([_case()], tmp_path / "l.html", batch_id="b01", reviewer="",
                        session_started_at="2026-07-22T00:00:00Z",
                        gap_um=DEFAULT_GAP_UM, cos_min=DEFAULT_COS_MIN)


def test_page_keys_by_uid(tmp_path):
    """Proposal ids repeat across wells; state must key on well/id."""
    from annotation_tools.qc_review.link_page import build_link_page

    out = build_link_page([_case("well_a/myotube_0001"), _case("well_b/myotube_0001")],
                          tmp_path / "l.html", batch_id="b01", reviewer="reviewer_01",
                          session_started_at="2026-07-22T00:00:00Z",
                          gap_um=DEFAULT_GAP_UM, cos_min=DEFAULT_COS_MIN)
    html = Path(out).read_text(encoding="utf-8")
    assert "state[c.uid]" in html and "state[c.id]" not in html
    assert "payload.decisions[c.uid]" in html


# ------------------------------------------------------------------ UI contract
# The operator reviewed 1,800 decisions in `page.py`. This page must match that
# instrument: one case at a time, decide-and-advance, arrow navigation, a
# position counter, and every control reachable from the keyboard.


def _html(tmp_path, cases=None):
    from annotation_tools.qc_review.link_page import build_link_page

    out = build_link_page(cases or [_case()], tmp_path / "l.html", batch_id="b01",
                          reviewer="reviewer_01",
                          session_started_at="2026-07-22T00:00:00Z",
                          gap_um=DEFAULT_GAP_UM, cos_min=DEFAULT_COS_MIN)
    return Path(out).read_text(encoding="utf-8")


def test_toggling_a_candidate_unlabels_it(tmp_path):
    """Pressing a letter twice must remove the link, not add it again."""
    html = _html(tmp_path)
    assert "s.picked.indexOf(key)" in html
    assert "s.picked.splice(i, 1)" in html, "no unlabel path"


def test_forward_and_backward_navigation_exist(tmp_path):
    html = _html(tmp_path)
    assert "function step(d)" in html
    assert 'onclick="step(-1)"' in html and 'onclick="step(1)"' in html
    assert "ArrowRight" in html and "ArrowLeft" in html


def test_position_counter_and_progress(tmp_path):
    html = _html(tmp_path)
    assert 'id="pos"' in html and "${idx+1} / ${n}" in html
    assert 'id="progress"' in html


def test_every_control_has_a_keyboard_shortcut(tmp_path):
    """Candidates, no-join, unsure, confirm, navigate, brightness, jump, help."""
    html = _html(tmp_path)
    for token in ("ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown",
                  "'Enter'", "=== 'N'", "=== 'U'", "=== 'J'", "=== '0'", "'?'"):
        assert token in html, f"missing keyboard binding: {token}"


def test_shortcut_help_panel_lists_every_binding(tmp_path):
    from annotation_tools.qc_review.link_page import _SHORTCUTS

    html = _html(tmp_path)
    assert 'id="kbmap"' in html
    assert len(_SHORTCUTS) >= 8
    for key, _desc in _SHORTCUTS:
        assert key in html, f"shortcut {key} not documented in the help panel"


def test_decide_and_advance(tmp_path):
    """N and U decide then move on, matching `page.py`'s decideNext behaviour."""
    html = _html(tmp_path)
    assert "toggle('__none'); step(1)" in html
    assert "toggle('__unsure'); step(1)" in html
    assert "function confirmNext()" in html


def test_jump_to_next_undecided(tmp_path):
    html = _html(tmp_path)
    assert "function nextUndecided()" in html


def test_brightness_contrast_controls(tmp_path):
    """Dim fibres are the hard cases; the viewer needs exposure control."""
    html = _html(tmp_path)
    assert "function nudge(" in html and "brightness(" in html and "contrast(" in html
    assert 'id="rBright"' in html and 'id="rContrast"' in html


def test_clear_returns_a_case_to_undecided(tmp_path):
    """'Unlabel' must include wiping the decision stamp, not just the picks."""
    html = _html(tmp_path)
    assert "function clearCase()" in html
    assert "{picked: [], none: false, unsure: false, t: null}" in html
    assert "=== 'C'" in html, "clear needs a keyboard shortcut"


def test_range_sliders_do_not_disable_shortcuts(tmp_path):
    """Regression: bailing on every INPUT meant one slider drag silently killed
    every keyboard shortcut for the rest of the session."""
    html = _html(tmp_path)
    assert "t.type !== 'range'" in html
    assert "activeElement.blur()" in html


def test_selection_state_is_echoed_in_words(tmp_path):
    """A toggle must be unambiguous, not inferred from a colour alone."""
    html = _html(tmp_path)
    assert 'id="status"' in html
    assert "LINKED TO" in html and "NO JOIN" in html


def test_image_is_not_reloaded_on_every_toggle(tmp_path):
    """Re-assigning a 140 KB data URL per toggle flashes the picture, which
    reads as 'my click did nothing'."""
    html = _html(tmp_path)
    assert "img.dataset.uid !== c.uid" in html


def test_outlines_are_a_separate_hideable_layer(tmp_path):
    """The operator must be able to see the bare stain.

    Judging whether a broken fibre truly continues across a gap requires an
    unobscured view of the Desmin/DAPI signal, so the outlines cannot be burned
    into the picture -- they live on their own layer that `L` hides.
    """
    html = _html(tmp_path)
    assert 'id="ovl"' in html, "no separate outline layer"
    assert "function toggleOverlay()" in html
    assert "=== 'L'" in html, "overlay toggle needs a shortcut"
    assert "ovl.classList.toggle('hidden', !overlayVisible)" in html
    # brightness must apply to the stain, not the outlines
    assert "img.style.filter" in html and "ovl.style.filter" not in html


def test_render_case_returns_clean_and_overlay_layers():
    """The clean layer must contain no outline pixels at all."""
    import numpy as np

    from annotation_tools.qc_review.link_page import FRAGMENT_RGB, render_case

    labels = np.zeros((80, 200), dtype=np.int32)
    labels[38:42, 10:70] = 1
    labels[38:42, 90:150] = 2
    fiber = np.zeros((80, 200), dtype=np.uint16)
    fiber[labels > 0] = 3000
    fiber[0, 0] = 5000                       # give percentile scaling a range
    clean, overlay, _bbox = render_case(fiber, None, labels, 1, [2], size=200)
    assert clean.startswith("data:image/jpeg;base64,")
    assert overlay.startswith("data:image/png;base64,")

    import base64
    import io
    from PIL import Image
    layer = Image.open(io.BytesIO(base64.b64decode(overlay.split(",", 1)[1])))
    assert layer.mode == "RGBA"
    arr = np.asarray(layer)
    assert (arr[..., 3] == 0).any(), "overlay must be mostly transparent"
    opaque = arr[arr[..., 3] > 0][:, :3]
    assert len(opaque), "overlay must actually draw outlines"
    # every opaque pixel is one of the declared outline colours
    assert any((opaque == np.array(FRAGMENT_RGB)).all(axis=1))
