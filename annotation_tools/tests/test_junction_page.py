"""Junction-splitting page: render contract + single-choice UI contract."""
from pathlib import Path

import numpy as np
import pytest

from annotation_tools.qc_review.junction_page import BRANCH_RGB, build_junction_page, render_case


def _case(uid="w/junction_000001"):
    return {"id": "junction_000001", "well": "w", "node": 131, "uid": uid,
            "dom_id": uid.replace("/", "__"),
            "img": "data:image/jpeg;base64,AA", "overlay": "data:image/png;base64,AA",
            "branches": [
                {"letter": letter, "rgb": ",".join(str(v) for v in rgb),
                 "branch_id": i, "length_um": 20.0 + i}
                for i, (letter, rgb) in enumerate(BRANCH_RGB)],
            "pairs": [
                {"key": "AB", "letters": ["A", "B"], "branches": [0, 1]},
                {"key": "AC", "letters": ["A", "C"], "branches": [0, 2]},
                {"key": "BC", "letters": ["B", "C"], "branches": [1, 2]},
            ]}


def _html(tmp_path, cases=None):
    out = build_junction_page(cases or [_case()], tmp_path / "j.html", batch_id="b01",
                              reviewer="reviewer_01",
                              session_started_at="2026-07-23T00:00:00Z")
    return Path(out).read_text(encoding="utf-8")


# ------------------------------------------------------------------ page contract


def test_page_requires_reviewer(tmp_path):
    with pytest.raises(ValueError, match="reviewer"):
        build_junction_page([_case()], tmp_path / "j.html", batch_id="b01", reviewer="",
                            session_started_at="2026-07-23T00:00:00Z")


def test_page_keys_by_uid(tmp_path):
    html = _html(tmp_path, [_case("well_a/junction_000001"), _case("well_b/junction_000001")])
    assert "state[c.uid]" in html and "state[c.id]" not in html
    assert "payload.decisions[c.uid]" in html


def test_choosing_a_pair_replaces_any_previous_choice(tmp_path):
    """A junction has one physical answer; picking a new pair must replace the old one."""
    html = _html(tmp_path)
    assert "s.choice = (s.choice === key) ? null : key" in html
    assert "s.picked.indexOf" not in html, "must not reuse the linker's multi-select model"


def test_declined_pairs_are_recorded_as_negatives(tmp_path):
    """Every pair not chosen must still export a label=0, not be silently dropped."""
    html = _html(tmp_path)
    assert "function pairLabel(" in html
    assert "label: s.t ? pairLabel(p, s) : null" in html
    assert "if(s.choice === p.key) return 1;" in html
    assert "return 0;" in html


def test_digit_keys_select_pairs(tmp_path):
    html = _html(tmp_path)
    assert "['1','2','3'].includes(e.key)" in html
    assert "cur().pairs[+e.key - 1]" in html


def test_branch_point_and_unsure_controls_exist(tmp_path):
    html = _html(tmp_path)
    assert "toggle('__none'); step(1)" in html
    assert "toggle('__unsure'); step(1)" in html
    assert "BRANCH POINT" in html


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
    html = _html(tmp_path)
    for token in ("ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown",
                  "'Enter'", "=== 'N'", "=== 'U'", "=== 'J'", "=== '0'", "'?'", "=== 'L'"):
        assert token in html, f"missing keyboard binding: {token}"


def test_shortcut_help_panel_lists_every_binding(tmp_path):
    from annotation_tools.qc_review.junction_page import _SHORTCUTS

    html = _html(tmp_path)
    assert 'id="kbmap"' in html
    assert len(_SHORTCUTS) >= 8
    for key, _desc in _SHORTCUTS:
        assert key in html, f"shortcut {key} not documented in the help panel"


def test_jump_to_next_undecided(tmp_path):
    html = _html(tmp_path)
    assert "function nextUndecided()" in html


def test_brightness_contrast_controls(tmp_path):
    html = _html(tmp_path)
    assert "function nudge(" in html and "brightness(" in html and "contrast(" in html
    assert 'id="rBright"' in html and 'id="rContrast"' in html


def test_clear_returns_a_case_to_undecided(tmp_path):
    html = _html(tmp_path)
    assert "function clearCase()" in html
    assert "{choice: null, unsure: false, t: null}" in html
    assert "=== 'C'" in html


def test_image_is_not_reloaded_on_every_toggle(tmp_path):
    html = _html(tmp_path)
    assert "img.dataset.uid !== c.uid" in html


def test_outlines_are_a_separate_hideable_layer(tmp_path):
    html = _html(tmp_path)
    assert 'id="ovl"' in html
    assert "function toggleOverlay()" in html
    assert "ovl.classList.toggle('hidden', !overlayVisible)" in html
    assert "img.style.filter" in html and "ovl.style.filter" not in html


def test_dot_product_is_never_sent_to_the_page(tmp_path):
    """The classical floor's current pairing must not anchor the operator's judgement."""
    html = _html(tmp_path)
    assert "tangent_cos" not in html
    assert '"dot"' not in html


# ------------------------------------------------------------------ render contract


def test_render_case_returns_clean_and_overlay_layers():
    coordinates = [
        np.array([[50.0, c] for c in range(10, 51)]),      # branch 0: horizontal, into junction
        np.array([[50.0, c] for c in range(50, 91)]),       # branch 1: horizontal, out of junction
        np.array([[float(r), 50.0] for r in range(10, 51)]),  # branch 2: vertical stub
    ]
    fiber = np.full((100, 100), 3000, dtype=np.uint16)
    fiber[0, 0] = 5000                     # give percentile scaling a range
    clean, overlay, _bbox = render_case(fiber, None, coordinates, (0, 1, 2),
                                        (50.0, 50.0), size=200, radius_um=30.0,
                                        pixel_um=0.6493)
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
    assert len(opaque), "overlay must actually draw branch paths"
    branch_colors = {rgb for _, rgb in BRANCH_RGB}
    for pixel in opaque[:: max(1, len(opaque) // 50)]:      # sample, don't check every pixel
        assert tuple(int(v) for v in pixel) in branch_colors
