"""Guardrails for the blinded over-merge review page.

The load-bearing property is the **blinding contract**: this page exists to get an
independent human verdict on three merges the linker already made, so anything
that tells the reviewer which objects are the flagged ones -- or what the model
thought -- destroys the evidence. Those checks are the reason this file exists;
the rendering tests are secondary.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from annotation_tools.qc_review.over_merge_page import (
    DECISIONS, PANELS, _FORBIDDEN_KEYS, _line, _outline, assert_no_separating_field,
    build_over_merge_page, count_references_in_view, crop_window, render_case_panels)


def _case(uid="om_001", **extra):
    """A payload case exactly as the packet builder emits it -- note there is no
    reference-mask count: it was removed after unblinding the first build."""
    return {"uid": uid, "n_fragments": 2, "gaps_um": [12.0],
            "panels": {p: f"data:image/jpeg;base64,{p}" for p in PANELS}, **extra}


# ------------------------------------------------------------------ blinding


@pytest.mark.parametrize("leak", ["well", "probability", "accepted_pairs", "is_control",
                                  "merged_label", "case_kind", "overlapping_references"])
def test_page_refuses_to_embed_unblinding_fields(tmp_path, leak):
    with pytest.raises(ValueError, match="unblind"):
        build_over_merge_page([_case(**{leak: "anything"})], tmp_path / "p.html",
                              batch_id="b", reviewer="reviewer_01",
                              session_started_at="2026-07-29T00:00:00Z", threshold=0.9)


def test_forbidden_keys_cover_every_field_the_key_file_carries():
    """If the extractor grows a field that identifies a case, the blinding check must
    already know about it -- otherwise it silently ships into the page."""
    key_file_fields = {"well", "merged_label", "case_kind", "fragment_ids",
                       "accepted_pairs", "overlapping_references"}
    assert key_file_fields <= _FORBIDDEN_KEYS


def test_page_embeds_no_probability_anywhere(tmp_path):
    """Belt and braces: the rendered HTML must not contain a link probability even
    if one reached the payload through a nested structure."""
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b",
                                reviewer="reviewer_01",
                                session_started_at="2026-07-29T00:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    for banned in ("probability", "0.9198", "19_B06", "22_B03", "over_merge_label"):
        assert banned not in html


def test_a_displayed_field_that_separates_the_groups_is_caught():
    """The real leak from the first build of this packet: no forbidden key present,
    yet the flagged cases had 7/7/4 reference masks in view against the controls'
    0-3, so ranking on one number unblinded the whole packet."""
    cases = [_case("a", n_refs_in_view=7), _case("b", n_refs_in_view=7),
             _case("c", n_refs_in_view=1), _case("d", n_refs_in_view=2)]
    kinds = ["over_merge", "over_merge", "control", "control"]
    with pytest.raises(ValueError, match="perfectly separates"):
        assert_no_separating_field(cases, kinds)


def test_overlapping_ranges_pass_the_separation_check():
    cases = [_case("a", n_refs_in_view=7), _case("b", n_refs_in_view=2),
             _case("c", n_refs_in_view=1), _case("d", n_refs_in_view=6)]
    kinds = ["over_merge", "over_merge", "control", "control"]
    report = assert_no_separating_field(cases, kinds)
    assert report["n_refs_in_view"]["flagged_range"] == [2, 7]
    assert report["n_refs_in_view"]["control_range"] == [1, 6]


def test_separation_check_uses_the_max_of_a_list_field():
    """`gaps_um` is a list; a reviewer reads the largest gap off the chip row."""
    cases = [_case("a", gaps_um=[40.0]), _case("b", gaps_um=[38.0]),
             _case("c", gaps_um=[5.0]), _case("d", gaps_um=[6.0, 7.0])]
    kinds = ["over_merge", "over_merge", "control", "control"]
    with pytest.raises(ValueError, match="gaps_um"):
        assert_no_separating_field(cases, kinds)


def test_a_field_only_the_flagged_cases_carry_is_caught():
    """The second leak: controls were extracted without `gap_um`, so the page showed a
    gap chip for the three real cases and nothing for the controls. Every *value* was
    innocuous; presence was the tell."""
    cases = [_case("a", gaps_um=[13.1]), _case("b", gaps_um=[34.1]),
             _case("c", gaps_um=[]), _case("d", gaps_um=[])]
    kinds = ["over_merge", "over_merge", "control", "control"]
    with pytest.raises(ValueError, match="present for one group"):
        assert_no_separating_field(cases, kinds)


def test_a_field_absent_from_the_flagged_cases_is_also_caught():
    cases = [_case("a", gaps_um=[]), _case("b", gaps_um=[]),
             _case("c", gaps_um=[9.0]), _case("d", gaps_um=[11.0])]
    kinds = ["over_merge", "over_merge", "control", "control"]
    with pytest.raises(ValueError, match="present for one group"):
        assert_no_separating_field(cases, kinds)


def test_mixed_presence_within_a_group_is_allowed():
    """Presence only unblinds when it is uniform within each group and differs between
    them; a ragged field is not a signal a reviewer can rank on. The values present
    must still interleave, which is the separate range rule."""
    cases = [_case("a", gaps_um=[10.0]), _case("b", gaps_um=[]), _case("c", gaps_um=[20.0]),
             _case("d", gaps_um=[]), _case("e", gaps_um=[15.0]), _case("f", gaps_um=[25.0])]
    kinds = ["over_merge"] * 3 + ["control"] * 3
    report = assert_no_separating_field(cases, kinds)
    assert report["gaps_um"] == {"flagged_range": [10.0, 20.0], "control_range": [15.0, 25.0]}


def test_separation_check_is_a_noop_without_both_groups():
    assert assert_no_separating_field([_case("a")], ["over_merge"]) == {}
    assert assert_no_separating_field([_case("a")], ["control"]) == {}


def test_separation_check_requires_one_kind_per_case():
    with pytest.raises(ValueError, match="one kind per case"):
        assert_no_separating_field([_case("a"), _case("b")], ["control"])


def test_the_page_does_not_display_the_reference_count(tmp_path):
    """Removed from the payload after it unblinded the first build. The reviewer can
    still see the outlines in the reference panel; they just cannot rank on a number."""
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b",
                                reviewer="reviewer_01",
                                session_started_at="2026-07-29T00:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    assert "reference mask(s) in view" not in html


# ------------------------------------------------------------------ crop geometry


def test_crop_window_clamps_to_the_field():
    assert crop_window((100, 100), [5, 5, 20, 20], 90) == (0, 0, 100, 100)
    assert crop_window((100, 100), [40, 40, 50, 50], 10) == (30, 30, 60, 60)


def test_count_references_in_view_agrees_with_the_renderer():
    """The matching code and the renderer must never disagree about what is in view,
    or controls get matched on a number the reviewer does not actually see."""
    fiber, kept, merged = _synthetic_field()
    ref = np.ones((10, 40), dtype=bool)
    references = [{"id": "a", "bbox": (50, 10, 60, 50), "mask": ref.copy()},
                  {"id": "b", "bbox": (50, 70, 60, 110), "mask": ref.copy()},
                  {"id": "c", "bbox": (5, 5, 15, 15), "mask": np.ones((10, 10), bool)}]
    bbox = [50, 10, 60, 110]
    counted = count_references_in_view(references, fiber.shape, bbox, 5)
    _panels, _crop, rendered = render_case_panels(
        fiber, None, kept, merged, references, fragment_ids=[7, 9], merged_label=7,
        links=[], bbox=bbox, pad_px=5, size=200)
    assert counted == rendered == 2


def test_count_references_ignores_an_all_false_mask_overlapping_the_crop():
    fiber, _kept, _merged = _synthetic_field()
    references = [{"id": "a", "bbox": (50, 10, 60, 50), "mask": np.zeros((10, 40), bool)}]
    assert count_references_in_view(references, fiber.shape, [50, 10, 60, 110], 5) == 0


# ------------------------------------------------------------------ provenance


def test_reviewer_is_required(tmp_path):
    with pytest.raises(ValueError, match="reviewer is required"):
        build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b", reviewer="",
                              session_started_at="2026-07-29T00:00:00Z", threshold=0.9)


def test_empty_packet_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no cases"):
        build_over_merge_page([], tmp_path / "p.html", batch_id="b", reviewer="reviewer_01",
                              session_started_at="2026-07-29T00:00:00Z", threshold=0.9)


def test_page_carries_reviewer_session_and_locked_threshold(tmp_path):
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="batch7",
                                reviewer="reviewer_01",
                                session_started_at="2026-07-29T12:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    assert "reviewer_01" in html and "2026-07-29T12:00:00Z" in html and "batch7" in html
    assert "LOCKED" in html, "the page must say the operating point is locked"


def test_export_records_a_timestamp_per_decision(tmp_path):
    """The export shape is evidence: decision plus decided_at, per case."""
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b",
                                reviewer="reviewer_01",
                                session_started_at="2026-07-29T00:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    assert "decided_at: s.t" in html
    assert "session_started_at: DATA.session_started_at" in html
    assert "exported_at: nowISO()" in html


# ------------------------------------------------------------------ vocabulary


def test_decision_vocabulary_is_the_three_agreed_verdicts():
    assert [d[0] for d in DECISIONS] == ["same_myotube", "different_myotubes", "ambiguous_2d"]


def test_ambiguous_is_presented_as_unresolved_not_as_agreement(tmp_path):
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b",
                                reviewer="reviewer_01",
                                session_started_at="2026-07-29T00:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    assert "UNRESOLVED" in html
    assert "never counted as safe" in html or "never as evidence" in html


def test_all_five_panels_are_required_and_ordered():
    assert PANELS == ["desmin", "fragments", "link", "linked", "references"]


def test_L_toggles_the_outlines(tmp_path):
    """Every other page in this tool binds L to hide/show outlines. The first build of
    this one bound nothing to L, so the reviewer pressed it, saw no change, and
    concluded the page had no overlays at all."""
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b",
                                reviewer="reviewer_01",
                                session_started_at="2026-07-29T00:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    assert "function toggleOverlay()" in html
    assert "k === 'l'" in html, "L must be bound in the keydown handler"
    assert 'onclick="toggleOverlay()"' in html, "and reachable without the keyboard"
    assert "outlines hidden" in html, "the hidden state must be visible, not silent"


def test_a_case_opens_on_an_overlay_view_not_the_clean_image(tmp_path):
    """Opening on raw Desmin made the packet look un-annotated. The reviewer has to see
    that overlays exist before they can think to look for them."""
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b",
                                reviewer="reviewer_01",
                                session_started_at="2026-07-29T00:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    assert "DATA.PANELS.indexOf('fragments')" in html
    assert "let idx = 0, panel = FIRST_OVERLAY" in html


def test_the_panel_bar_precedes_the_image_so_it_cannot_be_pushed_offscreen(tmp_path):
    """`body` is height:100vh/overflow:hidden. With the panel bar after <main> and a
    fixed calc() height on the image, a taller footer pushed the bar out of view --
    removing the only on-screen hint that other views exist."""
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b",
                                reviewer="reviewer_01",
                                session_started_at="2026-07-29T00:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    assert html.index('id="panelbar"') < html.index("<main>")
    assert "header,.panelbar,footer{flex:0 0 auto}" in html
    img_rule = html[html.index("#img{"):html.index("}", html.index("#img{"))]
    assert "calc(" not in img_rule, f"no viewport arithmetic on the image: {img_rule}"
    assert "max-height:100%" in img_rule


def test_the_image_cannot_paint_over_the_controls(tmp_path):
    """The image rendered at its natural size and covered the view bar and the decision
    buttons. Two independent reasons, both pinned here:

    * `max-height:100%` on the image is silently ignored unless its parent has a
      *definite* height -- which #stage only gets from `align-items:stretch` on main
      (`center` leaves it auto-height);
    * `overflow:hidden` on main is the backstop that clips rather than covers.
    """
    out = build_over_merge_page([_case()], tmp_path / "p.html", batch_id="b",
                                reviewer="reviewer_01",
                                session_started_at="2026-07-29T00:00:00Z", threshold=0.9)
    html = open(out, encoding="utf-8").read()
    main_rule = html[html.index("main{"):html.index("}", html.index("main{"))]
    assert "overflow:hidden" in main_rule, f"main must clip: {main_rule}"
    assert "align-items:stretch" in main_rule, (
        "align-items:center leaves #stage auto-height, which makes the image's "
        f"max-height:100% a no-op: {main_rule}")
    stage_rule = html[html.index("#stage{"):html.index("}", html.index("#stage{"))]
    assert "min-height:0" in stage_rule and "flex:1 1 auto" in stage_rule, stage_rule


def test_a_case_missing_panels_is_refused(tmp_path):
    bad = {"uid": "om_001", "n_fragments": 2}
    with pytest.raises(ValueError, match="panels"):
        build_over_merge_page([bad], tmp_path / "p.html", batch_id="b",
                              reviewer="reviewer_01",
                              session_started_at="2026-07-29T00:00:00Z", threshold=0.9)


# ------------------------------------------------------------------ rendering


def test_line_rasterises_between_the_two_endpoints():
    seg = _line((40, 40), (5, 5), (30, 30), thickness=1)
    assert seg[5, 5] and seg[30, 30] and seg[17, 17]
    assert not seg[5, 35]


def test_line_clips_outside_the_crop_instead_of_raising():
    seg = _line((20, 20), (-10, -10), (25, 25))
    assert seg.any()


def test_outline_is_hollow():
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:20, 10:20] = True
    edge = _outline(mask, width=1)
    assert edge[10, 10] and not edge[15, 15]


def _synthetic_field():
    fiber = np.zeros((120, 120), dtype=np.uint16)
    fiber[50:60, 10:110] = 3000
    kept = np.zeros((120, 120), dtype=np.int32)
    kept[50:60, 10:50] = 7          # fragment 7
    kept[50:60, 70:110] = 9         # fragment 9
    merged = np.where(kept > 0, 7, 0).astype(np.int32)
    return fiber, kept, merged


def test_render_produces_every_panel_and_counts_references_in_view():
    fiber, kept, merged = _synthetic_field()
    ref = np.zeros((10, 40), dtype=bool)
    ref[:, :] = True
    references = [{"id": "myotube_0001", "bbox": (50, 10, 60, 50), "mask": ref.copy()},
                  {"id": "myotube_0002", "bbox": (50, 70, 60, 110), "mask": ref.copy()},
                  {"id": "myotube_0003", "bbox": (5, 5, 15, 15), "mask":
                   np.ones((10, 10), dtype=bool)}]
    panels, crop, n_refs = render_case_panels(
        fiber, None, kept, merged, references, fragment_ids=[7, 9], merged_label=7,
        links=[((55, 50), (55, 70))], bbox=[50, 10, 60, 110], pad_px=5, size=200)
    assert set(panels) == set(PANELS)
    assert all(v.startswith("data:image/jpeg;base64,") for v in panels.values())
    assert n_refs == 2, "only references intersecting the crop are drawn"
    assert crop == (45, 5, 65, 115)


def test_reference_panel_is_built_the_same_way_when_nothing_overlaps():
    """A control with no reference masks in view must still get all five panels --
    an empty panel set would identify the controls at a glance."""
    fiber, kept, merged = _synthetic_field()
    panels, _crop, n_refs = render_case_panels(
        fiber, None, kept, merged, [], fragment_ids=[7, 9], merged_label=7,
        links=[], bbox=[50, 10, 60, 110], pad_px=5, size=200)
    assert set(panels) == set(PANELS)
    assert n_refs == 0


def test_desmin_panel_differs_from_the_overlay_panels():
    """Panel 1 is the untouched ground of appeal; if it were identical to an overlay
    panel the reviewer would have no unannotated view."""
    fiber, kept, merged = _synthetic_field()
    panels, _crop, _n = render_case_panels(
        fiber, None, kept, merged, [], fragment_ids=[7, 9], merged_label=7,
        links=[((55, 50), (55, 70))], bbox=[50, 10, 60, 110], pad_px=5, size=200)
    assert panels["desmin"] != panels["fragments"]
    assert panels["desmin"] != panels["linked"]
    assert panels["desmin"] != panels["link"]
