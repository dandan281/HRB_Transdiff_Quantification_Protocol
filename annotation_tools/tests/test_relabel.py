"""Relabelling: trace log replay, rasterisation, and the ignore mask.

The ignore-mask tests are the important ones. Everything else here is
bookkeeping; `unlabelled_fibre_ignore` is the piece that decides whether partial
annotation is safe to train on or actively harmful.
"""
from __future__ import annotations

import numpy as np
import pytest

from annotation_tools.relabel.raster import (compose_labels, polyline_pixels,
                                             ribbon_mask, snap_mask,
                                             unlabelled_fibre_ignore)
from annotation_tools.relabel.store import TraceStore


# ------------------------------------------------------------------- the log

def test_log_replays_add_edit_delete(tmp_path):
    s = TraceStore(tmp_path, "w1")
    a = s.append({"kind": "add", "points": [[0, 0], [0, 9]], "width_px": 6})
    s.append({"kind": "add", "points": [[5, 0], [5, 9]], "width_px": 6})
    assert len(s.current()["traces"]) == 2

    s.append({"kind": "edit", "replaces": a["trace_id"],
              "points": [[0, 0], [0, 20]], "width_px": 8})
    live = s.current()["traces"]
    assert len(live) == 2, "an edit replaces, it does not add"
    assert live[a["trace_id"]]["width_px"] == 8

    s.append({"kind": "delete", "replaces": a["trace_id"]})
    assert len(s.current()["traces"]) == 1


def test_log_is_append_only(tmp_path):
    """History survives deletion -- the log is the audit trail."""
    s = TraceStore(tmp_path, "w1")
    a = s.append({"kind": "add", "points": [[0, 0], [0, 9]]})
    s.append({"kind": "delete", "replaces": a["trace_id"]})
    assert len(s.records()) == 2
    assert s.current()["traces"] == {}


def test_reject_existing_and_undo(tmp_path):
    s = TraceStore(tmp_path, "w1")
    s.append({"kind": "reject_existing", "source_label": 7})
    assert 7 in s.current()["rejected_existing"]
    s.append({"kind": "reject_existing", "source_label": 7, "undo": True})
    assert s.current()["rejected_existing"] == {}


def test_unknown_kind_rejected(tmp_path):
    with pytest.raises(ValueError):
        TraceStore(tmp_path, "w1").append({"kind": "obliterate"})


# --------------------------------------------------------------- rasterising

def test_polyline_needs_two_points():
    with pytest.raises(ValueError):
        polyline_pixels([[1, 1]])


def test_ribbon_width_is_a_true_half_width():
    m = ribbon_mask([[20, 5], [20, 45]], (40, 50), width_px=8)
    col = m[:, 25]
    assert 7 <= col.sum() <= 10, "an 8 px ribbon should be ~8 px across"


def test_snap_hugs_signal_when_signal_is_there():
    img = np.zeros((40, 50), dtype=np.float32)
    img[18:23, 5:45] = 1.0                       # a 5 px wide bright fibre
    mask, info = snap_mask([[20, 5], [20, 44]], img, width_px=6)
    assert info["mode"] == "snap"
    # It should track the bright band rather than the drawn 6 px ribbon.
    assert mask[:, 25].sum() <= 8
    assert mask[19:22, 25].all()


def test_snap_falls_back_rather_than_erasing_a_faint_fibre():
    """A snap that would delete the object must yield to the operator."""
    img = np.zeros((40, 50), dtype=np.float32)   # no signal at all
    mask, info = snap_mask([[20, 5], [20, 44]], img, width_px=6)
    assert info["mode"] == "ribbon"
    assert mask.sum() > 0


# ------------------------------------------------------- THE ignore mask fix

def test_unlabelled_fibre_becomes_ignore_not_background():
    """The bug this package exists to fix.

    Two fibres in the territory, one labelled. The unlabelled one must be
    ignored -- if it stays background, the loss teaches the model to suppress
    exactly what it is supposed to detect.
    """
    terr = np.zeros((60, 60), dtype=bool)
    terr[10:14, 5:55] = True                     # labelled fibre
    terr[40:44, 5:55] = True                     # unlabelled fibre
    labels = np.zeros((60, 60), dtype=np.int32)
    labels[10:14, 5:55] = 1

    ignore, stats = unlabelled_fibre_ignore(terr, labels, halo_px=3)
    assert ignore[40:44, 5:55].all(), "unlabelled fibre must be ignored"
    assert not ignore[10:14, 5:55].any(), "the label itself is never ignored"
    assert stats["unlabelled_fibre_ignored_px"] > 0


def test_halo_keeps_the_label_edge_as_background():
    """The fibre/background transition is the distance-field signal; ignoring it
    would blur every edge the operator drew."""
    terr = np.ones((40, 40), dtype=bool)
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[18:22, 10:30] = 1

    ignore, _ = unlabelled_fibre_ignore(terr, labels, halo_px=5)
    assert not ignore[23, 20], "just outside the label stays background"
    assert ignore[35, 20], "far away is ignored"


def test_nothing_labelled_means_all_territory_ignored():
    terr = np.zeros((20, 20), dtype=bool)
    terr[5:9, :] = True
    ignore, _ = unlabelled_fibre_ignore(terr, np.zeros((20, 20), np.int32))
    assert (ignore == terr).all()


# ------------------------------------------------------------------ composing

def test_existing_certified_masks_win_over_new_traces():
    """A relabelling pass may add supervision; it may not silently rewrite a
    mask a human already certified."""
    base = np.zeros((40, 60), dtype=np.int32)
    base[18:22, 5:55] = 1
    img = np.zeros((40, 60), dtype=np.float32)
    trace = {"trace_id": "t1", "points": [[20, 5], [20, 54]],
             "width_px": 10, "mode": "ribbon"}

    labels, prov = compose_labels(base, [trace], img)
    assert (labels[18:22, 5:55] == 1).all(), "certified pixels are untouched"
    skipped = [p for p in prov if p.get("skipped")]
    assert len(labels_ids(labels)) >= 1
    # the overlapping trace either vanished or kept only its non-overlapping rim
    assert skipped or labels.max() == 2


def test_rejected_existing_is_dropped_and_ids_stay_contiguous():
    base = np.zeros((30, 30), dtype=np.int32)
    base[2:5, 2:20] = 1
    base[10:13, 2:20] = 2
    base[20:23, 2:20] = 3
    labels, prov = compose_labels(base, [], np.zeros((30, 30), np.float32),
                                  rejected={2})
    assert labels.max() == 2, "3 instances minus 1 rejected, renumbered 1..2"
    assert sorted(labels_ids(labels)) == [1, 2]
    assert all(p["origin"] == "bootstrap_v1" for p in prov)


def labels_ids(lab):
    return [v for v in np.unique(lab) if v]
