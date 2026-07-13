"""Tests for the napari-free I/O core (run in the base anaconda env; no napari/cellpose needed)."""
from __future__ import annotations

import numpy as np

from napari_myotube import (read_traces, write_traces, polylines_to_label_mask,
                            label_to_centerlines)


def test_traces_roundtrip(tmp_path):
    polys = [np.array([[10.0, 20.0], [10.5, 40.0], [11.0, 60.0]]),
             np.array([[100.0, 100.0], [200.0, 100.0]])]
    p = tmp_path / "t.txt"
    n = write_traces(p, polys)
    assert n == 2
    back = read_traces(p)
    assert len(back) == 2
    assert np.allclose(back[0], polys[0], atol=0.01)
    assert np.allclose(back[1], polys[1], atol=0.01)


def test_write_drops_degenerate(tmp_path):
    polys = [np.array([[1.0, 1.0]]),                     # < 2 vertices -> dropped
             np.array([[0.0, 0.0], [5.0, 5.0]])]
    p = tmp_path / "t.txt"
    assert write_traces(p, polys) == 1


def test_label_mask_assigns_unique_ids():
    # two horizontal fibers, well separated
    polys = [np.array([[10.0, 20.0], [80.0, 20.0]]),
             np.array([[10.0, 60.0], [80.0, 60.0]])]
    lab = polylines_to_label_mask(polys, (100, 100), fiber_width_px=6)
    assert lab.dtype == np.uint16
    assert set(np.unique(lab)) == {0, 1, 2}
    assert lab[20, 45] == 1 and lab[60, 45] == 2      # (row=y, col=x)


def test_mask_to_centerline_recovers_length():
    polys = [np.array([[10.0, 30.0], [90.0, 30.0]])]   # 80 px horizontal fiber
    lab = polylines_to_label_mask(polys, (60, 100), fiber_width_px=6)
    lines = label_to_centerlines(lab)
    assert len(lines) == 1
    line = lines[0]
    length = float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))
    assert 60 < length < 100                            # ~80 px, tolerant of skeleton ends
