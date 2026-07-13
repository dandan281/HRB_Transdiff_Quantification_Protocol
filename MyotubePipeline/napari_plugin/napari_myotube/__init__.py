"""napari-myotube -- review / curate myotube traces and bridge to Cellpose.

The pure-Python I/O (traces.txt, ImageJ ROI, Cellpose label masks) lives in ``_traces_io``
and ``_roi_io`` and imports NO napari, so it is usable from scripts and tests in the base env.
The napari dock widgets live in ``_widget`` (imported lazily by napari via napari.yaml).
"""
from __future__ import annotations

__version__ = "0.1.0"

from ._traces_io import read_traces, write_traces
from ._roi_io import (
    read_imagej_zip,
    write_imagej_zip,
    polylines_to_label_mask,
    export_cellpose_pair,
    label_to_centerlines,
)

__all__ = [
    "read_traces",
    "write_traces",
    "read_imagej_zip",
    "write_imagej_zip",
    "polylines_to_label_mask",
    "export_cellpose_pair",
    "label_to_centerlines",
]
