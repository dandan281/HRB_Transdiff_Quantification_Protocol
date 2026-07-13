"""napari widgets for the myotube pipeline.

Five dock widgets (contributed via napari.yaml):
  * Load run          -- open a runs/<stem>/ (composite image + editable trace layer)
  * Save traces       -- write an edited Shapes layer back to traces.txt
  * Import ROIs       -- load an ImageJ ROI .zip as editable traces
  * Export ROIs       -- write a Shapes layer to an ImageJ ROI .zip (Fiji round-trip)
  * Export training   -- turn image + curated traces into a Cellpose training pair
  * Cellpose trace    -- (experimental) run a Cellpose model and load its traces for review

napari uses (row, col) = (y, x) order for Shapes; the pipeline uses (x, y). Every boundary
here swaps with ``[:, ::-1]`` so on-disk files stay in pipeline (x, y) order.

Widgets are built with magicgui and only imported by napari at runtime, so the pure-Python
I/O modules stay importable (and testable) without napari installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from magicgui import magic_factory

from ._traces_io import read_traces, write_traces
from ._roi_io import (read_imagej_zip, write_imagej_zip, export_cellpose_pair,
                      label_to_centerlines)

_IMG_NAMES = ("_composite.png", "_composite_preview.png")   # preferred review images
_TRACE_ORDER = ("stage4_qc/final_traces.txt", "stage4_qc/combined_traces.txt",
                "stage2_bright/bright_traces.txt")


def _read_image(path: Path) -> np.ndarray:
    from skimage.io import imread
    return imread(str(path))


def _find_review_image(run_dir: Path) -> Path | None:
    s4 = run_dir / "stage4_qc"
    for suffix in _IMG_NAMES:
        hits = sorted(s4.glob(f"*{suffix}"))
        if hits:
            return hits[0]
    adj = sorted((run_dir / "stage1_threshold").glob("ch*_adjusted8.tif"))
    return adj[0] if adj else None


def _find_traces(run_dir: Path) -> Path | None:
    for rel in _TRACE_ORDER:
        p = run_dir / rel
        if p.exists():
            return p
    return None


def _paths_from_polys(polys):
    """(x, y) polylines -> napari path data (y, x)."""
    return [np.asarray(p, dtype=float)[:, ::-1] for p in polys]


def _polys_from_layer(layer):
    """napari Shapes -> list of (x, y) polylines, keeping only line/path shapes."""
    polys = []
    for data, kind in zip(layer.data, layer.shape_type):
        if kind in ("path", "line"):
            polys.append(np.asarray(data, dtype=float)[:, ::-1])   # (y, x) -> (x, y)
    return polys


@magic_factory(call_button="Load run", run_dir={"widget_type": "FileEdit", "mode": "d",
                                                "label": "runs/<stem>/"})
def load_run(viewer: "napari.Viewer", run_dir=Path()):  # noqa: F821
    """Load a run's review image + its traces as an editable Shapes layer."""
    run_dir = Path(run_dir)
    img_path = _find_review_image(run_dir)
    if img_path is None:
        raise FileNotFoundError(f"no composite/adjusted image under {run_dir}")
    viewer.add_image(_read_image(img_path), name=run_dir.name)
    tr_path = _find_traces(run_dir)
    if tr_path is not None:
        paths = _paths_from_polys(read_traces(tr_path))
        viewer.add_shapes(paths, shape_type="path", edge_color="cyan", edge_width=3,
                          name="traces", metadata={"source": str(tr_path)})


@magic_factory(call_button="Save traces", out_path={"widget_type": "FileEdit", "mode": "w",
                                                    "label": "traces.txt"})
def save_traces(traces: "napari.layers.Shapes", out_path=Path("final_traces.txt")):  # noqa: F821
    """Write an edited Shapes layer back to a pipeline traces.txt (x, y order)."""
    n = write_traces(out_path, _polys_from_layer(traces))
    print(f"[myotube] wrote {n} traces -> {out_path}")


@magic_factory(call_button="Import ROIs", roi_zip={"widget_type": "FileEdit",
                                                   "filter": "*.zip", "label": "ImageJ ROI .zip"})
def import_rois(viewer: "napari.Viewer", roi_zip=Path()):  # noqa: F821
    """Load an ImageJ RoiManager .zip (e.g. your ground truth) as editable traces."""
    paths = _paths_from_polys(read_imagej_zip(roi_zip))
    viewer.add_shapes(paths, shape_type="path", edge_color="yellow", edge_width=3,
                      name=Path(roi_zip).stem)


@magic_factory(call_button="Export ROIs", out_zip={"widget_type": "FileEdit", "mode": "w",
                                                   "filter": "*.zip", "label": "out .zip"})
def export_rois(traces: "napari.layers.Shapes", out_zip=Path("rois.zip")):  # noqa: F821
    """Write a Shapes layer to an ImageJ polyline-ROI .zip (opens in Fiji)."""
    n = write_imagej_zip(out_zip, _polys_from_layer(traces))
    print(f"[myotube] wrote {n} ROIs -> {out_zip}")


@magic_factory(call_button="Export training pair",
               out_dir={"widget_type": "FileEdit", "mode": "d", "label": "training/ dir"})
def export_training(image: "napari.layers.Image",           # noqa: F821
                    traces: "napari.layers.Shapes",          # noqa: F821
                    out_dir=Path("training"), stem="well", fiber_width_px=15.0):
    """image + curated traces -> Cellpose pair (<stem>.tif + <stem>_masks.tif)."""
    img = np.asarray(image.data)
    n = export_cellpose_pair(img, _polys_from_layer(traces), out_dir, stem, fiber_width_px)
    print(f"[myotube] {stem}: wrote image + {n}-label mask -> {out_dir}")


@magic_factory(call_button="Run Cellpose",
               model_path={"widget_type": "FileEdit", "label": "model (blank = cyto3)"})
def cellpose_trace(viewer: "napari.Viewer", image: "napari.layers.Image",   # noqa: F821
                   model_path=Path(), diameter=0.0, flow_threshold=0.4):
    """Experimental: run a Cellpose model on the image; load its traces for review.

    Leave ``model_path`` blank to use the pretrained ``cyto3``; point it at your fine-tuned
    model once trained. Masks are skeletonized to centerlines and loaded as an editable layer.
    """
    try:
        from cellpose import models
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Cellpose is not installed in this env "
                           "(`pip install cellpose`).") from exc
    img = np.asarray(image.data)
    mp = str(model_path)
    if mp and Path(mp).exists():
        model = models.CellposeModel(pretrained_model=mp, gpu=False)
    else:
        model = models.CellposeModel(model_type="cyto3", gpu=False)
    dia = diameter or None
    masks = model.eval(img, diameter=dia, flow_threshold=flow_threshold, channels=[0, 0])[0]
    paths = _paths_from_polys(label_to_centerlines(masks))
    viewer.add_shapes(paths, shape_type="path", edge_color="magenta", edge_width=3,
                      name="cellpose_traces")
    print(f"[myotube] cellpose: {len(paths)} objects")
