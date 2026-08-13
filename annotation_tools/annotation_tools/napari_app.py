"""napari / micro-sam view over :class:`AnnotationSession` (CL01).

The GUI is deliberately a thin shell: all state and every scientific guardrail
live in :mod:`annotation_tools.model`, so they are unit-tested without a display.
``napari`` and ``magicgui`` are imported lazily inside :func:`launch` so the rest
of the package (and its tests) run in a headless environment.

Design notes
------------
* **Prompt vs truth styling.** Automated proposals load in a ``prompts`` labels
  layer with low opacity and a distinct colormap, named "PROMPTS - not truth".
  Accepted instances render in a separate layer. A convenient click can never
  make a proposal look accepted.
* **Overlap-safe editing.** napari's Labels layer is mutually exclusive, so
  crossing arms are edited one at a time in an ``active edit`` scratch layer and
  committed back to the overlap-safe session. Two crossing myotubes therefore
  remain two full masks even though a single raster shows one colour per pixel.
* **No bulk authority.** There is no "mark all complete" button; status/review
  are set for the active instance only, mirroring the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import AnnotationSession
from .package import load_annotation_package, LoadedPackage
from ._schema_bridge import VALID_STATUSES


def _overview_label_image(session: AnnotationSession) -> tuple[np.ndarray, dict[int, str]]:
    """Build a display-only label raster of accepted instances (last-writer wins
    at overlaps; the authoritative session keeps both)."""
    raster = np.zeros(session.image_shape, dtype=np.int32)
    mapping: dict[int, str] = {}
    for value, inst in enumerate(session.accepted_instances(), start=1):
        r0, c0, r1, c1 = inst.mask.bbox
        raster[r0:r1, c0:c1][inst.mask.crop] = value
        mapping[value] = inst.id
    return raster, mapping


def _prompt_label_image(session: AnnotationSession) -> tuple[np.ndarray, dict[int, str]]:
    raster = np.zeros(session.image_shape, dtype=np.int32)
    mapping: dict[int, str] = {}
    for value, iid in enumerate(session.unresolved_prompt_ids(), start=1):
        inst = session.instances[iid]
        r0, c0, r1, c1 = inst.mask.bbox
        raster[r0:r1, c0:c1][inst.mask.crop] = value
        mapping[value] = iid
    return raster, mapping


def launch(package_dir: str | Path, export_path: str | Path | None = None):  # pragma: no cover
    """Open the annotation GUI on a Codex annotation package.

    Not exercised in headless CI; the logic it drives lives in the tested model.
    """
    import napari

    viewer, session = build_viewer(package_dir, export_path)
    napari.run()
    return session


def build_viewer(package_dir, export_path=None):  # pragma: no cover - needs Qt
    """Construct the annotation viewer and dock widgets without running the event
    loop. Separated from :func:`launch` so the wiring can be smoke-tested with an
    offscreen Qt platform.
    """
    import napari
    from magicgui import magicgui

    loaded: LoadedPackage = load_annotation_package(package_dir)
    session = loaded.session
    default_export = Path(export_path) if export_path else Path(package_dir) / f"{session.image_id}.instances.json"

    viewer = napari.Viewer(title=f"PrecisionMyotube annotation - {session.image_id}")

    # --- context channels (read-only, native resolution, intensity preserved) --
    if "fiber" in loaded.channels:
        viewer.add_image(loaded.channels["fiber"], name="fiber (Desmin)",
                         colormap="green", blending="additive",
                         contrast_limits=_p_limits(loaded.channels["fiber"]))
    if "dapi" in loaded.channels:
        viewer.add_image(loaded.channels["dapi"], name="DAPI (context)",
                         colormap="blue", blending="additive",
                         contrast_limits=_p_limits(loaded.channels["dapi"]))
    if "semantic_territory" in loaded.context_layers:
        viewer.add_image(loaded.context_layers["semantic_territory"].astype(float),
                         name="Desmin territory (PROMPT)", colormap="magenta",
                         opacity=0.30, blending="additive")

    # --- prompt (proposal) labels: visually distinct, low opacity ---
    prompt_raster, prompt_map = _prompt_label_image(session)
    prompts_layer = viewer.add_labels(prompt_raster, name="PROMPTS - not truth", opacity=0.35)

    # --- accepted instances overview (display only) ---
    overview_raster, overview_map = _overview_label_image(session)
    overview_layer = viewer.add_labels(overview_raster, name="accepted instances", opacity=0.55)

    # --- single-instance edit scratch (overlap-safe commit target) ---
    edit_layer = viewer.add_labels(np.zeros(session.image_shape, dtype=np.int32),
                                   name="active edit", opacity=0.6)
    state = {"active_id": None}

    def _refresh():
        overview_layer.data, m = _overview_label_image(session)
        overview_map.clear(); overview_map.update(m)
        prompts_layer.data, pm = _prompt_label_image(session)
        prompt_map.clear(); prompt_map.update(pm)

    @magicgui(call_button="Accept prompt under cursor -> instance")
    def accept_prompt():
        iid = _picked(prompts_layer, prompt_map)
        if iid is None:
            return _status(viewer, "Hover a prompt in the PROMPTS layer first.")
        session.accept_prompt(iid)
        state["active_id"] = iid
        _load_active(session, iid, edit_layer)
        _refresh()
        _status(viewer, f"Accepted {iid}; edit it in 'active edit', then Commit.")

    @magicgui(call_button="New empty instance for editing")
    def new_instance():
        edit_layer.data = np.zeros(session.image_shape, dtype=np.int32)
        state["active_id"] = "__new__"
        _status(viewer, "Draw the myotube body in 'active edit', then Commit.")

    @magicgui(call_button="Load accepted instance under cursor for editing")
    def edit_instance():
        iid = _picked(overview_layer, overview_map)
        if iid is None:
            return _status(viewer, "Hover an accepted instance first.")
        state["active_id"] = iid
        _load_active(session, iid, edit_layer)
        _status(viewer, f"Editing {iid}. Commit to save, overlap-safe.")

    @magicgui(call_button="Commit active edit")
    def commit_edit():
        mask = edit_layer.data > 0
        if not mask.any():
            return _status(viewer, "Active edit is empty; nothing to commit.")
        active = state["active_id"]
        if active in (None, "__new__"):
            new_id = session.create(mask, status="ambiguous", reviewer=session.reviewer)
            state["active_id"] = new_id
            _status(viewer, f"Created {new_id} (status=ambiguous).")
        else:
            session.refine(active, mask)
            _status(viewer, f"Updated {active}.")
        _refresh()

    @magicgui(call_button="Merge active with instance under cursor")
    def merge_instances():
        active = state["active_id"]
        other = _picked(overview_layer, overview_map)
        if active in (None, "__new__") or other is None or active == other:
            return _status(viewer, "Need an active instance and a different target.")
        new_id = session.merge(active, other)
        state["active_id"] = new_id
        _load_active(session, new_id, edit_layer); _refresh()
        _status(viewer, f"Merged into {new_id}.")

    @magicgui(status={"choices": sorted(VALID_STATUSES)},
              call_button="Apply status / review to active instance")
    def set_review(status: str = "ambiguous", reviewed: bool = False,
                   reviewer: str = "", notes: str = ""):
        active = state["active_id"]
        if active in (None, "__new__"):
            return _status(viewer, "Commit the active instance first.")
        if reviewer:
            session.set_reviewer(active, reviewer)
        session.set_status(active, status)
        session.set_notes(active, notes)
        session.set_reviewed(active, reviewed, reviewer=reviewer or None)
        _status(viewer, f"{active}: status={status} reviewed={reviewed}.")

    @magicgui(call_button="Export InstanceSet + review log")
    def export():
        try:
            info = session.save(default_export)
        except Exception as exc:  # AnnotationError -> surfaced to the reviewer
            return _status(viewer, f"Export blocked: {exc}")
        _status(viewer, f"Exported {info['n_exported']} instances "
                        f"({info['n_authoritative']} authoritative) -> {info['instances']}")

    for widget, name in [(new_instance, "New"), (edit_instance, "Edit"),
                         (commit_edit, "Commit"), (accept_prompt, "Accept prompt"),
                         (merge_instances, "Merge"), (set_review, "Status/review"),
                         (export, "Export")]:
        viewer.window.add_dock_widget(widget, name=name, area="right")

    return viewer, session


def _p_limits(arr: np.ndarray) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    return (lo, hi if hi > lo else lo + 1.0)


def _picked(layer, mapping):  # pragma: no cover - interactive
    """Map the layer's currently selected label to an instance id."""
    try:
        value = int(layer.selected_label)
    except Exception:
        return None
    return mapping.get(value)


def _load_active(session, iid, edit_layer):  # pragma: no cover - interactive
    edit_layer.data = session.instances[iid].mask.full().astype(np.int32)


def _status(viewer, message):  # pragma: no cover - interactive
    viewer.status = message
