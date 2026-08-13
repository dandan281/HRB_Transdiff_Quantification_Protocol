"""Load a Codex ``annotation-package`` directory into an editable session.

The package (produced by ``precision_myotube annotation-package``) contains:

* ``fiber_raw16.tif`` / ``dapi_raw16.tif`` -- native 16-bit channels (context).
* ``semantic_territory.tif``               -- Desmin territory (prompt/context).
* ``overlap_ignore.tif``                   -- overlap ignore hint.
* ``starting_labels.tif``                  -- mutually exclusive component
                                              proposals (loaded as prompts).
* ``instance_properties.csv``              -- per-label id/status/source/notes.
* ``README.json``                          -- image_id, pixel_um, warnings.

Every ``starting_labels`` component enters the session as a *prompt* instance:
distinct from accepted masks, never authoritative, and safe to delete without
touching the raw channels.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .model import AnnotationSession, SOURCE_PROMPT


@dataclass
class LoadedPackage:
    session: AnnotationSession
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    context_layers: dict[str, np.ndarray] = field(default_factory=dict)
    readme: dict = field(default_factory=dict)
    prompt_ids: list[str] = field(default_factory=list)


def _read_tif(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    import tifffile
    return np.asarray(tifffile.imread(str(path)))


def load_annotation_package(package_dir: str | Path) -> LoadedPackage:
    package_dir = Path(package_dir)
    if not package_dir.is_dir():
        raise FileNotFoundError(f"annotation package not found: {package_dir}")

    readme_path = package_dir / "README.json"
    readme = json.loads(readme_path.read_text(encoding="utf-8")) if readme_path.is_file() else {}
    image_id = readme.get("image_id") or package_dir.name
    pixel_um = readme.get("pixel_um")

    fiber = _read_tif(package_dir / "fiber_raw16.tif")
    dapi = _read_tif(package_dir / "dapi_raw16.tif")
    territory = _read_tif(package_dir / "semantic_territory.tif")
    overlap_ignore = _read_tif(package_dir / "overlap_ignore.tif")
    labels = _read_tif(package_dir / "starting_labels.tif")

    # Determine the field shape from whatever we have (channels preferred).
    shape_source = next((a for a in (fiber, dapi, labels, territory) if a is not None), None)
    if shape_source is None:
        raise ValueError("package has no channel or label image to size the field")
    image_shape = (int(shape_source.shape[0]), int(shape_source.shape[1]))

    session = AnnotationSession(image_shape, image_id, pixel_um=pixel_um)

    channels = {}
    if fiber is not None:
        channels["fiber"] = fiber
    if dapi is not None:
        channels["dapi"] = dapi
    context_layers = {}
    if territory is not None:
        context_layers["semantic_territory"] = territory
    if overlap_ignore is not None:
        context_layers["overlap_ignore"] = overlap_ignore

    # Per-label properties (status/source/notes/id) if provided.
    props: dict[int, dict] = {}
    props_path = package_dir / "instance_properties.csv"
    if props_path.is_file():
        with props_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                props[int(row["label"])] = row

    prompt_ids: list[str] = []
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 2:
            raise ValueError("starting_labels must be a 2-D label image")
        for label_id, slices in enumerate(ndi.find_objects(labels), start=1):
            if slices is None:
                continue
            local = labels[slices] == label_id
            row = props.get(label_id, {})
            iid = (row.get("id") or f"myotube_{label_id:04d}").strip()
            src = (row.get("source") or SOURCE_PROMPT).strip()
            origin = (int(slices[0].start), int(slices[1].start))
            from . import masks as _masks
            mask = _masks.SparseMask(origin, local, image_shape)
            prompt_ids.append(session.load_prompt(mask, source=src, instance_id=iid))

    return LoadedPackage(session=session, channels=channels,
                         context_layers=context_layers, readme=readme,
                         prompt_ids=prompt_ids)
