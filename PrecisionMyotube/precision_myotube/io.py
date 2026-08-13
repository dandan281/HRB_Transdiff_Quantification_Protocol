"""ND2 input, channel-role resolution, and immutable metadata."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from skimage.measure import label, regionprops


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def channel_scores(image: np.ndarray) -> dict:
    image = np.asarray(image)
    nonzero = image[image > 0]
    if not nonzero.size:
        return {"nuclei": 0, "fiber": 0.0, "p975": 0.0}
    bright_labels = label(image > np.percentile(nonzero, 97))
    areas = np.bincount(bright_labels.ravel())
    keep = areas >= 10
    keep[0] = False
    bright = keep[bright_labels]
    props = regionprops(label(bright))
    nuclei = sum(1 for p in props if p.area < 600 and p.eccentricity < 0.85)
    fiber = sum(float(p.area) for p in props
                if p.area >= 150 and p.eccentricity > 0.9 and p.major_axis_length > 40)
    return {"nuclei": int(nuclei), "fiber": fiber,
            "p975": float(np.percentile(nonzero, 97.5))}


def resolve_roles(channels: np.ndarray, force_fiber: int | None = None,
                  force_dapi: int | None = None) -> tuple[dict, dict]:
    if channels.ndim != 3 or channels.shape[0] < 2:
        raise ValueError("expected channels-first image (C,Y,X)")
    scores = {idx: channel_scores(channels[idx]) for idx in range(channels.shape[0])}
    dapi = force_dapi if force_dapi is not None else max(scores, key=lambda c: scores[c]["nuclei"])
    candidates = [c for c in scores if c != dapi]
    if force_fiber is not None:
        if force_fiber == dapi or force_fiber not in scores:
            raise ValueError("forced fiber channel conflicts with DAPI or is out of range")
        fiber = force_fiber
    else:
        # All currently validated Q-plate acquisitions use ch1 for Desmin. Morphology alone
        # cannot reliably distinguish Desmin from an elongated receptor channel, so preserve the
        # acquisition prior and record when the morphology score disagrees. New acquisition
        # layouts must use --fiber-ch explicitly.
        fiber = 1 if 1 in candidates else max(candidates, key=lambda c: scores[c]["fiber"])
    other = [c for c in scores if c not in {dapi, fiber}]
    return {"fiber": int(fiber), "dapi": int(dapi), "other": other}, scores


def load_nd2(path: str | Path) -> tuple[np.ndarray, float, dict]:
    import nd2
    path = Path(path).resolve()
    with nd2.ND2File(path) as handle:
        sizes = dict(handle.sizes)
        if any(axis in sizes for axis in ("T", "Z", "P")):
            raise ValueError(
                f"this validated workflow accepts one 2-D field only; ND2 dimensions are {sizes}"
            )
        array = handle.asarray()
        voxel = handle.voxel_size()
        pixel_um = float(voxel.x)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"expected C,Y,X ND2 data; got shape {array.shape}")
    return array, pixel_um, sizes


def prepare_run(nd2_path: str | Path, out_dir: str | Path, *,
                force_fiber: int | None = None, force_dapi: int | None = None) -> dict:
    path = Path(nd2_path).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metadata.json").exists():
        raise FileExistsError(
            f"{out / 'metadata.json'} already exists; use a new run directory to preserve provenance"
        )
    channels, pixel_um, sizes = load_nd2(path)
    roles, scores = resolve_roles(channels, force_fiber, force_dapi)
    for idx, channel in enumerate(channels):
        np.save(out / f"ch{idx}_raw16.npy", channel)
    metadata = {
        "schema_version": "1.0",
        "pipeline": "precision-myotube",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_nd2": path.as_posix(),
        "source_sha256": sha256_file(path),
        "image_id": path.stem,
        "image_shape": [int(channels.shape[1]), int(channels.shape[2])],
        "nd2_sizes": sizes,
        "dtype": str(channels.dtype),
        "pixel_um": pixel_um,
        "channels": roles,
        "channel_scores": {str(k): v for k, v in scores.items()},
        "channel_role_method": {
            "fiber": "explicit" if force_fiber is not None else "validated Q-plate ch1 prior",
            "dapi": "explicit" if force_dapi is not None else "nuclei morphology score",
            "requires_human_confirmation": force_fiber is None,
        },
        "authoritative_policy": "reviewed complete instances only",
        "canonical_parameters": {
            "minimum_real_fiber_length_um": 50.0,
            "nucleus_area_um2": [50.0, 500.0],
            "conversion_overlap_fractions": [0.4, 0.5, 0.6],
            "primary_conversion_overlap_fraction": 0.5,
            "instance_assignment_fraction": 0.5,
            "instance_assignment_margin": 0.25,
        },
        "software_versions": {
            name: _package_version(name)
            for name in ("precision-myotube", "numpy", "scipy", "scikit-image", "nd2",
                         "cellpose", "skan", "networkx")
        },
        "stage_metadata": ["territory_metadata.json", "nuclei_metadata.json"],
        "qc_history": "qc_history.jsonl",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_metadata(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / "metadata.json").read_text(encoding="utf-8"))


def load_run_channel(run_dir: str | Path, role: str) -> np.ndarray:
    run = Path(run_dir)
    metadata = load_metadata(run)
    channel = metadata["channels"][role]
    return np.load(run / f"ch{channel}_raw16.npy", mmap_mode="r")
