"""Semantic territory, nuclei segmentation, and explicitly non-authoritative proposals."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import apply_hysteresis_threshold, sato
from skimage.measure import label
from skimage.morphology import disk, white_tophat

from .io import load_metadata, load_run_channel
from .schema import from_label_image


def _remove_small_foreground(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Version-stable equivalent of remove_small_objects for a binary image."""
    components, _ = ndi.label(mask)
    areas = np.bincount(components.ravel())
    keep = areas >= int(min_size)
    keep[0] = False
    return keep[components]


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Fill enclosed background components smaller than max_area without bridging objects."""
    holes, _ = ndi.label(~mask)
    areas = np.bincount(holes.ravel())
    touches_border = np.zeros(len(areas), dtype=bool)
    border = np.concatenate((holes[0], holes[-1], holes[:, 0], holes[:, -1]))
    touches_border[np.unique(border)] = True
    fill = (areas < int(max_area)) & ~touches_border
    fill[0] = False
    return mask | fill[holes]


def _background_and_clahe(image: np.ndarray, radius: int = 40) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, (1, 99.9))
    if hi <= lo:
        raise ValueError("fiber channel is blank or constant")
    scaled = rescale_intensity(image, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(np.float32)
    background_subtracted = white_tophat(scaled, disk(radius)).astype(np.float32)
    kernel = max(32, min(image.shape) // 16)
    enhanced = equalize_adapthist(background_subtracted, kernel_size=kernel,
                                  clip_limit=0.01).astype(np.float32)
    return background_subtracted, enhanced


def semantic_territory(image: np.ndarray, highs=(86.0, 89.0, 92.0, 94.0, 96.0),
                       band: float = 14.0, gate_pct: float = 90.0,
                       min_object_px: int = 180) -> tuple[np.ndarray, dict]:
    """Create Desmin territory and choose a threshold stability plateau.

    Connected-component counts are used only to choose a stable semantic threshold. They are
    never described as independent-myotube counts.
    """
    started = time.time()
    bg, enhanced = _background_and_clahe(image)
    response = rescale_intensity(
        sato(enhanced, sigmas=(1, 2, 4), black_ridges=False), out_range=(0.0, 1.0)
    ).astype(np.float32)
    nonzero = response[response > 0]
    signal = bg[bg > 0]
    if not nonzero.size or not signal.size:
        raise ValueError("fiber channel contains no detectable signal")
    gate = ndi.binary_dilation(bg > np.percentile(signal, gate_pct), structure=disk(1))
    masks: dict[float, np.ndarray] = {}
    records = []
    for high in highs:
        ridge = apply_hysteresis_threshold(
            response, np.percentile(nonzero, high - band), np.percentile(nonzero, high)
        )
        mask = _remove_small_foreground(ridge & gate, min_object_px)
        mask = _fill_small_holes(mask, min_object_px)
        masks[float(high)] = mask
        records.append({
            "high_pct": float(high), "components": int(label(mask).max()),
            "coverage_pct": float(mask.mean() * 100.0),
        })
    best_index = 0
    best_key = None
    for index in range(1, len(records) - 1):
        sensitivity = abs(records[index - 1]["components"] - records[index + 1]["components"])
        key = (sensitivity, -records[index]["coverage_pct"])
        if best_key is None or key < best_key:
            best_index, best_key = index, key
    selected = records[best_index]
    debug = {
        "method": "tophat-clahe-sato-hysteresis-intensity-gate",
        "selected_high_pct": selected["high_pct"],
        "gate_pct": gate_pct,
        "sweep": records,
        "seconds": round(time.time() - started, 3),
        "warning": "component counts are not independent-myotube counts",
    }
    return masks[selected["high_pct"]], debug


def create_territory(run_dir: str | Path) -> Path:
    run = Path(run_dir)
    mask, debug = semantic_territory(load_run_channel(run, "fiber"))
    metadata = load_metadata(run)
    from .fiber_gate import length_gated_territory
    real, gate_debug = length_gated_territory(mask, float(metadata["pixel_um"]), 50.0)
    np.save(run / "desmin_semantic_mask.npy", mask)
    path = run / "myotube_territory.npy"
    np.save(path, real)
    debug["real_myotube_gate"] = gate_debug
    (run / "territory_metadata.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
    return path


def segment_nuclei(image: np.ndarray, cellprobs=(-2.0, -1.0, 0.0, 1.0, 2.0),
                   model_path: str | None = None) -> tuple[np.ndarray, dict]:
    try:
        import torch
        from cellpose import models
    except ImportError as exc:
        raise RuntimeError("Cellpose is required unless --nuclei-masks is supplied") from exc
    kwargs = {"gpu": bool(torch.cuda.is_available())}
    if model_path:
        kwargs["pretrained_model"] = model_path
    model = models.CellposeModel(**kwargs)
    masks_by_cp = {}
    counts = []
    started = time.time()
    for threshold in cellprobs:
        result = model.eval(np.asarray(image, dtype=np.float32),
                            cellprob_threshold=float(threshold))
        masks = result[0]
        masks_by_cp[float(threshold)] = masks.astype(np.int32, copy=False)
        counts.append(int(masks.max()))
    best_index = min(range(1, len(cellprobs) - 1),
                     key=lambda i: (abs(counts[i - 1] - counts[i + 1]), abs(cellprobs[i])))
    selected = float(cellprobs[best_index])
    return masks_by_cp[selected], {
        "model": model_path or "Cellpose-SAM default",
        "gpu": kwargs["gpu"], "selected_cellprob": selected,
        "sweep": [{"cellprob": float(cp), "count": count}
                  for cp, count in zip(cellprobs, counts)],
        "seconds": round(time.time() - started, 3),
    }


def create_nuclei(run_dir: str | Path, model_path: str | None = None) -> Path:
    run = Path(run_dir)
    masks, debug = segment_nuclei(load_run_channel(run, "dapi"), model_path=model_path)
    path = run / "nuclei_masks.npy"
    np.save(path, masks)
    (run / "nuclei_metadata.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
    return path


def create_component_proposals(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """Create review-only connected-territory proposals.

    Every proposal is ambiguous and unreviewed by construction. This command is an annotation
    convenience, not an independent-myotube detector.
    """
    run = Path(run_dir)
    metadata = load_metadata(run)
    territory_path = run / "myotube_territory.npy"
    if not territory_path.exists():
        create_territory(run)
    proposal_mask = run / "desmin_semantic_mask.npy"
    labels = label(np.load(proposal_mask if proposal_mask.exists() else territory_path))
    instances = from_label_image(labels, metadata["image_id"], source="semantic_component_proposal",
                                 reviewed=False, default_status="ambiguous")
    path = Path(out_path) if out_path else run / "instance_proposals.json"
    instances.save(path)
    return path
