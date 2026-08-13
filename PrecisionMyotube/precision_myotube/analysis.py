"""Authoritative measurements from reviewed instances plus independent field metrics."""
from __future__ import annotations

from dataclasses import asdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from .geometry import measure_mask
from .io import load_metadata, sha256_file
from .schema import InstanceSet


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def valid_nuclei(labels: np.ndarray, pixel_um: float, amin_um2: float,
                 amax_um2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int32)
    areas_px = np.bincount(labels.ravel())
    areas_um2 = areas_px.astype(float) * pixel_um * pixel_um
    valid = (areas_um2 >= amin_um2) & (areas_um2 <= amax_um2)
    if valid.size:
        valid[0] = False
    return valid, areas_px, areas_um2


def nucleus_centroids(labels: np.ndarray, n_labels: int) -> dict[int, tuple[float, float]]:
    indices = list(range(1, n_labels + 1))
    centers = ndi.center_of_mass(np.ones_like(labels, dtype=np.uint8), labels, indices)
    return {idx: (float(center[1]), float(center[0]))
            for idx, center in zip(indices, centers) if np.isfinite(center).all()}


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    holes, _ = ndi.label(~mask)
    areas = np.bincount(holes.ravel())
    touches_border = np.zeros(len(areas), dtype=bool)
    border = np.concatenate((holes[0], holes[-1], holes[:, 0], holes[:, -1]))
    touches_border[np.unique(border)] = True
    fill = (areas < int(max_area)) & ~touches_border
    fill[0] = False
    return mask | fill[holes]


def analyze(run_dir: str | Path, instances_path: str | Path, *,
            nuclei_masks_path: str | Path | None = None,
            territory_path: str | Path | None = None,
            amin_um2: float = 50.0, amax_um2: float = 500.0,
            assignment_fraction: float = 0.5, assignment_margin: float = 0.25) -> dict:
    run = Path(run_dir)
    metadata = load_metadata(run)
    pixel_um = float(metadata["pixel_um"])
    instance_set = InstanceSet.load(instances_path)
    if tuple(instance_set.image_shape) != tuple(metadata["image_shape"]):
        raise ValueError("instance annotations do not match the ND2 image shape")

    nuclei_path = Path(nuclei_masks_path) if nuclei_masks_path else run / "nuclei_masks.npy"
    territory_file = Path(territory_path) if territory_path else run / "myotube_territory.npy"
    if not nuclei_path.exists():
        raise FileNotFoundError("nuclei masks missing; run `nuclei` or supply --nuclei-masks")
    if not territory_file.exists():
        raise FileNotFoundError("myotube territory missing; run `territory` or supply --territory")
    nuclei = np.load(nuclei_path).astype(np.int32, copy=False)
    territory = np.load(territory_file).astype(bool, copy=False)
    if nuclei.shape != territory.shape or nuclei.shape != tuple(instance_set.image_shape):
        raise ValueError("nuclei, territory, and instance masks must have the same shape")
    canonical_instances = run / "myotube_instances.json"
    if Path(instances_path).resolve() != canonical_instances.resolve():
        instance_set.save(canonical_instances)
    canonical_nuclei = run / "nuclei_masks.npy"
    if nuclei_path.resolve() != canonical_nuclei.resolve():
        np.save(canonical_nuclei, nuclei)
    canonical_territory = run / "myotube_territory.npy"
    if territory_file.resolve() != canonical_territory.resolve():
        np.save(canonical_territory, territory)

    valid, areas_px, areas_um2 = valid_nuclei(nuclei, pixel_um, amin_um2, amax_um2)
    n_labels = len(areas_px) - 1
    centers = nucleus_centroids(nuclei, n_labels)
    # Fill only nucleus-sized holes; do not bridge separate Desmin structures.
    max_hole_px = int(np.ceil(amax_um2 / (pixel_um * pixel_um)))
    fusion_territory = _fill_small_holes(territory, max_hole_px)
    territory_overlap_px = np.bincount(
        nuclei.ravel(), weights=fusion_territory.ravel().astype(np.uint8), minlength=len(areas_px)
    )
    territory_fraction = np.divide(territory_overlap_px, areas_px,
                                   out=np.zeros_like(territory_overlap_px, dtype=float),
                                   where=areas_px > 0)

    instance_rows: list[dict] = []
    instance_masks: dict[str, tuple[tuple[int, int, int, int], np.ndarray]] = {}
    overlaps_by_nucleus: dict[int, list[tuple[str, float]]] = {}
    qc_flags: list[dict] = []
    image_shape = tuple(instance_set.image_shape)
    for record, bbox, mask in instance_set.cropped_masks():
        r0, c0, r1, c1 = bbox
        geometry = measure_mask(mask, pixel_um)
        touches_image_border = bool((r0 == 0 and mask[0].any()) or
                                    (r1 == image_shape[0] and mask[-1].any()) or
                                    (c0 == 0 and mask[:, 0].any()) or
                                    (c1 == image_shape[1] and mask[:, -1].any()))
        effective_status = record.status
        if touches_image_border and record.status == "complete":
            effective_status = "border_truncated"
            qc_flags.append({"type": "border_status_conflict", "instance_id": record.id,
                             "severity": "required"})
        if geometry.components != 1:
            qc_flags.append({"type": "disconnected_instance", "instance_id": record.id,
                             "severity": "required", "components": geometry.components})
        if not record.reviewed:
            qc_flags.append({"type": "unreviewed_instance", "instance_id": record.id,
                             "severity": "required"})
        if record.confidence is not None and record.confidence < 0.95:
            qc_flags.append({"type": "low_model_confidence", "instance_id": record.id,
                             "severity": "required", "confidence": record.confidence})
        if geometry.width_cv > 0.75:
            qc_flags.append({"type": "unexpected_width_variation", "instance_id": record.id,
                             "severity": "review", "width_cv": round(geometry.width_cv, 4)})

        counts = np.bincount(nuclei[r0:r1, c0:c1][mask], minlength=len(areas_px))
        for nucleus_id in np.flatnonzero(counts):
            if nucleus_id == 0 or nucleus_id >= len(valid) or not valid[nucleus_id]:
                continue
            fraction = float(counts[nucleus_id] / areas_px[nucleus_id])
            if fraction > 0:
                overlaps_by_nucleus.setdefault(int(nucleus_id), []).append((record.id, fraction))
        geometry_values = asdict(geometry)
        geometry_values["touches_border"] = touches_image_border
        row = {"id": record.id, "status": effective_status, "annotated_status": record.status,
               "reviewed": record.reviewed, "source": record.source,
               "confidence": record.confidence if record.confidence is not None else "",
               "notes": record.notes, **geometry_values, "nuclei_count": 0,
               "authoritative": record.is_authoritative(effective_status)}
        instance_rows.append(row)
        instance_masks[record.id] = (bbox, mask)

    authoritative_ids = {row["id"] for row in instance_rows if row["authoritative"]}
    assignments: dict[int, tuple[str | None, str, float, float]] = {}
    for nucleus_id in np.flatnonzero(valid):
        all_choices = sorted(overlaps_by_nucleus.get(int(nucleus_id), []),
                             key=lambda item: item[1], reverse=True)
        choices = [item for item in all_choices if item[0] in authoritative_ids]
        best_id, best = choices[0] if choices else (None, 0.0)
        second = choices[1][1] if len(choices) > 1 else 0.0
        unresolved_best = all_choices[0][1] if all_choices else 0.0
        if not choices and unresolved_best >= assignment_fraction:
            assignments[int(nucleus_id)] = (None, "assignment_ambiguous", unresolved_best, 0.0)
            qc_flags.append({"type": "nucleus_overlaps_non_authoritative_instance",
                             "nucleus_id": int(nucleus_id), "severity": "required",
                             "best_fraction": round(unresolved_best, 4)})
        elif best < assignment_fraction:
            assignments[int(nucleus_id)] = (None, "outside_instances", best, second)
        elif best - second < assignment_margin:
            assignments[int(nucleus_id)] = (None, "assignment_ambiguous", best, second)
            qc_flags.append({"type": "ambiguous_nucleus_assignment", "nucleus_id": int(nucleus_id),
                             "severity": "required", "best_fraction": round(best, 4),
                             "second_fraction": round(second, 4)})
        else:
            assignments[int(nucleus_id)] = (best_id, "assigned", best, second)

    row_by_id = {row["id"]: row for row in instance_rows}
    for assigned_id, status, *_ in assignments.values():
        if status == "assigned" and assigned_id in row_by_id:
            row_by_id[assigned_id]["nuclei_count"] += 1

    nucleus_rows = []
    for nucleus_id in np.flatnonzero(valid):
        assigned, status, best, second = assignments[int(nucleus_id)]
        x, y = centers.get(int(nucleus_id), (float("nan"), float("nan")))
        nucleus_rows.append({
            "id": int(nucleus_id), "x_px": round(x, 3), "y_px": round(y, 3),
            "area_um2": round(float(areas_um2[nucleus_id]), 4),
            "territory_overlap_fraction": round(float(territory_fraction[nucleus_id]), 5),
            "in_myotube_40": bool(territory_fraction[nucleus_id] >= 0.4),
            "in_myotube_50": bool(territory_fraction[nucleus_id] >= 0.5),
            "in_myotube_60": bool(territory_fraction[nucleus_id] >= 0.6),
            "assigned_myotube": assigned or "", "assignment_status": status,
            "best_instance_fraction": round(best, 5), "second_instance_fraction": round(second, 5),
        })

    authoritative = [r for r in instance_rows if r["authoritative"]]
    counts = np.asarray([r["nuclei_count"] for r in authoritative], dtype=int)
    total_valid = int(valid.sum())
    summary = {
        "image_id": metadata["image_id"], "pixel_um": pixel_um,
        "total_nuclei": total_valid,
        "nuclei_in_myotube_40": int(np.count_nonzero(valid & (territory_fraction >= 0.4))),
        "nuclei_in_myotube_50": int(np.count_nonzero(valid & (territory_fraction >= 0.5))),
        "nuclei_in_myotube_60": int(np.count_nonzero(valid & (territory_fraction >= 0.6))),
        "conversion_efficiency_40": (float(np.mean(territory_fraction[valid] >= 0.4)) if total_valid else 0.0),
        "conversion_efficiency_50": (float(np.mean(territory_fraction[valid] >= 0.5)) if total_valid else 0.0),
        "conversion_efficiency_60": (float(np.mean(territory_fraction[valid] >= 0.6)) if total_valid else 0.0),
        "instances_total": len(instance_rows),
        "instances_authoritative": len(authoritative),
        "instances_ambiguous": sum(r["status"] == "ambiguous" for r in instance_rows),
        "instances_truncated": sum(r["status"] == "border_truncated" for r in instance_rows),
        "nuclei_assignment_ambiguous": sum(r["assignment_status"] == "assignment_ambiguous"
                                             for r in nucleus_rows),
        "multinucleation_n_instances": int(counts.size),
        "multinucleation_median": (float(np.median(counts)) if counts.size else None),
        "multinucleation_q25": (float(np.percentile(counts, 25)) if counts.size else None),
        "multinucleation_q75": (float(np.percentile(counts, 75)) if counts.size else None),
        "multinucleation_pct_ge2": (float(np.mean(counts >= 2)) if counts.size else None),
        "multinucleation_bins": {
            "1": int(np.count_nonzero(counts == 1)), "2": int(np.count_nonzero(counts == 2)),
            "3_to_5": int(np.count_nonzero((counts >= 3) & (counts <= 5))),
            "gt_5": int(np.count_nonzero(counts > 5)),
        },
        "required_qc_flags": sum(f["severity"] == "required" for f in qc_flags),
    }
    _write_csv(run / "myotubes.csv", instance_rows)
    _write_csv(run / "nuclei.csv", nucleus_rows)
    _write_csv(run / "field_summary.csv", [summary])
    (run / "qc_flags.json").write_text(json.dumps(qc_flags, indent=2), encoding="utf-8")
    (run / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    history_entry = {
        "analyzed_utc": datetime.now(timezone.utc).isoformat(),
        "instances": {"path": str(Path(instances_path).resolve()),
                      "sha256": sha256_file(instances_path)},
        "nuclei_masks": {"path": str(nuclei_path.resolve()), "sha256": sha256_file(nuclei_path)},
        "territory": {"path": str(territory_file.resolve()),
                      "sha256": sha256_file(territory_file)},
        "parameters": {"nucleus_area_um2": [amin_um2, amax_um2],
                       "assignment_fraction": assignment_fraction,
                       "assignment_margin": assignment_margin,
                       "conversion_overlap_fractions": [0.4, 0.5, 0.6]},
        "required_qc_flags": summary["required_qc_flags"],
        "instances_authoritative": summary["instances_authoritative"],
    }
    with (run / "qc_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_entry, separators=(",", ":")) + "\n")
    return {"summary": summary, "myotubes": instance_rows, "nuclei": nucleus_rows,
            "qc_flags": qc_flags, "instance_masks": instance_masks,
            "nuclei_labels": nuclei, "territory": fusion_territory, "valid_nuclei": valid}
