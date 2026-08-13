"""Topology-aware myotube geometry measurements."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

NEIGHBORS = [
    (-1, -1, math.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0),                              (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)),  (1, 0, 1.0),  (1, 1, math.sqrt(2.0)),
]


@dataclass
class Geometry:
    area_um2: float
    length_um: float
    total_skeleton_um: float
    branch_count: int
    endpoint_count: int
    width_median_um: float
    width_q25_um: float
    width_q75_um: float
    width_p10_um: float
    width_p90_um: float
    width_area_over_length_um: float
    width_cv: float
    components: int
    touches_border: bool


def _graph(skeleton: np.ndarray):
    coords = np.argwhere(skeleton)
    lookup = {tuple(rc): i for i, rc in enumerate(coords)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in coords]
    edge_total = 0.0
    degrees = np.zeros(len(coords), dtype=np.int16)
    for i, (r, c) in enumerate(coords):
        for dr, dc, weight in NEIGHBORS:
            j = lookup.get((int(r + dr), int(c + dc)))
            if j is not None:
                adjacency[i].append((j, weight))
                if j > i:
                    edge_total += weight
        degrees[i] = len(adjacency[i])
    return coords, adjacency, degrees, edge_total


def _dijkstra(adjacency, start: int):
    dist = np.full(len(adjacency), np.inf)
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance != dist[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < dist[neighbor]:
                dist[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return dist


def _longest_geodesic(adjacency, degrees) -> float:
    if not adjacency:
        return 0.0
    endpoints = np.flatnonzero(degrees == 1)
    candidates = endpoints if endpoints.size >= 2 else np.arange(len(adjacency))
    # Exact endpoint diameter is affordable for normal fibers. For pathological meshes,
    # use a deterministic two-sweep approximation rather than quadratic all-pairs work.
    if len(candidates) <= 128:
        diameter = 0.0
        candidate_set = np.asarray(candidates, dtype=int)
        for start in candidate_set:
            d = _dijkstra(adjacency, int(start))[candidate_set]
            finite = d[np.isfinite(d)]
            if finite.size:
                diameter = max(diameter, float(finite.max()))
        return diameter
    d0 = _dijkstra(adjacency, int(candidates[0]))
    far = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    d1 = _dijkstra(adjacency, far)
    return float(d1[np.isfinite(d1)].max())


def measure_mask(mask: np.ndarray, pixel_um: float, endpoint_exclusion_um: float = 10.0) -> Geometry:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("myotube mask must be a non-empty 2-D array")
    component_labels, components = ndi.label(mask)
    skeleton = skeletonize(mask)
    coords, adjacency, degrees, total_px = _graph(skeleton)
    length_px = _longest_geodesic(adjacency, degrees)

    endpoint_map = np.zeros_like(mask)
    branch_map = np.zeros_like(mask)
    if len(coords):
        endpoints = coords[degrees == 1]
        branches = coords[degrees >= 3]
        if len(endpoints):
            endpoint_map[endpoints[:, 0], endpoints[:, 1]] = True
        if len(branches):
            branch_map[branches[:, 0], branches[:, 1]] = True
    branch_regions = int(ndi.label(branch_map)[1])

    edt = ndi.distance_transform_edt(mask)
    sample = skeleton.copy()
    if endpoint_map.any():
        sample &= ndi.distance_transform_edt(~endpoint_map) * pixel_um >= endpoint_exclusion_um
    if branch_map.any():
        median_radius_px = float(np.median(edt[skeleton])) if skeleton.any() else 1.0
        sample &= ndi.distance_transform_edt(~branch_map) >= max(1.0, median_radius_px)
    if not sample.any():
        sample = skeleton
    widths = 2.0 * edt[sample] * pixel_um
    if not widths.size:
        widths = np.array([0.0])

    area_um2 = float(mask.sum()) * pixel_um * pixel_um
    length_um = length_px * pixel_um
    touches = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    mean_width = float(np.mean(widths))
    return Geometry(
        area_um2=area_um2,
        length_um=length_um,
        total_skeleton_um=float(total_px * pixel_um),
        branch_count=branch_regions,
        endpoint_count=int(np.count_nonzero(degrees == 1)),
        width_median_um=float(np.median(widths)),
        width_q25_um=float(np.percentile(widths, 25)),
        width_q75_um=float(np.percentile(widths, 75)),
        width_p10_um=float(np.percentile(widths, 10)),
        width_p90_um=float(np.percentile(widths, 90)),
        width_area_over_length_um=(area_um2 / length_um if length_um > 0 else float("nan")),
        width_cv=(float(np.std(widths) / mean_width) if mean_width > 0 else float("nan")),
        components=int(components),
        touches_border=touches,
    )
