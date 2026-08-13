"""Validated crossing-aware traced-fiber length gate from Conversion Efficiency."""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

SPUR_UM = 10.0
STRAIGHT_DOT = -0.5


def _unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def trace_fibers(mask: np.ndarray, pixel_um: float):
    """Trace whole fibers through junctions by pairing anti-parallel branch directions."""
    skeleton = skeletonize(mask > 0)
    if skeleton.sum() < 3:
        nearest = distance_transform_edt(~skeleton, return_distances=False, return_indices=True)
        return skeleton, nearest, []
    skan_skeleton = Skeleton(skeleton)
    table = summarize(skan_skeleton, separator="-")
    distance_col = next(c for c in table.columns if "branch" in c and "distance" in c)
    type_col = next(c for c in table.columns if "branch" in c and "type" in c)
    source_col = next(c for c in table.columns if "node" in c and "src" in c)
    destination_col = next(c for c in table.columns if "node" in c and "dst" in c)
    lengths_px = table[distance_col].to_numpy(dtype=np.float64)
    types = table[type_col].to_numpy()
    sources = table[source_col].to_numpy(); destinations = table[destination_col].to_numpy()
    coordinates = [np.asarray(skan_skeleton.path_coordinates(i), float)
                   for i in range(len(lengths_px))]

    graph = nx.Graph()
    node_ends = {}
    for index, length_px in enumerate(lengths_px):
        if types[index] == 1 and length_px * pixel_um < SPUR_UM:
            continue
        graph.add_edge(("branch", index, "source"), ("branch", index, "destination"),
                       weight=float(length_px) * pixel_um, branch=index)
        coords = coordinates[index]
        step = min(3, len(coords) - 1)
        source_direction = _unit(coords[step] - coords[0]) if len(coords) > 1 else np.zeros(2)
        destination_direction = _unit(coords[-1 - step] - coords[-1]) if len(coords) > 1 else np.zeros(2)
        node_ends.setdefault(int(sources[index]), []).append((index, "source", source_direction))
        node_ends.setdefault(int(destinations[index]), []).append(
            (index, "destination", destination_direction))

    for ends in node_ends.values():
        pair_candidates = []
        for first in range(len(ends)):
            for second in range(first + 1, len(ends)):
                pair_candidates.append((float(np.dot(ends[first][2], ends[second][2])),
                                        first, second))
        used = set()
        for dot, first, second in sorted(pair_candidates):
            if dot > STRAIGHT_DOT:
                break
            if first in used or second in used:
                continue
            used.update((first, second))
            a, a_end, _ = ends[first]; b, b_end, _ = ends[second]
            graph.add_edge(("branch", a, a_end), ("branch", b, b_end), weight=0.0)

    fibers = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        branch_ids, length_um = set(), 0.0
        for _, _, data in subgraph.edges(data=True):
            length_um += data["weight"]
            if "branch" in data:
                branch_ids.add(data["branch"])
        if length_um <= 0:
            continue
        pixels = np.concatenate([coordinates[i] for i in branch_ids]).round().astype(int)
        fibers.append((float(length_um), pixels))
    nearest = distance_transform_edt(~skeleton, return_distances=False, return_indices=True)
    return skeleton, nearest, fibers


def length_gated_territory(mask: np.ndarray, pixel_um: float,
                           minimum_length_um: float = 50.0) -> tuple[np.ndarray, dict]:
    skeleton, nearest, fibers = trace_fibers(mask, pixel_um)
    kept = np.zeros_like(skeleton, dtype=bool)
    lengths = []
    for length_um, pixels in fibers:
        if length_um >= minimum_length_um:
            kept[pixels[:, 0], pixels[:, 1]] = True
            lengths.append(length_um)
    nearest_kept = kept[nearest[0], nearest[1]]
    territory = binary_fill_holes((mask > 0) & nearest_kept)
    return territory, {
        "minimum_fiber_length_um": minimum_length_um,
        "traced_fibers_total": len(fibers),
        "traced_fibers_retained": len(lengths),
        "retained_length_median_um": float(np.median(lengths)) if lengths else None,
        "coverage_pct": float(territory.mean() * 100.0),
    }
