"""T02 candidate 1 - deterministic classical ridge/graph instance segmenter.

This is the *reproducible floor* required first by the T02 contract
(`coordination/requests/claude/2026-07-21-t02-start.md`). It contains no learned
weights: given the same image and the same parameters it always returns the same
instances.

Pipeline
--------
A. **Territory** - canonical `precision_myotube.segmentation.semantic_territory`
   (tophat -> CLAHE -> Sato tubeness -> hysteresis -> intensity gate, with the
   threshold-plateau selection). Imported unchanged; this stage is expensive
   (~2 min per 3636x3636 field) and independent of every candidate parameter, so
   callers cache it per well.
B. **Graph trace** - skeletonise the territory, build the skan branch graph, and
   pair anti-parallel branch ends at each junction so a fibre that passes
   straight through a crossing stays one object.
C. **Territory assignment** - every territory pixel joins its nearest *retained
   fibre* pixel, then fibres are filtered by traced length and instance area.

Relationship to the canonical code
----------------------------------
Stage B is a **parameterised re-implementation** of
`precision_myotube.fiber_gate.trace_fibers`, which hard-codes `SPUR_UM = 10.0`
and `STRAIGHT_DOT = -0.5` as module constants. The canonical file is Codex-owned
and is deliberately left untouched; exposing those two values as fold-fitted
parameters is the entire point of the candidate, because measured fragmentation
(~4 predicted pieces per reviewed myotube) says junction pairing is where the
classical floor actually loses. With `spur_um=10.0, straight_dot=-0.5` stage B
reproduces the canonical tracer's grouping.

One deliberate fix vs. the canonical helper: `length_gated_territory` computes
nearest-skeleton indices over the *whole* skeleton, so pixels whose nearest
skeleton pixel belongs to a pruned spur are orphaned (measured: 32% of territory
area on `23_B02_ctrl`). Stage C measures distance to retained fibre pixels only,
which assigns 100% of the territory and lifts median best-overlap IoU from 0.490
to 0.588 on that well.

All outputs are unreviewed proposals. A connected component or an instance count
from this module is never an authoritative independent-myotube count.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import time

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

__all__ = [
    "TracerParams", "FilterParams", "FieldTrace",
    "semantic_territory_cached", "trace_fibers_parameterised",
    "build_branch_graph", "junction_candidates", "pair_junction_ends",
    "assign_territory", "filter_assigned", "assign_and_filter", "PARAM_GRID",
]


@dataclass(frozen=True)
class TracerParams:
    """Stage-B parameters. Defaults reproduce the canonical tracer's grouping."""
    spur_um: float = 10.0        # dead-end branches shorter than this are pruned
    straight_dot: float = -0.5   # pair branch ends whose direction dot <= this
    direction_step: int = 3      # px used to estimate a branch-end direction

    def key(self) -> str:
        return f"spur{self.spur_um:g}_dot{self.straight_dot:g}_step{self.direction_step}"


@dataclass(frozen=True)
class FilterParams:
    """Stage-C parameters."""
    min_length_um: float = 50.0  # traced fibre length gate (validated default)
    min_area_px: int = 180       # instance area gate (matches min_object_px)


@dataclass
class FieldTrace:
    """Everything stage C needs, for one well at one set of tracer parameters."""
    territory: np.ndarray        # bool (H, W)
    fiber_id: np.ndarray         # int32 (H, W); 0 = not a retained fibre pixel
    lengths_um: np.ndarray       # float64, index i = traced length of fibre i
    pixel_um: float
    seconds: float = 0.0

    @property
    def n_fibers(self) -> int:
        return int(len(self.lengths_um) - 1)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def semantic_territory_cached(fiber_image: np.ndarray, cache_path: str | Path | None = None,
                              ) -> tuple[np.ndarray, dict]:
    """Canonical Desmin territory, memoised to ``cache_path`` (stage A).

    Stage A costs ~2 minutes per full field and depends on no candidate
    parameter, so the fold runner computes it once per well and reuses it across
    the whole parameter grid.
    """
    import json

    from precision_myotube.segmentation import semantic_territory

    if cache_path is not None:
        cache_path = Path(cache_path)
        meta_path = cache_path.with_suffix(".json")
        if cache_path.is_file() and meta_path.is_file():
            return np.load(cache_path), json.loads(meta_path.read_text(encoding="utf-8"))

    mask, debug = semantic_territory(np.asarray(fiber_image))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, mask)
        cache_path.with_suffix(".json").write_text(
            __import__("json").dumps(debug, indent=2), encoding="utf-8")
    return mask, debug


def build_branch_graph(skeleton: np.ndarray, pixel_um: float, params: TracerParams,
                       ) -> tuple[nx.Graph, dict[int, list], list[np.ndarray]]:
    """Retained branches as weighted graph edges, plus per-junction end directions.

    Returns ``(graph, node_ends, coordinates)``. ``node_ends`` maps a skan node id
    to the list of ``(branch_index, end, direction_unit_vector)`` meeting there --
    the exact candidate set :func:`pair_junction_ends` (and any junction-level
    diagnostic) consumes, so callers never need to re-derive it from the skan
    table. ``coordinates[i]`` is branch ``i``'s pixel path. Short dead-end spurs
    (skan type 1 shorter than ``params.spur_um``) are dropped before they can
    manufacture a spurious junction.
    """
    skan_skeleton = Skeleton(skeleton)
    table = summarize(skan_skeleton, separator="-")
    distance_col = next(c for c in table.columns if "branch" in c and "distance" in c)
    type_col = next(c for c in table.columns if "branch" in c and "type" in c)
    source_col = next(c for c in table.columns if "node" in c and "src" in c)
    destination_col = next(c for c in table.columns if "node" in c and "dst" in c)
    lengths_px = table[distance_col].to_numpy(dtype=np.float64)
    types = table[type_col].to_numpy()
    sources = table[source_col].to_numpy()
    destinations = table[destination_col].to_numpy()
    coordinates = [np.asarray(skan_skeleton.path_coordinates(i), float)
                   for i in range(len(lengths_px))]

    graph = nx.Graph()
    node_ends: dict[int, list] = {}
    for index, length_px in enumerate(lengths_px):
        # type 1 == dead-end branch; prune short spurs so noise makes no junction
        if types[index] == 1 and length_px * pixel_um < params.spur_um:
            continue
        graph.add_edge(("branch", index, "source"), ("branch", index, "destination"),
                       weight=float(length_px) * pixel_um, branch=index)
        coords = coordinates[index]
        step = min(params.direction_step, len(coords) - 1)
        if len(coords) > 1:
            source_direction = _unit(coords[step] - coords[0])
            destination_direction = _unit(coords[-1 - step] - coords[-1])
        else:
            source_direction = destination_direction = np.zeros(2)
        node_ends.setdefault(int(sources[index]), []).append(
            (index, "source", source_direction))
        node_ends.setdefault(int(destinations[index]), []).append(
            (index, "destination", destination_direction))
    return graph, node_ends, coordinates


def junction_candidates(ends: list) -> list[tuple[float, int, int]]:
    """All pairwise direction dots for the branch-ends at one junction, ascending.

    Ascending order means the most anti-parallel (straightest through) candidate
    sorts first -- both :func:`pair_junction_ends` and diagnostics that want the
    "next-best" alternative to the winning pair rely on this order.
    """
    return sorted((float(np.dot(ends[a][2], ends[b][2])), a, b)
                  for a in range(len(ends)) for b in range(a + 1, len(ends)))


def pair_junction_ends(ends: list, straight_dot: float) -> list[tuple[int, int]]:
    """Greedy most-anti-parallel-first matching of one junction's branch-ends.

    Consumes candidates straightest-through first (ascending dot) and joins a
    pair only once, skipping any end already used by a straighter pair. A pair is
    only joined when its dot is ``<= straight_dot``. Returns the chosen
    ``(a, b)`` index pairs into ``ends``; unmatched ends are where a fibre
    genuinely stops at this junction.
    """
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for dot, a, b in junction_candidates(ends):
        if dot > straight_dot:
            break
        if a in used or b in used:
            continue
        used.update((a, b))
        pairs.append((a, b))
    return pairs


def trace_fibers_parameterised(territory: np.ndarray, pixel_um: float,
                               params: TracerParams = TracerParams(),
                               junction_decider=None) -> FieldTrace:
    """Stage B: skeleton graph trace with fold-fittable junction behaviour.

    Branch ends meeting at a junction are paired most-anti-parallel-first; a pair
    is only joined when its direction dot product is ``<= straight_dot``, so a
    stricter (more negative) value keeps only near-straight pass-throughs and a
    looser value also joins bent fibres. Union-find over those joins turns branch
    segments into whole fibres.

    ``junction_decider`` swaps out *only* that pairing rule. It is called as
    ``decider(node, ends, coordinates)`` and returns the ``(a, b)`` index pairs
    into ``ends`` to join, exactly like :func:`pair_junction_ends`. ``None``
    (the default) keeps the deterministic classical rule, so the sealed floor
    is untouched; a learned junction classifier is injected here rather than by
    forking the tracer, which keeps both arms of a comparison identical in
    every other respect.
    """
    started = time.time()
    territory = np.asarray(territory, dtype=bool)
    height, width = territory.shape
    skeleton = skeletonize(territory)
    empty = FieldTrace(territory, np.zeros((height, width), np.int32),
                       np.zeros(1, np.float64), float(pixel_um))
    if skeleton.sum() < 3:
        empty.seconds = time.time() - started
        return empty

    graph, node_ends, coordinates = build_branch_graph(skeleton, pixel_um, params)

    for node, ends in node_ends.items():
        pairs = (pair_junction_ends(ends, params.straight_dot) if junction_decider is None
                 else junction_decider(node, ends, coordinates))
        for a, b in pairs:
            first, first_end, _ = ends[a]
            second, second_end, _ = ends[b]
            graph.add_edge(("branch", first, first_end), ("branch", second, second_end),
                           weight=0.0)

    fiber_id = np.zeros((height, width), dtype=np.int32)
    lengths = [0.0]
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        branch_ids: set[int] = set()
        length_um = 0.0
        for _, _, data in subgraph.edges(data=True):
            length_um += data["weight"]
            if "branch" in data:
                branch_ids.add(data["branch"])
        if length_um <= 0 or not branch_ids:
            continue
        pixels = np.concatenate([coordinates[i] for i in branch_ids]).round().astype(int)
        lengths.append(float(length_um))
        fiber_id[pixels[:, 0], pixels[:, 1]] = len(lengths) - 1

    return FieldTrace(territory, fiber_id, np.asarray(lengths, dtype=np.float64),
                      float(pixel_um), time.time() - started)


def assign_territory(trace: FieldTrace) -> tuple[np.ndarray, np.ndarray]:
    """Stage C part 1: label every territory pixel with its nearest fibre id.

    Distance is measured to the nearest **retained fibre** pixel rather than the
    nearest skeleton pixel, so territory whose closest skeleton pixel sits on a
    pruned spur is not orphaned. Independent of :class:`FilterParams`, so the
    fold runner computes it once per (well, tracer-params) and reuses it for
    every filter combination.

    Returns ``(assigned, areas)`` where ``areas[i]`` is the pixel area of fibre
    ``i`` and index 0 is unassigned background.
    """
    if trace.n_fibers == 0:
        return np.zeros(trace.territory.shape, dtype=np.int32), np.zeros(1, dtype=np.int64)
    nearest = ndi.distance_transform_edt(trace.fiber_id == 0, return_distances=False,
                                         return_indices=True)
    assigned = np.where(trace.territory, trace.fiber_id[nearest[0], nearest[1]], 0)
    areas = np.bincount(assigned.ravel(), minlength=trace.n_fibers + 1)
    return assigned.astype(np.int32, copy=False), areas


def filter_assigned(trace: FieldTrace, assigned: np.ndarray, areas: np.ndarray,
                    filters: FilterParams = FilterParams(),
                    ) -> tuple[list[int], dict]:
    """Stage C part 2: apply the traced-length and area gates (cheap).

    Returns **kept fibre ids**, not masks. A full field is 3636x3636, so one
    boolean mask per instance is ~13 MB and a dense well yields thousands of
    instances -- materialising them as a list costs tens of gigabytes. Callers
    stream masks with :func:`iter_masks` or read pixels with
    :func:`instance_positions` instead.
    """
    keep = [i for i in range(1, trace.n_fibers + 1)
            if trace.lengths_um[i] >= filters.min_length_um
            and areas[i] >= filters.min_area_px]
    territory_px = int(trace.territory.sum())
    debug = {
        "n_fibers_traced": trace.n_fibers,
        "n_instances": len(keep),
        "assigned_fraction": (float((assigned > 0).sum()) / territory_px
                              if territory_px else 0.0),
        "instance_area_px": int(sum(int(areas[i]) for i in keep)),
        "territory_px": territory_px,
        "tracer_seconds": round(trace.seconds, 2),
    }
    return keep, debug


def iter_masks(assigned: np.ndarray, kept_ids: list[int]):
    """Yield one full-field boolean mask at a time (constant peak memory)."""
    for fiber_id in kept_ids:
        yield assigned == fiber_id


def instance_positions(assigned: np.ndarray, kept_ids: list[int]
                       ) -> list[tuple[int, np.ndarray]]:
    """Fortran-flat foreground positions per kept fibre, in one pass.

    Sorting the non-zero labels once is far cheaper than ``assigned == i`` per
    instance (which rescans the whole field every time) and never allocates a
    full-field array per instance.
    """
    height = assigned.shape[0]
    rows, cols = np.nonzero(assigned)
    if not rows.size:
        return []
    values = assigned[rows, cols]
    order = np.argsort(values, kind="stable")
    rows, cols, values = rows[order], cols[order], values[order]
    positions = rows.astype(np.int64) + cols.astype(np.int64) * height
    boundaries = np.searchsorted(values, np.asarray(kept_ids, dtype=values.dtype),
                                 side="left")
    ends = np.searchsorted(values, np.asarray(kept_ids, dtype=values.dtype), side="right")
    return [(int(fiber_id), positions[start:stop])
            for fiber_id, start, stop in zip(kept_ids, boundaries, ends) if stop > start]


def assign_and_filter(trace: FieldTrace, filters: FilterParams = FilterParams(),
                      ) -> tuple[list[np.ndarray], dict]:
    """Stage C: nearest-retained-fibre assignment followed by the gates.

    Returns ``(masks, debug)``. ``masks`` are mutually exclusive boolean arrays;
    the classical floor cannot represent a crossing as two overlapping instances,
    which is itself a finding rather than a defect to hide.

    This convenience **materialises every mask**, which is only safe for small
    fields and tests. Full-field callers must use :func:`assign_territory` +
    :func:`filter_assigned` and then stream via :func:`iter_masks` or
    :func:`instance_positions`.
    """
    if trace.n_fibers == 0:
        return [], {"n_fibers_traced": 0, "n_instances": 0, "assigned_fraction": 0.0}
    assigned, areas = assign_territory(trace)
    kept, debug = filter_assigned(trace, assigned, areas, filters)
    return list(iter_masks(assigned, kept)), debug


# Fold-fitted grid. Kept small and explicit so the search is reproducible and
# cheap: stages B/C are seconds, and stage A is cached per well.
PARAM_GRID: list[tuple[TracerParams, FilterParams]] = [
    (TracerParams(spur_um=spur, straight_dot=dot), FilterParams(min_length_um=length))
    for spur in (4.0, 10.0)
    for dot in (-0.5, -0.2, 0.0, 0.3)
    for length in (25.0, 50.0, 75.0, 100.0)
]


def params_as_dict(tracer: TracerParams, filters: FilterParams) -> dict:
    return {"tracer": asdict(tracer), "filters": asdict(filters)}
