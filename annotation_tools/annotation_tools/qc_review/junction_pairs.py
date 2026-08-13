"""Junction candidates for the operator-labeling round: which pairing is right?

The classical floor resolves every skeleton-graph junction with one hand-tuned
rule (`classical.ridge_graph`'s anti-parallel `STRAIGHT_DOT` pairing). The
measurement in `classical.junction_ambiguity` found that rule genuinely
ambiguous at 615 of 893 fibre-scale candidate junctions across the six
bootstrap wells (`coordination/reports/claude_junction_ambiguity_measurement_
2026-07-23.md`) -- those are this round's pool.

**Scoped to degree-3 junctions.** `degree_ge4` never fired on a single
fibre-scale candidate across all six wells (measured), so a degree-3 junction
covers every real case: exactly three branch-ends meet, and exactly three
pairings are possible -- (0,1), (0,2), (1,2). The operator's decision is a
single choice among those three pairs, or "none" (a genuine branch point), or
"unsure". This is the "split" sibling of the fragment linker's join decision,
built on the same declined-candidates-are-negatives principle: whichever pair
is NOT chosen becomes a labelled negative automatically.

Round 1 pool (recommended, see the measurement report): the union of
``near_threshold_winner`` (the algorithm's actual decision was itself close to
the boundary -- 167 junctions) and ``width_or_intensity_conflict`` (direction
disagrees with width/intensity continuity even when the winner was decisive --
105 junctions, only 27 overlapping the first group) -- 245 junctions total.
The broader ``near_threshold`` (any candidate near the boundary, 615 junctions)
is available via ``reasons=None`` for a later round if more data is needed,
mirroring the linker's iterative widen-the-window pattern.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from classical.junction_ambiguity import (
    CONFLICT_RATIO, MIN_BRANCH_UM, THRESHOLD_MARGIN, _branch_length_um,
    evaluate_junction)
from classical.ridge_graph import TracerParams, build_branch_graph

# Round-1 pool: the adopted decision was borderline, or direction disagrees
# with an independent cue (width/intensity). See module docstring.
ROUND1_REASONS = ("near_threshold_winner", "width_or_intensity_conflict")


@dataclass
class JunctionCase:
    well: str
    node: int
    reasons: list                        # which criteria flagged this junction
    branch_ids: tuple                    # (branch_0, branch_1, branch_2)
    branch_ends: tuple                   # ("source"|"destination",) * 3, same order
    branch_lengths_um: tuple             # matches branch_ids order
    pair_keys: tuple                     # (("0","1"), ("0","2"), ("1","2")) end-index pairs
    centroid_rc: tuple

    def case_id(self) -> str:
        return f"junction_{self.node:06d}"


def _matches_pool(info: dict, reasons) -> bool:
    if reasons is None:
        return bool(info["reasons"])                    # any criterion at all
    return any(r in info["reasons"] for r in reasons) or (
        "near_threshold_winner" in reasons and info["near_threshold_winner"])


def find_junction_cases(well: str, territory: np.ndarray, fiber: np.ndarray,
                        pixel_um: float, params: TracerParams = TracerParams(),
                        min_branch_um: float = MIN_BRANCH_UM,
                        threshold_margin: float = THRESHOLD_MARGIN,
                        conflict_ratio: float = CONFLICT_RATIO,
                        reasons=ROUND1_REASONS,
                        ) -> tuple[list[JunctionCase], list, dict]:
    """Degree-3 fibre-scale junctions matching ``reasons``, ready for the page.

    Returns ``(cases, coordinates, node_ends)``:

    * ``coordinates`` (branch id -> pixel-path array, indexed exactly like
      `build_branch_graph`'s own return) lets the page renderer draw branch
      paths without rebuilding the branch graph a second time.
    * ``node_ends`` (node id -> ``[(branch_id, end, direction), ...]``, also
      `build_branch_graph`'s own return) lets a caller that needs direction
      vectors -- e.g. to score a case's candidate pairs with a trained model,
      as the active-learning round does -- look them up by
      ``case.node`` instead of re-walking the branch graph.

    Only degree-3 junctions are considered (see module docstring); a
    degree-4+ junction, if one ever appears, is silently skipped here rather
    than mis-modelled as a single 3-way choice.
    """
    territory = np.asarray(territory, dtype=bool)
    skeleton = skeletonize(territory)
    if skeleton.sum() < 3:
        return [], [], {}

    graph, node_ends, coordinates = build_branch_graph(skeleton, pixel_um, params)
    distance_to_bg = ndi.distance_transform_edt(territory)
    branch_lengths = [_branch_length_um(c, pixel_um) for c in coordinates]

    cases: list[JunctionCase] = []
    for node, ends in node_ends.items():
        if len(ends) != 3:
            continue
        if min(branch_lengths[b] for b, _, _ in ends) < min_branch_um:
            continue
        info = evaluate_junction(node, ends, coordinates, distance_to_bg, fiber,
                                 pixel_um, params, threshold_margin, conflict_ratio)
        if not _matches_pool(info, reasons):
            continue
        branch_ids = tuple(b for b, _, _ in ends)
        branch_ends = tuple(e for _, e, _ in ends)
        cases.append(JunctionCase(
            well=well, node=int(node), reasons=info["reasons"],
            branch_ids=branch_ids, branch_ends=branch_ends,
            branch_lengths_um=tuple(round(branch_lengths[b], 1) for b in branch_ids),
            pair_keys=(("0", "1"), ("0", "2"), ("1", "2")),
            centroid_rc=tuple(info["centroid_rc"])))
    return cases, coordinates, node_ends
