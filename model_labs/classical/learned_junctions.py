"""Inject the learned junction classifier into the classical floor's tracer.

The classifier has so far only been scored on *junction decisions in
isolation* (64.5% vs the fixed rule's 23.8%). That is a proxy. What the
project actually cares about is the instance-level readout -- myotube counts,
lengths, precision/recall/IoU -- so this module makes the substitution
measurable end to end by handing
`ridge_graph.trace_fibers_parameterised` a ``junction_decider``. Nothing else
in the pipeline changes, so any metric difference is attributable to the
junction rule alone.

Scope and fallbacks, matching exactly what the classifier was trained on:

* **degree != 3** -> classical rule. The labeling rounds only ever covered
  degree-3 junctions (degree>=4 never fired on a fibre-scale candidate), so the
  model has no basis to decide anything else.
* **any incident branch below ``min_branch_um``** -> classical rule. These are
  sub-fibre-scale skeletonisation whiskers, deliberately excluded from the
  labeling pool; ~90% of raw degree-3 nodes are in this class.
* otherwise -> two-stage learned decision: the branch-point gate first, then
  the pairwise model's best pair.

A junction the gate calls a branch point returns **no pairing at all**, which
is the classical rule's own representation of "nothing continues through here"
-- so downstream union-find, territory assignment and filtering are unchanged.
"""
from __future__ import annotations

import numpy as np

from classical.junction_ambiguity import MIN_BRANCH_UM, _branch_length_um
from classical.ridge_graph import TracerParams, pair_junction_ends

PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
PAIR_KEYS = ("AB", "AC", "BC")


def make_learned_decider(pair_model, gate_model, gate_threshold: float,
                         fiber: np.ndarray, distance_to_bg: np.ndarray,
                         pixel_um: float, params: TracerParams = TracerParams(),
                         min_branch_um: float = MIN_BRANCH_UM, stats: dict | None = None):
    """A ``junction_decider`` for :func:`ridge_graph.trace_fibers_parameterised`.

    ``stats``, if given, accumulates how often each branch was taken -- worth
    recording because if the learned rule only fires on a handful of junctions
    its instance-level effect is bounded no matter how accurate it is.
    """
    from annotation_tools.qc_review.junction_features import (
        compute_junction_features, compute_pair_features, node_position)
    from annotation_tools.qc_review.junction_model import (
        JunctionExample, JunctionPairExample)

    counters = stats if stats is not None else {}
    for key in ("classical_degree", "classical_short", "learned_branch_point",
                "learned_pair", "total"):
        counters.setdefault(key, 0)

    def decide(node, ends, coordinates):
        counters["total"] += 1
        if len(ends) != 3:
            counters["classical_degree"] += 1
            return pair_junction_ends(ends, params.straight_dot)
        lengths = [_branch_length_um(coordinates[b], pixel_um) for b, _e, *_ in ends]
        if min(lengths) < min_branch_um:
            counters["classical_short"] += 1
            return pair_junction_ends(ends, params.straight_dot)

        junction_feats = compute_junction_features(
            coordinates, ends, distance_to_bg, fiber, pixel_um, lengths)
        if gate_model.score(JunctionExample("", int(node), junction_feats, None)) >= gate_threshold:
            counters["learned_branch_point"] += 1
            return []                       # nothing continues through: no pairing

        node_rc = node_position(coordinates, ends)
        best_pair, best_proba = None, -1.0
        for key, (a, b) in zip(PAIR_KEYS, PAIR_INDICES):
            branch_a, end_a, *_ = ends[a]
            branch_b, end_b, *_ = ends[b]
            feats = compute_pair_features(
                coordinates, branch_a, end_a, branch_b, end_b, distance_to_bg,
                fiber, pixel_um, lengths[a], lengths[b], node_rc=node_rc)
            proba = pair_model.score(JunctionPairExample("", int(node), key, feats, None))
            if proba > best_proba:
                best_pair, best_proba = (a, b), proba
        counters["learned_pair"] += 1
        return [best_pair]

    return decide
