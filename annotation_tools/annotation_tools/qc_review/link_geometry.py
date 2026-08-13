"""Geometric constraints the fragment linker declares but did not enforce.

The linker's candidate rule says a partner qualifies when each fragment's outward
direction points at the other. That is a statement about the two *endpoints*, and
it was the only geometry ever checked. Two things follow, and the control-only
safety round (`over_merge_c1`, 2026-08-04) found both in the wild:

* **the fragments' own axes were never compared.** A vertical fibre whose tip
  happens to bend toward a horizontal one satisfies the endpoint rule. Twelve of
  the fifty-five reviewed objects contained a fragment pair beyond the declared
  45 degree window, and the operator called all twelve two different myotubes;
* **union-find then discarded even the endpoint constraint.** Merging is the
  transitive closure of accepted pairs, so A-B and B-C put A and C in one object
  without anyone testing A against C. Across the reviewed objects, 1342 of 1584
  within-object fragment pairs (85%) were never directly scored. In one case the
  chain ran 4 -> 25 -> 39 -> 0 degrees between neighbours and left its two ends
  59 degrees apart.

Nothing here introduces a new operating parameter. `cos_min` is the window the
linker already declares; these functions apply it to the fragments and to the
merged object instead of only to a pair of endpoints. That distinction matters:
the safety round may not be used to fit a threshold, and this does not fit one.

`MIN_AXIS_EXTENT_PX` is likewise derived rather than chosen. A direction is
estimated from a neighbourhood of `ENDPOINT_LOCAL_PX` pixels, so a fragment
shorter than two such windows has no axis to speak of -- reading one off it
returns noise. Objects whose least elongated fragment was under 3:1 were called
wrong 14 times out of 16. Where the axis cannot be estimated the constraint
cannot be verified, and an unverifiable merge is refused rather than assumed
safe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Two endpoint-direction windows must fit along a fragment before its principal
# axis means anything. Pinned against link_candidates.ENDPOINT_LOCAL_PX by a test
# so the two cannot drift apart.
MIN_AXIS_EXTENT_PX = 24


@dataclass(frozen=True)
class Axis:
    direction: np.ndarray        # unit principal direction, sign-free
    extent_px: float             # length of the fragment along that direction
    elongation: float            # major/minor spread; 1.0 is a disc


def fragment_axis(mask: np.ndarray) -> Axis | None:
    """Principal axis of one fragment, or None when it has no estimable one."""
    positions = np.argwhere(np.asarray(mask, dtype=bool)).astype(float)
    if len(positions) < 3:
        return None
    centred = positions - positions.mean(0)
    _, spread, components = np.linalg.svd(centred, full_matrices=False)
    direction = components[0]
    projection = centred @ direction
    extent = float(projection.max() - projection.min())
    if extent < MIN_AXIS_EXTENT_PX:
        return None
    minor = float(spread[1]) if spread[1] > 1e-9 else 1e-9
    return Axis(direction=direction, extent_px=extent,
                elongation=float(spread[0]) / minor)


def axes_agree(first: Axis | None, second: Axis | None, cos_min: float) -> bool:
    """Do two fragment axes lie within the linker's declared angular window?

    Undirected: a fibre traced head-to-tail and one traced tail-to-head describe
    the same line, so the sign of the dot product carries no information.

    An unestimable axis is **not** agreement. The linker's failure is
    over-merging, so the case it cannot check is the case it must decline.
    """
    if first is None or second is None:
        return False
    return abs(float(np.dot(first.direction, second.direction))) >= cos_min


@dataclass
class MergeResult:
    components: dict[int, list[int]]      # representative -> sorted members
    refused: list[dict]                   # edges declined, with the reason


def constrained_merge(fragment_ids, edges, axes: dict[int, Axis | None], *,
                      cos_min: float) -> MergeResult:
    """Union accepted edges, but never build an object that violates `cos_min`.

    Edges are taken in descending probability so the linker's own preference
    decides which of two conflicting joins survives, and ties break on the
    fragment ids so the result does not depend on dictionary order.

    Before each union the *cross product* of the two components is checked, not just
    the edge being added. That is the whole point: the previous merge enforced the
    constraint on edges and then let the closure carry objects past it.
    """
    parent = {int(f): int(f) for f in fragment_ids}
    members = {int(f): [int(f)] for f in fragment_ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    refused: list[dict] = []
    ordered = sorted(edges, key=lambda e: (-e[0], int(e[1]), int(e[2])))
    for probability, first, second in ordered:
        first, second = int(first), int(second)
        if first not in parent or second not in parent:
            continue
        root_a, root_b = find(first), find(second)
        if root_a == root_b:
            continue
        clash = next((
            (a, b) for a in members[root_a] for b in members[root_b]
            if not axes_agree(axes.get(a), axes.get(b), cos_min)), None)
        if clash is not None:
            reason = ("axis pair outside the declared window"
                      if axes.get(clash[0]) and axes.get(clash[1])
                      else "axis not estimable, constraint unverifiable")
            refused.append({"fragments": [first, second],
                            "probability": float(probability),
                            "blocking_pair": [int(clash[0]), int(clash[1])],
                            "reason": reason})
            continue
        low, high = sorted((root_a, root_b))
        parent[high] = low
        members[low] = sorted(members[low] + members[high])
        members.pop(high)

    components = {root: sorted(group) for root, group in members.items()}
    return MergeResult(components=components, refused=refused)
