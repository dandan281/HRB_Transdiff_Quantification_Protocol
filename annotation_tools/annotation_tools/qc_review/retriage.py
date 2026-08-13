"""Re-triage the 839 first-pass `ambiguous` proposals into actionable categories.

Why
---
The first pass collapsed several very different situations into one bucket. The
operator's instruction was explicitly conservative -- "<120um -> default Ambiguous
unless unmistakable" -- and **601 of the 839 ambiguous proposals fall below
120um**, so a large share were *defaulted*, not judged unresolvable. Measured
first-pass accept rate rises steadily with length (14% under 60um to 63% at or
above 250um), and 775 of the 839 are geometrically "clean" (elongated, solid, not
border-touching); the clean bucket is statistically indistinguishable from masks
the operator accepted.

That data is recoverable, and on a 375-mask training set recovering even a
fraction of it outweighs any architecture change.

Categories
----------
Six, each mapping to a distinct training role so nothing reviewed is discarded:

========================  ==================================================
category                  training role
========================  ==================================================
`complete`                promote -> new full-area target
`branched_one_myotube`    promote -> new target; one object that happens to branch
`fragment_too_short`      not a target; ignore + fragment-linking signal
`merged_too_long`         not a target; over-merge negative; splittable
`unresolvable`            stays ignore
`not_myotube`             background (an informative negative)
========================  ==================================================

The machine pre-classifies every case so the operator **confirms rather than
authors**; the prediction is a starting position, never a decision. Cases are
served highest-expected-yield first so the operator can stop after any batch and
still have captured most of the recoverable data.

Nothing here mutates the frozen first-pass artifacts. Re-triage is a new,
separately versioned annotation round with its own provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

CATEGORIES = (
    "complete",
    "branched_one_myotube",
    "fragment_too_short",
    "merged_too_long",
    "unresolvable",
    "not_myotube",
)

PROMOTES_TO_TARGET = ("complete", "branched_one_myotube")

# First-pass accept rate by length band, measured over all six wells. Used only to
# order the queue (highest expected yield first) -- never to decide anything.
ACCEPT_RATE_BY_BAND = (
    (0.0, 60.0, 0.14),
    (60.0, 90.0, 0.25),
    (90.0, 120.0, 0.37),
    (120.0, 160.0, 0.40),
    (160.0, 250.0, 0.47),
    (250.0, float("inf"), 0.63),
)

# Thresholds calibrated against the 408 first-pass ACCEPTED masks -- known-good
# complete myotubes. Any rule set must classify those overwhelmingly as
# promotable, or the operator would be fighting a wrong suggestion on every card.
# The calibrated rules below put 85% of accepted masks in complete/branched; the
# ~10% residual in `fragment_too_short` is by construction, since SHORT_UM is
# their 10th percentile.
SHORT_UM = 56.0     # 10th percentile length of accepted masks
LONG_UM = 481.0     # 95th percentile length of accepted masks

# CALIBRATION FINDING -- do not use `traced_fiber_count >= 2` alone as evidence of
# merging. It fires on **39% (161/408) of known-good accepted masks**, because the
# crossing tracer fragments real single myotubes (independently measured in the
# classical T02 candidate at ~4 fragments per reviewed myotube). A first version of
# this classifier used that rule and misfiled 39% of accepted masks as
# `merged_too_long`. Merging is now asserted only when an object is *also*
# unusually long and not solid.


def band_yield(length_um: float) -> float:
    for low, high, rate in ACCEPT_RATE_BY_BAND:
        if low <= length_um < high:
            return rate
    return 0.0


@dataclass
class SkeletonEvidence:
    """Topology of one proposal, used to separate branched from merged."""
    n_junctions: int = 0
    n_endpoints: int = 0
    n_branches: int = 0
    traced_fiber_count: int = 0     # how many whole fibres the tracer sees
    skeleton_px: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def skeleton_evidence(mask: np.ndarray, min_spur: int = 12) -> SkeletonEvidence:
    """Junction/endpoint topology plus how many fibres the tracer resolves.

    A *branched* object has junctions but the tracer still carries it through as
    one fibre. A *merged* object is one the tracer splits into two or more.
    """
    from skimage.morphology import skeletonize

    from .pipeline import _prune_spurs, trace_hypotheses

    mask = np.asarray(mask, dtype=bool)
    skeleton = skeletonize(mask)
    if skeleton.sum() < 3:
        return SkeletonEvidence(skeleton_px=int(skeleton.sum()))
    skeleton = _prune_spurs(skeleton, min_spur)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbours = ndi.convolve(skeleton.astype(int), kernel, mode="constant") * skeleton
    junction = skeleton & (neighbours >= 3)
    endpoint = skeleton & (neighbours == 1)
    structure = np.ones((3, 3))
    _, n_junction_clusters = ndi.label(junction, structure=structure)
    branch_pixels = skeleton & ~ndi.binary_dilation(junction, structure=structure)
    _, n_branches = ndi.label(branch_pixels, structure=structure)

    traced = trace_hypotheses(mask)
    if traced is None:
        fiber_count = 1
    else:
        _, hypotheses = traced
        fiber_count = len(set(hypotheses[1])) if len(hypotheses) > 1 else 1

    return SkeletonEvidence(
        n_junctions=int(n_junction_clusters),
        n_endpoints=int(ndi.label(endpoint, structure=structure)[1]),
        n_branches=int(n_branches),
        traced_fiber_count=int(fiber_count),
        skeleton_px=int(skeleton.sum()),
    )


def classify(features: dict, evidence: SkeletonEvidence) -> tuple[str, str]:
    """Pre-classify one ambiguous proposal. Returns ``(category, why)``.

    Deliberately transparent and rule-based: the operator sees *why* the machine
    proposed a category, which makes a wrong suggestion easy to overrule and keeps
    the tool honest. This is a labelling accelerator, not a decision-maker.
    """
    length = float(features.get("length_um", 0.0))
    aspect = float(features.get("aspect", 0.0))
    solidity = float(features.get("solidity", 1.0))
    width = float(features.get("width_um", 0.0))
    border = float(features.get("touches_border", 0.0)) > 0

    # Not myotube-shaped at all: squat and wide rather than elongated.
    if aspect < 3.0 and width > 20.0:
        return "not_myotube", f"low aspect {aspect:.1f} with width {width:.0f}um - not fibre-shaped"

    # Tangled, low-solidity mass the tracer cannot decompose. Requires unusual
    # length too, so a merely knotty short object is not written off.
    if solidity < 0.25 and evidence.n_junctions >= 3 and length >= LONG_UM:
        return ("unresolvable",
                f"solidity {solidity:.2f}, {evidence.n_junctions} junctions, "
                f"length {length:.0f}um - tangled and unusually long")

    # Merged: unusually long AND multi-traced AND not solid. All three, because
    # multi-traced alone fires on 39% of known-good accepted masks (see above).
    if length >= LONG_UM and evidence.traced_fiber_count >= 2 and solidity < 0.45:
        return ("merged_too_long",
                f"length {length:.0f}um exceeds the {LONG_UM:.0f}um 95th-percentile of "
                f"accepted masks, tracer resolves {evidence.traced_fiber_count} fibres, "
                f"solidity {solidity:.2f}")

    # Short: more likely a piece of a longer fibre than a whole one.
    if length < SHORT_UM and not border:
        return ("fragment_too_short",
                f"length {length:.0f}um is below the {SHORT_UM:.0f}um "
                "10th-percentile of accepted masks")

    # Junctions present but not merge-shaped -> branched, likely one object.
    if evidence.n_junctions >= 1:
        return ("branched_one_myotube",
                f"{evidence.n_junctions} junction(s), length {length:.0f}um - "
                "branch rather than a merge of separate fibres")

    return ("complete",
            f"clean single fibre, length {length:.0f}um, aspect {aspect:.1f}, "
            f"solidity {solidity:.2f}")


def queue_priority(features: dict, category: str) -> float:
    """Higher = review sooner. Ordered by expected promotion yield.

    Yield is dominated by length band (measured 14% -> 63%), boosted when the
    machine already thinks the case is promotable and damped when it does not, so
    the operator meets the richest cases first and can stop after any batch.
    """
    length = float(features.get("length_um", 0.0))
    priority = band_yield(length)
    if category in PROMOTES_TO_TARGET:
        priority += 0.35
    elif category in ("not_myotube", "unresolvable"):
        priority -= 0.25
    if float(features.get("touches_border", 0.0)) > 0:
        priority -= 0.10           # border cases can never become complete targets
    return priority


def load_first_pass(package_dir: Path, stem: str) -> dict[str, dict]:
    """First-pass decisions for one well, keyed by proposal id."""
    path = Path(package_dir) / f"{stem}.decisions.json"
    return json.loads(path.read_text(encoding="utf-8")).get("decisions", {})


def select_ambiguous(decisions: dict[str, dict]) -> list[str]:
    """Ids whose first-pass decision was `ambiguous` (sorted for determinism)."""
    return sorted(pid for pid, d in decisions.items() if d.get("action") == "ambiguous")


def batch(cases: list[dict], size: int) -> list[list[dict]]:
    """Split an ordered queue into bounded batches."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [cases[i:i + size] for i in range(0, len(cases), size)]


def build_queue(packages: list[Path], *, thumb_px: int = 190, edit_px: int = 460,
                progress=None) -> list[dict]:
    """Assemble the full ordered re-triage queue across every well.

    For each well: read the first-pass decisions, keep the `ambiguous` ones,
    compute skeleton evidence and a pre-classification for each, and render its
    imagery. The result is sorted by expected promotion yield, highest first, so
    a partially completed queue is still the most valuable part of it.
    """
    from scipy import ndimage as ndi
    import tifffile

    from .cli import _load_package
    from .pipeline import build_cases

    queue: list[dict] = []
    for package_dir in packages:
        package_dir = Path(package_dir)
        stem, labels, fiber, territory, pixel_um, dapi = _load_package(package_dir)
        decisions = load_first_pass(package_dir, stem)
        wanted = select_ambiguous(decisions)
        if not wanted:
            continue
        label_ids = {int(pid.split("_")[-1]) for pid in wanted}

        slices = ndi.find_objects(labels)
        evidence: dict[str, SkeletonEvidence] = {}
        for pid in wanted:
            label_id = int(pid.split("_")[-1])
            if label_id > len(slices) or slices[label_id - 1] is None:
                continue
            sl = slices[label_id - 1]
            evidence[pid] = skeleton_evidence(labels[sl] == label_id)

        cases = build_cases(labels, fiber, pixel_um, territory, dapi=dapi,
                            thumb_px=thumb_px, edit_px=edit_px, only_ids=label_ids)
        for case in cases:
            pid = case["id"]
            if pid not in evidence:
                continue
            features = decisions.get(pid, {}).get("features", case["features"])
            category, why = classify(features, evidence[pid])
            case.update({
                "well": stem,
                # Proposal ids are only unique WITHIN a well -- `myotube_0161`
                # exists in several wells. The page keys its state and its export
                # by `uid`, so that two wells' cards cannot collide and silently
                # overwrite each other's decision.
                "uid": f"{stem}/{pid}",
                "dom_id": f"{stem}__{pid}".replace("/", "_").replace(".", "_"),
                "features": features,
                "machine_category": category,
                "machine_why": why,
                "skeleton": evidence[pid].as_dict(),
                "priority": round(queue_priority(features, category), 4),
            })
            # imagery only: drop the editor payloads this page does not use
            case.pop("mask_rle", None)
            case.pop("segments", None)
            case.pop("hypotheses", None)
            case.pop("geom", None)
            queue.append(case)
        if progress:
            progress(stem, len(cases))

    # Highest expected yield first; uid break for a fully deterministic order.
    queue.sort(key=lambda c: (-c["priority"], c["uid"]))
    seen = {c["uid"] for c in queue}
    if len(seen) != len(queue):
        raise RuntimeError("re-triage queue contains duplicate uids")
    return queue
