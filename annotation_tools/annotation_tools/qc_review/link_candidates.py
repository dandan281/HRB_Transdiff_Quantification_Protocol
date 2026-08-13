"""Find gap-bridging link candidates for operator-confirmed fragments.

Why gap bridging and not tracer tuning
--------------------------------------
The operator labelled 65 long objects (median 205 um) `fragment_too_short` --
"a piece of something longer". Measured against the T02 v1 predictions, the
relaxed tracer assembles **0 of those 65** (length ratio 1.00). That is
structural, not a tuning failure: junction pairing only merges *within* one
connected component, and these fragments are **separate components split by gaps
in the Desmin signal**. No value of `straight_dot` can bridge a gap.

So the fix is a linker, and the fragments are its training data. Measured
feasibility on the 65: 74% have 1-3 collinear partners within 40 um (median 1),
22% have none. "Does this join that one?" is therefore a well-posed glance.

Candidate rule
--------------
For each skeleton endpoint of a fragment, a partner endpoint qualifies when

* the gap between the two endpoints is under ``gap_um``;
* the fragment's outward direction points at the partner (``cos >= cos_min``);
* the partner's outward direction points back (``cos >= cos_min``).

Requiring *both* directions rejects a fibre that merely passes nearby, which a
distance-only rule would happily propose.

Rejected candidates matter as much as accepted ones: a candidate the operator
declines is a labelled **negative** for the linker, so the full offered set is
always recorded, never just the accepted links.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

DEFAULT_GAP_UM = 40.0
DEFAULT_COS_MIN = 0.80
MIN_PROPOSAL_PX = 60
ENDPOINT_LOCAL_PX = 12          # neighbourhood used to estimate an outward direction
CANDIDATE_LETTERS_MAX = 5       # A-E; beyond this the card stops being a fast glance


@dataclass
class LinkCandidate:
    fragment_id: str
    candidate_id: str
    gap_um: float
    cos_fragment: float          # fragment endpoint points at the partner
    cos_candidate: float         # partner endpoint points back
    fragment_endpoint: tuple[int, int]
    candidate_endpoint: tuple[int, int]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["fragment_endpoint"] = list(self.fragment_endpoint)
        d["candidate_endpoint"] = list(self.candidate_endpoint)
        return d


def endpoint_directions(mask: np.ndarray, origin: tuple[int, int] = (0, 0),
                        min_spur: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    """Skeleton endpoints in full-field coordinates with outward unit directions."""
    from skimage.morphology import skeletonize

    from .pipeline import _prune_spurs

    skeleton = _prune_spurs(skeletonize(np.asarray(mask, dtype=bool)), min_spur)
    if skeleton.sum() < 3:
        return []
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbours = ndi.convolve(skeleton.astype(int), kernel, mode="constant") * skeleton
    endpoints = np.argwhere(skeleton & (neighbours == 1))
    coordinates = np.argwhere(skeleton).astype(float)
    out = []
    for point in endpoints:
        local = coordinates[((coordinates - point) ** 2).sum(1) <= ENDPOINT_LOCAL_PX ** 2]
        if len(local) < 2:
            continue
        vector = point - local.mean(0)                  # points outward, away from the body
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            continue
        out.append((point + np.asarray(origin), vector / norm))
    return out


def find_link_candidates(labels: np.ndarray, fragment_ids: list[int], pixel_um: float,
                         gap_um: float = DEFAULT_GAP_UM,
                         cos_min: float = DEFAULT_COS_MIN,
                         id_fmt: str = "myotube_{:04d}",
                         require_axis_agreement: bool = True,
                         ) -> dict[str, list[LinkCandidate]]:
    """Collinear partners for each fragment, keyed by fragment proposal id.

    Every proposal in the field is a possible partner, not only other fragments:
    a fragment may well continue into an object that was accepted in round one.

    ``require_axis_agreement`` additionally demands that the two fragments' own
    principal axes lie within ``cos_min`` of each other, not merely that their
    endpoints point at one another. Without it a vertical fibre whose tip bends
    toward a horizontal one is a legal candidate -- see `link_geometry`. It is a
    parameter rather than an unconditional rule only so the pre-fix behaviour
    stays reachable for reproducing earlier runs; new work wants it on.
    """
    from .link_geometry import axes_agree, fragment_axis

    labels = np.asarray(labels)
    gap_px = gap_um / pixel_um
    slices = ndi.find_objects(labels)

    ends: dict[int, list] = {}
    axes: dict[int, object] = {}
    for label_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        mask = labels[sl] == label_id
        if mask.sum() < MIN_PROPOSAL_PX:
            continue
        ends[label_id] = endpoint_directions(mask, (sl[0].start, sl[1].start))
        axes[label_id] = fragment_axis(mask)

    out: dict[str, list[LinkCandidate]] = {}
    for fragment in fragment_ids:
        found: dict[int, LinkCandidate] = {}
        for point, direction in ends.get(fragment, []):
            for other, other_ends in ends.items():
                if other == fragment:
                    continue
                if require_axis_agreement and not axes_agree(
                        axes.get(fragment), axes.get(other), cos_min):
                    continue
                for other_point, other_direction in other_ends:
                    delta = other_point - point
                    gap = float(np.linalg.norm(delta))
                    if gap < 1.0 or gap > gap_px:
                        continue
                    toward = delta / gap
                    cos_fragment = float(np.dot(direction, toward))
                    cos_candidate = float(np.dot(other_direction, -toward))
                    if cos_fragment < cos_min or cos_candidate < cos_min:
                        continue
                    previous = found.get(other)
                    if previous is not None and previous.gap_um <= gap * pixel_um:
                        continue
                    found[other] = LinkCandidate(
                        fragment_id=id_fmt.format(fragment),
                        candidate_id=id_fmt.format(other),
                        gap_um=round(gap * pixel_um, 1),
                        cos_fragment=round(cos_fragment, 3),
                        cos_candidate=round(cos_candidate, 3),
                        fragment_endpoint=(int(point[0]), int(point[1])),
                        candidate_endpoint=(int(other_point[0]), int(other_point[1])))
        # nearest first: the most plausible join is offered as option A
        out[id_fmt.format(fragment)] = sorted(found.values(), key=lambda c: c.gap_um)
    return out


def load_fragments(round2_dir: str | Path) -> dict[str, list[str]]:
    """Operator-confirmed `fragment_too_short` ids per well, from the round-2 bank."""
    round2_dir = Path(round2_dir)
    manifest = json.loads((round2_dir / "round2_manifest.json").read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for well in manifest["wells"]:
        classes = json.loads((round2_dir / f"{well}.round2_classes.json"
                              ).read_text(encoding="utf-8"))
        ids = [c["id"] for c in classes["ignore"]
               if c["category"] == "fragment_too_short"]
        if ids:
            out[well] = sorted(ids)
    return out
