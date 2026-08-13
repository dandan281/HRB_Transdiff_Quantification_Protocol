"""Features for the junction classifier: does this pair of branches pass straight through?

The classical floor already computes one feature -- the direction dot product
-- and uses it alone (`STRAIGHT_DOT` in `classical.ridge_graph`). The
ambiguity measurement (`classical.junction_ambiguity`) found 105 of 893
fibre-scale candidate junctions where direction and width/intensity actively
disagree, so a learned classifier needs the independent cues too, not just a
sharper threshold on the same one signal.

Features (all computed from the branch-end pair + the field image, reusing the
branch-graph geometry `classical.ridge_graph.build_branch_graph` already
produced -- nothing here re-skeletonises):

``tangent_cos``       dot product of the two branches' outward-from-junction
                     directions, each estimated over ``DIRECTION_WINDOW_PX``
                     pixels of that branch's path. A real straight-through
                     fibre is anti-parallel here (near -1). Same *kind* of
                     signal the classical floor thresholds, but measured over a
                     much longer window -- see the note below, this is the
                     single most important choice in the whole feature set.
``turn_angle_deg``   0 for a perfectly straight pass-through, 180 for a full
                     reversal. A monotonic reparameterisation of ``tangent_cos``
                     (``arccos(-tangent_cos)`` in degrees) -- kept as a separate
                     feature because a linear model reads an angular scale
                     differently than a raw cosine near its extremes.
``width_ratio``      min/max of the two branches' local width in microns
                     (distance-to-background at the branch-end pixels, doubled).
                     A real myotube keeps roughly constant width through a
                     crossing; a false pairing often joins a thick fibre to a
                     thin one. 1 = identical width, low = mismatched.
``intensity_ratio``  min/max of the two branches' local mean Desmin intensity
                     at the branch-end window. Stain continuity, independent of
                     geometry -- the same principle as the linker's
                     `bridge_over_bg`, but sampled at the ends rather than
                     across a gap (junctions have no gap to bridge).
``length_min_um``    the shorter of the two branches' traced lengths. Short
                     branches give noisy direction/width/intensity estimates
                     (a 2-3 px branch cannot support a stable tangent), so this
                     lets the model discount low-length pairs rather than
                     trusting them at face value.
``node_intensity_ratio``  mean Desmin in a small square at the junction node,
                     over the mean of the pair's two branch-end intensities --
                     i.e. how the stain right at the node compares with the
                     stain on the two branches meeting there. Added 2026-07-28
                     to attack the dominant over-merge error; worth +0.010
                     LOWO AUC (0.892 -> 0.902).

                     **Read it as a local contrast, not as "a dip means
                     branch point".** The branch-end sampling window starts
                     *at* the node, so it overlaps the node square; which way
                     the ratio moves for a given stain pattern depends on how
                     the two windows overlap it, and is not a clean
                     "continues through / does not" signal. It earns its place
                     empirically, and the tests below assert only the
                     invariants that actually hold (neutral 1.0 on a uniform
                     field, responsive to node-region stain) rather than a
                     direction the geometry does not guarantee.
                     A variant sampling each branch 8-20 px *away* from the
                     node was also measured: weaker alone (0.896) but
                     complementary (0.907 with both). Banked, not adopted --
                     see the round-2 report.

Junction-level features
-----------------------
`JUNCTION_FEATURE_KEYS` / :func:`compute_junction_features` describe the
junction as a whole rather than one candidate pair -- the three tangents
sorted, the branch length/width spread, and the node intensity. They feed the
branch-point gate in `junction_model`, which needs to answer "is this a
junction where *nothing* continues through?" -- a question no per-pair feature
can express, because calling a branch point requires all three pairs to fail
*together*. All are built from sorted statistics, so they do not depend on
the arbitrary A/B/C labelling of the three branches.

All the continuity features (tangent_cos, width_ratio, intensity_ratio) use
the exact sampling this pair would use if it were the classical floor's
adopted pairing, whether or not it actually was -- so a declined pair's
features are directly comparable to an accepted one's.

Why the direction window is 15 px and not the tracer's 3
--------------------------------------------------------
Rounds 1+2 (395 junctions) plateaued at LOWO AUC 0.693 / junction accuracy
37%, and a learning curve showed the plateau was NOT a data shortage: AUC was
flat from ~120 junctions onward, gaining +0.005 per additional 100 junctions.
The cause was this feature. It was originally computed from the direction
vectors `build_branch_graph` attaches to each branch end, which use
``TracerParams.direction_step = 3`` -- a *3-pixel* window, chosen for the
tracer's own spur/junction bookkeeping, not for discriminating myotubes. Over
3 px a skeleton direction is dominated by pixelation noise.

Measured on all 341 labeled junctions, LOWO by well, changing nothing else:

    window   3 px -> AUC 0.606     (the original)
    window   8 px -> AUC 0.833
    window  15 px -> AUC 0.858     <- chosen
    window  30 px -> AUC 0.852
    window  60 px -> AUC 0.844
    whole-branch PCA -> AUC 0.805

and in the full feature set, 3 px -> 15 px moves LOWO AUC **0.693 -> 0.892**
and junction accuracy **37% -> 58%**, cutting wrong-pair errors from 89 to 30.
Adding multi-scale windows and a whole-branch PCA axis on top gained nothing
further (0.893), so the single 15 px window is kept -- fewer features, same
result. 15 px ~= 10 um at 0.6493 um/px, roughly one myotube width: long enough
to average out pixelation, short enough that a genuinely curving fibre is not
straightened away.

This repeats the fragment linker's own history, which found its local
endpoint tangent (`min_cos`, a 12-px patch) too noisy and was rescued by a
whole-object axis feature. The lesson generalises: **estimate direction over a
fibre-scale window, not a pixel-scale one.** Hence this module computes its
own directions from the branch paths rather than reusing the tracer's, so the
classifier's window can never be silently changed by a tracer parameter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

FEATURE_KEYS = ("tangent_cos", "turn_angle_deg", "width_ratio", "intensity_ratio",
               "length_min_um", "node_intensity_ratio")

JUNCTION_FEATURE_KEYS = ("best_tan", "second_tan", "worst_tan", "tan_margin", "tan_spread",
                        "len_min_um", "len_mid_um", "len_max_um", "len_ratio_min_max",
                        "width_min_um", "width_max_um", "width_ratio_min_max",
                        "node_intensity_over_min", "node_intensity_over_max",
                        "intensity_ratio_min_max")

SAMPLE_PX = 5              # branch-end window for width/intensity; matches junction_ambiguity
DIRECTION_WINDOW_PX = 15   # branch-end window for direction; see module docstring.
                           # Deliberately NOT TracerParams.direction_step (3 px).
NODE_DISK_PX = 3           # half-width of the square sampled at the junction node


@dataclass
class JunctionPairFeatures:
    tangent_cos: float
    turn_angle_deg: float
    width_ratio: float
    intensity_ratio: float
    length_min_um: float
    node_intensity_ratio: float = 1.0

    def vector(self, keys=FEATURE_KEYS) -> list[float]:
        d = asdict(self)
        return [float(d[k]) for k in keys]


@dataclass
class JunctionFeatures:
    """Whole-junction description for the branch-point gate. See module docstring."""
    best_tan: float
    second_tan: float
    worst_tan: float
    tan_margin: float
    tan_spread: float
    len_min_um: float
    len_mid_um: float
    len_max_um: float
    len_ratio_min_max: float
    width_min_um: float
    width_max_um: float
    width_ratio_min_max: float
    node_intensity_over_min: float
    node_intensity_over_max: float
    intensity_ratio_min_max: float

    def vector(self, keys=JUNCTION_FEATURE_KEYS) -> list[float]:
        d = asdict(self)
        return [float(d[k]) for k in keys]


def _end_sample_points(coords: np.ndarray, end: str, window: int = SAMPLE_PX) -> np.ndarray:
    if end == "source":
        return coords[:window]
    return coords[-window:]


def _end_width_um(distance_to_bg: np.ndarray, coords: np.ndarray, end: str,
                  pixel_um: float) -> float:
    points = _end_sample_points(coords, end).round().astype(int)
    rows = np.clip(points[:, 0], 0, distance_to_bg.shape[0] - 1)
    cols = np.clip(points[:, 1], 0, distance_to_bg.shape[1] - 1)
    return float(np.mean(distance_to_bg[rows, cols])) * 2.0 * pixel_um


def _end_intensity(fiber: np.ndarray, coords: np.ndarray, end: str) -> float:
    points = _end_sample_points(coords, end).round().astype(int)
    rows = np.clip(points[:, 0], 0, fiber.shape[0] - 1)
    cols = np.clip(points[:, 1], 0, fiber.shape[1] - 1)
    return float(np.mean(fiber[rows, cols].astype(np.float64)))


def _ratio(a: float, b: float) -> float:
    lo, hi = min(a, b), max(a, b)
    return lo / hi if hi > 1e-9 else 1.0


def turn_angle_deg(tangent_cos: float) -> float:
    """0 = perfectly straight through, 180 = full reversal. See module docstring."""
    return float(np.degrees(np.arccos(np.clip(-tangent_cos, -1.0, 1.0))))


def end_direction(coords: np.ndarray, end: str,
                  window: int = DIRECTION_WINDOW_PX) -> np.ndarray:
    """Outward unit direction at a branch end, over ``window`` pixels of its path.

    "Outward" means pointing away from the junction, so two branches that
    continue straight through each other are anti-parallel (dot ~= -1). The
    window is clamped to the branch's own length, so a branch shorter than
    ``window`` uses its whole path rather than erroring -- ``length_min_um``
    is the feature that tells the model such an estimate is less trustworthy.
    """
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return np.zeros(2)
    window = min(window, len(coords) - 1)
    vector = (coords[window] - coords[0]) if end == "source" else (coords[-1 - window] - coords[-1])
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.zeros(2)


def node_position(coordinates: list, ends: list) -> tuple[float, float]:
    """The junction's own (row, col): the shared endpoint of its incident branches."""
    branch, end, *_ = ends[0]
    path = coordinates[branch]
    point = path[0] if end == "source" else path[-1]
    return (float(point[0]), float(point[1]))


def node_intensity(fiber: np.ndarray, node_rc, radius: int = NODE_DISK_PX) -> float:
    """Mean stain in a small square at the junction node."""
    rows, cols = fiber.shape
    r0 = max(0, int(node_rc[0]) - radius); r1 = min(rows, int(node_rc[0]) + radius + 1)
    c0 = max(0, int(node_rc[1]) - radius); c1 = min(cols, int(node_rc[1]) + radius + 1)
    patch = fiber[r0:r1, c0:c1]
    return float(np.mean(patch)) if patch.size else 0.0


def compute_pair_features(coordinates: list, branch_a: int, end_a: str,
                          branch_b: int, end_b: str,
                          distance_to_bg: np.ndarray, fiber: np.ndarray,
                          pixel_um: float, branch_length_a_um: float,
                          branch_length_b_um: float,
                          node_rc=None,
                          direction_window: int = DIRECTION_WINDOW_PX) -> JunctionPairFeatures:
    """Features for one candidate branch-end pair. See module docstring.

    ``tangent_cos`` is computed here from the branch paths rather than taken
    from the caller: the tracer's own end-direction vectors use a 3-px window
    that measured far worse (AUC 0.606 vs 0.858), and computing it in one
    place makes it impossible for training and inference to drift onto
    different windows.

    ``node_rc`` is the junction's own position (see :func:`node_position`).
    When omitted, ``node_intensity_ratio`` stays at its neutral 1.0 so a
    caller with only branch geometry still gets the other features.
    """
    tangent_cos = float(np.dot(
        end_direction(coordinates[branch_a], end_a, direction_window),
        end_direction(coordinates[branch_b], end_b, direction_window)))
    width_a = _end_width_um(distance_to_bg, coordinates[branch_a], end_a, pixel_um)
    width_b = _end_width_um(distance_to_bg, coordinates[branch_b], end_b, pixel_um)
    intensity_a = _end_intensity(fiber, coordinates[branch_a], end_a)
    intensity_b = _end_intensity(fiber, coordinates[branch_b], end_b)
    if node_rc is None:
        node_ratio = 1.0
    else:
        mean_end = 0.5 * (intensity_a + intensity_b)
        node_ratio = node_intensity(fiber, node_rc) / mean_end if mean_end > 1e-9 else 1.0
    return JunctionPairFeatures(
        tangent_cos=round(tangent_cos, 4),
        turn_angle_deg=round(turn_angle_deg(tangent_cos), 2),
        width_ratio=round(_ratio(width_a, width_b), 4),
        intensity_ratio=round(_ratio(intensity_a, intensity_b), 4),
        length_min_um=round(min(branch_length_a_um, branch_length_b_um), 2),
        node_intensity_ratio=round(node_ratio, 4))


def compute_junction_features(coordinates: list, ends: list, distance_to_bg: np.ndarray,
                              fiber: np.ndarray, pixel_um: float,
                              branch_lengths_um: list,
                              direction_window: int = DIRECTION_WINDOW_PX) -> JunctionFeatures:
    """Whole-junction features for the branch-point gate. See module docstring.

    Everything is derived from *sorted* per-branch statistics, so the result
    is invariant to which incident branch happens to be labelled A, B or C.
    """
    widths = sorted(_end_width_um(distance_to_bg, coordinates[b], e, pixel_um)
                    for b, e, *_ in ends)
    intensities = sorted(_end_intensity(fiber, coordinates[b], e) for b, e, *_ in ends)
    lengths = sorted(branch_lengths_um)
    directions = [end_direction(coordinates[b], e, direction_window) for b, e, *_ in ends]
    tangents = sorted(float(np.dot(directions[a], directions[b]))
                      for a, b in ((0, 1), (0, 2), (1, 2)))
    node_i = node_intensity(fiber, node_position(coordinates, ends))
    return JunctionFeatures(
        best_tan=round(tangents[0], 4), second_tan=round(tangents[1], 4),
        worst_tan=round(tangents[2], 4),
        tan_margin=round(tangents[1] - tangents[0], 4),
        tan_spread=round(tangents[2] - tangents[0], 4),
        len_min_um=round(lengths[0], 2), len_mid_um=round(lengths[1], 2),
        len_max_um=round(lengths[2], 2),
        len_ratio_min_max=round(lengths[0] / lengths[2], 4) if lengths[2] > 1e-9 else 1.0,
        width_min_um=round(widths[0], 3), width_max_um=round(widths[2], 3),
        width_ratio_min_max=round(widths[0] / widths[2], 4) if widths[2] > 1e-9 else 1.0,
        node_intensity_over_min=round(node_i / intensities[0], 4) if intensities[0] > 1e-9 else 1.0,
        node_intensity_over_max=round(node_i / intensities[2], 4) if intensities[2] > 1e-9 else 1.0,
        intensity_ratio_min_max=round(intensities[0] / intensities[2], 4) if intensities[2] > 1e-9 else 1.0)
