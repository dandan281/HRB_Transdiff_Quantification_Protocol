"""Measure how ambiguous the classical floor's junction pairing is (pre-build).

The classical floor (`ridge_graph.trace_fibers_parameterised`) resolves every
skeleton-graph junction with a single hand-tuned rule: pair branch-ends
straightest-through first, joining a pair only when its direction dot is
``<= straight_dot`` (canonical constant -0.5). This script does not change that
rule -- it audits it, so the size of the disagreement pool is known *before* any
learned junction classifier is built.

`build_branch_graph` only prunes short *dead-end* branches (skan type 1, the
canonical spur rule); it never prunes short junction-to-junction branches, so a
raw degree>=3 node count includes enormous numbers of sub-pixel skeletonisation
whiskers off an irregular territory boundary that are not real fibre crossings
(measured on one well: median incident-branch length 3.9 um / ~6 px, and only
~1% of nodes have every incident branch >= 10 um). A junction is only counted
as a **candidate** at all when every incident branch clears ``MIN_BRANCH_UM``
(10 um, matching the codebase's existing ``spur_um`` convention for "real
fibre-scale", not skeleton noise).

Among candidate junctions, one is flagged **ambiguous** when the direction-only
rule's chosen pairing is not clearly the right call, by any of three
independent criteria:

``near_threshold``   the winning pair's dot is within ``THRESHOLD_MARGIN`` of
                     ``straight_dot`` -- a small change in the fold-fitted
                     constant would have paired it differently.
``degree_ge4``       4+ branch-ends meet here (an X-crossing or busier), so more
                     than one simultaneous pairing is geometrically possible and
                     the greedy match is not guaranteed optimal.
``width_or_intensity_conflict``  the winning pair's local branch width or mean
                     Desmin intensity disagree sharply (ratio below
                     ``CONFLICT_RATIO``) despite being paired by direction --
                     width and stain continuity are independent evidence the
                     direction-only rule ignores.

Any junction meeting >=1 criterion joins the labeling pool. Run once per well
using the already-cached stage-A territory (`classical/_runs/v1/_territory_cache`)
and the bootstrap fiber images, at the canonical `TracerParams()` defaults --
the same pairing rule actually shipped, not a fold-fitted variant.

Usage (from repo root, PYTHONPATH includes model_labs)::

    python model_labs/classical/junction_ambiguity.py \\
        --territory-cache model_labs/classical/_runs/v1/_territory_cache \\
        --bootstrap PrecisionMyotube/annotation_work/bootstrap_v1 \\
        --out model_labs/classical/_runs/junction_ambiguity_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from classical.ridge_graph import (
    TracerParams, build_branch_graph, junction_candidates, pair_junction_ends)

WELLS = ["19_B06_act104_trka", "22_B03_act104_egfrc", "23_B02_ctrl",
         "29_C05_br223_egfrc", "32_C08_br223_igf1r", "33_C09_br223_trka"]

THRESHOLD_MARGIN = 0.15   # dot within this of straight_dot counts as "near"
CONFLICT_RATIO = 0.5      # min/max width or intensity below this counts as "conflict"
SAMPLE_PX = 5              # branch-end window used for width/intensity, in pixels
MIN_BRANCH_UM = 10.0       # a junction only counts if every incident branch clears
                           # this fibre-scale length; matches the codebase's own
                           # spur_um convention (see below)


def _end_sample_points(coords: np.ndarray, end: str, window: int = SAMPLE_PX) -> np.ndarray:
    """Up to ``window`` pixel coordinates nearest a branch's source/destination end."""
    if end == "source":
        return coords[:window]
    return coords[-window:]


def _end_width_um(distance_to_bg: np.ndarray, coords: np.ndarray, end: str,
                  pixel_um: float) -> float:
    points = _end_sample_points(coords, end).round().astype(int)
    rows = np.clip(points[:, 0], 0, distance_to_bg.shape[0] - 1)
    cols = np.clip(points[:, 1], 0, distance_to_bg.shape[1] - 1)
    # distance-to-background at a centreline pixel ~= local half-width
    return float(np.mean(distance_to_bg[rows, cols])) * 2.0 * pixel_um


def _end_intensity(fiber: np.ndarray, coords: np.ndarray, end: str) -> float:
    points = _end_sample_points(coords, end).round().astype(int)
    rows = np.clip(points[:, 0], 0, fiber.shape[0] - 1)
    cols = np.clip(points[:, 1], 0, fiber.shape[1] - 1)
    return float(np.mean(fiber[rows, cols].astype(np.float64)))


def _ratio(a: float, b: float) -> float:
    lo, hi = min(a, b), max(a, b)
    return lo / hi if hi > 1e-9 else 1.0


def _branch_length_um(coords: np.ndarray, pixel_um: float) -> float:
    if len(coords) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))) * pixel_um


def evaluate_junction(node: int, ends: list, coordinates: list,
                      distance_to_bg: np.ndarray, fiber: np.ndarray, pixel_um: float,
                      params: TracerParams = TracerParams(),
                      threshold_margin: float = THRESHOLD_MARGIN,
                      conflict_ratio: float = CONFLICT_RATIO) -> dict:
    """Full diagnostic for one degree>=3 junction: every candidate pair + reasons.

    Exposed separately from :func:`audit_well` so the operator-labeling round
    (`annotation_tools.qc_review.junction_pairs`) can reuse the exact same
    ambiguity criteria and per-pair diagnostics instead of re-deriving them --
    the labeling pool must be defined identically to what was measured and
    reported. Caller applies its own fibre-scale (``min_branch_um``) gate before
    calling this; it evaluates whatever ``ends`` it is given.
    """
    degree = len(ends)
    candidates = junction_candidates(ends)
    pairs = pair_junction_ends(ends, params.straight_dot)

    # any candidate (won or lost) within the margin means a slightly different
    # fold-fitted straight_dot would have paired this junction differently. This
    # is deliberately evaluated over every candidate pair, not just the winner:
    # the classifier's training unit is a candidate *pair*, and a junction whose
    # losing candidates are also decisively far from the boundary offers the
    # labeler nothing the winner didn't already establish.
    near_threshold = any(abs(dot - params.straight_dot) <= threshold_margin
                         for dot, _, _ in candidates)
    # stricter sub-flag: was the pairing actually adopted (or the best available
    # candidate, if nothing qualified) itself close to flipping?
    winner_dots = [dot for dot, a, b in candidates if (a, b) in pairs] or (
        [candidates[0][0]] if candidates else [])
    near_threshold_winner = any(abs(dot - params.straight_dot) <= threshold_margin
                                for dot in winner_dots)
    degree_ge4 = degree >= 4

    conflict = False
    adopted_diagnostics = []
    all_pair_diagnostics = []
    for dot, a, b in candidates:
        branch_a, end_a, _ = ends[a]
        branch_b, end_b, _ = ends[b]
        width_a = _end_width_um(distance_to_bg, coordinates[branch_a], end_a, pixel_um)
        width_b = _end_width_um(distance_to_bg, coordinates[branch_b], end_b, pixel_um)
        intensity_a = _end_intensity(fiber, coordinates[branch_a], end_a)
        intensity_b = _end_intensity(fiber, coordinates[branch_b], end_b)
        width_ratio = _ratio(width_a, width_b)
        intensity_ratio = _ratio(intensity_a, intensity_b)
        is_adopted = (a, b) in pairs
        if is_adopted and (width_ratio < conflict_ratio or intensity_ratio < conflict_ratio):
            conflict = True
        entry = {"end_indices": [a, b], "branches": [branch_a, branch_b],
                 "branch_ends": [end_a, end_b], "dot": round(dot, 4),
                 "width_ratio": round(width_ratio, 3),
                 "intensity_ratio": round(intensity_ratio, 3), "adopted": is_adopted}
        all_pair_diagnostics.append(entry)
        if is_adopted:
            adopted_diagnostics.append(entry)

    reasons = []
    if near_threshold:
        reasons.append("near_threshold")
    if degree_ge4:
        reasons.append("degree_ge4")
    if conflict:
        reasons.append("width_or_intensity_conflict")

    centroid = coordinates[ends[0][0]][0 if ends[0][1] == "source" else -1]
    return {
        "node": int(node), "degree": degree,
        "centroid_rc": [round(float(centroid[0]), 1), round(float(centroid[1]), 1)],
        "reasons": reasons, "near_threshold_winner": near_threshold_winner,
        "n_pairs_chosen": len(pairs), "n_ends_unpaired": degree - 2 * len(pairs),
        "pairs": adopted_diagnostics, "all_pairs": all_pair_diagnostics,
    }


def audit_well(well: str, territory: np.ndarray, fiber: np.ndarray, pixel_um: float,
               params: TracerParams = TracerParams(),
               min_branch_um: float = MIN_BRANCH_UM) -> dict:
    territory = np.asarray(territory, dtype=bool)
    skeleton = skeletonize(territory)
    if skeleton.sum() < 3:
        return {"well": well, "n_junctions": 0, "n_candidates": 0, "n_ambiguous": 0,
               "junctions": []}

    graph, node_ends, coordinates = build_branch_graph(skeleton, pixel_um, params)
    distance_to_bg = ndi.distance_transform_edt(territory)
    branch_lengths = [_branch_length_um(c, pixel_um) for c in coordinates]

    n_candidates = 0
    junctions = []
    for node, ends in node_ends.items():
        degree = len(ends)
        if degree < 3:
            continue  # degree<=2 is a pass-through or a dead end, never ambiguous
        if min(branch_lengths[b] for b, _, _ in ends) < min_branch_um:
            continue  # skeletonisation whisker, not a real fibre-scale junction
        n_candidates += 1
        info = evaluate_junction(node, ends, coordinates, distance_to_bg, fiber, pixel_um,
                                 params, THRESHOLD_MARGIN, CONFLICT_RATIO)
        if info["reasons"]:
            junctions.append(info)

    n_total_junctions = sum(1 for ends in node_ends.values() if len(ends) >= 3)
    return {"well": well, "n_junctions_raw": n_total_junctions, "n_candidates": n_candidates,
           "n_ambiguous": len(junctions), "junctions": junctions}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--territory-cache", default="model_labs/classical/_runs/v1/_territory_cache")
    parser.add_argument("--bootstrap", default="PrecisionMyotube/annotation_work/bootstrap_v1")
    parser.add_argument("--out", default="model_labs/classical/_runs/junction_ambiguity_v1.json")
    parser.add_argument("--wells", nargs="*", default=WELLS)
    args = parser.parse_args(argv)

    cache_dir = Path(args.territory_cache)
    bootstrap_dir = Path(args.bootstrap)
    results = []
    for well in args.wells:
        territory = np.load(cache_dir / f"{well}.territory.npy")
        fiber = tifffile.imread(bootstrap_dir / well / "image_fiber.tif")
        pixel_um = 0.6493  # constant across the bootstrap wells (verified per-well READMEs)
        result = audit_well(well, territory, fiber, pixel_um)
        del territory, fiber
        by_reason = {"near_threshold": 0, "degree_ge4": 0, "width_or_intensity_conflict": 0}
        n_near_threshold_winner = 0
        for j in result["junctions"]:
            for r in j["reasons"]:
                by_reason[r] += 1
            n_near_threshold_winner += j["near_threshold_winner"]
        result["by_reason"] = by_reason
        result["n_near_threshold_winner"] = n_near_threshold_winner
        results.append(result)
        print(f"  {well:22s} raw={result['n_junctions_raw']:4d} "
             f"candidates(>={MIN_BRANCH_UM:g}um)={result['n_candidates']:4d} "
             f"ambiguous={result['n_ambiguous']:4d}  {by_reason}")

    summary = {
        "params": {"straight_dot": TracerParams().straight_dot,
                  "threshold_margin": THRESHOLD_MARGIN,
                  "conflict_ratio": CONFLICT_RATIO, "sample_px": SAMPLE_PX,
                  "min_branch_um": MIN_BRANCH_UM},
        "total_junctions_raw": sum(r["n_junctions_raw"] for r in results),
        "total_candidates": sum(r["n_candidates"] for r in results),
        "total_ambiguous": sum(r["n_ambiguous"] for r in results),
        "total_near_threshold_winner": sum(r["n_near_threshold_winner"] for r in results),
        "wells": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ntotal raw degree>=3 nodes: {summary['total_junctions_raw']} "
         "(mostly sub-pixel skeletonisation whiskers, not real crossings)")
    print(f"total fibre-scale candidate junctions (>= {MIN_BRANCH_UM:g}um every branch): "
         f"{summary['total_candidates']}")
    print(f"total ambiguous among those, any-candidate-near-boundary "
         f"(labeling pool): {summary['total_ambiguous']}")
    print(f"  of which the adopted/best pairing itself was near the boundary: "
         f"{summary['total_near_threshold_winner']}")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
