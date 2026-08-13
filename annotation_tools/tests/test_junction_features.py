"""Junction-pair features: tangent_cos, turn_angle, width/intensity ratio, length_min."""
import numpy as np
import pytest

from annotation_tools.qc_review.junction_features import (
    FEATURE_KEYS, compute_pair_features, turn_angle_deg)

PIXEL_UM = 0.6493


def test_turn_angle_zero_for_perfect_straight_through():
    """Anti-parallel outward directions (dot=-1) means a fibre continues dead straight."""
    assert turn_angle_deg(-1.0) == pytest.approx(0.0, abs=1e-6)


def test_turn_angle_90_for_perpendicular_branches():
    assert turn_angle_deg(0.0) == pytest.approx(90.0, abs=1e-6)


def test_turn_angle_180_for_a_reversal():
    """Same-direction outward vectors (dot=+1) is a full U-turn, not a pass-through."""
    assert turn_angle_deg(1.0) == pytest.approx(180.0, abs=1e-6)


def test_turn_angle_is_monotonic_in_tangent_cos():
    dots = np.linspace(-1.0, 1.0, 21)
    angles = [turn_angle_deg(d) for d in dots]
    assert angles == sorted(angles), "turn_angle must increase as tangent_cos increases"


def _straight_coords():
    """Two collinear branches on either side of column 50: a pass-through pair.

    Columns are kept well clear of each other (0-40 vs 60-100) so each
    branch's 5px end-sample window never straddles a test fixture's column
    boundary at 50.
    """
    left = np.array([[50.0, c] for c in range(0, 41)])           # destination end near col 40
    right = np.array([[50.0, c] for c in range(60, 101)])        # source end near col 60
    return left, right


def test_features_for_a_clean_straight_pass_through():
    left, right = _straight_coords()
    coordinates = [left, right]
    fiber = np.full((110, 110), 3000, dtype=np.uint16)
    distance_to_bg = np.full((110, 110), 3.0)   # uniform width everywhere

    feats = compute_pair_features(coordinates, 0, "destination", 1, "source",
                                  distance_to_bg=distance_to_bg,
                                  fiber=fiber, pixel_um=PIXEL_UM,
                                  branch_length_a_um=26.0, branch_length_b_um=26.6)
    assert feats.tangent_cos == pytest.approx(-1.0)
    assert feats.turn_angle_deg == pytest.approx(0.0, abs=1e-2)
    assert feats.width_ratio == pytest.approx(1.0)
    assert feats.intensity_ratio == pytest.approx(1.0)
    assert feats.length_min_um == pytest.approx(26.0)


def test_width_ratio_reflects_mismatched_branch_thickness():
    left, right = _straight_coords()
    coordinates = [left, right]
    fiber = np.full((110, 110), 3000, dtype=np.uint16)
    distance_to_bg = np.zeros((110, 110))
    distance_to_bg[:, :50] = 5.0     # thick on the left branch's end window
    distance_to_bg[:, 50:] = 1.0     # thin on the right branch's end window

    feats = compute_pair_features(coordinates, 0, "destination", 1, "source",
                                  distance_to_bg=distance_to_bg,
                                  fiber=fiber, pixel_um=PIXEL_UM,
                                  branch_length_a_um=26.0, branch_length_b_um=26.0)
    assert feats.width_ratio == pytest.approx(1.0 / 5.0, abs=1e-3)


def test_intensity_ratio_reflects_mismatched_stain():
    left, right = _straight_coords()
    coordinates = [left, right]
    fiber = np.zeros((110, 110), dtype=np.uint16)
    fiber[:, :50] = 4000     # bright on the left branch's end window
    fiber[:, 50:] = 1000     # dim on the right branch's end window
    distance_to_bg = np.full((110, 110), 3.0)

    feats = compute_pair_features(coordinates, 0, "destination", 1, "source",
                                  distance_to_bg=distance_to_bg,
                                  fiber=fiber, pixel_um=PIXEL_UM,
                                  branch_length_a_um=26.0, branch_length_b_um=26.0)
    assert feats.intensity_ratio == pytest.approx(1000.0 / 4000.0, abs=1e-3)


def test_length_min_takes_the_shorter_branch():
    left, right = _straight_coords()
    coordinates = [left, right]
    fiber = np.full((110, 110), 3000, dtype=np.uint16)
    distance_to_bg = np.full((110, 110), 3.0)

    feats = compute_pair_features(coordinates, 0, "destination", 1, "source",
                                  distance_to_bg=distance_to_bg,
                                  fiber=fiber, pixel_um=PIXEL_UM,
                                  branch_length_a_um=12.5, branch_length_b_um=40.0)
    assert feats.length_min_um == pytest.approx(12.5)


def test_vector_respects_feature_key_order():
    left, right = _straight_coords()
    feats = compute_pair_features([left, right], 0, "destination", 1, "source",
                                  distance_to_bg=np.full((110, 110), 2.0),
                                  fiber=np.full((110, 110), 1000, dtype=np.uint16),
                                  pixel_um=PIXEL_UM, branch_length_a_um=15.0,
                                  branch_length_b_um=20.0)
    vec = feats.vector()
    assert len(vec) == len(FEATURE_KEYS) == 6
    assert vec[0] == pytest.approx(feats.tangent_cos)              # tangent_cos is first
    assert vec[4] == pytest.approx(15.0)                           # length_min_um
    assert vec[5] == pytest.approx(feats.node_intensity_ratio)     # node ratio is last


# ------------------------------------------------- direction window (the 2026-07-28 fix)


def test_end_direction_points_outward_from_the_junction():
    """Two collinear branches meeting at a junction must be ANTI-parallel: each
    points away from the shared node, so their dot is -1, not +1."""
    from annotation_tools.qc_review.junction_features import end_direction

    left, right = _straight_coords()
    d_left = end_direction(left, "destination")      # junction is at left's high-col end
    d_right = end_direction(right, "source")         # junction is at right's low-col end
    assert float(np.dot(d_left, d_right)) == pytest.approx(-1.0, abs=1e-6)


def test_direction_window_is_not_the_tracers_three_pixel_step():
    """Regression guard on the finding that drove AUC 0.693 -> 0.892.

    The tracer's `TracerParams.direction_step` is 3 px, tuned for its own spur
    bookkeeping; over 3 px a skeleton direction is pixelation noise. If this
    default is ever silently reset to the tracer's, the classifier quietly
    loses ~0.2 AUC with no test failing -- so pin it.
    """
    from classical.ridge_graph import TracerParams
    from annotation_tools.qc_review.junction_features import DIRECTION_WINDOW_PX

    assert DIRECTION_WINDOW_PX == 15
    assert DIRECTION_WINDOW_PX > TracerParams().direction_step * 4


def test_longer_window_is_robust_to_a_pixelation_wobble():
    """The whole point of the longer window: one jittered pixel near the
    junction must not flip the measured direction."""
    from annotation_tools.qc_review.junction_features import end_direction

    straight = np.array([[50.0, c] for c in range(0, 41)])
    wobbled = straight.copy()
    wobbled[-1] = [51.0, 40.0]        # last pixel steps off-axis, as skeletons do

    tiny = float(np.dot(end_direction(straight, "destination", window=2),
                        end_direction(wobbled, "destination", window=2)))
    wide = float(np.dot(end_direction(straight, "destination", window=15),
                        end_direction(wobbled, "destination", window=15)))
    assert wide > tiny, "a longer window must be less sensitive to endpoint jitter"
    assert wide > 0.99


def test_window_is_clamped_to_short_branches():
    """A branch shorter than the window uses its whole path instead of erroring."""
    from annotation_tools.qc_review.junction_features import end_direction

    short = np.array([[50.0, 0.0], [50.0, 1.0], [50.0, 2.0]])
    d = end_direction(short, "source", window=15)
    assert float(np.linalg.norm(d)) == pytest.approx(1.0)


def test_single_pixel_branch_returns_zero_direction():
    from annotation_tools.qc_review.junction_features import end_direction

    assert np.allclose(end_direction(np.array([[5.0, 5.0]]), "source"), np.zeros(2))


# ------------------------------------------ node intensity + junction-level features


def _three_branch_ends(coordinates):
    """(branch, end, direction) triples in the shape build_branch_graph returns."""
    return [(0, "destination", None), (1, "source", None), (2, "source", None)]


def _t_coordinates():
    """Three branches meeting at (50, 50): two collinear horizontals + a vertical."""
    left = np.array([[50.0, c] for c in range(10, 51)])          # ends AT the node
    right = np.array([[50.0, c] for c in range(50, 91)])         # starts AT the node
    stub = np.array([[float(r), 50.0] for r in range(50, 21, -1)])  # starts AT the node
    return [left, right, stub]


def test_node_position_is_the_shared_endpoint():
    from annotation_tools.qc_review.junction_features import node_position

    coords = _t_coordinates()
    assert node_position(coords, _three_branch_ends(coords)) == pytest.approx((50.0, 50.0))


def test_node_intensity_ratio_is_neutral_on_a_uniform_field():
    """No node-vs-branch contrast anywhere means exactly 1.0."""
    coords = _t_coordinates()
    feats = compute_pair_features(coords, 0, "destination", 1, "source",
                                  distance_to_bg=np.full((110, 110), 3.0),
                                  fiber=np.full((110, 110), 3000, dtype=np.uint16),
                                  pixel_um=PIXEL_UM, branch_length_a_um=26.0,
                                  branch_length_b_um=26.0, node_rc=(50.0, 50.0))
    assert feats.node_intensity_ratio == pytest.approx(1.0, abs=1e-6)


def test_node_intensity_ratio_responds_to_stain_at_the_node():
    """It must actually measure the node region, not silently ignore it.

    Deliberately asserts only that the value MOVES off neutral -- not which
    way. The branch-end window starts at the node and so overlaps the node
    square, which makes the sign geometry-dependent; see the module docstring.
    """
    coords = _t_coordinates()
    uniform = np.full((110, 110), 3000, dtype=np.uint16)
    altered = uniform.copy()
    altered[48:53, 48:53] = 300              # change the stain right at the junction

    common = dict(distance_to_bg=np.full((110, 110), 3.0), pixel_um=PIXEL_UM,
                  branch_length_a_um=26.0, branch_length_b_um=26.0, node_rc=(50.0, 50.0))
    flat = compute_pair_features(coords, 0, "destination", 1, "source",
                                 fiber=uniform, **common)
    changed = compute_pair_features(coords, 0, "destination", 1, "source",
                                    fiber=altered, **common)
    assert flat.node_intensity_ratio == pytest.approx(1.0, abs=1e-6)
    assert abs(changed.node_intensity_ratio - 1.0) > 0.05


def test_node_intensity_samples_the_node_not_the_whole_field():
    from annotation_tools.qc_review.junction_features import node_intensity

    field = np.full((110, 110), 1000, dtype=np.uint16)
    field[47:54, 47:54] = 9000               # bright only at the node square
    assert node_intensity(field, (50.0, 50.0)) == pytest.approx(9000.0)
    assert node_intensity(field, (10.0, 10.0)) == pytest.approx(1000.0)


def test_node_intensity_is_clipped_at_the_image_edge():
    from annotation_tools.qc_review.junction_features import node_intensity

    field = np.full((20, 20), 500, dtype=np.uint16)
    assert node_intensity(field, (0.0, 0.0)) == pytest.approx(500.0)
    assert node_intensity(field, (19.0, 19.0)) == pytest.approx(500.0)


def test_node_intensity_ratio_is_neutral_without_a_node():
    """A caller with only branch geometry still gets the other features."""
    coords = _t_coordinates()
    feats = compute_pair_features(coords, 0, "destination", 1, "source",
                                  distance_to_bg=np.full((110, 110), 3.0),
                                  fiber=np.full((110, 110), 3000, dtype=np.uint16),
                                  pixel_um=PIXEL_UM, branch_length_a_um=26.0,
                                  branch_length_b_um=26.0)
    assert feats.node_intensity_ratio == pytest.approx(1.0)


def test_junction_features_are_invariant_to_branch_ordering():
    """Built from sorted statistics, so relabelling A/B/C cannot change them."""
    from annotation_tools.qc_review.junction_features import (
        JUNCTION_FEATURE_KEYS, compute_junction_features)

    coords = _t_coordinates()
    ends = _three_branch_ends(coords)
    fiber = np.full((110, 110), 3000, dtype=np.uint16)
    d2bg = np.full((110, 110), 3.0)
    lengths = [26.0, 26.0, 18.0]

    a = compute_junction_features(coords, ends, d2bg, fiber, PIXEL_UM, lengths)
    # permute the branches (and their lengths consistently)
    order = [2, 0, 1]
    b = compute_junction_features(coords, [ends[i] for i in order], d2bg, fiber,
                                  PIXEL_UM, [lengths[i] for i in order])
    assert a.vector(JUNCTION_FEATURE_KEYS) == pytest.approx(b.vector(JUNCTION_FEATURE_KEYS))


def test_junction_features_rank_the_three_tangents():
    from annotation_tools.qc_review.junction_features import compute_junction_features

    coords = _t_coordinates()
    f = compute_junction_features(coords, _three_branch_ends(coords),
                                  np.full((110, 110), 3.0),
                                  np.full((110, 110), 3000, dtype=np.uint16),
                                  PIXEL_UM, [26.0, 26.0, 18.0])
    assert f.best_tan <= f.second_tan <= f.worst_tan
    assert f.tan_margin == pytest.approx(f.second_tan - f.best_tan)
    assert f.tan_spread == pytest.approx(f.worst_tan - f.best_tan)
    # the two collinear horizontals are the straight-through pair: dot ~ -1
    assert f.best_tan == pytest.approx(-1.0, abs=1e-6)
