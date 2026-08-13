"""Junction candidate finder: degree-3 gate, fibre-scale gate, ambiguity gate."""
import numpy as np

from annotation_tools.qc_review.junction_pairs import find_junction_cases

PIXEL_UM = 0.6493


def _t_field(shape=(200, 200), left_thickness=9, right_thickness=3, stub_len=42,
            stub_thickness=5):
    """A T-junction with a width step between its two horizontal halves.

    Genuinely ambiguous: the width transition bends the skeleton path near the
    node, so the winning pair's direction dot drifts off a clean -1 and lands
    near the `straight_dot` boundary (measured: -0.61).
    """
    mask = np.zeros(shape, dtype=bool)
    mid = shape[0] // 2
    lh, rh = left_thickness // 2, right_thickness // 2
    mask[mid - lh:mid + lh + 1, 10:100] = True
    mask[mid - rh:mid + rh + 1, 100:190] = True
    sh = stub_thickness // 2
    mask[mid - stub_len:mid, mid - sh:mid + sh + 1] = True
    return mask


def _clean_t_field(shape=(200, 200), thickness=5, stub_len=42):
    """A uniform-width T-junction: dot=-1 exactly, matched width/intensity.

    Unambiguous by every criterion -- must never reach the labeling pool.
    """
    mask = np.zeros(shape, dtype=bool)
    mid = shape[0] // 2
    half = thickness // 2
    mask[mid - half:mid + half + 1, 10:190] = True
    mask[mid - stub_len:mid, mid - half:mid + half + 1] = True
    return mask


def _cross_field(shape=(200, 200), thickness=5):
    """Two long fibres crossing -- a degree-4 node, out of scope for round 1."""
    mask = np.zeros(shape, dtype=bool)
    mid = shape[0] // 2
    half = thickness // 2
    mask[mid - half:mid + half + 1, 10:190] = True
    mask[10:190, mid - half:mid + half + 1] = True
    return mask


def _fiber(shape=(200, 200), value=3000):
    return np.full(shape, value, dtype=np.uint16)


def test_ambiguous_t_junction_is_found_with_three_branches():
    cases, coordinates, node_ends = find_junction_cases("well_a", _t_field(), _fiber(), PIXEL_UM,
                                                        reasons=None)
    assert len(cases) == 1
    case = cases[0]
    assert case.well == "well_a"
    assert len(case.branch_ids) == 3
    assert len(set(case.branch_ids)) == 3, "the three branches must be distinct"
    assert case.pair_keys == (("0", "1"), ("0", "2"), ("1", "2"))
    assert all(bid < len(coordinates) for bid in case.branch_ids)
    assert case.reasons, "this fixture is constructed to be genuinely ambiguous"
    assert len(node_ends[case.node]) == 3, "node_ends must be keyed by the same node id"


def test_unambiguous_junction_is_never_offered():
    """A perfectly clean straight-through T (dot=-1, matched width) is not
    genuinely ambiguous and must not reach the labeling pool at any breadth."""
    cases, _coords, _ends = find_junction_cases("well_a", _clean_t_field(), _fiber(), PIXEL_UM,
                                                reasons=None)
    assert cases == [], "an unambiguous junction must not be offered for labeling"


def test_degree_four_junction_is_out_of_scope():
    """A cross is degree 4; round 1 is scoped to degree-3 junctions only."""
    cases, _coords, _ends = find_junction_cases("well_a", _cross_field(), _fiber(), PIXEL_UM,
                                                reasons=None)
    assert cases == []


def test_min_branch_um_gate_excludes_short_stub():
    """A stub shorter than the fibre-scale floor must not create a candidate."""
    short_stub = _t_field(stub_len=6)   # ~3.9 um at this pixel size, well under 10 um
    cases, _coords, _ends = find_junction_cases("well_a", short_stub, _fiber(), PIXEL_UM,
                                                reasons=None, min_branch_um=10.0)
    assert cases == [], "a sub-fibre-scale stub must not reach the labeling pool"


def test_case_id_is_stable_and_unique():
    cases, _coords, _ends = find_junction_cases("well_a", _t_field(), _fiber(), PIXEL_UM,
                                                reasons=None)
    assert cases[0].case_id().startswith("junction_")


def test_round1_pool_is_a_subset_of_the_full_ambiguous_pool():
    """Round 1's (near_threshold_winner | width_or_intensity_conflict) pool must
    never exceed the full any-criterion (near_threshold) pool."""
    from annotation_tools.qc_review.junction_pairs import ROUND1_REASONS

    all_cases, _coords, _ends = find_junction_cases("well_a", _t_field(), _fiber(), PIXEL_UM,
                                                    reasons=None)
    round1_cases, _coords2, _ends2 = find_junction_cases("well_a", _t_field(), _fiber(), PIXEL_UM,
                                                         reasons=ROUND1_REASONS)
    all_nodes = {c.node for c in all_cases}
    round1_nodes = {c.node for c in round1_cases}
    assert round1_nodes <= all_nodes
    assert round1_nodes, "the fixture's junction must itself qualify for round 1"


def test_empty_territory_returns_no_cases():
    cases, coordinates, node_ends = find_junction_cases(
        "well_a", np.zeros((64, 64), dtype=bool), _fiber((64, 64)), PIXEL_UM, reasons=None)
    assert cases == [] and coordinates == [] and node_ends == {}
