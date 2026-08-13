"""Active-learning junction round 2: uncertainty ranking, exclusion, end-to-end.

Uses small synthetic fields for exact properties, mirroring test_link_active.py's
philosophy: the one thing these cannot cover is the AUC on the real 245 labels;
that is checked by running the CLI against the bootstrap (see the session
report), not in unit tests.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from annotation_tools.qc_review.junction_active import (
    already_offered, build_active_round, score_new_candidates, uncertainty)
from annotation_tools.qc_review.junction_features import JunctionPairFeatures
from annotation_tools.qc_review.junction_model import (
    JunctionPairExample, fit_junction_classifier)

PIXEL_UM = 0.6493


def _draw_t(mask, r, c):
    """Draw one ambiguous T-junction with its node at ``(r, c)``.

    Same shape the other junction tests use: a horizontal fibre with a width
    step at the node (thick left, thin right) plus a vertical stub. The width
    step bends the skeleton near the node, so the winning pair's direction dot
    lands near the `straight_dot` boundary -- i.e. genuinely ambiguous, so it
    reaches the candidate pool.
    """
    mask[r - 4:r + 5, c - 90:c] = True        # thick left horizontal
    mask[r - 1:r + 2, c:c + 90] = True        # thin right horizontal
    mask[r - 42:r, c - 2:c + 3] = True        # vertical stub, going up


def _one_junction_field(shape=(200, 220)):
    mask = np.zeros(shape, dtype=bool)
    _draw_t(mask, 100, 110)
    return mask


def _many_junction_field(n, cols=4, cell=260):
    rows = -(-n // cols)
    mask = np.zeros((rows * cell + 80, cols * cell + 80), dtype=bool)
    placed = 0
    for r in range(rows):
        for c in range(cols):
            if placed >= n:
                break
            _draw_t(mask, 60 + r * cell, 120 + c * cell)
            placed += 1
    return mask


def _fiber(shape, value=3000):
    return np.full(shape, value, dtype=np.uint16)


def _junction_nodes(territory):
    from classical.ridge_graph import TracerParams, build_branch_graph
    from skimage.morphology import skeletonize

    _graph, node_ends, _coords = build_branch_graph(
        skeletonize(territory), PIXEL_UM, TracerParams())
    return sorted(n for n, e in node_ends.items() if len(e) == 3), node_ends


# ------------------------------------------------------------------ uncertainty


def test_uncertainty_peaks_at_one_half():
    assert uncertainty(0.5) == pytest.approx(1.0)
    assert uncertainty(0.0) == pytest.approx(0.0)
    assert uncertainty(1.0) == pytest.approx(0.0)
    assert uncertainty(0.55) > uncertainty(0.9)


# ---------------------------------------------------------------- already_offered


def test_already_offered_includes_every_outcome(tmp_path):
    """Branch points and unsure cases must be excluded from re-serving too, not
    just decided through-pairs -- re-serving an 'unsure' case wastes the
    operator's time on something they already looked at and could not call."""
    export = {"decisions": {
        "w1/junction_000001": {"well": "w1", "node": 1, "chosen_pair": "AB",
                               "branch_point": False, "unsure": False},
        "w1/junction_000002": {"well": "w1", "node": 2, "chosen_pair": None,
                               "branch_point": True, "unsure": False},
        "w1/junction_000003": {"well": "w1", "node": 3, "chosen_pair": None,
                               "branch_point": False, "unsure": True},
    }}
    export_path = tmp_path / "e.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    assert already_offered([export_path]) == {("w1", 1), ("w1", 2), ("w1", 3)}


def test_already_offered_merges_multiple_exports(tmp_path):
    e1 = {"decisions": {"w1/junction_000001": {"well": "w1", "node": 1}}}
    e2 = {"decisions": {"w1/junction_000002": {"well": "w1", "node": 2}}}
    p1, p2 = tmp_path / "e1.json", tmp_path / "e2.json"
    p1.write_text(json.dumps(e1), encoding="utf-8")
    p2.write_text(json.dumps(e2), encoding="utf-8")
    assert already_offered([p1, p2]) == {("w1", 1), ("w1", 2)}


# ------------------------------------------------------------- score_new_candidates


def _synthetic_model():
    """Trivially-fit model on tangent_cos alone -- no images, exact and fast."""
    examples = []
    for wi, well in enumerate(("wA", "wB", "wC")):
        for i, cos in enumerate(np.linspace(-1, 1, 12)):
            feats = JunctionPairFeatures(float(cos), 0.0, 1.0, 1.0, 20.0)
            examples.append(JunctionPairExample(well, wi * 100 + i, "AB", feats,
                                                int(cos < -0.3)))
    return fit_junction_classifier(examples, ("tangent_cos",))


def test_score_new_candidates_returns_a_scored_junction():
    territory = _one_junction_field()
    served, counts = score_new_candidates("w1", territory, _fiber(territory.shape),
                                          PIXEL_UM, _synthetic_model(), exclude=set())
    assert counts["candidates"] == 1 and counts["new"] == 1
    assert len(served) == 1
    sj = served[0]
    assert sj.best_pair_key in ("AB", "AC", "BC")
    assert 0.0 <= sj.best_proba <= 1.0
    assert 0.0 <= sj.uncertainty <= 1.0


def test_score_new_candidates_excludes_already_offered():
    territory = _one_junction_field()
    nodes, _ends = _junction_nodes(territory)
    served, counts = score_new_candidates("w1", territory, _fiber(territory.shape),
                                          PIXEL_UM, _synthetic_model(),
                                          exclude={("w1", nodes[0])})
    assert served == []
    assert counts["candidates"] == 1, "the junction is still a candidate..."
    assert counts["new"] == 0, "...but must not be served a second time"


# ------------------------------------------------------------------ end to end


def _two_well_fixture(tmp_path, n_per_well=10):
    """Two identical wells on disk (the project's split policy is LOWO by well,
    so a fixture must have >= 2 wells for the model to fit at all)."""
    territory = _many_junction_field(n=n_per_well)
    fiber = _fiber(territory.shape)
    nodes, node_ends = _junction_nodes(territory)
    assert len(nodes) == n_per_well

    cache_dir = tmp_path / "territory_cache"
    cache_dir.mkdir()
    bootstrap = tmp_path / "bootstrap"
    for well in ("w1", "w2"):
        np.save(cache_dir / f"{well}.territory.npy", territory)
        well_dir = bootstrap / well
        well_dir.mkdir(parents=True)
        tifffile.imwrite(well_dir / "image_fiber.tif", fiber)
    return cache_dir, bootstrap, nodes, node_ends


def _decision(well, node, ends, chosen):
    labels = {"AB": 0, "AC": 0, "BC": 0}
    labels[chosen] = 1
    return {"well": well, "node": node, "decided_at": "2026-01-01T00:00:00Z",
            "chosen_pair": chosen, "branch_point": False, "unsure": False,
            "pairs": [
                {"key": "AB", "branches": [ends[0][0], ends[1][0]], "label": labels["AB"]},
                {"key": "AC", "branches": [ends[0][0], ends[2][0]], "label": labels["AC"]},
                {"key": "BC", "branches": [ends[1][0], ends[2][0]], "label": labels["BC"]}]}


def _write_export(path, decisions):
    path.write_text(json.dumps({
        "schema": "junction_pairs.v1", "batch_id": "round1", "reviewer": "r",
        "session_started_at": "2026-01-01T00:00:00Z",
        "exported_at": "2026-01-01T00:00:00Z",
        "n_cases": len(decisions), "n_explicitly_decided": len(decisions),
        "decisions": decisions}), encoding="utf-8")
    return path


def test_build_active_round_serves_only_unlabeled_junctions(tmp_path):
    """9 of 10 junctions per well become round-1 training data; the 10th in each
    is the only thing round 2 should discover and serve."""
    cache_dir, bootstrap, nodes, node_ends = _two_well_fixture(tmp_path)

    decisions = {}
    for well in ("w1", "w2"):
        for i, node in enumerate(nodes[:9]):
            chosen = "AB" if i % 2 == 0 else "BC"     # both classes present per well
            decisions[f"{well}/junction_{node:06d}"] = _decision(well, node,
                                                                 node_ends[node], chosen)
    export_path = _write_export(tmp_path / "round1.junctions.json", decisions)

    out_path = tmp_path / "round2.html"
    manifest = build_active_round(cache_dir, bootstrap, [export_path], out_path,
                                  reviewer="r2", wells=["w1", "w2"], max_junctions=50)

    assert {row["node"] for row in manifest["served_junctions"]} == {nodes[9]}
    assert manifest["pool"]["new_junctions_total"] == 2      # the 10th in each well
    assert manifest["pool"]["dropped_least_uncertain"] == 0
    assert out_path.is_file()
    assert Path(out_path).with_suffix(".manifest.json").is_file()


def test_build_active_round_caps_and_reports_what_it_dropped(tmp_path):
    """A bounded round must SAY what it truncated rather than silently dropping."""
    cache_dir, bootstrap, nodes, node_ends = _two_well_fixture(tmp_path)

    decisions = {}
    for well in ("w1", "w2"):
        for i, node in enumerate(nodes[:6]):
            chosen = "AB" if i % 2 == 0 else "BC"
            decisions[f"{well}/junction_{node:06d}"] = _decision(well, node,
                                                                 node_ends[node], chosen)
    export_path = _write_export(tmp_path / "round1.junctions.json", decisions)

    manifest = build_active_round(cache_dir, bootstrap, [export_path],
                                  tmp_path / "round2.html", reviewer="r2",
                                  wells=["w1", "w2"], max_junctions=3)
    pool = manifest["pool"]
    assert pool["new_junctions_total"] == 8       # (10 - 6) unlabeled x 2 wells
    assert pool["served"] == 3
    assert pool["dropped_least_uncertain"] == 5


def test_select_feature_set_error_names_the_single_well_cause():
    """A single-well label set cannot support leave-one-well-out; the error must
    say so rather than surfacing a numpy zero-size reduction."""
    from annotation_tools.qc_review.junction_model import select_feature_set

    examples = [
        JunctionPairExample("only_well", i, "AB",
                            JunctionPairFeatures(float(c), 0.0, 1.0, 1.0, 20.0),
                            int(c < 0))
        for i, c in enumerate(np.linspace(-1, 1, 12))]
    with pytest.raises(RuntimeError, match="leave-one-well-out needs at least 2"):
        select_feature_set(examples)
