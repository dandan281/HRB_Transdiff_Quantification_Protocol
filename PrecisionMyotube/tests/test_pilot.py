import json

import numpy as np

from precision_myotube.pilot import (build_pilot_candidates, build_pilot_handoff,
                                     evaluate_g1, select_pilot_tasks)
from precision_myotube.schema import InstanceRecord, InstanceSet, encode_rle


def _candidates():
    rows = []
    plates = ["PLATE_23", "PLATE_28", "PLATE_32", "PLATE_26"]
    for index in range(140):
        rows.append({
            "image_id": f"field_{index // 10}",
            "object_id": f"proposal_{index}",
            "plate": plates[index % len(plates)],
            "density": "sparse" if index % 2 else "dense",
            "intensity": "dim" if index % 3 else "bright",
            "length_class": "short" if index % 5 else "long",
            "hard_case": index % 3 == 0,
        })
    return rows


def test_pilot_selection_is_deterministic_stratified_and_excludes_plate26(tmp_path):
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps({"candidates": _candidates()}), encoding="utf-8")
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    a = select_pilot_tasks(source, first, target=80, minimum_hard=25, seed="locked")
    b = select_pilot_tasks(source, second, target=80, minimum_hard=25, seed="locked")
    assert [x["task_id"] for x in a["tasks"]] == [x["task_id"] for x in b["tasks"]]
    assert a["audit"]["ready_for_dual_annotation"]
    assert "PLATE_26" not in a["audit"]["plates"]


def test_task_identity_includes_plate_for_repeated_well_names(tmp_path):
    rows = _candidates()
    rows[1]["image_id"] = rows[0]["image_id"]
    rows[1]["object_id"] = rows[0]["object_id"]
    rows[1]["plate"] = "PLATE_28"
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps({"candidates": rows}), encoding="utf-8")
    result = select_pilot_tasks(source, tmp_path / "pilot.json", target=80, minimum_hard=25)
    assert result["audit"]["requirements"]["unique_task_ids"]


def test_g1_fails_closed_without_human_evidence(tmp_path):
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps({"candidates": _candidates()}), encoding="utf-8")
    pilot = tmp_path / "pilot.json"
    select_pilot_tasks(source, pilot, target=80, minimum_hard=25)
    evidence = tmp_path / "g1.json"
    evidence.write_text(json.dumps({
        "pilot_manifest": "pilot.json",
        "synthetic_overlap_roundtrip_passed": True,
        "critical_tool_defects_open": 0,
        "schema_validation_passed": True,
    }), encoding="utf-8")
    result = evaluate_g1(evidence)
    assert not result["passed"]
    assert not result["requirements"]["biological_reviewers_approved"]


def test_candidate_builder_derives_proposal_strata_without_claiming_truth(tmp_path):
    samples = []
    for field_index, coverage_width in enumerate((8, 14, 22)):
        run = tmp_path / f"run_{field_index}"
        run.mkdir()
        shape = (30, 40)
        fiber = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        np.save(run / "ch0_raw16.npy", fiber)
        np.save(run / "ch1_raw16.npy", np.zeros(shape, dtype=np.uint16))
        territory = np.zeros(shape, bool)
        territory[5:20, 2:2 + coverage_width] = True
        np.save(run / "myotube_territory.npy", territory)
        image_id = f"field_{field_index}"
        (run / "metadata.json").write_text(json.dumps({
            "image_id": image_id, "image_shape": list(shape), "pixel_um": 1.0,
            "channels": {"fiber": 0, "dapi": 1},
        }), encoding="utf-8")
        a = np.zeros(shape, bool); a[6:10, 3:10] = True
        b = np.zeros(shape, bool); b[10:14, 3:18] = True
        instances = InstanceSet(shape, image_id, [
            InstanceRecord("a", "ambiguous", encode_rle(a)),
            InstanceRecord("b", "ambiguous", encode_rle(b)),
        ])
        instances.save(run / "instance_proposals.json")
        samples.append({"plate": f"PLATE_{23 + field_index}", "run_dir": str(run)})
    manifest = tmp_path / "runs.json"
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    output = tmp_path / "candidates.json"
    result = build_pilot_candidates(manifest, output)
    assert len(result["candidates"]) == 6
    assert result["purpose"].endswith("not biological labels")
    assert {"sparse", "dense"} <= {x["density"] for x in result["candidates"]}
    assert all("hard_case" in x and "length_class" in x for x in result["candidates"])


def test_pilot_handoff_binds_and_hashes_package(tmp_path):
    package = tmp_path / "package"; package.mkdir()
    shape = (4, 5)
    for name in ("fiber_raw16.tif", "dapi_raw16.tif", "semantic_territory.tif",
                 "starting_labels.tif", "overlap_ignore.tif"):
        (package / name).write_bytes(name.encode())
    (package / "README.json").write_text(json.dumps({"image_id": "field"}))
    (package / "instance_properties.csv").write_text(
        "label,id,status,reviewed,source,notes\n1,myotube_0001,ambiguous,false,prompt,\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{
        "task_id": "PLATE_23::field::myotube_0001", "field_key": "PLATE_23::field",
        "object_id": "myotube_0001"}]}))
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps({"samples": [{"plate": "PLATE_23",
                                              "package_dir": str(package)}]}))
    result = build_pilot_handoff(manifest, runs, tmp_path / "handoff.json")
    assert result["task_count"] == 1
    assert result["fields"][0]["pilot_object_ids"] == ["myotube_0001"]
    assert len(result["fields"][0]["package_artifact_sha256"]) == 7
