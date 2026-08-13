"""CL02 -- overlap-safe annotation round-trip proof."""
import numpy as np

from annotation_tools.roundtrip import run_overlap_roundtrip, make_synthetic_crossing
from annotation_tools.package import load_annotation_package


def test_overlap_roundtrip_all_checks_pass(tmp_path):
    report = run_overlap_roundtrip(tmp_path)
    assert report.passed, report.checks
    # The individually required properties from the manual:
    assert report.checks["shared_pixels_nonzero"]
    assert report.checks["both_full_through_crossing"]
    assert report.checks["untouched_hash_stable"]
    assert report.checks["untouched_iou_is_1"]
    assert report.checks["edited_iou_matches_intended_edit"]
    assert report.checks["overlap_survived_json"]
    assert report.checks["flat_tiff_loses_overlap"]


def test_synthetic_crossing_has_shared_pixels():
    a, b = make_synthetic_crossing((48, 48), thickness=5)
    assert np.logical_and(a, b).sum() > 0
    assert a.sum() > 0 and b.sum() > 0


def test_flat_tiff_loses_overlap_information():
    # A flat mutually exclusive TIFF cannot keep both crossing arms.
    a, b = make_synthetic_crossing((64, 64), thickness=5)
    flat = np.zeros((64, 64), dtype=np.int32)
    flat[a] = 1
    flat[b] = 2
    a_from_flat = flat == 1
    lost = int((a & ~a_from_flat).sum())
    assert lost > 0


def test_annotation_package_roundtrip_via_written_package(tmp_path):
    """Build a minimal annotation package, load it, accept + review, export."""
    import json
    import tifffile

    shape = (32, 32)
    fiber = (np.random.default_rng(0).random(shape) * 1000).astype(np.uint16)
    dapi = (np.random.default_rng(1).random(shape) * 1000).astype(np.uint16)
    labels = np.zeros(shape, dtype=np.int32)
    labels[4:12, 4:8] = 1
    labels[18:28, 20:24] = 2

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    tifffile.imwrite(pkg / "fiber_raw16.tif", fiber)
    tifffile.imwrite(pkg / "dapi_raw16.tif", dapi)
    tifffile.imwrite(pkg / "starting_labels.tif", labels)
    (pkg / "README.json").write_text(json.dumps({"image_id": "pkg_img", "pixel_um": 0.6493}))
    (pkg / "instance_properties.csv").write_text(
        "label,id,status,reviewed,source,notes\n"
        "1,myotube_0001,ambiguous,False,semantic_component_proposal,\n"
        "2,myotube_0002,ambiguous,False,semantic_component_proposal,\n")

    loaded = load_annotation_package(pkg)
    assert set(loaded.channels) == {"fiber", "dapi"}
    assert len(loaded.prompt_ids) == 2
    # Prompts are not authoritative and excluded from a default export.
    assert loaded.session.to_instance_set().instances == []

    # Accept one proposal, review it, and export.
    session = loaded.session
    session.accept_prompt("myotube_0001", status="complete", reviewer="cara")
    session.set_reviewed("myotube_0001", True)
    out = tmp_path / "pkg_img.instances.json"
    info = session.save(out)
    assert info["n_exported"] == 1
    assert info["n_authoritative"] == 1
