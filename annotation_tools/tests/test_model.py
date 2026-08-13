"""CL01 -- assisted annotation interface guardrails and schema compatibility."""
import numpy as np
import pytest

from annotation_tools.model import AnnotationSession, AnnotationError
from annotation_tools.masks import SparseMask
from annotation_tools._schema_bridge import InstanceSet


def _bar(shape, r0, r1, c0, c1):
    m = np.zeros(shape, dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def test_create_and_export_roundtrips_through_canonical_schema(tmp_path):
    shape = (40, 40)
    s = AnnotationSession(shape, "img_a", pixel_um=0.6493)
    iid = s.create(_bar(shape, 5, 30, 10, 14), status="complete", reviewer="anna")
    s.set_reviewed(iid, True, reviewer="anna")

    out = tmp_path / "a.instances.json"
    info = s.save(out)
    # Codex's own loader/validator must accept the output unchanged.
    reloaded = InstanceSet.load(out)
    assert reloaded.image_id == "img_a"
    assert [r.id for r in reloaded.instances] == [iid]
    assert reloaded.instances[0].status == "complete"
    assert info["n_authoritative"] == 1
    # Review log carries reviewer provenance the compact record does not.
    log_text = out.with_suffix(".review_log.jsonl").read_text()
    assert "anna" in log_text


def test_export_refuses_reviewed_without_reviewer():
    shape = (20, 20)
    s = AnnotationSession(shape, "img_b")
    iid = s.create(_bar(shape, 2, 10, 2, 6))
    # Forcing reviewed True without a reviewer is rejected at the source.
    with pytest.raises(AnnotationError):
        s.set_reviewed(iid, True)


def test_export_refuses_missing_status():
    shape = (20, 20)
    s = AnnotationSession(shape, "img_c")
    iid = s.create(_bar(shape, 2, 10, 2, 6))
    # Corrupt the status to simulate a missing/invalid state and confirm export blocks.
    s.instances[iid].status = "not_a_status"
    with pytest.raises(AnnotationError):
        s.to_instance_set()


def test_no_bulk_mark_all_complete_exists():
    # The class must not expose any bulk authority-granting operation.
    forbidden = [name for name in dir(AnnotationSession)
                 if ("all" in name.lower() and ("complete" in name.lower()
                     or "review" in name.lower()))]
    assert forbidden == []


def test_prompts_are_not_exported_and_not_authoritative(tmp_path):
    shape = (30, 30)
    s = AnnotationSession(shape, "img_d")
    p = s.load_prompt(_bar(shape, 3, 20, 5, 9))
    # A raw prompt cannot receive a status directly.
    with pytest.raises(AnnotationError):
        s.set_status(p, "complete")
    # Default export excludes prompts entirely.
    exported = s.to_instance_set()
    assert exported.instances == []
    assert s.authoritative_ids() == []


def test_accept_prompt_then_review_makes_authoritative():
    shape = (30, 30)
    s = AnnotationSession(shape, "img_e")
    p = s.load_prompt(_bar(shape, 3, 25, 5, 9))
    s.accept_prompt(p, status="complete", reviewer="bea")
    s.set_reviewed(p, True)
    assert s.authoritative_ids() == [p]


def test_merge_split_erase_change_geometry():
    shape = (40, 40)
    s = AnnotationSession(shape, "img_f")
    a = s.create(_bar(shape, 2, 10, 2, 6))
    b = s.create(_bar(shape, 20, 30, 2, 6))
    merged = s.merge(a, b)
    assert set(s.instances) == {merged}
    assert s.instances[merged].mask.area == 8 * 4 + 10 * 4

    # split back into two disjoint parts
    part1 = _bar(shape, 2, 10, 2, 6)
    part2 = _bar(shape, 20, 30, 2, 6)
    id1, id2 = s.split(merged, part1, part2)
    assert set(s.instances) == {id1, id2}

    # erase trims pixels
    before = s.instances[id1].mask.area
    s.erase(id1, _bar(shape, 2, 3, 2, 6))
    assert s.instances[id1].mask.area == before - 4


def test_overlap_safe_two_crossing_instances_share_pixels():
    shape = (40, 40)
    s = AnnotationSession(shape, "img_g")
    horiz = _bar(shape, 18, 22, 0, 40)
    vert = _bar(shape, 0, 40, 18, 22)
    ida = s.create(horiz, status="complete", reviewer="x")
    idb = s.create(vert, status="complete", reviewer="x")
    shared = np.logical_and(s.instances[ida].mask.full(),
                            s.instances[idb].mask.full()).sum()
    assert shared == 16   # 4x4 crossing preserved in both masks


def test_sparse_mask_memory_is_object_proportional():
    shape = (3636, 3636)
    m = np.zeros((10, 10), dtype=bool)
    m[2:8, 2:8] = True
    sm = SparseMask((100, 100), m, shape)
    # Stored crop is tiny even though the field is huge.
    assert sm.crop.size == 36
    assert sm.image_shape == shape
    assert sm.to_rle()["size"] == [3636, 3636]
