"""Loss-masking policy: ambiguous/border/excluded must be ignored, not background."""
import numpy as np
import pytest

from _shared.training_masks import build_ignore_mask
from _shared.schema_bridge import InstanceRecord, InstanceSet, encode_sparse_positions

SHAPE = (64, 64)


def _rec(id_, rows, cols, status, reviewed):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[rows, cols] = True
    r, c = np.nonzero(mask)
    positions = r.astype(np.int64) + c.astype(np.int64) * SHAPE[0]
    return InstanceRecord(id=id_, status=status, reviewed=reviewed, source="qc_review",
                          rle=encode_sparse_positions(SHAPE, positions))


def _write(tmp_path, records):
    path = tmp_path / "well.qc.instances.json"
    InstanceSet(SHAPE, "well", records).save(path)
    return path


def test_ambiguous_and_border_are_ignored_not_background(tmp_path):
    path = _write(tmp_path, [
        _rec("m1", slice(2, 6), slice(2, 40), "complete", True),
        _rec("m2", slice(20, 24), slice(2, 40), "ambiguous", False),
        _rec("m3", slice(40, 44), slice(2, 40), "border_truncated", True),
    ])
    ignore, stats = build_ignore_mask(path, SHAPE)
    assert ignore[22, 10] and ignore[42, 10], "ambiguous/border must be ignored"
    assert not ignore[4, 10], "a complete record is a target, never ignored"
    assert stats["counts_by_status"] == {"ambiguous": 1, "border_truncated": 1}


def test_complete_targets_are_protected_from_ignore(tmp_path):
    """An ambiguous proposal overlapping a reviewed target must not erode it."""
    path = _write(tmp_path, [
        _rec("m1", slice(10, 14), slice(2, 40), "complete", True),
        _rec("m2", slice(10, 14), slice(2, 40), "ambiguous", False),   # exact overlap
    ])
    labels = np.zeros(SHAPE, dtype=np.int32)
    labels[10:14, 2:40] = 1
    ignore, stats = build_ignore_mask(path, SHAPE, labels=labels)
    assert not ignore[12, 10], "reviewed target pixel was ignored"
    assert stats["target_px_protected_from_ignore"] == int((labels > 0).sum())


def test_binding_exclusions_are_ignored_not_background(tmp_path):
    """The two `training_exclude.json` ids still read complete/reviewed.

    Without explicit handling they fall through to background, asserting "empty
    field" over a fibre the operator re-reviewed as ambiguous.
    """
    path = _write(tmp_path, [
        _rec("keep", slice(2, 6), slice(2, 40), "complete", True),
        _rec("myotube_0377", slice(30, 34), slice(2, 40), "complete", True),
    ])
    plain, _ = build_ignore_mask(path, SHAPE)
    assert not plain[32, 10], "precondition: excluded id is not ignored by default"

    ignore, stats = build_ignore_mask(path, SHAPE, excluded_ids=("myotube_0377",))
    assert ignore[32, 10], "binding exclusion must be ignored"
    assert not ignore[4, 10], "the kept target is unaffected"
    assert stats["counts_by_status"]["binding_exclusion"] == 1


def test_unknown_excluded_id_is_an_error(tmp_path):
    path = _write(tmp_path, [_rec("m1", slice(2, 6), slice(2, 40), "complete", True)])
    with pytest.raises(ValueError, match="not found"):
        build_ignore_mask(path, SHAPE, excluded_ids=("nope",))


def test_instance_overlap_pixels_are_ignored(tmp_path):
    path = _write(tmp_path, [_rec("m1", slice(2, 6), slice(2, 40), "complete", True)])
    overlap = np.zeros(SHAPE, dtype=bool)
    overlap[50, 50] = True
    ignore, stats = build_ignore_mask(path, SHAPE, overlap_ignore=overlap)
    assert ignore[50, 50]
    assert stats["counts_by_status"]["instance_overlap_px"] == 1


def test_rejected_proposals_stay_background(tmp_path):
    """Rejected == operator asserted 'not a myotube' == informative negative.

    `rejected` is not a canonical status (VALID_STATUSES is complete /
    border_truncated / occluded / ambiguous); a rejected proposal is simply
    absent from the exported InstanceSet. So this asserts the *default*: an
    un-recorded region stays background and is never swept into ignore.
    """
    path = _write(tmp_path, [_rec("m1", slice(2, 6), slice(2, 40), "complete", True)])
    ignore, _ = build_ignore_mask(path, SHAPE)
    assert not ignore[22, 10], "an unrecorded (rejected) region must stay background"
    assert ignore.sum() == 0


def test_shape_mismatch_is_rejected(tmp_path):
    path = _write(tmp_path, [_rec("m1", slice(2, 6), slice(2, 40), "complete", True)])
    with pytest.raises(ValueError, match="shape"):
        build_ignore_mask(path, (32, 32))
