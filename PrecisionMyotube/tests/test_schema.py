import numpy as np

from precision_myotube.schema import (InstanceRecord, decode_rle, decode_rle_cropped,
                                      encode_rle, encode_sparse_positions,
                                      from_label_image)


def test_rle_round_trip_with_fortran_layout():
    mask = np.zeros((7, 9), dtype=bool)
    mask[1:5, 2] = True
    mask[4, 2:8] = True
    assert np.array_equal(mask, decode_rle(encode_rle(mask)))


def test_imported_labels_are_not_authoritative_by_default():
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[2:5, 2:6] = 1
    result = from_label_image(labels, "example")
    assert len(result.instances) == 1
    assert result.instances[0].status == "ambiguous"
    assert not result.instances[0].reviewed


def test_only_reviewed_complete_state_is_authoritative():
    mask = np.ones((3, 4), dtype=bool)
    for status in ("complete", "border_truncated", "occluded", "ambiguous"):
        for reviewed in (False, True):
            record = InstanceRecord("x", status, encode_rle(mask), reviewed=reviewed)
            assert record.is_authoritative() is (reviewed and status == "complete")
    complete = InstanceRecord("x", "complete", encode_rle(mask), reviewed=True)
    assert not complete.is_authoritative("border_truncated")


def test_large_sparse_rle_decodes_only_tight_crop():
    shape = (20_000, 30_000)
    positions = np.array([
        10 + 15 * shape[0], 11 + 15 * shape[0],
        10 + 16 * shape[0], 11 + 16 * shape[0],
    ])
    bbox, crop = decode_rle_cropped(encode_sparse_positions(shape, positions))
    assert bbox == (10, 15, 12, 17)
    assert crop.shape == (2, 2)
    assert crop.all()
