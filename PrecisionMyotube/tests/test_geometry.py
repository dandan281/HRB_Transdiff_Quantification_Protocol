import numpy as np

from precision_myotube.geometry import measure_mask


def test_rectangle_geometry_is_physical_and_unbranched():
    mask = np.zeros((80, 120), dtype=bool)
    mask[30:40, 10:110] = True
    result = measure_mask(mask, pixel_um=1.0, endpoint_exclusion_um=5.0)
    assert 90 <= result.length_um <= 105
    assert 9 <= result.width_median_um <= 12
    assert result.branch_count == 0
    assert result.endpoint_count == 2
    assert result.components == 1
    assert not result.touches_border


def test_border_contact_is_detected():
    mask = np.zeros((40, 40), dtype=bool)
    mask[0:8, 2:30] = True
    assert measure_mask(mask, 1.0).touches_border


def test_geometry_scales_in_physical_units():
    mask = np.zeros((40, 80), dtype=bool)
    mask[15:25, 10:70] = True
    one = measure_mask(mask, 1.0, endpoint_exclusion_um=2.0)
    half = measure_mask(mask, 0.5, endpoint_exclusion_um=1.0)
    assert np.isclose(half.length_um, one.length_um / 2)
    assert np.isclose(half.width_median_um, one.width_median_um / 2)
    assert np.isclose(half.area_um2, one.area_um2 / 4)
