import numpy as np

from colorfm.utils.color_conversions import wavelengths_380_780_10, reflectance_to_xyz, reflectance_to_lab, delta_e_76, delta_e_2000


def test_wavelength_grid():
    wl = wavelengths_380_780_10()
    assert wl[0] == 380
    assert wl[-1] == 780
    assert len(wl) == 41


def test_reflectance_to_lab_shape_and_range():
    wl = wavelengths_380_780_10()
    refl = np.ones_like(wl, dtype=np.float32) * 0.5
    lab = reflectance_to_lab(refl, wl)
    assert lab.shape == (3,)
    assert 0 <= lab[0] <= 100


def test_delta_e_zero_for_identical_colors():
    lab = np.array([[50.0, 2.0, -3.0]], dtype=np.float32)
    assert float(delta_e_76(lab, lab)[0]) == 0.0


def test_delta_e_2000_zero_for_identical_colors():
    lab = np.array([[50.0, 2.0, -3.0]], dtype=np.float32)
    assert float(delta_e_2000(lab, lab)[0]) == 0.0
