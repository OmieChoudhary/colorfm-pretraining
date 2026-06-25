"""Small color-science utilities for demo experiments.

These functions are intentionally dependency-light. The approximate CMFs below are not a replacement
for a production-grade color-science library; they are enough for a reproducible portfolio demo.
"""
from __future__ import annotations

import numpy as np


def wavelengths_380_780_10() -> np.ndarray:
    return np.arange(380, 781, 10, dtype=np.float32)


def approximate_cmf(wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate CIE-like color matching functions with smooth Gaussians."""
    wl = wavelengths.astype(np.float32)
    xbar = 1.065 * np.exp(-0.5 * ((wl - 595.8) / 33.3) ** 2) + 0.366 * np.exp(-0.5 * ((wl - 446.8) / 19.4) ** 2)
    ybar = 1.014 * np.exp(-0.5 * ((wl - 556.3) / 28.0) ** 2)
    zbar = 1.839 * np.exp(-0.5 * ((wl - 449.8) / 19.0) ** 2)
    return xbar.astype(np.float32), ybar.astype(np.float32), zbar.astype(np.float32)


def d65_like_illuminant(wavelengths: np.ndarray) -> np.ndarray:
    """Simple daylight-like illuminant curve normalized to mean 1."""
    wl = wavelengths.astype(np.float32)
    illum = 0.85 + 0.25 * np.exp(-0.5 * ((wl - 460) / 60) ** 2) + 0.10 * np.exp(-0.5 * ((wl - 610) / 120) ** 2)
    return (illum / illum.mean()).astype(np.float32)


def reflectance_to_xyz(reflectance: np.ndarray, wavelengths: np.ndarray | None = None) -> np.ndarray:
    """Convert reflectance spectra to approximate XYZ tristimulus values.

    reflectance shape can be (n_wavelengths,) or (batch, n_wavelengths).
    """
    if wavelengths is None:
        wavelengths = wavelengths_380_780_10()
    refl = np.asarray(reflectance, dtype=np.float32)
    one_d = refl.ndim == 1
    if one_d:
        refl = refl[None, :]
    xbar, ybar, zbar = approximate_cmf(wavelengths)
    illum = d65_like_illuminant(wavelengths)
    k = 100.0 / np.sum(illum * ybar)
    X = k * np.sum(refl * illum * xbar, axis=1)
    Y = k * np.sum(refl * illum * ybar, axis=1)
    Z = k * np.sum(refl * illum * zbar, axis=1)
    xyz = np.stack([X, Y, Z], axis=1).astype(np.float32)
    return xyz[0] if one_d else xyz


def xyz_to_lab(xyz: np.ndarray, white_xyz: np.ndarray | None = None) -> np.ndarray:
    """Convert XYZ to CIE Lab using a D65-like whitepoint."""
    xyz = np.asarray(xyz, dtype=np.float32)
    one_d = xyz.ndim == 1
    if one_d:
        xyz = xyz[None, :]
    if white_xyz is None:
        white_xyz = np.array([95.047, 100.0, 108.883], dtype=np.float32)
    ratio = xyz / white_xyz[None, :]
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(ratio > eps, np.cbrt(np.maximum(ratio, 0)), (kappa * ratio + 16) / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    lab = np.stack([L, a, b], axis=1).astype(np.float32)
    return lab[0] if one_d else lab


def reflectance_to_lab(reflectance: np.ndarray, wavelengths: np.ndarray | None = None) -> np.ndarray:
    return xyz_to_lab(reflectance_to_xyz(reflectance, wavelengths))


def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Simple DeltaE CIE76 distance. Good demo metric; add CIEDE2000 for serious work."""
    lab1 = np.asarray(lab1, dtype=np.float32)
    lab2 = np.asarray(lab2, dtype=np.float32)
    return np.linalg.norm(lab1 - lab2, axis=-1)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 color difference.

    Implementation follows the standard Sharma et al. equations. It is included so the demo can
    report a perceptual color-difference metric beyond Euclidean Lab distance.
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    kL = kC = kH = 1.0
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0
    G = 0.5 * (1.0 - np.sqrt((C_bar**7) / (C_bar**7 + 25.0**7 + 1e-12)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = (np.degrees(np.arctan2(b1, a1p)) + 360.0) % 360.0
    h2p = (np.degrees(np.arctan2(b2, a2p)) + 360.0) % 360.0
    h1p = np.where(C1p == 0, 0.0, h1p)
    h2p = np.where(C2p == 0, 0.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180.0, dh - 360.0, dh)
    dh = np.where(dh < -180.0, dh + 360.0, dh)
    dh = np.where((C1p * C2p) == 0, 0.0, dh)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2.0))

    Lp_bar = (L1 + L2) / 2.0
    Cp_bar = (C1p + C2p) / 2.0
    hp_sum = h1p + h2p
    hp_diff = np.abs(h1p - h2p)
    hp_bar = np.where(
        (C1p * C2p) == 0,
        hp_sum,
        np.where(hp_diff <= 180.0, hp_sum / 2.0, np.where(hp_sum < 360.0, (hp_sum + 360.0) / 2.0, (hp_sum - 360.0) / 2.0)),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )
    delta_theta = 30.0 * np.exp(-((hp_bar - 275.0) / 25.0) ** 2)
    RC = 2.0 * np.sqrt((Cp_bar**7) / (Cp_bar**7 + 25.0**7 + 1e-12))
    SL = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cp_bar
    SH = 1.0 + 0.015 * Cp_bar * T
    RT = -np.sin(np.radians(2.0 * delta_theta)) * RC

    de = np.sqrt(
        (dLp / (kL * SL)) ** 2
        + (dCp / (kC * SC)) ** 2
        + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )
    return de.astype(np.float32)
