"""
VeraLux Silentium implementation - Noise suppression.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Silentium.py

Uses Stationary Wavelet Transform (SWT) with soft thresholding for noise reduction.
Includes detail guard to preserve fine structure and separate chroma processing.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pywt
from astropy.io import fits

from siril_job_runner.config import Config
from siril_job_runner.protocols import SirilInterface
from siril_job_runner.veralux_core import (
    lab_to_rgb,
    mad_sigma,
    rgb_to_lab,
)


def _estimate_noise_level(data: np.ndarray) -> float:
    """
    Estimate noise level from the finest wavelet scale.

    Uses the HH (diagonal detail) subband of first SWT level.

    Args:
        data: 2D image array

    Returns:
        Estimated noise sigma
    """
    coeffs = pywt.swt2(data, "db4", level=1, trim_approx=True)
    HH = coeffs[0][2]
    return mad_sigma(HH)


def _soft_threshold(data: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft thresholding to wavelet coefficients.

    Values below threshold are set to zero, values above are shrunk.

    Args:
        data: Wavelet coefficients
        threshold: Threshold value

    Returns:
        Thresholded coefficients
    """
    sign = np.sign(data)
    magnitude = np.abs(data) - threshold
    return sign * np.maximum(magnitude, 0)


def _compute_detail_guard(L: np.ndarray, guard_strength: float) -> np.ndarray:
    """
    Compute detail protection mask based on local variance.

    High-variance regions (edges, details) get less noise reduction.

    Args:
        L: Luminance channel (2D array)
        guard_strength: Detail guard strength (0-100)

    Returns:
        Protection mask in [0, 1] where higher = more protection
    """
    from scipy.ndimage import gaussian_filter, uniform_filter

    local_mean = uniform_filter(L, size=7)
    local_sq_mean = uniform_filter(L**2, size=7)
    local_var = np.maximum(local_sq_mean - local_mean**2, 0)
    local_std = np.sqrt(local_var)

    normalized_var = local_std / (np.max(local_std) + 1e-6)
    protection = normalized_var ** (1.0 / (1.0 + guard_strength / 50.0))
    return gaussian_filter(protection, sigma=2)


def _compute_shadow_smoothing(L: np.ndarray, shadow_smooth: float) -> np.ndarray:
    """
    Compute additional smoothing weight for shadow regions.

    Dark areas often have more visible noise, so they get more reduction.

    Args:
        L: Luminance channel (0-100 range)
        shadow_smooth: Shadow smoothing strength (0-100)

    Returns:
        Multiplier for threshold in [1, 1 + shadow_smooth/100]
    """
    L_norm = L / 100.0
    median = np.median(L_norm)
    shadow_weight = np.clip(1.0 - L_norm / (median + 0.1), 0, 1)
    return 1.0 + (shadow_smooth / 100.0) * shadow_weight


def denoise_channel(
    channel: np.ndarray,
    intensity: float = 25.0,
    detail_guard: float = 50.0,
    shadow_smooth: float = 10.0,
    n_levels: int = 4,
) -> np.ndarray:
    """
    Denoise a single channel using SWT with soft thresholding.

    Args:
        channel: 2D image channel
        intensity: Noise reduction intensity (0-100)
        detail_guard: Detail protection strength (0-100)
        shadow_smooth: Extra smoothing for shadows (0-100)
        n_levels: Number of SWT decomposition levels

    Returns:
        Denoised channel
    """
    original_shape = channel.shape
    pad_h = (2**n_levels - channel.shape[0] % (2**n_levels)) % (2**n_levels)
    pad_w = (2**n_levels - channel.shape[1] % (2**n_levels)) % (2**n_levels)

    if pad_h > 0 or pad_w > 0:
        channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="reflect")

    noise_sigma = _estimate_noise_level(channel)
    base_threshold = noise_sigma * (intensity / 100.0) * 3.0

    detail_mask = _compute_detail_guard(channel, detail_guard)
    shadow_mult = _compute_shadow_smoothing(channel, shadow_smooth)

    coeffs = pywt.swt2(channel, "db4", level=n_levels, trim_approx=True)

    denoised_coeffs = [coeffs[0]]

    for level_idx in range(1, len(coeffs)):
        LH, HL, HH = coeffs[level_idx]
        scale_factor = 2 ** (n_levels - level_idx + 1)
        level_threshold = base_threshold * np.sqrt(scale_factor)

        effective_threshold = level_threshold * shadow_mult * (1.0 - 0.5 * detail_mask)

        LH_d = _soft_threshold(LH, effective_threshold)
        HL_d = _soft_threshold(HL, effective_threshold)
        HH_d = _soft_threshold(HH, effective_threshold)

        denoised_coeffs.append((LH_d, HL_d, HH_d))

    result = pywt.iswt2(denoised_coeffs, "db4")

    if pad_h > 0 or pad_w > 0:
        result = result[: original_shape[0], : original_shape[1]]

    return result


def denoise_image(
    data: np.ndarray,
    intensity: float = 25.0,
    detail_guard: float = 50.0,
    chroma: float = 30.0,
    shadow_smooth: float = 10.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Denoise RGB image with separate luminance and chroma processing.

    Args:
        data: RGB image data, shape (3, H, W), values in [0, 1]
        intensity: Luminance noise reduction (0-100)
        detail_guard: Detail protection strength (0-100)
        chroma: Chroma noise reduction (0-100)
        shadow_smooth: Extra shadow smoothing (0-100)

    Returns:
        Tuple of (denoised_data, stats_dict)
    """
    lab = rgb_to_lab(data)
    L, a, b = lab[0], lab[1], lab[2]

    L_noise = _estimate_noise_level(L / 100.0) * 100
    a_noise = _estimate_noise_level(a)
    b_noise = _estimate_noise_level(b)

    L_denoised = denoise_channel(
        L,
        intensity=intensity,
        detail_guard=detail_guard,
        shadow_smooth=shadow_smooth,
    )

    chroma_intensity = chroma
    a_denoised = denoise_channel(
        a,
        intensity=chroma_intensity,
        detail_guard=detail_guard * 0.5,
        shadow_smooth=0,
    )
    b_denoised = denoise_channel(
        b,
        intensity=chroma_intensity,
        detail_guard=detail_guard * 0.5,
        shadow_smooth=0,
    )

    L_denoised = np.clip(L_denoised, 0, 100)

    lab_denoised = np.stack([L_denoised, a_denoised, b_denoised], axis=0)
    rgb_denoised = lab_to_rgb(lab_denoised)

    stats = {
        "L_noise_estimate": L_noise,
        "a_noise_estimate": a_noise,
        "b_noise_estimate": b_noise,
        "intensity": intensity,
        "chroma_intensity": chroma_intensity,
    }

    return rgb_denoised, stats


def apply_silentium(
    siril: SirilInterface,
    image_path: Path,
    config: Config,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[bool, dict[str, float]]:
    """
    Apply VeraLux Silentium noise reduction to an image.

    Loads the image, applies SWT-based denoising, and saves back.

    Args:
        siril: SirilWrapper instance (used for loading context)
        image_path: Path to the image to denoise
        config: Configuration with silentium parameters
        log_fn: Optional logging function

    Returns:
        Tuple of (success, stats_dict)
    """

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    log(
        f"Silentium: intensity={config.veralux_silentium_intensity}, "
        f"detail_guard={config.veralux_silentium_detail_guard}, "
        f"chroma={config.veralux_silentium_chroma}"
    )

    with fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()

    if data.max() > 1.5:
        data = data / 65535.0

    if data.ndim != 3 or data.shape[0] != 3:
        log("Silentium: Image must be RGB (3, H, W)")
        return False, {}

    denoised, stats = denoise_image(
        data,
        intensity=config.veralux_silentium_intensity,
        detail_guard=config.veralux_silentium_detail_guard,
        chroma=config.veralux_silentium_chroma,
        shadow_smooth=config.veralux_silentium_shadow_smooth,
    )

    out_data = np.clip(denoised * 65535, 0, 65535).astype(np.uint16)
    hdu = fits.PrimaryHDU(out_data, header=header)
    hdu.writeto(image_path, overwrite=True)

    log(
        f"Silentium applied: L_noise={stats['L_noise_estimate']:.4f}, "
        f"chroma_noise={stats['a_noise_estimate']:.4f}"
    )

    if not siril.load(str(image_path)):
        return False, stats

    return True, stats
