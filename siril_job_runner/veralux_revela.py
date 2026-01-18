"""
VeraLux Revela implementation - Detail enhancement.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Revela.py

Uses a trous wavelet transform (ATWT) to enhance fine details (texture)
and medium-scale structures while protecting shadows and stars.
"""

from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits

from siril_job_runner.config import Config
from siril_job_runner.protocols import SirilInterface
from siril_job_runner.veralux_core import (
    atrous_decomposition,
    atrous_reconstruction,
    lab_to_rgb,
    mad_sigma,
    rgb_to_lab,
)


def _compute_star_mask(L: np.ndarray, threshold_sigma: float = 5.0) -> np.ndarray:
    """
    Compute a mask identifying stars based on high luminance peaks.

    Args:
        L: Luminance channel (2D array)
        threshold_sigma: Number of sigma above median for star detection

    Returns:
        Float mask where 1 = star region (for protection)
    """
    from scipy.ndimage import binary_dilation, gaussian_filter

    median = np.median(L)
    sigma = mad_sigma(L)
    threshold = median + threshold_sigma * sigma

    star_cores = threshold < L
    dilated = binary_dilation(star_cores, iterations=3)
    mask = gaussian_filter(dilated.astype(np.float64), sigma=2)
    return np.clip(mask, 0, 1)


def _compute_shadow_mask(
    L: np.ndarray, shadow_auth: float, threshold_sigma: float = 2.0
) -> np.ndarray:
    """
    Compute shadow protection mask.

    Areas below threshold get reduced enhancement based on shadow_auth.

    Args:
        L: Luminance channel (2D array, 0-100 range)
        shadow_auth: Shadow authority (0-100), higher = more shadow protection
        threshold_sigma: Sigma threshold for shadow detection

    Returns:
        Float mask in [0, 1] where lower values reduce enhancement
    """
    from scipy.ndimage import gaussian_filter

    L_norm = L / 100.0
    median = np.median(L_norm)
    sigma = mad_sigma(L_norm)
    threshold = median + threshold_sigma * sigma

    shadow_raw = 1.0 - np.clip((threshold - L_norm) / (threshold + 1e-6), 0, 1)
    protection_strength = shadow_auth / 100.0
    mask = 1.0 - protection_strength * (1.0 - shadow_raw)
    return gaussian_filter(mask, sigma=3)


def enhance_details(
    data: np.ndarray,
    texture: float = 50.0,
    structure: float = 50.0,
    shadow_auth: float = 25.0,
    protect_stars: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Enhance image details using ATWT wavelet decomposition.

    Args:
        data: RGB image data, shape (3, H, W), values in [0, 1]
        texture: Fine detail enhancement (0-100), affects scales 0-1
        structure: Medium structure enhancement (0-100), affects scales 2-4
        shadow_auth: Shadow protection strength (0-100)
        protect_stars: If True, reduce enhancement in star regions

    Returns:
        Tuple of (enhanced_data, stats_dict)
    """
    lab = rgb_to_lab(data)
    L = lab[0]

    planes, residual = atrous_decomposition(L, n_scales=6)

    shadow_mask = _compute_shadow_mask(L, shadow_auth)
    star_mask = _compute_star_mask(L) if protect_stars else np.zeros_like(L)

    protection = shadow_mask * (1.0 - 0.8 * star_mask)

    texture_boost = 1.0 + (texture / 100.0) * 0.5
    structure_boost = 1.0 + (structure / 100.0) * 0.5

    texture_scales = [0, 1]
    structure_scales = [2, 3, 4]

    for i in texture_scales:
        planes[i] = planes[i] * (1.0 + (texture_boost - 1.0) * protection)

    for i in structure_scales:
        planes[i] = planes[i] * (1.0 + (structure_boost - 1.0) * protection)

    L_enhanced = atrous_reconstruction(planes, residual)
    L_enhanced = np.clip(L_enhanced, 0, 100)

    lab_enhanced = np.stack([L_enhanced, lab[1], lab[2]], axis=0)
    rgb_enhanced = lab_to_rgb(lab_enhanced)

    stats = {
        "texture_boost": texture_boost,
        "structure_boost": structure_boost,
        "shadow_coverage": float(np.mean(shadow_mask < 0.5)),
        "star_coverage": float(np.mean(star_mask > 0.5)),
    }

    return rgb_enhanced, stats


def apply_revela(
    siril: SirilInterface,
    image_path: Path,
    config: Config,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[bool, dict[str, float]]:
    """
    Apply VeraLux Revela detail enhancement to an image.

    Loads the image, applies ATWT-based enhancement, and saves back.

    Args:
        siril: SirilWrapper instance (used for loading context)
        image_path: Path to the image to enhance
        config: Configuration with revela parameters
        log_fn: Optional logging function

    Returns:
        Tuple of (success, stats_dict)
    """

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    log(
        f"Revela: texture={config.veralux_revela_texture}, "
        f"structure={config.veralux_revela_structure}, "
        f"shadow_auth={config.veralux_revela_shadow_auth}"
    )

    with fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()

    if data.max() > 1.5:
        data = data / 65535.0

    if data.ndim != 3 or data.shape[0] != 3:
        log("Revela: Image must be RGB (3, H, W)")
        return False, {}

    enhanced, stats = enhance_details(
        data,
        texture=config.veralux_revela_texture,
        structure=config.veralux_revela_structure,
        shadow_auth=config.veralux_revela_shadow_auth,
        protect_stars=config.veralux_revela_protect_stars,
    )

    out_data = np.clip(enhanced * 65535, 0, 65535).astype(np.uint16)
    hdu = fits.PrimaryHDU(out_data, header=header)
    hdu.writeto(image_path, overwrite=True)

    log(
        f"Revela applied: texture_boost={stats['texture_boost']:.2f}, "
        f"structure_boost={stats['structure_boost']:.2f}"
    )

    if not siril.load(str(image_path)):
        return False, stats

    return True, stats
