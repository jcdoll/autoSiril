"""
VeraLux Vectra implementation - Smart saturation.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Vectra.py

Applies per-vector (hue-based) saturation control in LCH color space
with shadow protection and optional star protection.
"""

from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits

from siril_job_runner.config import Config
from siril_job_runner.protocols import SirilInterface
from siril_job_runner.veralux_core import (
    lab_to_lch,
    lab_to_rgb,
    lch_to_lab,
    mad_sigma,
    rgb_to_lab,
)

# Hue centers in radians for the 6 color vectors
# Order: Red, Yellow, Green, Cyan, Blue, Magenta
_HUE_CENTERS = np.array(
    [0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3]
)
_HUE_NAMES = ["red", "yellow", "green", "cyan", "blue", "magenta"]


def _compute_hue_weights(H: np.ndarray, sigma: float = 0.5) -> list[np.ndarray]:
    """
    Compute Gaussian-weighted influence for each hue vector.

    Args:
        H: Hue values in radians, shape (H, W)
        sigma: Width of Gaussian influence (radians)

    Returns:
        List of 6 weight arrays, one per hue vector
    """
    weights = []
    for center in _HUE_CENTERS:
        diff = np.abs(H - center)
        diff = np.minimum(diff, 2 * np.pi - diff)
        w = np.exp(-0.5 * (diff / sigma) ** 2)
        weights.append(w)
    return weights


def _compute_shadow_mask(L: np.ndarray, shadow_auth: float) -> np.ndarray:
    """
    Compute shadow protection mask for saturation.

    Args:
        L: Luminance channel (0-100 range)
        shadow_auth: Shadow authority (0-100)

    Returns:
        Float mask in [0, 1] where lower values reduce saturation boost
    """
    from scipy.ndimage import gaussian_filter

    L_norm = L / 100.0
    median = np.median(L_norm)
    sigma = mad_sigma(L_norm)
    threshold = median + 2.0 * sigma

    shadow_raw = np.clip(L_norm / (threshold + 1e-6), 0, 1)
    protection_strength = shadow_auth / 100.0
    mask = 1.0 - protection_strength * (1.0 - shadow_raw)
    return gaussian_filter(mask, sigma=2)


def _compute_star_mask(L: np.ndarray) -> np.ndarray:
    """
    Compute star protection mask for saturation.

    Stars should have reduced saturation boost to avoid color bloat.

    Args:
        L: Luminance channel (0-100 range)

    Returns:
        Float mask in [0, 1] where lower values = more star protection
    """
    from scipy.ndimage import binary_dilation, gaussian_filter

    median = np.median(L)
    sigma = mad_sigma(L)
    threshold = median + 5.0 * sigma

    star_cores = threshold < L
    dilated = binary_dilation(star_cores, iterations=2)
    star_mask = gaussian_filter(dilated.astype(np.float64), sigma=2)

    return 1.0 - np.clip(star_mask, 0, 1) * 0.8


def apply_saturation(
    data: np.ndarray,
    saturation: float = 25.0,
    shadow_auth: float = 0.0,
    protect_stars: bool = True,
    per_vector: dict[str, float | None] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Apply smart saturation enhancement in LCH color space.

    Args:
        data: RGB image data, shape (3, H, W), values in [0, 1]
        saturation: Global saturation boost (0-100)
        shadow_auth: Shadow protection strength (0-100)
        protect_stars: If True, reduce saturation boost in star regions
        per_vector: Optional dict with per-hue overrides (red, yellow, etc.)

    Returns:
        Tuple of (saturated_data, stats_dict)
    """
    lab = rgb_to_lab(data)
    L, C, H = lab_to_lch(lab)

    shadow_mask = _compute_shadow_mask(L, shadow_auth)
    star_mask = _compute_star_mask(L) if protect_stars else np.ones_like(L)
    protection = shadow_mask * star_mask

    base_boost = 1.0 + (saturation / 100.0) * 0.5

    per_vector = per_vector or {}
    vector_boosts = []
    for name in _HUE_NAMES:
        override = per_vector.get(name)
        boost = 1.0 + (override / 100.0) * 0.5 if override is not None else base_boost
        vector_boosts.append(boost)

    hue_weights = _compute_hue_weights(H)

    total_boost = np.zeros_like(C)
    total_weight = np.zeros_like(C)

    for weight, boost in zip(hue_weights, vector_boosts, strict=True):
        total_boost += weight * boost
        total_weight += weight

    total_weight = np.maximum(total_weight, 1e-6)
    effective_boost = total_boost / total_weight

    C_boosted = C * (1.0 + (effective_boost - 1.0) * protection)

    lab_out = lch_to_lab(L, C_boosted, H)
    rgb_out = lab_to_rgb(lab_out)

    stats = {
        "base_boost": base_boost,
        "effective_boost_mean": float(np.mean(effective_boost)),
        "shadow_protection": float(np.mean(1.0 - shadow_mask)),
        "star_protection": float(np.mean(1.0 - star_mask)),
    }

    for i, name in enumerate(_HUE_NAMES):
        stats[f"{name}_boost"] = vector_boosts[i]

    return rgb_out, stats


def apply_vectra(
    siril: SirilInterface,
    image_path: Path,
    config: Config,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[bool, dict[str, float]]:
    """
    Apply VeraLux Vectra smart saturation to an image.

    Loads the image, applies LCH-based saturation enhancement, and saves back.

    Args:
        siril: SirilWrapper instance (used for loading context)
        image_path: Path to the image to enhance
        config: Configuration with vectra parameters
        log_fn: Optional logging function

    Returns:
        Tuple of (success, stats_dict)
    """

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    log(
        f"Vectra: saturation={config.veralux_vectra_saturation}, "
        f"shadow_auth={config.veralux_vectra_shadow_auth}"
    )

    with fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()

    if data.max() > 1.5:
        data = data / 65535.0

    if data.ndim != 3 or data.shape[0] != 3:
        log("Vectra: Image must be RGB (3, H, W)")
        return False, {}

    per_vector = {}
    for name in _HUE_NAMES:
        attr_name = f"veralux_vectra_{name}"
        if hasattr(config, attr_name):
            value = getattr(config, attr_name)
            if value is not None:
                per_vector[name] = value

    enhanced, stats = apply_saturation(
        data,
        saturation=config.veralux_vectra_saturation,
        shadow_auth=config.veralux_vectra_shadow_auth,
        protect_stars=config.veralux_vectra_protect_stars,
        per_vector=per_vector if per_vector else None,
    )

    out_data = np.clip(enhanced * 65535, 0, 65535).astype(np.uint16)
    hdu = fits.PrimaryHDU(out_data, header=header)
    hdu.writeto(image_path, overwrite=True)

    log(
        f"Vectra applied: base_boost={stats['base_boost']:.2f}, "
        f"mean_boost={stats['effective_boost_mean']:.2f}"
    )

    if not siril.load(str(image_path)):
        return False, stats

    return True, stats
