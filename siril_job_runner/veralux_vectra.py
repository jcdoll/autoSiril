"""
VeraLux Vectra implementation - Smart saturation.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Vectra.py

Applies per-vector (hue-based) saturation and hue control in LCH color space
with shadow protection, star protection, and chroma stability gating.

Ported from standalone VeraLux reference script. Algorithmic functions are
intentionally self-contained to allow independent validation against the
upstream source. Shared math utilities (color space, wavelets) live in
veralux_colorspace.py and veralux_wavelet.py.
"""

from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve

from siril_job_runner.config import Config
from siril_job_runner.protocols import SirilInterface
from siril_job_runner.veralux_colorspace import (
    lab_to_lch,
    lab_to_rgb,
    lch_to_lab,
    rgb_to_lab,
)
from siril_job_runner.veralux_wavelet import atrous_decomposition

# Hue centers in radians for the 6 color vectors
# Order: Red, Yellow, Green, Cyan, Blue, Magenta
HUE_CENTERS_RAD = np.array(
    [0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3]
)
HUE_NAMES = ["red", "yellow", "green", "cyan", "blue", "magenta"]

# Reference: sigma_ang = 30 degrees = pi/6 radians
HUE_SIGMA_RAD = np.pi / 6.0


def _compute_hue_weights(H: np.ndarray) -> list[np.ndarray]:
    """
    Compute Gaussian-weighted influence for each hue vector.

    Reference: weight = exp(-(delta_angle^2) / (2 * sigma^2))
    where sigma = 30 degrees.

    Args:
        H: Hue values in radians, shape (H, W)

    Returns:
        List of 6 weight arrays, one per hue vector
    """
    weights = []
    for center in HUE_CENTERS_RAD:
        diff = np.abs(H - center)
        # Handle circular wraparound
        diff = np.minimum(diff, 2 * np.pi - diff)
        w = np.exp(-0.5 * (diff / HUE_SIGMA_RAD) ** 2)
        weights.append(w)
    return weights


def _compute_signal_mask(L: np.ndarray, shadow_auth: float) -> np.ndarray:
    """
    Compute signal mask using 25th percentile baseline + MAD-based threshold.

    Reference formula:
        sigma_thresh = shadow_auth / 20.0
        bg = 25th percentile of L_norm
        noise_floor = bg + (sigma_thresh * sigma)
        mask = (L_norm - noise_floor) / (2 * sigma)

    Args:
        L: Luminance channel (0-100 range)
        shadow_auth: Shadow authority (0-100), controls protection strength

    Returns:
        Float mask in [0, 1] where higher = more saturation allowed
    """
    L_norm = L / 100.0

    # Reference: sigma_thresh = shadow_auth / 20.0
    sigma_thresh = shadow_auth / 20.0

    # MAD-based sigma estimate
    median = np.median(L_norm)
    mad = np.median(np.abs(L_norm - median))
    sigma = 1.4826 * mad
    if sigma < 1e-6:
        sigma = 1e-6

    # Reference: 25th percentile as background baseline
    bg = np.percentile(L_norm, 25.0)
    noise_floor = bg + (sigma_thresh * sigma)

    # Reference: mask = (L_norm - noise_floor) / (2 * sigma)
    raw_mask = (L_norm - noise_floor) / (2.0 * sigma + 1e-9)
    raw_mask = np.clip(raw_mask, 0, 1)

    # Apply 3x3 convolution softening (reference: nd_convolve with 3x3/9)
    kernel = np.ones((3, 3)) / 9.0
    signal_mask = convolve(raw_mask, kernel, mode="reflect")

    return np.clip(signal_mask, 0, 1)


def _compute_star_mask_energy(L: np.ndarray) -> np.ndarray:
    """
    Compute star protection mask using ATWT wavelet energy detection.

    Reference formula:
        planes = atrous_decomposition(L, n_scales=2)
        energy = |plane[0]| + |plane[1]|
        star_map = clip((energy - 1.5) * 0.5, 0, 1)
        protection = 1 - star_map

    Args:
        L: Luminance channel (0-100 range)

    Returns:
        Float mask in [0, 1] where 1 = no star (full saturation allowed)
    """
    L_norm = L / 100.0

    # Reference: 2-scale ATWT decomposition
    planes, _ = atrous_decomposition(L_norm, n_scales=2)

    # Reference: energy = |plane[0]| + |plane[1]|
    energy = np.abs(planes[0]) + np.abs(planes[1])

    # Reference: star_map = clip((energy - 1.5) * 0.5, 0, 1)
    star_map = np.clip((energy - 1.5) * 0.5, 0, 1)

    # Reference: protection = 1 - star_map
    protection = 1.0 - star_map

    return np.clip(protection, 0, 1)


def _compute_chroma_stability(
    C: np.ndarray, L: np.ndarray, signal_mask: np.ndarray
) -> np.ndarray:
    """
    Compute chroma stability gate using luminance-relative chroma.

    Reference formula:
        C_rel = C / (L + 1.0)
        chroma_stability = clip((C_rel - 0.015) / 0.07, 0, 1)
        assist = 0.25 * clip((signal_mask - 0.10) / 0.30, 0, 1)
        chroma_stability = max(chroma_stability, assist)

    This distinguishes faint chromatic signal (e.g., Ha arms) from neutral noise.

    Args:
        C: Chroma channel
        L: Luminance channel (0-100 range)
        signal_mask: Signal mask for assist gating

    Returns:
        Float mask in [0, 1] where higher = more stable chroma
    """
    # Reference: C_rel = C / (L + 1.0)
    C_rel = C / (L + 1.0)

    # Reference: chroma_stability = clip((C_rel - 0.015) / 0.07, 0, 1)
    chroma_stability = np.clip((C_rel - 0.015) / 0.07, 0, 1)

    # Reference: Weak-chroma assist gated by signal_mask
    # assist = 0.25 * clip((signal_mask - 0.10) / 0.30, 0, 1)
    assist = 0.25 * np.clip((signal_mask - 0.10) / 0.30, 0, 1)

    # Reference: chroma_stability = max(chroma_stability, assist)
    chroma_stability = np.maximum(chroma_stability, assist)

    return chroma_stability


def _compute_global_mask(
    L: np.ndarray,
    C: np.ndarray,
    shadow_auth: float,
    protect_stars: bool,
) -> np.ndarray:
    """
    Compute combined global protection mask.

    Reference formula:
    global_mask = star_mask * chroma_stability * (0.15 + 0.85 * signal_mask)

    Args:
        L: Luminance channel (0-100 range)
        C: Chroma channel
        shadow_auth: Shadow authority (0-100)
        protect_stars: Whether to apply star protection

    Returns:
        Float mask in [0, 1] where higher = more saturation allowed
    """
    # Signal mask (shadow authority)
    signal_mask = _compute_signal_mask(L, shadow_auth)

    # Star mask
    star_mask = _compute_star_mask_energy(L) if protect_stars else np.ones_like(L)

    # Chroma stability (needs signal_mask for assist gating)
    chroma_stability = _compute_chroma_stability(C, L, signal_mask)

    # Reference formula for global mask
    # Ensures: stars stay white, background stays neutral, faint emission responds
    global_mask = star_mask * chroma_stability * (0.15 + 0.85 * signal_mask)

    return np.clip(global_mask, 0, 1)


def apply_saturation(
    data: np.ndarray,
    saturation: float = 25.0,
    shadow_auth: float = 0.0,
    protect_stars: bool = True,
    per_vector_sat: dict[str, float | None] | None = None,
    per_vector_hue: dict[str, float | None] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Apply smart saturation and hue enhancement in LCH color space.

    Args:
        data: RGB image data, shape (3, H, W), values in [0, 1]
        saturation: Global saturation boost (0-100)
        shadow_auth: Shadow protection strength (0-100)
        protect_stars: If True, reduce saturation boost in star regions
        per_vector_sat: Optional dict with per-hue saturation overrides
        per_vector_hue: Optional dict with per-hue shift overrides (degrees)

    Returns:
        Tuple of (saturated_data, stats_dict)
    """
    # Convert to LCH
    lab = rgb_to_lab(data)
    L, C, H = lab_to_lch(lab)

    # Compute combined protection mask
    global_mask = _compute_global_mask(L, C, shadow_auth, protect_stars)

    # Base saturation boost (reference: scales by 0.5 for 100% input)
    base_sat_boost = saturation / 100.0 * 0.5

    # Prepare per-vector parameters
    per_vector_sat = per_vector_sat or {}
    per_vector_hue = per_vector_hue or {}

    sat_boosts = []
    hue_shifts = []
    for name in HUE_NAMES:
        # Saturation
        sat_override = per_vector_sat.get(name)
        if sat_override is not None:
            sat_boosts.append(sat_override / 100.0 * 0.5)
        else:
            sat_boosts.append(base_sat_boost)

        # Hue shift (degrees to radians)
        hue_override = per_vector_hue.get(name)
        if hue_override is not None:
            hue_shifts.append(np.radians(hue_override))
        else:
            hue_shifts.append(0.0)

    # Compute hue weights
    hue_weights = _compute_hue_weights(H)

    # Accumulate weighted saturation and hue changes
    total_sat_boost = np.zeros_like(C)
    total_hue_shift = np.zeros_like(H)
    total_weight = np.zeros_like(C)

    for weight, sat_boost, hue_shift in zip(
        hue_weights, sat_boosts, hue_shifts, strict=True
    ):
        total_sat_boost += weight * sat_boost
        total_hue_shift += weight * hue_shift
        total_weight += weight

    # Normalize by total weight
    total_weight = np.maximum(total_weight, 1e-6)
    effective_sat_boost = total_sat_boost / total_weight
    effective_hue_shift = total_hue_shift / total_weight

    # Apply with protection mask
    # Reference: C_final = C * (1 + delta_sat * mask)
    C_boosted = C * (1.0 + effective_sat_boost * global_mask)

    # Reference: H_final = H + (delta_hue * mask)
    H_shifted = H + effective_hue_shift * global_mask

    # Wrap hue to [0, 2*pi)
    H_shifted = H_shifted % (2 * np.pi)

    # Convert back to RGB
    lab_out = lch_to_lab(L, C_boosted, H_shifted)
    rgb_out = lab_to_rgb(lab_out)

    # Collect statistics
    stats = {
        "base_sat_boost": float(base_sat_boost),
        "effective_sat_boost_mean": float(np.mean(effective_sat_boost)),
        "effective_hue_shift_mean_deg": float(np.degrees(np.mean(effective_hue_shift))),
        "shadow_protection_mean": float(
            1.0 - np.mean(_compute_signal_mask(L, shadow_auth))
        ),
        "global_mask_mean": float(np.mean(global_mask)),
    }

    for i, name in enumerate(HUE_NAMES):
        stats[f"{name}_sat_boost"] = sat_boosts[i]
        stats[f"{name}_hue_shift_deg"] = float(np.degrees(hue_shifts[i]))

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

    # Collect per-vector saturation overrides
    per_vector_sat = {}
    for name in HUE_NAMES:
        attr_name = f"veralux_vectra_{name}"
        if hasattr(config, attr_name):
            value = getattr(config, attr_name)
            if value is not None:
                per_vector_sat[name] = value

    enhanced, stats = apply_saturation(
        data,
        saturation=config.veralux_vectra_saturation,
        shadow_auth=config.veralux_vectra_shadow_auth,
        protect_stars=config.veralux_vectra_protect_stars,
        per_vector_sat=per_vector_sat if per_vector_sat else None,
    )

    out_data = np.clip(enhanced * 65535, 0, 65535).astype(np.uint16)
    hdu = fits.PrimaryHDU(out_data, header=header)
    hdu.writeto(image_path, overwrite=True)

    log(
        f"Vectra applied: sat_boost={stats['base_sat_boost']:.3f}, "
        f"mean_boost={stats['effective_sat_boost_mean']:.3f}, "
        f"mask_mean={stats['global_mask_mean']:.3f}"
    )

    if not siril.load(str(image_path)):
        return False, stats

    return True, stats
