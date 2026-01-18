"""
VeraLux StarComposer implementation - Star compositing.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_StarComposer.py

Blends a starless image with a star mask using hyperbolic stretch for star
intensity control and screen/linear blend modes.
"""

from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits

from siril_job_runner.config import Config
from siril_job_runner.protocols import SirilInterface


class BlendMode(Enum):
    """Star blending modes."""

    SCREEN = "screen"
    LINEAR_ADD = "linear_add"


def hyperbolic_stretch(
    data: np.ndarray, log_d: float = 1.0, hardness: float = 6.0
) -> np.ndarray:
    """
    Apply hyperbolic stretch to star mask for intensity control.

    Uses asinh-based stretch with controllable intensity and hardness.

    Args:
        data: Star mask data, values in [0, 1]
        log_d: Log stretch intensity (0-2), higher = brighter stars
        hardness: Profile hardness (1-100), higher = sharper star profiles

    Returns:
        Stretched star mask
    """
    D = 10.0**log_d
    b = max(hardness, 0.1)

    term1 = np.arcsinh(D * data + b)
    term2 = np.arcsinh(b)
    norm_factor = np.arcsinh(D + b) - term2

    if abs(norm_factor) < 1e-6:
        norm_factor = 1e-6

    return (term1 - term2) / norm_factor


def blend_screen(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """
    Screen blend mode: 1 - (1-base) * (1-layer).

    Lightens the base image, good for bright stars on dark backgrounds.

    Args:
        base: Base (starless) image
        layer: Overlay (star mask) image

    Returns:
        Blended result
    """
    return 1.0 - (1.0 - base) * (1.0 - layer)


def blend_linear_add(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """
    Linear add blend mode: base + layer.

    Simple additive blend, can cause clipping on bright areas.

    Args:
        base: Base (starless) image
        layer: Overlay (star mask) image

    Returns:
        Blended result (clipped to [0, 1])
    """
    return np.clip(base + layer, 0, 1)


def apply_color_grip(stars_rgb: np.ndarray, color_grip: float = 0.5) -> np.ndarray:
    """
    Apply color grip to control star colorfulness.

    Interpolates between scalar (grayscale) and vector (full color) star modes.

    Args:
        stars_rgb: RGB star mask, shape (3, H, W)
        color_grip: 0 = grayscale stars, 1 = full color stars

    Returns:
        Color-adjusted star mask
    """
    luminance = 0.2126 * stars_rgb[0] + 0.7152 * stars_rgb[1] + 0.0722 * stars_rgb[2]
    scalar = np.stack([luminance, luminance, luminance], axis=0)
    return (1.0 - color_grip) * scalar + color_grip * stars_rgb


def compose_stars(
    starless: np.ndarray,
    starmask: np.ndarray,
    log_d: float = 1.0,
    hardness: float = 6.0,
    color_grip: float = 0.5,
    blend_mode: BlendMode = BlendMode.SCREEN,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Compose stars onto a starless image.

    Args:
        starless: Starless image, shape (3, H, W), values in [0, 1]
        starmask: Star mask (linear), shape (3, H, W), values in [0, 1]
        log_d: Star intensity (0-2)
        hardness: Star profile hardness (1-100)
        color_grip: Color vs grayscale (0-1)
        blend_mode: Screen or linear add

    Returns:
        Tuple of (composed_image, stats_dict)
    """
    stars_stretched = np.zeros_like(starmask)
    for c in range(3):
        stars_stretched[c] = hyperbolic_stretch(starmask[c], log_d, hardness)

    stars_colored = apply_color_grip(stars_stretched, color_grip)

    if blend_mode == BlendMode.SCREEN:
        result = blend_screen(starless, stars_colored)
    else:
        result = blend_linear_add(starless, stars_colored)

    result = np.clip(result, 0, 1)

    stats = {
        "log_d": log_d,
        "hardness": hardness,
        "color_grip": color_grip,
        "blend_mode": blend_mode.value,
        "star_brightness_mean": float(np.mean(stars_stretched)),
        "result_brightness_mean": float(np.mean(result)),
    }

    return result, stats


def apply_starcomposer(
    siril: SirilInterface,
    starless_path: Path,
    starmask_path: Path,
    config: Config,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[bool, Path]:
    """
    Apply VeraLux StarComposer to combine starless image with star mask.

    Loads both images, applies star stretching and blending, saves result.

    Args:
        siril: SirilWrapper instance
        starless_path: Path to starless image
        starmask_path: Path to star mask (linear)
        config: Configuration with starcomposer parameters
        log_fn: Optional logging function

    Returns:
        Tuple of (success, output_path)
    """

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    log(
        f"StarComposer: log_d={config.veralux_starcomposer_log_d}, "
        f"hardness={config.veralux_starcomposer_hardness}, "
        f"color_grip={config.veralux_starcomposer_color_grip}"
    )

    with fits.open(starless_path) as hdul:
        starless = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()

    with fits.open(starmask_path) as hdul:
        starmask = hdul[0].data.astype(np.float64)

    if starless.max() > 1.5:
        starless = starless / 65535.0
    if starmask.max() > 1.5:
        starmask = starmask / 65535.0

    if starless.ndim != 3 or starless.shape[0] != 3:
        log("StarComposer: Starless image must be RGB (3, H, W)")
        return False, starless_path

    if starmask.ndim != 3 or starmask.shape[0] != 3:
        log("StarComposer: Star mask must be RGB (3, H, W)")
        return False, starless_path

    if starless.shape != starmask.shape:
        log(
            f"StarComposer: Shape mismatch - starless {starless.shape} "
            f"vs starmask {starmask.shape}"
        )
        return False, starless_path

    blend_mode_str = config.veralux_starcomposer_blend_mode
    try:
        blend_mode = BlendMode(blend_mode_str)
    except ValueError:
        log(f"StarComposer: Unknown blend mode '{blend_mode_str}', using screen")
        blend_mode = BlendMode.SCREEN

    composed, stats = compose_stars(
        starless=starless,
        starmask=starmask,
        log_d=config.veralux_starcomposer_log_d,
        hardness=config.veralux_starcomposer_hardness,
        color_grip=config.veralux_starcomposer_color_grip,
        blend_mode=blend_mode,
    )

    output_path = starless_path.parent / f"{starless_path.stem}_stars.fit"
    out_data = np.clip(composed * 65535, 0, 65535).astype(np.uint16)
    hdu = fits.PrimaryHDU(out_data, header=header)
    hdu.writeto(output_path, overwrite=True)

    log(
        f"StarComposer applied: blend={blend_mode.value}, "
        f"star_brightness={stats['star_brightness_mean']:.4f}"
    )

    if not siril.load(str(output_path)):
        return False, output_path

    return True, output_path
