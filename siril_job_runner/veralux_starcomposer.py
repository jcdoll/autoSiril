"""
VeraLux StarComposer implementation - Star compositing.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_StarComposer.py

Blends a starless image with a star mask using rational stretch for star
intensity control, hybrid scalar/vector engine for color grip, and
screen/linear blend modes.

Ported from standalone VeraLux reference script. Algorithmic functions are
intentionally self-contained to allow independent validation against the
upstream source. Shared math utilities (color space, wavelets) live in
veralux_colorspace.py and veralux_wavelet.py.
"""

from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from siril_job_runner.config import DEFAULTS, BlendMode, Config
from siril_job_runner.protocols import SirilInterface
from siril_job_runner.veralux_stretch import get_sensor_weights

from .veralux_core import normalized_image

# Default VeraLux brightness weights (Rec.709), independent of SPCC calibration.
DEFAULT_WEIGHTS = (0.2126, 0.7152, 0.0722)


def _apply_micro_blur(data: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """
    Apply micro-blur for transition smoothing.

    Reference: cv2.GaussianBlur with sigma=0.5

    Args:
        data: Input data, shape (3, H, W)
        sigma: Gaussian blur sigma

    Returns:
        Blurred data
    """
    result = np.zeros_like(data)
    for c in range(data.shape[0]):
        result[c] = gaussian_filter(data[c], sigma=sigma, mode="mirror")
    return result


def _calculate_anchor_adaptive(
    data: np.ndarray, weights: tuple[float, float, float] = DEFAULT_WEIGHTS
) -> float:
    """
    Calculate adaptive anchor (black point) using percentile method.

    Reference: Samples at stride, computes 5th percentile if sparsity >= 5%.

    Args:
        data: RGB data, shape (3, H, W)
        weights: RGB luminance weights tuple

    Returns:
        Anchor value (black point)
    """
    r_w, g_w, b_w = weights

    # Compute luminance
    L = r_w * data[0] + g_w * data[1] + b_w * data[2]

    # Sample for efficiency on large images
    stride = max(1, data.size // 1000000)
    sampled = L.flat[::stride]

    # Filter valid pixels (L > 0)
    valid = sampled[sampled > 0]

    if len(valid) < 0.05 * len(sampled):
        # Less than 5% valid - no anchoring
        return 0.0

    # 5th percentile as anchor
    anchor = np.percentile(valid, 5.0)
    return max(0.0, anchor)


def rational_stretch(data: np.ndarray, D: float, b: float) -> np.ndarray:
    """Apply the bounded StarComposer 2.1.1 rational curve to linear stars.

    D is 10**log_d; log_d 1 is identity at neutral hardness 50.
    Higher intensity lifts faint stars without a second gamma conversion.
    """
    x = np.clip(data, 0.0, 1.0)
    strength = np.clip((np.log10(max(D, 0.1)) - 1.0) / 2.0, 0.0, 12.0)
    k = 3.0**strength
    u = np.clip((max(b, 0.1) - 50.0) / 50.0, -1.5, 1.5)
    toe = max(1.0 + 0.60 * u**3, 1e-3)
    shaped = x / np.maximum(x + toe * (1.0 - x), 1e-9)
    return np.clip(k * shaped / ((k - 1.0) * shaped + 1.0), 0.0, 1.0)


def _stretch_scalar(img_anchored: np.ndarray, D: float, b: float) -> np.ndarray:
    """
    Scalar branch: stretch each channel independently.

    Produces crisp white star cores.

    Args:
        img_anchored: Anchored RGB data, shape (3, H, W)
        D: Stretch intensity
        b: Profile hardness

    Returns:
        Stretched RGB data
    """
    result = np.zeros_like(img_anchored)
    for c in range(3):
        result[c] = rational_stretch(img_anchored[c], D, b)
    return np.clip(result, 0, 1)


def _stretch_vector(
    img_anchored: np.ndarray,
    D: float,
    b: float,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> np.ndarray:
    """
    Vector branch: preserve color ratios via luminance stretching.

    Produces color-preserved stars.

    Args:
        img_anchored: Anchored RGB data, shape (3, H, W)
        D: Stretch intensity
        b: Profile hardness
        weights: RGB luminance weights tuple

    Returns:
        Stretched RGB data with preserved color ratios
    """
    r_w, g_w, b_w = weights

    # Extract luminance from anchored data
    L = r_w * img_anchored[0] + g_w * img_anchored[1] + b_w * img_anchored[2]

    # Stretch luminance
    L_stretched = rational_stretch(L, D, b)

    # Compute color ratios
    epsilon = 1e-9
    r_ratio = img_anchored[0] / (L + epsilon)
    g_ratio = img_anchored[1] / (L + epsilon)
    b_ratio = img_anchored[2] / (L + epsilon)

    # Reconstruct RGB
    result = np.zeros_like(img_anchored)
    result[0] = L_stretched * r_ratio
    result[1] = L_stretched * g_ratio
    result[2] = L_stretched * b_ratio

    return np.clip(result, 0, 1)


def _apply_color_grip(
    scalar: np.ndarray,
    vector: np.ndarray,
    grip: float,
    shadow_conv: float = 0.0,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> np.ndarray:
    """
    Blend scalar and vector branches with shadow convergence damping.

    Reference formula:
        grip_map = grip * (L_ref ^ shadow_conv)
        final = vector * grip_map + scalar * (1 - grip_map)

    Args:
        scalar: Scalar branch result (crisp white)
        vector: Vector branch result (color preserved)
        grip: Color grip 0-1 (0=scalar, 1=vector)
        shadow_conv: Shadow convergence exponent 0-3
        weights: RGB luminance weights tuple

    Returns:
        Blended result
    """
    r_w, g_w, b_w = weights

    if grip <= 0.001:
        return scalar

    # Start with uniform grip
    grip_map = np.full_like(scalar[0], grip)

    # Apply shadow convergence damping
    if shadow_conv > 0.01:
        # Use scalar luminance as reference
        L_ref = r_w * scalar[0] + g_w * scalar[1] + b_w * scalar[2]
        # Damping reduces vector influence in dark areas
        damping = np.power(np.clip(L_ref, 0, 1), shadow_conv)
        grip_map = grip_map * damping

    # Blend: vector where grip_map is high, scalar where low
    final = vector * grip_map + scalar * (1.0 - grip_map)

    return np.clip(final, 0, 1)


def blend_screen(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """
    Screen blend mode: 1 - (1-base) * (1-layer).

    Lightens the base image, good for bright stars on dark backgrounds.
    Prevents clipping and preserves galaxy cores.

    Args:
        base: Base (starless) image
        layer: Overlay (processed stars) image

    Returns:
        Blended result
    """
    return 1.0 - (1.0 - base) * (1.0 - layer)


def blend_linear_add(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """
    Linear add blend mode: base + layer.

    Physical light addition, can cause clipping on bright areas.

    Args:
        base: Base (starless) image
        layer: Overlay (processed stars) image

    Returns:
        Blended result (clipped to [0, 1])
    """
    return np.clip(base + layer, 0, 1)


def compose_stars(
    starless: np.ndarray,
    starmask: np.ndarray,
    log_d: float = DEFAULTS.veralux_starcomposer_log_d,
    hardness: float = DEFAULTS.veralux_starcomposer_hardness,
    color_grip: float = 0.5,
    shadow_conv: float = 0.0,
    blend_mode: BlendMode = BlendMode.SCREEN,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """
    Compose stars onto a starless image using hybrid scalar/vector engine.

    Pipeline:
    1. Bound linear input pixels to [0, 1]
    2. Micro-blur for transition smoothing
    3. Adaptive anchoring (black point)
    4. Scalar branch: stretch R,G,B independently (crisp)
    5. Vector branch: stretch luminance, apply ratios (color)
    6. Color grip blend with shadow convergence damping
    7. Final blend with starless image

    Args:
        starless: Starless image, shape (3, H, W), values in [0, 1]
        starmask: Star mask (linear), shape (3, H, W), values in [0, 1]
        log_d: Star intensity as log10(D), range 1-21
        hardness: Star profile hardness, range 1-100
        color_grip: Color vs grayscale, 0=scalar (crisp), 1=vector (color)
        shadow_conv: Shadow convergence damping, 0-3
        blend_mode: Screen or linear add
        weights: VeraLux RGB brightness weights, independent of SPCC calibration

    Returns:
        Tuple of (composed_image, stats_dict)
    """
    # Input is already linear; gamma conversion would suppress faint stars.
    starmask_linear = np.clip(starmask, 0.0, 1.0)
    starless = np.clip(starless, 0.0, 1.0)

    # Step 2: Micro-blur for transition smoothing
    starmask_blur = _apply_micro_blur(starmask_linear, sigma=0.5)

    # Step 3: Adaptive anchoring
    anchor = _calculate_anchor_adaptive(starmask_blur, weights)
    starmask_anchored = np.maximum(starmask_blur - anchor, 0.0)

    # Compute D and b
    D = 10.0**log_d
    b = max(hardness, 0.1)

    # Step 4 & 5: Hybrid engine - scalar and vector branches
    scalar = _stretch_scalar(starmask_anchored, D, b)
    vector = _stretch_vector(starmask_anchored, D, b, weights)

    # Step 6: Color grip blend with shadow convergence
    stars_processed = _apply_color_grip(
        scalar, vector, color_grip, shadow_conv, weights
    )

    # Step 7: Blend with starless image
    if blend_mode == BlendMode.SCREEN:
        result = blend_screen(starless, stars_processed)
    else:
        result = blend_linear_add(starless, stars_processed)

    result = np.clip(result, 0, 1)

    stats = {
        "log_d": log_d,
        "D": D,
        "hardness": hardness,
        "color_grip": color_grip,
        "shadow_conv": shadow_conv,
        "anchor": anchor,
        "blend_mode": blend_mode.value,
        "star_brightness_mean": float(np.mean(stars_processed)),
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

    Loads both images, applies hybrid star processing and blending, saves result.

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
        f"color_grip={config.veralux_starcomposer_color_grip}, "
        f"shadow_conv={config.veralux_starcomposer_shadow_conv}"
    )

    with fits.open(starless_path) as hdul:
        starless = normalized_image(hdul[0].data)
        header = hdul[0].header.copy()

    with fits.open(starmask_path) as hdul:
        starmask = normalized_image(hdul[0].data)

    if starless.ndim != 3 or starless.shape[0] != 3:
        log("StarComposer: Starless image must be RGB (3, H, W)")
        return False, starless_path

    if starmask.ndim == 2:
        starmask = np.repeat(starmask[np.newaxis, ...], 3, axis=0)

    if starmask.ndim != 3 or starmask.shape[0] != 3:
        log("StarComposer: Star mask must be RGB (3, H, W)")
        return False, starless_path

    if starless.shape != starmask.shape:
        log(
            f"StarComposer: Shape mismatch - starless {starless.shape} "
            f"vs starmask {starmask.shape}"
        )
        return False, starless_path

    blend_mode = config.veralux_starcomposer_blend_mode

    # Use the same VeraLux brightness weights as the stretch, not SPCC response curves.
    weights = get_sensor_weights(config.veralux_sensor_profile)

    composed, stats = compose_stars(
        starless=starless,
        starmask=starmask,
        log_d=config.veralux_starcomposer_log_d,
        hardness=config.veralux_starcomposer_hardness,
        color_grip=config.veralux_starcomposer_color_grip,
        shadow_conv=config.veralux_starcomposer_shadow_conv,
        blend_mode=blend_mode,
        weights=weights,
    )

    output_path = starless_path.parent / f"{starless_path.stem}_stars.fit"
    out_data = np.clip(composed * 65535, 0, 65535).astype(np.uint16)
    hdu = fits.PrimaryHDU(out_data, header=header)
    hdu.writeto(output_path, overwrite=True)

    log(
        f"StarComposer applied: anchor={stats['anchor']:.4f}, "
        f"star_brightness={stats['star_brightness_mean']:.4f}"
    )

    if not siril.load(str(output_path)):
        return False, output_path

    return True, output_path
