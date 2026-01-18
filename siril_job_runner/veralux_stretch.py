"""
VeraLux HyperMetric Stretch implementation.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_HyperMetric_Stretch.py

This module implements the core stretch algorithm for use with Siril's PixelMath.
"""

import math
from pathlib import Path
from typing import Callable

from siril_job_runner.config import Config
from siril_job_runner.siril_wrapper import SirilWrapper


def hyperbolic_stretch_value(
    value: float, D: float, b: float, SP: float = 0.0
) -> float:
    """
    Apply hyperbolic stretch to a single value.

    Formula: (asinh(D*(value-SP)+b) - asinh(b)) / (asinh(D*(1-SP)+b) - asinh(b))

    Args:
        value: Input value (0-1 range)
        D: Stretch intensity (1 to 10^7)
        b: Highlight protection / curve knee (typically 6.0)
        SP: Shadow point anchor (typically 0.0)

    Returns:
        Stretched value (0-1 range)
    """
    D = max(D, 0.1)
    b = max(b, 0.1)

    term1 = math.asinh(D * (value - SP) + b)
    term2 = math.asinh(b)
    norm_factor = math.asinh(D * (1.0 - SP) + b) - term2

    if norm_factor == 0:
        norm_factor = 1e-6

    return (term1 - term2) / norm_factor


def solve_log_d(
    median_in: float,
    target_median: float,
    b: float,
    log_d_min: float = 0.0,
    log_d_max: float = 7.0,
    max_iterations: int = 40,
    tolerance: float = 0.0001,
) -> float:
    """
    Binary search to find log_d that achieves target median.

    Args:
        median_in: Input image median (0-1)
        target_median: Desired output median (typically 0.20)
        b: Highlight protection parameter
        log_d_min: Minimum log_d bound (D = 10^log_d)
        log_d_max: Maximum log_d bound
        max_iterations: Maximum binary search iterations
        tolerance: Convergence tolerance

    Returns:
        Optimal log_d value
    """
    if median_in < 1e-9:
        return 2.0  # Default for near-zero input

    low_log = log_d_min
    high_log = log_d_max
    best_log_d = 2.0

    for _ in range(max_iterations):
        mid_log = (low_log + high_log) / 2.0
        mid_D = 10.0**mid_log

        test_val = hyperbolic_stretch_value(median_in, mid_D, b)

        if abs(test_val - target_median) < tolerance:
            best_log_d = mid_log
            break

        if test_val < target_median:
            low_log = mid_log
        else:
            high_log = mid_log

        best_log_d = mid_log

    return best_log_d


def build_pixelmath_formula(D: float, b: float) -> str:
    """
    Build Siril PixelMath formula for hyperbolic stretch.

    With SP=0, the formula simplifies to:
    (asinh(D*$T+b) - asinh(b)) / (asinh(D+b) - asinh(b))

    Args:
        D: Stretch intensity
        b: Highlight protection parameter

    Returns:
        PixelMath formula string
    """
    # Pre-calculate normalization factor for efficiency
    norm = math.asinh(D + b) - math.asinh(b)
    asinh_b = math.asinh(b)

    return f"(asinh({D}*$T+{b})-{asinh_b})/{norm}"


def apply_stretch(
    siril: SirilWrapper,
    image_path: Path,
    config: Config,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[bool, float]:
    """
    Apply VeraLux HyperMetric stretch to an image.

    Args:
        siril: SirilWrapper instance
        image_path: Path to the image to stretch
        config: Configuration with veralux parameters
        log_fn: Optional logging function

    Returns:
        Tuple of (success, log_d_used)
    """

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    # Get image statistics
    stats = siril.get_image_stats(image_path)
    median_in = stats["median"]

    log(f"Input median: {median_in:.4f}, target: {config.veralux_target_median}")

    # Solve for optimal log_d
    log_d = solve_log_d(
        median_in=median_in,
        target_median=config.veralux_target_median,
        b=config.veralux_b,
        log_d_min=config.veralux_log_d_min,
        log_d_max=config.veralux_log_d_max,
    )

    D = 10.0**log_d
    log(f"Calculated log_d={log_d:.2f} (D={D:.1f})")

    # Load image
    if not siril.load(str(image_path)):
        return False, log_d

    # Apply stretch via PixelMath
    formula = build_pixelmath_formula(D, config.veralux_b)
    log(f"Applying PixelMath: {formula[:60]}...")

    if not siril.pm(formula):
        return False, log_d

    return True, log_d
