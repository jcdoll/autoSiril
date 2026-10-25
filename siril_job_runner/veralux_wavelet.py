"""
Wavelet transform utilities for VeraLux image processing.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/

Provides a trous wavelet decomposition and reconstruction.
"""

import numpy as np
from scipy.ndimage import convolve1d

# B3-spline kernel for a trous wavelet transform
_B3_KERNEL_1D = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0


def atrous_decomposition(
    img: np.ndarray, n_scales: int = 6
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Perform a trous wavelet decomposition with B3-spline kernel.

    Args:
        img: 2D array (grayscale) or 3D array (color, channel-first)
        n_scales: Number of wavelet scales to compute

    Returns:
        Tuple of (wavelet_planes, residual) where:
        - wavelet_planes: List of n_scales detail planes
        - residual: Low-frequency residual (approximation)
    """
    planes = []
    current = img.astype(np.float64)

    for scale in range(n_scales):
        step = 2**scale
        kernel = np.zeros(4 * step + 1, dtype=np.float64)
        kernel[::step] = _B3_KERNEL_1D
        # The B3 kernel separates into row and column passes, without mixing color.
        smoothed = convolve1d(current, kernel, axis=-2, mode="reflect")
        smoothed = convolve1d(smoothed, kernel, axis=-1, mode="reflect")

        plane = current - smoothed
        planes.append(plane)
        current = smoothed

    return planes, current


def atrous_reconstruction(planes: list[np.ndarray], residual: np.ndarray) -> np.ndarray:
    """
    Reconstruct image from a trous wavelet planes.

    Args:
        planes: List of wavelet detail planes
        residual: Low-frequency residual

    Returns:
        Reconstructed image
    """
    result = residual.copy()
    for plane in planes:
        result = result + plane
    return result
