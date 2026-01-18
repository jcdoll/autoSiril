"""
Wavelet transform utilities for VeraLux image processing.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/

Provides a trous wavelet decomposition and reconstruction.
"""

import numpy as np
from scipy.ndimage import convolve

# B3-spline kernel for a trous wavelet transform
_B3_KERNEL_1D = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0


def _build_atrous_kernel(scale: int) -> np.ndarray:
    """
    Build 2D a trous kernel for given scale.

    The a trous algorithm inserts zeros between kernel elements,
    effectively doubling the kernel size at each scale.
    """
    step = 2**scale
    size = 4 * step + 1

    kernel_2d = np.zeros((size, size), dtype=np.float64)
    for i, vi in enumerate(_B3_KERNEL_1D):
        for j, vj in enumerate(_B3_KERNEL_1D):
            kernel_2d[i * step, j * step] = vi * vj

    return kernel_2d


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
        kernel = _build_atrous_kernel(scale)
        if current.ndim == 3:
            smoothed = np.stack(
                [convolve(current[c], kernel, mode="reflect") for c in range(3)], axis=0
            )
        else:
            smoothed = convolve(current, kernel, mode="reflect")

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
