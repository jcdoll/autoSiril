"""
VeraLux core utilities for image processing.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/

Provides shared statistics and I/O helpers.
"""

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

# Re-export color space conversions for backwards compatibility
from .veralux_colorspace import (
    lab_to_lch,
    lab_to_rgb,
    lch_to_lab,
    rgb_to_lab,
    rgb_to_xyz,
    xyz_to_rgb,
)

# Re-export wavelet functions for backwards compatibility
from .veralux_wavelet import atrous_decomposition, atrous_reconstruction

__all__ = [
    # Color space
    "rgb_to_xyz",
    "xyz_to_rgb",
    "rgb_to_lab",
    "lab_to_rgb",
    "lab_to_lch",
    "lch_to_lab",
    # Wavelets
    "atrous_decomposition",
    "atrous_reconstruction",
    # Statistics
    "mad_sigma",
    "compute_signal_mask",
    # I/O
    "load_fits_data",
    "save_fits_data",
]


def mad_sigma(data: np.ndarray) -> float:
    """
    Compute robust noise estimate using Median Absolute Deviation.

    MAD is scaled to be consistent with standard deviation for Gaussian noise.

    Args:
        data: Input array

    Returns:
        Robust noise estimate (sigma)
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return float(mad * 1.4826)


def compute_signal_mask(L: np.ndarray, threshold_sigma: float = 3.0) -> np.ndarray:
    """
    Compute a mask identifying signal regions based on luminance.

    Uses MAD-based thresholding to separate signal from background.

    Args:
        L: Luminance channel (2D array)
        threshold_sigma: Number of sigma above background

    Returns:
        Boolean mask where True = signal region
    """
    median = np.median(L)
    sigma = mad_sigma(L)
    threshold = median + threshold_sigma * sigma
    return threshold < L


def load_fits_data(filepath: Path) -> np.ndarray:
    """
    Load image data from a FITS file.

    Args:
        filepath: Path to FITS file

    Returns:
        Image data as float64 array normalized to [0, 1].
        For RGB images, returns shape (3, H, W).
    """
    with fits.open(filepath) as hdul:
        data = hdul[0].data.astype(np.float64)

    # Normalize to [0, 1] if needed
    if data.max() > 1.5:
        data = data / 65535.0

    return data


def save_fits_data(
    filepath: Path, data: np.ndarray, header: Any = None, as_uint16: bool = True
) -> bool:
    """
    Save image data to a FITS file.

    Args:
        filepath: Output path
        data: Image data (assumed [0, 1] range for float)
        header: Optional FITS header to preserve
        as_uint16: If True, save as uint16 (0-65535), else as float32

    Returns:
        True if successful
    """
    if as_uint16:
        out_data = np.clip(data * 65535, 0, 65535).astype(np.uint16)
    else:
        out_data = data.astype(np.float32)

    hdu = fits.PrimaryHDU(out_data, header=header)
    hdu.writeto(filepath, overwrite=True)
    return True
