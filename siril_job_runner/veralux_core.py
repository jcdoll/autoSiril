"""
VeraLux core utilities for image processing.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/

Provides shared color space conversions, wavelet utilities, and I/O helpers.
"""

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve

# CIE Lab reference white point (D65 illuminant)
_REF_X = 95.047
_REF_Y = 100.0
_REF_Z = 108.883

# B3-spline kernel for a trous wavelet transform
_B3_KERNEL_1D = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to CIE XYZ color space.

    Args:
        rgb: Array of shape (3, H, W) with values in [0, 1]

    Returns:
        XYZ array of shape (3, H, W)
    """
    # sRGB to linear RGB
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    # Linear RGB to XYZ (sRGB D65)
    r, g, b = linear[0], linear[1], linear[2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    return np.stack([x * 100, y * 100, z * 100], axis=0)


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    """
    Convert CIE XYZ to RGB color space.

    Args:
        xyz: Array of shape (3, H, W)

    Returns:
        RGB array of shape (3, H, W) with values clipped to [0, 1]
    """
    x, y, z = xyz[0] / 100, xyz[1] / 100, xyz[2] / 100

    # XYZ to linear RGB (sRGB D65)
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    # Linear RGB to sRGB
    linear = np.stack([r, g, b], axis=0)
    srgb = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055
    )

    return np.clip(srgb, 0, 1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to CIE LAB color space.

    Args:
        rgb: Array of shape (3, H, W) with values in [0, 1]

    Returns:
        LAB array of shape (3, H, W) where L is [0, 100], a/b are ~[-128, 127]
    """
    xyz = rgb_to_xyz(rgb)
    x, y, z = xyz[0] / _REF_X, xyz[1] / _REF_Y, xyz[2] / _REF_Z

    epsilon = 0.008856
    kappa = 903.3

    fx = np.where(x > epsilon, np.cbrt(x), (kappa * x + 16) / 116)
    fy = np.where(y > epsilon, np.cbrt(y), (kappa * y + 16) / 116)
    fz = np.where(z > epsilon, np.cbrt(z), (kappa * z + 16) / 116)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=0)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIE LAB to RGB color space.

    Args:
        lab: Array of shape (3, H, W) where L is [0, 100], a/b are ~[-128, 127]

    Returns:
        RGB array of shape (3, H, W) with values clipped to [0, 1]
    """
    L, a, b = lab[0], lab[1], lab[2]

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    epsilon = 0.008856
    kappa = 903.3

    x3 = fx**3
    z3 = fz**3

    x = np.where(x3 > epsilon, x3, (116 * fx - 16) / kappa)
    y = np.where(kappa * epsilon < L, ((L + 16) / 116) ** 3, L / kappa)
    z = np.where(z3 > epsilon, z3, (116 * fz - 16) / kappa)

    xyz = np.stack([x * _REF_X, y * _REF_Y, z * _REF_Z], axis=0)
    return xyz_to_rgb(xyz)


def lab_to_lch(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert LAB to LCH (cylindrical representation).

    Args:
        lab: Array of shape (3, H, W)

    Returns:
        Tuple of (L, C, H) arrays where H is in radians
    """
    L, a, b = lab[0], lab[1], lab[2]
    C = np.sqrt(a**2 + b**2)
    H = np.arctan2(b, a)
    return L, C, H


def lch_to_lab(L: np.ndarray, C: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Convert LCH back to LAB.

    Args:
        L: Lightness array
        C: Chroma array
        H: Hue array in radians

    Returns:
        LAB array of shape (3, H, W)
    """
    a = C * np.cos(H)
    b = C * np.sin(H)
    return np.stack([L, a, b], axis=0)


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
