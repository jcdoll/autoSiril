"""
Color space conversions for VeraLux image processing.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/

Provides RGB, XYZ, LAB, LCH color space conversions.
"""

import numpy as np

# CIE Lab reference white point (D65 illuminant)
_REF_X = 95.047
_REF_Y = 100.0
_REF_Z = 108.883


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
