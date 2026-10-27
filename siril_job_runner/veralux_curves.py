# Copyright (c) 2025 Riccardo Paterniti
# SPDX-License-Identifier: GPL-3.0-or-later
"""VeraLux Curves 1.0.1 RGB/K curve and brightness-range mask.

Ported from Riccardo Paterniti's GPL-3.0-or-later script:
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Curves.py
Individual channel, Lab and HSV curves and the GUI are not ported.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import Akima1DInterpolator
from scipy.special import expit

from .config import Config
from .protocols import SirilInterface
from .veralux_core import normalized_image


def curve_image(image: np.ndarray, config: Config) -> np.ndarray:
    """Apply upstream's LUT and optional soft mask to normalized CHW RGB pixels.

    The mask uses mean RGB, not the HyperMetric brightness weights or SPCC.
    Akima interpolation can overshoot between points; upstream clips the LUT
    to [0, 1]. This is a tone curve, not a detector of sky or nebula pixels.
    """
    data = normalized_image(image).astype(np.float32)
    if data.ndim != 3 or data.shape[0] != 3:
        raise ValueError("VeraLux Curves requires an RGB image (3, H, W)")
    points = config.veralux_curves_points
    if all(x == y for x, y in points) and points[0][0] == 0 and points[-1][0] == 1:
        return data

    x, y = zip(*points, strict=True)
    domain = np.linspace(0, 1, 65536, dtype=np.float32)
    lut = np.interp(domain, x, y).astype(np.float32)
    if len(points) > 2:
        inside = (domain >= x[0]) & (domain <= x[-1])
        lut[inside] = Akima1DInterpolator(x, y)(domain[inside])
    lut = np.clip(lut, 0, 1)
    transformed = np.interp(data.ravel(), domain, lut).reshape(data.shape)

    low, high = config.veralux_curves_range
    if low == 0 and high == 1:
        return transformed.astype(np.float32)
    brightness = np.mean(data, axis=0)
    feather = config.veralux_curves_feather
    if feather < 1e-6:
        mask = ((brightness >= low) & (brightness <= high)).astype(np.float32)
    else:
        mask = np.ones_like(brightness)
        if low > 0:
            mask = np.minimum(mask, expit(2.5 * (brightness - low) / feather))
        if high < 1:
            mask = np.minimum(mask, expit(2.5 * (high - brightness) / feather))
    return (data * (1 - mask) + transformed * mask).astype(np.float32)


def apply_curves(
    siril: SirilInterface,
    image_path: Path,
    config: Config,
    log_fn: Callable[[str], None],
) -> None:
    """Finish a stretched RGB working file without quantizing its pixels."""
    points = config.veralux_curves_points
    if all(x == y for x, y in points) and points[0][0] == 0 and points[-1][0] == 1:
        return
    log_fn(
        f"Applying VeraLux Curves: points={points}, range={config.veralux_curves_range}"
    )
    with fits.open(image_path, memmap=False) as hdul:
        result = curve_image(hdul[0].data, config)
        header = hdul[0].header.copy()
    fits.PrimaryHDU(result, header=header).writeto(image_path, overwrite=True)
    if not siril.load(str(image_path)):
        raise RuntimeError("VeraLux Curves reload failed during composition")
