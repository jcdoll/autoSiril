"""Check separable wavelets against the original two-dimensional definition."""

from math import comb

import numpy as np
import pytest
from scipy.ndimage import convolve

from siril_job_runner.veralux_wavelet import atrous_decomposition, atrous_reconstruction


@pytest.mark.parametrize("shape", [(31, 43), (3, 31, 43)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_wavelets_match_2d_filter_at_every_scale(
    shape: tuple[int, ...], dtype: type[np.floating]
) -> None:
    """Preserve edge reflection, color separation, precision and reconstruction."""
    image = np.random.default_rng(42).random(shape).astype(dtype)
    # Exercise reflected boundaries at scales wider than the image.
    image[..., 0, 0] = 10
    planes, residual = atrous_decomposition(image, n_scales=6)
    expected = image.astype(np.float64)
    for scale, plane in enumerate(planes):
        # B3 is the normalized fourth-order binomial kernel in each spatial axis.
        taps = np.zeros(4 * 2**scale + 1)
        taps[:: 2**scale] = [comb(4, i) / 16 for i in range(5)]
        kernel = np.outer(taps, taps)
        if len(shape) == 3:
            kernel = kernel[np.newaxis, :, :]
        smoothed = convolve(expected, kernel, mode="reflect")
        np.testing.assert_allclose(plane, expected - smoothed, atol=1e-14, rtol=0)
        expected = smoothed
    np.testing.assert_allclose(residual, expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(
        atrous_reconstruction(planes, residual), image, atol=1e-14, rtol=0
    )
