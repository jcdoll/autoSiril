"""Tests for veralux_core module."""

import numpy as np

from siril_job_runner.veralux_core import (
    atrous_decomposition,
    atrous_reconstruction,
    compute_signal_mask,
    lab_to_lch,
    lab_to_rgb,
    lch_to_lab,
    mad_sigma,
    rgb_to_lab,
)


class TestColorSpaceConversions:
    """Tests for color space conversion functions."""

    def test_rgb_to_lab_black(self):
        """Black should convert to L=0."""
        rgb = np.zeros((3, 10, 10))
        lab = rgb_to_lab(rgb)
        assert lab.shape == (3, 10, 10)
        np.testing.assert_array_almost_equal(lab[0], 0, decimal=3)

    def test_rgb_to_lab_white(self):
        """White should convert to L=100."""
        rgb = np.ones((3, 10, 10))
        lab = rgb_to_lab(rgb)
        np.testing.assert_array_almost_equal(lab[0], 100, decimal=1)
        np.testing.assert_array_almost_equal(lab[1], 0, decimal=1)
        np.testing.assert_array_almost_equal(lab[2], 0, decimal=1)

    def test_rgb_lab_roundtrip(self):
        """RGB -> LAB -> RGB should be identity."""
        np.random.seed(42)
        rgb = np.random.rand(3, 20, 20) * 0.8 + 0.1
        lab = rgb_to_lab(rgb)
        rgb_back = lab_to_rgb(lab)
        np.testing.assert_array_almost_equal(rgb, rgb_back, decimal=4)

    def test_lab_to_lch_and_back(self):
        """LAB -> LCH -> LAB should be identity."""
        lab = np.array([[[50.0]], [[25.0]], [[-30.0]]])
        L, C, H = lab_to_lch(lab)
        lab_back = lch_to_lab(L, C, H)
        np.testing.assert_array_almost_equal(lab, lab_back, decimal=10)

    def test_lch_chroma_computation(self):
        """Chroma should be sqrt(a^2 + b^2)."""
        lab = np.array([[[50.0]], [[3.0]], [[4.0]]])
        L, C, H = lab_to_lch(lab)
        np.testing.assert_almost_equal(C[0, 0], 5.0, decimal=10)


class TestAtrousWavelet:
    """Tests for a trous wavelet decomposition."""

    def test_decomposition_reconstruction_identity(self):
        """Decomposition followed by reconstruction should be identity."""
        np.random.seed(42)
        img = np.random.rand(64, 64) * 0.5 + 0.25

        planes, residual = atrous_decomposition(img, n_scales=4)
        reconstructed = atrous_reconstruction(planes, residual)

        np.testing.assert_array_almost_equal(img, reconstructed, decimal=10)

    def test_decomposition_returns_correct_count(self):
        """Should return n_scales planes plus residual."""
        img = np.random.rand(32, 32)
        n_scales = 5
        planes, residual = atrous_decomposition(img, n_scales=n_scales)

        assert len(planes) == n_scales
        assert residual.shape == img.shape

    def test_decomposition_3d_input(self):
        """Should handle 3D (color) input."""
        img = np.random.rand(3, 32, 32)
        planes, residual = atrous_decomposition(img, n_scales=3)

        assert len(planes) == 3
        assert all(p.shape == (3, 32, 32) for p in planes)
        assert residual.shape == (3, 32, 32)

    def test_wavelet_planes_sum_to_zero_mean(self):
        """Wavelet detail planes should have approximately zero mean."""
        img = np.random.rand(64, 64)
        planes, _ = atrous_decomposition(img, n_scales=4)

        for i, plane in enumerate(planes[:-1]):
            mean = np.abs(np.mean(plane))
            assert mean < 0.01, f"Plane {i} has mean {mean}"


class TestStatistics:
    """Tests for statistical functions."""

    def test_mad_sigma_gaussian(self):
        """MAD sigma should approximate std for Gaussian noise."""
        np.random.seed(42)
        data = np.random.normal(0, 1.0, size=10000)
        sigma = mad_sigma(data)
        np.testing.assert_almost_equal(sigma, 1.0, decimal=1)

    def test_mad_sigma_with_outliers(self):
        """MAD should be robust to outliers."""
        np.random.seed(42)
        data = np.random.normal(0, 1.0, size=1000)
        data_with_outliers = np.concatenate([data, [100, -100, 50]])

        sigma_clean = mad_sigma(data)
        sigma_outliers = mad_sigma(data_with_outliers)

        np.testing.assert_almost_equal(sigma_clean, sigma_outliers, decimal=1)

    def test_compute_signal_mask(self):
        """Signal mask should identify bright regions."""
        img = np.zeros((100, 100))
        img[40:60, 40:60] = 1.0

        mask = compute_signal_mask(img, threshold_sigma=2.0)

        assert mask[50, 50]
        assert not mask[10, 10]

    def test_compute_signal_mask_threshold(self):
        """Higher threshold should select fewer pixels."""
        np.random.seed(42)
        img = np.random.rand(100, 100)

        mask_low = compute_signal_mask(img, threshold_sigma=1.0)
        mask_high = compute_signal_mask(img, threshold_sigma=3.0)

        assert np.sum(mask_low) > np.sum(mask_high)
