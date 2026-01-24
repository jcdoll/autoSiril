"""Tests for veralux_revela module."""

import numpy as np

from siril_job_runner.veralux_revela import (
    _compute_shadow_mask,
    _compute_star_mask,
    enhance_details,
)


class TestStarMask:
    """Tests for star mask computation."""

    def test_star_mask_identifies_bright_peaks(self):
        """Bright isolated peaks should be detected as stars."""
        L = np.zeros((100, 100))
        L[50, 50] = 100.0

        mask = _compute_star_mask(L, threshold_sigma=3.0)
        assert mask[50, 50] > 0.5

    def test_star_mask_ignores_background(self):
        """Uniform low background should not trigger star detection."""
        L = np.ones((100, 100)) * 10.0
        mask = _compute_star_mask(L, threshold_sigma=3.0)
        assert np.mean(mask) < 0.1


class TestShadowMask:
    """Tests for shadow mask computation."""

    def test_shadow_mask_protects_dark_areas(self):
        """Dark areas should have lower mask values (more protection)."""
        L = np.zeros((100, 100))
        L[:50, :] = 10.0
        L[50:, :] = 80.0

        mask = _compute_shadow_mask(L, shadow_auth=50.0)

        dark_avg = np.mean(mask[:50, :])
        bright_avg = np.mean(mask[50:, :])
        assert dark_avg < bright_avg

    def test_shadow_mask_zero_auth_no_protection(self):
        """Zero shadow authority should give uniform mask."""
        L = np.zeros((100, 100))
        L[:50, :] = 10.0
        L[50:, :] = 80.0

        mask = _compute_shadow_mask(L, shadow_auth=0.0)
        assert np.std(mask) < 0.1


class TestEnhanceDetails:
    """Tests for the main enhancement function."""

    def test_enhance_returns_correct_shape(self):
        """Output should have same shape as input."""
        data = np.random.rand(3, 64, 64) * 0.5 + 0.25
        enhanced, stats = enhance_details(data)
        assert enhanced.shape == data.shape

    def test_enhance_returns_valid_range(self):
        """Output should be clipped to [0, 1]."""
        data = np.random.rand(3, 64, 64) * 0.5 + 0.25
        enhanced, stats = enhance_details(data, texture=100, structure=100)
        assert enhanced.min() >= 0.0
        assert enhanced.max() <= 1.0

    def test_enhance_zero_params_near_identity(self):
        """Zero enhancement should be close to identity."""
        np.random.seed(42)
        data = np.random.rand(3, 32, 32) * 0.5 + 0.25
        enhanced, stats = enhance_details(data, texture=0, structure=0)
        np.testing.assert_array_almost_equal(data, enhanced, decimal=2)

    def test_enhance_returns_stats(self):
        """Should return enhancement statistics."""
        data = np.random.rand(3, 32, 32)
        _, stats = enhance_details(data)

        assert "texture_boost" in stats
        assert "structure_boost" in stats
        assert "shadow_coverage" in stats
        assert "star_coverage" in stats

    def test_enhance_texture_affects_fine_scales(self):
        """Higher texture should increase fine detail contrast."""
        np.random.seed(42)
        data = np.random.rand(3, 64, 64) * 0.5 + 0.25

        _, stats_low = enhance_details(data, texture=10, structure=0)
        _, stats_high = enhance_details(data, texture=90, structure=0)

        assert stats_high["texture_boost"] > stats_low["texture_boost"]

    def test_enhance_with_star_protection(self):
        """Star protection should affect star coverage stat."""
        data = np.zeros((3, 64, 64)) + 0.1
        data[:, 32, 32] = 1.0

        _, stats_on = enhance_details(data, protect_stars=True)
        _, stats_off = enhance_details(data, protect_stars=False)

        assert stats_on["star_coverage"] > stats_off["star_coverage"]
