"""Tests for veralux_vectra module."""

import numpy as np
import pytest

from siril_job_runner.veralux_vectra import (
    _compute_hue_weights,
    _compute_shadow_mask,
    _compute_star_mask,
    apply_saturation,
)


class TestHueWeights:
    """Tests for hue weight computation."""

    def test_hue_weights_returns_six_arrays(self):
        """Should return 6 weight arrays for 6 color vectors."""
        H = np.random.rand(32, 32) * 2 * np.pi
        weights = _compute_hue_weights(H)
        assert len(weights) == 6

    def test_hue_weights_peak_at_centers(self):
        """Weight should be highest at the corresponding hue center."""
        H = np.array([[0.0]])
        weights = _compute_hue_weights(H, sigma=0.5)
        assert weights[0][0, 0] > weights[1][0, 0]
        assert weights[0][0, 0] > weights[5][0, 0]

    def test_hue_weights_sum_positive(self):
        """All weights should be positive."""
        H = np.random.rand(32, 32) * 2 * np.pi
        weights = _compute_hue_weights(H)
        for w in weights:
            assert np.all(w >= 0)


class TestShadowMask:
    """Tests for shadow mask in vectra."""

    def test_shadow_mask_shape(self):
        """Shadow mask should match input shape."""
        L = np.random.rand(64, 64) * 100
        mask = _compute_shadow_mask(L, shadow_auth=50.0)
        assert mask.shape == L.shape

    def test_shadow_mask_range(self):
        """Shadow mask should be in [0, 1]."""
        L = np.random.rand(64, 64) * 100
        mask = _compute_shadow_mask(L, shadow_auth=50.0)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_zero_shadow_auth_no_protection(self):
        """Zero shadow authority should give near-uniform mask."""
        L = np.random.rand(64, 64) * 100
        mask = _compute_shadow_mask(L, shadow_auth=0.0)
        np.testing.assert_almost_equal(np.mean(mask), 1.0, decimal=1)


class TestStarMask:
    """Tests for star mask in vectra."""

    def test_star_mask_reduces_at_bright_peaks(self):
        """Star mask should have lower values at bright peaks."""
        L = np.ones((64, 64)) * 20.0
        L[32, 32] = 100.0

        mask = _compute_star_mask(L)
        assert mask[32, 32] < mask[10, 10]


class TestApplySaturation:
    """Tests for the main saturation function."""

    def test_saturation_returns_correct_shape(self):
        """Output should match input shape."""
        data = np.random.rand(3, 64, 64) * 0.5 + 0.25
        result, stats = apply_saturation(data)
        assert result.shape == data.shape

    def test_saturation_returns_valid_range(self):
        """Output should be in [0, 1]."""
        data = np.random.rand(3, 64, 64) * 0.5 + 0.25
        result, stats = apply_saturation(data, saturation=100)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_zero_saturation_near_identity(self):
        """Zero saturation should be close to identity."""
        np.random.seed(42)
        data = np.random.rand(3, 32, 32) * 0.5 + 0.25
        result, stats = apply_saturation(data, saturation=0)
        np.testing.assert_array_almost_equal(data, result, decimal=2)

    def test_saturation_returns_stats(self):
        """Should return relevant statistics."""
        data = np.random.rand(3, 32, 32)
        _, stats = apply_saturation(data)

        assert "base_boost" in stats
        assert "effective_boost_mean" in stats
        assert "shadow_protection" in stats

    def test_per_vector_override(self):
        """Per-vector override should affect specific hue stats."""
        data = np.random.rand(3, 32, 32)
        _, stats = apply_saturation(
            data, saturation=25, per_vector={"red": 80.0, "blue": 10.0}
        )

        assert stats["red_boost"] > stats["blue_boost"]

    def test_saturation_increases_chroma(self):
        """Higher saturation should generally increase chroma."""
        np.random.seed(42)
        data = np.random.rand(3, 32, 32) * 0.5 + 0.25

        result_low, _ = apply_saturation(data, saturation=10)
        result_high, _ = apply_saturation(data, saturation=90)

        from siril_job_runner.veralux_core import lab_to_lch, rgb_to_lab

        lab_low = rgb_to_lab(result_low)
        lab_high = rgb_to_lab(result_high)
        _, C_low, _ = lab_to_lch(lab_low)
        _, C_high, _ = lab_to_lch(lab_high)

        assert np.mean(C_high) > np.mean(C_low)
