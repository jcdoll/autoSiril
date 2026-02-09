"""Tests for veralux_vectra module."""

import numpy as np

from siril_job_runner.veralux_vectra import (
    HUE_NAMES,
    HUE_SIGMA_RAD,
    _compute_chroma_stability,
    _compute_global_mask,
    _compute_hue_weights,
    _compute_signal_mask,
    _compute_star_mask_energy,
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
        H = np.array([[0.0]])  # Red hue = 0
        weights = _compute_hue_weights(H)
        # Red weight should be highest at hue=0
        assert weights[0][0, 0] > weights[1][0, 0]  # Red > Yellow
        assert weights[0][0, 0] > weights[5][0, 0]  # Red > Magenta

    def test_hue_weights_sum_positive(self):
        """All weights should be positive."""
        H = np.random.rand(32, 32) * 2 * np.pi
        weights = _compute_hue_weights(H)
        for w in weights:
            assert np.all(w >= 0)

    def test_hue_sigma_is_30_degrees(self):
        """Hue sigma should be exactly 30 degrees (pi/6 radians)."""
        np.testing.assert_almost_equal(HUE_SIGMA_RAD, np.pi / 6.0)


class TestSignalMask:
    """Tests for signal mask (shadow authority)."""

    def test_signal_mask_shape(self):
        """Signal mask should match input shape."""
        L = np.random.rand(64, 64) * 100
        mask = _compute_signal_mask(L, shadow_auth=50.0)
        assert mask.shape == L.shape

    def test_signal_mask_range(self):
        """Signal mask should be in [0, 1]."""
        L = np.random.rand(64, 64) * 100
        mask = _compute_signal_mask(L, shadow_auth=50.0)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_zero_shadow_auth_allows_bright(self):
        """Zero shadow authority should allow bright regions through."""
        # Reference: sigma_thresh = shadow_auth / 20.0 = 0
        # noise_floor = bg (25th percentile) + 0 = bg
        # So bright areas above bg should have positive mask
        # Need data with variation for MAD to be meaningful
        np.random.seed(42)
        L = np.random.rand(64, 64) * 30 + 70  # Bright data (70-100)
        mask = _compute_signal_mask(L, shadow_auth=0.0)
        # Should have significant mask values for bright data above bg
        assert np.mean(mask) > 0.0

    def test_high_shadow_auth_protects_dark(self):
        """High shadow authority should protect dark regions more."""
        # Reference: sigma_thresh = shadow_auth / 20.0
        # Higher shadow_auth = higher noise_floor = more protection
        L = np.zeros((64, 64))
        L[:32, :] = 20.0  # Dark half
        L[32:, :] = 80.0  # Bright half

        mask_low = _compute_signal_mask(L, shadow_auth=20.0)
        mask_high = _compute_signal_mask(L, shadow_auth=80.0)

        # With higher shadow_auth, more protection (lower mean mask)
        assert np.mean(mask_low) >= np.mean(mask_high)


class TestStarMaskEnergy:
    """Tests for wavelet energy-based star mask."""

    def test_star_mask_shape(self):
        """Star mask should match input shape."""
        L = np.random.rand(64, 64) * 100
        mask = _compute_star_mask_energy(L)
        assert mask.shape == L.shape

    def test_star_mask_range(self):
        """Star mask should be in [0, 1]."""
        L = np.random.rand(64, 64) * 100
        mask = _compute_star_mask_energy(L)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_star_mask_uniform_for_smooth(self):
        """Star mask should be near 1 for smooth uniform data (no stars)."""
        # Reference formula: protection = 1 - star_map
        # For uniform data, wavelet energy is ~0, star_map ~0, protection ~1
        L = np.ones((64, 64)) * 50.0
        mask = _compute_star_mask_energy(L)
        # Should be close to 1 (full saturation allowed) for smooth data
        assert np.mean(mask) > 0.9


class TestChromaStability:
    """Tests for chroma stability gate."""

    def test_chroma_stability_shape(self):
        """Chroma stability should match input shape."""
        C = np.random.rand(64, 64) * 50
        L = np.random.rand(64, 64) * 100
        signal_mask = np.ones((64, 64))
        stability = _compute_chroma_stability(C, L, signal_mask)
        assert stability.shape == C.shape

    def test_chroma_stability_range(self):
        """Chroma stability should be in [0, 1]."""
        C = np.random.rand(64, 64) * 50
        L = np.random.rand(64, 64) * 100
        signal_mask = np.ones((64, 64))
        stability = _compute_chroma_stability(C, L, signal_mask)
        assert stability.min() >= 0.0
        assert stability.max() <= 1.0

    def test_weak_chroma_with_strong_signal_has_assist(self):
        """Weak chroma with strong signal should have assist floor ~0.25."""
        # Reference: assist = 0.25 * clip((signal_mask - 0.10) / 0.30, 0, 1)
        # With signal_mask = 1.0: assist = 0.25 * clip(0.9/0.3, 0, 1) = 0.25
        C = np.zeros((32, 32))  # Zero chroma
        L = np.ones((32, 32)) * 50
        signal_mask = np.ones((32, 32))  # Strong signal
        stability = _compute_chroma_stability(C, L, signal_mask)
        # Should be at least 0.25 from assist term
        np.testing.assert_almost_equal(stability.min(), 0.25)


class TestGlobalMask:
    """Tests for combined global protection mask."""

    def test_global_mask_shape(self):
        """Global mask should match input shape."""
        L = np.random.rand(64, 64) * 100
        C = np.random.rand(64, 64) * 50
        mask = _compute_global_mask(L, C, shadow_auth=50.0, protect_stars=True)
        assert mask.shape == L.shape

    def test_global_mask_range(self):
        """Global mask should be in [0, 1]."""
        L = np.random.rand(64, 64) * 100
        C = np.random.rand(64, 64) * 50
        mask = _compute_global_mask(L, C, shadow_auth=50.0, protect_stars=True)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_global_mask_without_star_protection(self):
        """Without star protection, star mask component should be 1."""
        L = np.random.rand(64, 64) * 100
        C = np.random.rand(64, 64) * 50

        mask_with = _compute_global_mask(L, C, shadow_auth=50.0, protect_stars=True)
        mask_without = _compute_global_mask(L, C, shadow_auth=50.0, protect_stars=False)

        # Without star protection, mask should generally be higher
        assert np.mean(mask_without) >= np.mean(mask_with)


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

        assert "base_sat_boost" in stats
        assert "effective_sat_boost_mean" in stats
        assert "global_mask_mean" in stats

    def test_per_vector_override(self):
        """Per-vector override should affect specific hue stats."""
        data = np.random.rand(3, 32, 32)
        _, stats = apply_saturation(
            data, saturation=25, per_vector_sat={"red": 80.0, "blue": 10.0}
        )

        assert stats["red_sat_boost"] > stats["blue_sat_boost"]

    def test_saturation_increases_chroma(self):
        """Higher saturation should generally increase chroma."""
        np.random.seed(42)
        data = np.random.rand(3, 32, 32) * 0.5 + 0.25

        result_low, _ = apply_saturation(data, saturation=10)
        result_high, _ = apply_saturation(data, saturation=90)

        from siril_job_runner.veralux_colorspace import lab_to_lch, rgb_to_lab

        lab_low = rgb_to_lab(result_low)
        lab_high = rgb_to_lab(result_high)
        _, C_low, _ = lab_to_lch(lab_low)
        _, C_high, _ = lab_to_lch(lab_high)

        assert np.mean(C_high) > np.mean(C_low)

    def test_hue_shift_support(self):
        """Per-vector hue shift should be recorded in stats."""
        data = np.random.rand(3, 32, 32)
        _, stats = apply_saturation(
            data, saturation=25, per_vector_hue={"red": 15.0, "cyan": -10.0}
        )

        np.testing.assert_almost_equal(stats["red_hue_shift_deg"], 15.0)
        np.testing.assert_almost_equal(stats["cyan_hue_shift_deg"], -10.0)

    def test_all_hue_names_in_stats(self):
        """All 6 hue names should appear in stats."""
        data = np.random.rand(3, 32, 32)
        _, stats = apply_saturation(data)

        for name in HUE_NAMES:
            assert f"{name}_sat_boost" in stats
            assert f"{name}_hue_shift_deg" in stats
