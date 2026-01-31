"""Tests for veralux_starcomposer module."""

import numpy as np

from siril_job_runner.veralux_starcomposer import (
    BlendMode,
    _apply_color_grip,
    _apply_gamma_conditioning,
    _apply_micro_blur,
    _calculate_anchor_adaptive,
    _stretch_scalar,
    _stretch_vector,
    blend_linear_add,
    blend_screen,
    compose_stars,
    hyperbolic_stretch,
)


class TestHyperbolicStretch:
    """Tests for hyperbolic stretch function."""

    def test_stretch_preserves_zero(self):
        """Zero input should give zero output."""
        data = np.array([0.0])
        result = hyperbolic_stretch(data, D=10.0, b=6.0)
        np.testing.assert_almost_equal(result[0], 0.0, decimal=5)

    def test_stretch_preserves_one(self):
        """One input should give one output (normalized)."""
        data = np.array([1.0])
        result = hyperbolic_stretch(data, D=10.0, b=6.0)
        np.testing.assert_almost_equal(result[0], 1.0, decimal=5)

    def test_stretch_monotonic(self):
        """Stretch should be monotonically increasing."""
        data = np.linspace(0, 1, 100)
        result = hyperbolic_stretch(data, D=10.0, b=6.0)
        assert np.all(np.diff(result) >= 0)

    def test_higher_D_brighter(self):
        """Higher D should give brighter output for mid-values."""
        data = np.array([0.5])
        low_D = hyperbolic_stretch(data, D=5.0, b=6.0)
        high_D = hyperbolic_stretch(data, D=50.0, b=6.0)
        assert high_D[0] > low_D[0]


class TestBlendModes:
    """Tests for blend mode functions."""

    def test_screen_brightens(self):
        """Screen blend should brighten the base."""
        base = np.array([0.5])
        layer = np.array([0.5])
        result = blend_screen(base, layer)
        assert result[0] > base[0]

    def test_screen_zero_layer_identity(self):
        """Screen with zero layer should equal base."""
        base = np.array([0.5])
        layer = np.array([0.0])
        result = blend_screen(base, layer)
        np.testing.assert_almost_equal(result[0], base[0])

    def test_linear_add_sums(self):
        """Linear add should sum values (before clipping)."""
        base = np.array([0.3])
        layer = np.array([0.4])
        result = blend_linear_add(base, layer)
        np.testing.assert_almost_equal(result[0], 0.7)

    def test_linear_add_clips(self):
        """Linear add should clip to [0, 1]."""
        base = np.array([0.8])
        layer = np.array([0.5])
        result = blend_linear_add(base, layer)
        assert result[0] == 1.0


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_gamma_conditioning_darkens(self):
        """Gamma 2.4 should darken mid-tones."""
        data = np.array([[[0.5]]])
        result = _apply_gamma_conditioning(data, gamma=2.4)
        assert result[0, 0, 0] < 0.5

    def test_gamma_conditioning_preserves_extremes(self):
        """Gamma should preserve 0 and 1."""
        data = np.array([[[0.0, 1.0]]])
        result = _apply_gamma_conditioning(data, gamma=2.4)
        np.testing.assert_almost_equal(result[0, 0, 0], 0.0)
        np.testing.assert_almost_equal(result[0, 0, 1], 1.0)

    def test_micro_blur_smooths(self):
        """Micro-blur should reduce sharp transitions."""
        data = np.zeros((3, 32, 32))
        data[:, 16, 16] = 1.0  # Single bright pixel
        result = _apply_micro_blur(data, sigma=0.5)
        # Bright pixel should spread to neighbors
        assert result[0, 16, 17] > 0.0
        assert result[0, 16, 16] < 1.0

    def test_anchor_adaptive_returns_reasonable_value(self):
        """Anchor should be a small positive value for typical data."""
        data = np.random.rand(3, 64, 64) * 0.5 + 0.1
        anchor = _calculate_anchor_adaptive(data)
        assert 0.0 <= anchor < 0.5


class TestHybridEngine:
    """Tests for scalar/vector branches and color grip."""

    def test_scalar_stretches_channels_independently(self):
        """Scalar branch should stretch each channel independently."""
        data = np.zeros((3, 32, 32))
        data[0] = 0.5  # R
        data[1] = 0.3  # G
        data[2] = 0.1  # B

        result = _stretch_scalar(data, D=10.0, b=6.0)

        # Each channel should have different output based on its input
        assert result[0].mean() > result[1].mean() > result[2].mean()

    def test_vector_preserves_ratios(self):
        """Vector branch should preserve color ratios."""
        data = np.ones((3, 32, 32))
        data[0] = 0.6  # R
        data[1] = 0.3  # G
        data[2] = 0.1  # B

        result = _stretch_vector(data, D=10.0, b=6.0)

        # Ratios should be approximately preserved
        input_ratio = data[0, 0, 0] / data[1, 0, 0]
        output_ratio = result[0, 0, 0] / result[1, 0, 0]
        np.testing.assert_almost_equal(input_ratio, output_ratio, decimal=1)

    def test_color_grip_zero_uses_scalar(self):
        """Zero grip should give scalar result."""
        scalar = np.array([[[0.8]], [[0.8]], [[0.8]]])
        vector = np.array([[[0.6]], [[0.4]], [[0.2]]])

        result = _apply_color_grip(scalar, vector, grip=0.0)

        np.testing.assert_array_almost_equal(result, scalar)

    def test_color_grip_one_uses_vector(self):
        """Full grip should give vector result."""
        scalar = np.array([[[0.8]], [[0.8]], [[0.8]]])
        vector = np.array([[[0.6]], [[0.4]], [[0.2]]])

        result = _apply_color_grip(scalar, vector, grip=1.0)

        np.testing.assert_array_almost_equal(result, vector)

    def test_shadow_conv_reduces_vector_in_dark(self):
        """Shadow convergence should reduce vector influence in dark areas."""
        # Scalar and vector with different values
        scalar = np.array([[[0.2]], [[0.2]], [[0.2]]])  # Dark, uniform
        vector = np.array([[[0.3]], [[0.2]], [[0.1]]])  # Dark, colored

        # With no shadow conv, grip=1 should give vector
        result_no_shadow = _apply_color_grip(scalar, vector, grip=1.0, shadow_conv=0.0)

        # With shadow conv, dark areas should lean toward scalar
        result_with_shadow = _apply_color_grip(scalar, vector, grip=1.0, shadow_conv=2.0)

        # Result with shadow should be closer to scalar than result without
        diff_no_shadow = np.abs(result_no_shadow - scalar).mean()
        diff_with_shadow = np.abs(result_with_shadow - scalar).mean()

        assert diff_with_shadow < diff_no_shadow


class TestComposeStars:
    """Tests for the main composition function."""

    def test_compose_returns_correct_shape(self):
        """Output should match input shape."""
        starless = np.random.rand(3, 64, 64) * 0.5
        starmask = np.random.rand(3, 64, 64) * 0.1
        result, stats = compose_stars(starless, starmask)
        assert result.shape == starless.shape

    def test_compose_returns_valid_range(self):
        """Output should be in [0, 1]."""
        starless = np.random.rand(3, 64, 64) * 0.5
        starmask = np.random.rand(3, 64, 64) * 0.1
        result, stats = compose_stars(starless, starmask)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_compose_returns_stats(self):
        """Should return relevant statistics."""
        starless = np.random.rand(3, 64, 64)
        starmask = np.random.rand(3, 64, 64) * 0.1
        _, stats = compose_stars(starless, starmask)

        assert "log_d" in stats
        assert "D" in stats
        assert "hardness" in stats
        assert "color_grip" in stats
        assert "shadow_conv" in stats
        assert "anchor" in stats
        assert "blend_mode" in stats
        assert "star_brightness_mean" in stats

    def test_compose_with_screen_blend(self):
        """Screen blend mode should be applied."""
        starless = np.ones((3, 32, 32)) * 0.3
        starmask = np.ones((3, 32, 32)) * 0.2
        result, stats = compose_stars(
            starless, starmask, blend_mode=BlendMode.SCREEN
        )
        assert stats["blend_mode"] == "screen"
        assert result[0, 16, 16] > starless[0, 16, 16]

    def test_compose_with_linear_add_blend(self):
        """Linear add blend mode should be applied."""
        starless = np.ones((3, 32, 32)) * 0.3
        starmask = np.ones((3, 32, 32)) * 0.2
        result, stats = compose_stars(
            starless, starmask, blend_mode=BlendMode.LINEAR_ADD
        )
        assert stats["blend_mode"] == "linear_add"

    def test_zero_starmask_preserves_starless(self):
        """Zero star mask should give approximately starless image."""
        starless = np.random.rand(3, 32, 32) * 0.5 + 0.25
        starmask = np.zeros((3, 32, 32))
        result, _ = compose_stars(starless, starmask)
        np.testing.assert_array_almost_equal(starless, result, decimal=3)

    def test_color_grip_affects_output(self):
        """Different color grip values should produce different results."""
        starless = np.ones((3, 32, 32)) * 0.2
        starmask = np.ones((3, 32, 32))
        starmask[0] = 0.3  # R
        starmask[1] = 0.2  # G
        starmask[2] = 0.1  # B

        result_scalar, _ = compose_stars(starless, starmask, color_grip=0.0)
        result_vector, _ = compose_stars(starless, starmask, color_grip=1.0)

        # Results should be different
        assert not np.allclose(result_scalar, result_vector)

    def test_shadow_conv_affects_output(self):
        """Shadow convergence should affect dark regions."""
        starless = np.ones((3, 32, 32)) * 0.1  # Dark image
        starmask = np.ones((3, 32, 32)) * 0.1
        starmask[0] = 0.2  # Colored stars

        result_no_shadow, _ = compose_stars(
            starless, starmask, color_grip=1.0, shadow_conv=0.0
        )
        result_with_shadow, _ = compose_stars(
            starless, starmask, color_grip=1.0, shadow_conv=2.0
        )

        # Results should be different
        assert not np.allclose(result_no_shadow, result_with_shadow)
