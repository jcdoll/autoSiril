"""Tests for veralux_starcomposer module."""

import numpy as np
import pytest

from siril_job_runner.veralux_starcomposer import (
    BlendMode,
    apply_color_grip,
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
        result = hyperbolic_stretch(data, log_d=1.0, hardness=6.0)
        np.testing.assert_almost_equal(result[0], 0.0, decimal=5)

    def test_stretch_preserves_one(self):
        """One input should give one output (normalized)."""
        data = np.array([1.0])
        result = hyperbolic_stretch(data, log_d=1.0, hardness=6.0)
        np.testing.assert_almost_equal(result[0], 1.0, decimal=5)

    def test_stretch_monotonic(self):
        """Stretch should be monotonically increasing."""
        data = np.linspace(0, 1, 100)
        result = hyperbolic_stretch(data, log_d=1.0, hardness=6.0)
        assert np.all(np.diff(result) >= 0)

    def test_higher_log_d_brighter(self):
        """Higher log_d should give brighter output for mid-values."""
        data = np.array([0.5])
        low_d = hyperbolic_stretch(data, log_d=0.5, hardness=6.0)
        high_d = hyperbolic_stretch(data, log_d=1.5, hardness=6.0)
        assert high_d[0] > low_d[0]


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


class TestColorGrip:
    """Tests for color grip function."""

    def test_color_grip_zero_gives_grayscale(self):
        """Zero color grip should give grayscale stars."""
        stars = np.array([[[0.8]], [[0.4]], [[0.2]]])
        result = apply_color_grip(stars, color_grip=0.0)
        np.testing.assert_almost_equal(result[0], result[1])
        np.testing.assert_almost_equal(result[1], result[2])

    def test_color_grip_one_preserves_color(self):
        """Full color grip should preserve original colors."""
        stars = np.array([[[0.8]], [[0.4]], [[0.2]]])
        result = apply_color_grip(stars, color_grip=1.0)
        np.testing.assert_array_almost_equal(stars, result)

    def test_color_grip_intermediate(self):
        """Intermediate color grip should be between grayscale and color."""
        stars = np.array([[[0.8]], [[0.4]], [[0.2]]])
        result = apply_color_grip(stars, color_grip=0.5)

        gray = apply_color_grip(stars, color_grip=0.0)
        color = apply_color_grip(stars, color_grip=1.0)

        # Red channel should be between gray and full color
        assert gray[0, 0, 0] <= result[0, 0, 0] <= color[0, 0, 0]


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
        assert "hardness" in stats
        assert "color_grip" in stats
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
