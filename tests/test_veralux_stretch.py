"""Tests for veralux_stretch module."""

import numpy as np

from siril_job_runner.veralux_stretch import (
    SENSOR_PROFILES,
    _apply_mtf,
    _calculate_anchor_morphological,
    _calculate_anchor_percentile,
    _hyperbolic_stretch_array,
    _soft_clip_channel,
    estimate_star_pressure,
    get_sensor_weights,
    hyperbolic_stretch_value,
    solve_log_d,
)


class TestSensorProfiles:
    """Tests for sensor profile functions."""

    def test_get_sensor_weights_rec709(self):
        """Rec709 should return standard weights."""
        weights = get_sensor_weights("rec709")
        assert weights == (0.2126, 0.7152, 0.0722)

    def test_get_sensor_weights_imx571(self):
        """IMX571 should return sensor-specific weights."""
        weights = get_sensor_weights("imx571")
        assert weights == (0.2944, 0.5021, 0.2035)

    def test_get_sensor_weights_case_insensitive(self):
        """Profile lookup should be case-insensitive."""
        weights1 = get_sensor_weights("IMX571")
        weights2 = get_sensor_weights("imx571")
        assert weights1 == weights2

    def test_get_sensor_weights_unknown_returns_rec709(self):
        """Unknown profile should fall back to rec709."""
        weights = get_sensor_weights("unknown_sensor")
        assert weights == SENSOR_PROFILES["rec709"]

    def test_all_profiles_sum_to_one(self):
        """All profiles should have weights summing to ~1.0."""
        for name, weights in SENSOR_PROFILES.items():
            total = sum(weights)
            np.testing.assert_almost_equal(
                total, 1.0, decimal=3, err_msg=f"Profile {name}"
            )


class TestHyperbolicStretch:
    """Tests for hyperbolic stretch function."""

    def test_stretch_preserves_zero(self):
        """Zero input should give zero output."""
        result = hyperbolic_stretch_value(0.0, D=10.0, b=6.0)
        np.testing.assert_almost_equal(result, 0.0, decimal=5)

    def test_stretch_preserves_one(self):
        """One input should give one output (normalized)."""
        result = hyperbolic_stretch_value(1.0, D=10.0, b=6.0)
        np.testing.assert_almost_equal(result, 1.0, decimal=5)

    def test_stretch_monotonic(self):
        """Stretch should be monotonically increasing."""
        values = np.linspace(0, 1, 100)
        results = [hyperbolic_stretch_value(v, D=10.0, b=6.0) for v in values]
        assert all(results[i] <= results[i + 1] for i in range(len(results) - 1))

    def test_higher_D_brighter(self):
        """Higher D should give brighter output for mid-values."""
        low_D = hyperbolic_stretch_value(0.5, D=5.0, b=6.0)
        high_D = hyperbolic_stretch_value(0.5, D=50.0, b=6.0)
        assert high_D > low_D

    def test_array_stretch_matches_scalar(self):
        """Array stretch should match scalar computation."""
        data = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        D, b = 10.0, 6.0
        array_result = _hyperbolic_stretch_array(data, D, b)
        scalar_results = [hyperbolic_stretch_value(v, D, b) for v in data]
        np.testing.assert_array_almost_equal(array_result, scalar_results, decimal=10)


class TestStarPressure:
    """Tests for star pressure estimation."""

    def test_uniform_image_low_pressure(self):
        """Uniform image should have low star pressure."""
        data = np.full((100, 100), 0.1)
        pressure = estimate_star_pressure(data)
        assert pressure < 0.2

    def test_image_with_bright_pixels_higher_pressure(self):
        """Image with bright point sources should have higher pressure."""
        data = np.full((100, 100), 0.1)
        # Add some "stars" - bright isolated points
        data[10, 10] = 0.9
        data[30, 30] = 0.95
        data[50, 50] = 0.85
        pressure = estimate_star_pressure(data)
        # Should have elevated pressure due to bright points
        assert pressure > 0.0

    def test_pressure_in_valid_range(self):
        """Star pressure should always be in [0, 1]."""
        np.random.seed(42)
        data = np.random.rand(100, 100)
        pressure = estimate_star_pressure(data)
        assert 0.0 <= pressure <= 1.0

    def test_empty_image_returns_zero(self):
        """Nearly empty image should return zero pressure."""
        data = np.zeros((100, 100))
        pressure = estimate_star_pressure(data)
        assert pressure == 0.0


class TestAnchorCalculation:
    """Tests for anchor (black point) calculation."""

    def test_percentile_anchor_positive(self):
        """Percentile anchor should be non-negative."""
        np.random.seed(42)
        data = np.random.rand(3, 64, 64)
        anchor = _calculate_anchor_percentile(data)
        assert anchor >= 0.0

    def test_percentile_anchor_below_median(self):
        """Anchor should be well below the median."""
        np.random.seed(42)
        data = np.random.rand(3, 64, 64) * 0.5 + 0.25
        anchor = _calculate_anchor_percentile(data)
        median = np.median(data)
        assert anchor < median

    def test_morphological_anchor_positive(self):
        """Morphological anchor should be non-negative."""
        np.random.seed(42)
        data = np.random.rand(3, 64, 64)
        weights = (0.2126, 0.7152, 0.0722)
        anchor = _calculate_anchor_morphological(data, weights)
        assert anchor >= 0.0

    def test_morphological_anchor_reasonable_range(self):
        """Morphological anchor should be in reasonable range."""
        np.random.seed(42)
        data = np.random.rand(3, 128, 128) * 0.5  # Data mostly 0-0.5
        weights = (0.2126, 0.7152, 0.0722)
        anchor = _calculate_anchor_morphological(data, weights)
        # Anchor should be less than half the median
        assert anchor < 0.25


class TestSolver:
    """Tests for log_d solver."""

    def test_solver_finds_solution(self):
        """Solver should find a reasonable solution."""
        median_in = 0.01
        target_median = 0.20
        b = 6.0
        log_d = solve_log_d(median_in, target_median, b)
        # Should find a positive log_d
        assert log_d > 0.0

    def test_solver_respects_bounds(self):
        """Solver should stay within bounds."""
        log_d = solve_log_d(
            median_in=0.01,
            target_median=0.20,
            b=6.0,
            log_d_min=1.0,
            log_d_max=5.0,
        )
        assert 1.0 <= log_d <= 5.0

    def test_solver_handles_zero_median(self):
        """Solver should handle zero median gracefully."""
        log_d = solve_log_d(median_in=0.0, target_median=0.20, b=6.0)
        assert log_d == 2.0  # Default value

    def test_star_pressure_damping_formula(self):
        """Star pressure damping should be applied before solver per reference.

        Reference formula (lines 1534-1536):
            if star_pressure > 0.6:
                target_temp *= (1.0 - 0.15 * star_pressure)
        """
        median_in = 0.01
        target = 0.20
        b = 6.0

        # No damping when star_pressure <= 0.6
        log_d_low = solve_log_d(median_in, target, b)

        # With high star pressure (0.8), damping reduces effective target
        star_pressure = 0.8
        damped_target = target * (1.0 - 0.15 * star_pressure)
        log_d_damped = solve_log_d(median_in, damped_target, b)

        # Damped target is lower, so log_d should be lower
        assert log_d_damped < log_d_low


class TestMTF:
    """Tests for Midtone Transfer Function."""

    def test_mtf_preserves_zero(self):
        """MTF should preserve zero."""
        data = np.array([0.0])
        result = _apply_mtf(data, m=0.5)
        np.testing.assert_almost_equal(result[0], 0.0, decimal=5)

    def test_mtf_preserves_one(self):
        """MTF should preserve one."""
        data = np.array([1.0])
        result = _apply_mtf(data, m=0.5)
        np.testing.assert_almost_equal(result[0], 1.0, decimal=5)

    def test_mtf_midtone_shift(self):
        """MTF with m<0.5 should brighten, m>0.5 should darken."""
        data = np.array([0.5])
        bright = _apply_mtf(data, m=0.3)
        dark = _apply_mtf(data, m=0.7)
        assert bright[0] > 0.5
        assert dark[0] < 0.5


class TestSoftClip:
    """Tests for soft-clip function."""

    def test_soft_clip_below_threshold_unchanged(self):
        """Values below threshold should pass through."""
        data = np.array([0.5, 0.7, 0.9])
        result = _soft_clip_channel(data, thresh=0.95)
        np.testing.assert_array_almost_equal(data, result, decimal=10)

    def test_soft_clip_above_threshold_compressed(self):
        """Values above threshold should be compressed."""
        data = np.array([0.99, 1.0, 1.1])
        result = _soft_clip_channel(data, thresh=0.98)
        # All should be below 1.0
        assert result.max() <= 1.0
        # Should preserve ordering
        assert result[0] <= result[1] <= result[2]

    def test_soft_clip_output_range(self):
        """Soft clip should always output [0, 1]."""
        data = np.array([0.0, 0.5, 0.98, 0.99, 1.0, 1.5])
        result = _soft_clip_channel(data)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
