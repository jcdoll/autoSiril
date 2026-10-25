"""Regression coverage for FITS units at the enhancement file boundary."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from siril_job_runner.config import Config
from siril_job_runner.veralux_revela import apply_revela
from siril_job_runner.veralux_silentium import apply_silentium
from siril_job_runner.veralux_vectra import apply_vectra


@pytest.mark.parametrize("enhancement", ["silentium", "revela", "vectra"])
@pytest.mark.parametrize("storage", ["float_outlier", "uint16"])
def test_enhancement_preserves_background_scale(
    tmp_path: Path, enhancement: str, storage: str
) -> None:
    """Bright float outliers must not change the units of ordinary pixels."""
    pixels = np.full((3, 128, 128), 0.08, dtype=np.float32)
    if storage == "float_outlier":
        pixels[:, 0, 0] = 124.0
        pixels[:, -1, -1] = -1.0
    else:
        pixels = (pixels * 65535).astype(np.uint16)
    path = tmp_path / "nebula.fit"
    fits.writeto(path, pixels)
    apply = {
        "silentium": apply_silentium,
        "revela": apply_revela,
        "vectra": apply_vectra,
    }[enhancement]
    success, _ = apply(MagicMock(), path, Config())
    assert success
    with fits.open(path, memmap=False) as hdus:
        result = hdus[0].data
        if np.issubdtype(result.dtype, np.integer):
            result = result / np.iinfo(result.dtype).max
    assert np.isfinite(result).all()
    assert 0.06 < np.median(result[:, 32:96, 32:96]) < 0.10


def test_stretch_does_not_rescale_other_pixels_at_float_threshold(
    tmp_path: Path,
) -> None:
    """Crossing the old float maximum threshold cannot change image-wide units."""
    from siril_job_runner.veralux_stretch import apply_stretch

    y, x = np.mgrid[-1:1:128j, -1:1:128j]
    signal = 0.02 + 0.1 * np.exp(-(x * x + y * y) / 0.1)
    original = np.stack([signal * 1.2, signal, signal * 0.8]).astype(np.float32)
    outputs = []
    for index, peak in enumerate([1.0999, 1.1001]):
        pixels = original.copy()
        pixels[:, 0, 0] = peak
        path = tmp_path / f"input{index}.fit"
        fits.writeto(path, pixels)
        success, _ = apply_stretch(MagicMock(), path, Config())
        assert success
        outputs.append(
            fits.getdata(path.with_stem(f"input{index}_stretched"), memmap=False)
        )
    np.testing.assert_array_equal(outputs[0], outputs[1])


@pytest.mark.parametrize("rgb", [False, True])
def test_automatic_stretch_matches_upstream_reference(
    tmp_path: Path, rgb: bool
) -> None:
    """Check public output against independently executed HyperMetric 1.5.2."""
    from siril_job_runner.veralux_stretch import apply_stretch

    y, x = np.mgrid[-1:1:64j, -1:1:64j]
    signal = 0.02 + 0.05 * np.exp(-(x * x + y * y) / 0.2) + 0.002 * (x + 1)
    data = np.stack([signal * 1.3, signal, signal * 0.6]) if rgb else signal
    path = tmp_path / "reference.fit"
    fits.writeto(path, data.astype(np.float32))
    config = Config(
        veralux_target_median=0.10,
        veralux_adaptive_anchor=False,
        veralux_sensor_profile="imx571",
    )
    success, _ = apply_stretch(MagicMock(), path, config)
    assert success
    result = fits.getdata(path.with_stem("reference_stretched"), memmap=False) / 65535

    # Executed upstream AutoSolverThread.run + process_veralux_v6 with target 0.10,
    # IMX571, statistical anchor and ready_to_use mode. Source and hash:
    # https://gitlab.com/free-astro/siril-scripts/-/raw/main/VeraLux/VeraLux_HyperMetric_Stretch.py
    # SHA256: 5b9704c0dac81dac71fb0be4b5c0efae7276cf68a7285f777f341b62fb90c571
    # Samples cover faint background, bright signal and unequal RGB channels.
    expected = (
        [
            [0.174069077, 1.0, 0.207399726],
            [0.047659017, 0.723682404, 0.073972687],
            [0.0, 0.311463326, 0.0],
        ]
        if rgb
        else [0.064109348, 1.0, 0.097725704]
    )
    np.testing.assert_allclose(
        result[..., [10, 32, 53], [17, 32, 48]], expected, atol=2 / 65535, rtol=0
    )


@pytest.mark.parametrize("storage", ["uint16", "float32"])
def test_image_statistics_preserve_units(tmp_path: Path, storage: str) -> None:
    """Integer full scale and float outliers cannot misidentify the weakest channel."""
    from siril_job_runner.siril_wrapper import SirilWrapper

    pixels = np.full((16, 16), 0.2, dtype=np.float32)
    pixels[0, 0] = 1.6 if storage == "float32" else 1.0
    if storage == "uint16":
        pixels = (pixels * 65535).astype(np.uint16)
    path = tmp_path / "stats.fit"
    fits.writeto(path, pixels)
    stats = SirilWrapper(MagicMock()).get_image_stats(path)
    assert stats["median"] == pytest.approx(0.2, abs=1 / 65535)
    assert stats["max"] == 1.0


@pytest.mark.parametrize(
    "rgb, expected_log_d, expected",
    [
        (False, 3.829833984375, [0.085123979, 1.0, 0.182062551]),
        (
            True,
            3.81787109375,
            [
                [0.310479969, 1.0, 0.384774566],
                [0.039544575, 0.890989065, 0.128958598],
                [0.0, 0.591479778, 0.0],
            ],
        ),
    ],
)
def test_adaptive_stretch_matches_upstream_on_faint_signal(
    tmp_path: Path,
    rgb: bool,
    expected_log_d: float,
    expected: list[float] | list[list[float]],
) -> None:
    """The solver and final image must both use the low-signal histogram anchor."""
    from siril_job_runner.veralux_stretch import apply_stretch

    y, x = np.mgrid[-1:1:64j, -1:1:64j]
    signal = 0.00065 + 0.003 * np.exp(-(x * x + y * y) / 0.12) + 0.00008 * (x + 1)
    data = np.stack([signal * 1.3, signal, signal * 0.6]) if rgb else signal
    path = tmp_path / "faint.fit"
    fits.writeto(path, data.astype(np.float32))
    success, log_d = apply_stretch(MagicMock(), path, Config())
    assert success
    # Independently executed HyperMetric 1.5.2 AutoSolverThread.run and
    # process_veralux_v6: adaptive anchor, target 0.20, Rec.709, ready_to_use.
    # https://gitlab.com/free-astro/siril-scripts/-/raw/main/VeraLux/VeraLux_HyperMetric_Stretch.py
    # SHA256: 5b9704c0dac81dac71fb0be4b5c0efae7276cf68a7285f777f341b62fb90c571
    # This faint histogram peaks below bin 100 and selects a histogram anchor,
    # exercising the adaptive path rather than only its percentile fallback.
    assert log_d == pytest.approx(expected_log_d)
    result = fits.getdata(path.with_stem("faint_stretched"), memmap=False) / 65535
    np.testing.assert_allclose(
        result[..., [10, 32, 53], [17, 32, 48]], expected, atol=2 / 65535, rtol=0
    )


@pytest.mark.parametrize("operation", ["stretch", "statistics"])
def test_nonfinite_pixels_fail_before_stretch_or_statistics(
    tmp_path: Path, operation: str
) -> None:
    """Invalid values must not be converted into plausible finite image results."""
    from siril_job_runner.siril_wrapper import SirilWrapper
    from siril_job_runner.veralux_stretch import apply_stretch

    path = tmp_path / "bad.fit"
    fits.writeto(path, np.full((3, 8, 8), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        if operation == "stretch":
            apply_stretch(MagicMock(), path, Config())
        else:
            SirilWrapper(MagicMock()).get_image_stats(path)


@pytest.mark.parametrize("rgb", [False, True])
def test_linear_denoise_keeps_faint_signal_and_is_not_reapplied_after_stretch(
    tmp_path: Path, rgb: bool
) -> None:
    """Linear output must survive below one 16-bit step and bypass post-stretch NR."""
    from siril_job_runner.stretch_helpers import (
        apply_enhancements,
        apply_linear_noise_reduction,
    )

    pixels = np.full((3, 64, 64) if rgb else (64, 64), 0.000002, dtype=np.float32)
    path = tmp_path / "linear.fit"
    fits.writeto(path, pixels)
    cfg = Config(veralux_revela_enabled=False, veralux_vectra_enabled=False)
    siril = MagicMock()
    apply_linear_noise_reduction(siril, path, cfg, lambda _: None)
    result = fits.getdata(path, memmap=False)
    np.testing.assert_allclose(result, pixels, atol=1e-9, rtol=0)

    saved = path.read_bytes()
    apply_enhancements(siril, path, cfg, lambda _: None)
    assert path.read_bytes() == saved


def test_linear_denoise_failure_stops_processing(tmp_path: Path) -> None:
    """A failed Siril reload must not let processing use the previous image."""
    from siril_job_runner.stretch_helpers import apply_linear_noise_reduction

    path = tmp_path / "linear.fit"
    fits.writeto(path, np.full((32, 32), 0.01, dtype=np.float32))
    siril = MagicMock()
    siril.load.return_value = False
    with pytest.raises(RuntimeError, match="Silentium"):
        apply_linear_noise_reduction(siril, path, Config(), lambda _: None)
