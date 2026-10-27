"""Public finishing behavior: reference curves, precision and failure handling."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from siril_job_runner.config import Config, with_overrides
from siril_job_runner.fits_utils import check_clipping
from siril_job_runner.stretch_helpers import apply_enhancements
from siril_job_runner.veralux_curves import curve_image
from siril_job_runner.veralux_revela import apply_revela
from siril_job_runner.veralux_starcomposer import apply_starcomposer
from siril_job_runner.veralux_vectra import apply_vectra


def test_curves_match_independently_executed_upstream() -> None:
    """Compare RGB/K output with upstream 1.0.1, including soft and hard masks."""
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/veralux_curves.json").read_text()
    )
    for case in fixture["cases"]:
        config = with_overrides(
            {
                "veralux_curves_points": fixture["points"],
                "veralux_curves_range": case["range"],
                "veralux_curves_feather": case["feather"],
            }
        )
        actual = curve_image(np.array(fixture["input"], dtype=np.float32), config)
        np.testing.assert_allclose(actual, case["expected"], atol=1e-7, rtol=0)


def test_curve_identity_endpoints_and_input_preservation() -> None:
    """Identity preserves float pixels; restricted point domains extend constantly."""
    image = np.tile(np.linspace(0, 1, 31, dtype=np.float32), (3, 2, 1))
    original = image.copy()
    np.testing.assert_array_equal(curve_image(image, Config()), image)
    result = curve_image(image, Config(veralux_curves_points=((0.2, 0.1), (0.8, 0.9))))
    np.testing.assert_allclose(result[..., 0], 0.1)
    np.testing.assert_allclose(result[..., -1], 0.9)
    assert np.all(np.diff(result, axis=-1) >= 0)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    "overrides",
    [
        {"veralux_curves_points": [[0, 0]]},
        {"veralux_curves_points": [[0, 0], [0, 0.5], [1, 1]]},
        {"veralux_curves_points": [[0, 0], [float("nan"), 1]]},
        {"veralux_curves_range": [0.8, 0.2]},
        {"veralux_curves_feather": -0.1},
    ],
)
def test_invalid_curve_config_is_rejected(overrides: dict) -> None:
    """Ambiguous curves must fail at configuration loading, before processing."""
    with pytest.raises(ValueError):
        with_overrides(overrides)


@pytest.mark.parametrize("stage", ["revela", "vectra", "starcomposer"])
def test_working_files_keep_sub_16_bit_signal(tmp_path: Path, stage: str) -> None:
    """Neutral stages must not erase a signal smaller than one 16-bit step."""
    image = np.full((3, 32, 32), 0.000003, dtype=np.float32)
    path = tmp_path / "image.fit"
    fits.writeto(path, image, header=fits.Header({"OBJECT": "precision test"}))
    siril = MagicMock()
    config = Config()
    if stage == "revela":
        ok, _ = apply_revela(siril, path, config)
    elif stage == "vectra":
        ok, _ = apply_vectra(siril, path, config)
    else:
        stars = tmp_path / "stars.fit"
        fits.writeto(stars, np.zeros_like(image))
        ok, path = apply_starcomposer(siril, path, stars, config)
    assert ok
    result = fits.getdata(path, memmap=False)
    assert result.dtype.kind == "f"
    np.testing.assert_allclose(result, image, atol=1e-8, rtol=0)
    assert fits.getheader(path)["OBJECT"] == "precision test"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_clipping_counts_float_fits_pixels(tmp_path: Path, dtype: type) -> None:
    """FITS byte order must not turn normalized pixels into all-black reports."""
    path = tmp_path / "float.fit"
    fits.writeto(path, np.array([[0, 0.005, 0.2, 0.6, 0.995, 1]], dtype=dtype))
    result = check_clipping(path)
    assert result is not None
    assert (result.clipped_low, result.clipped_high, result.total_pixels) == (2, 2, 6)


@pytest.mark.parametrize("stage", ["revela", "vectra"])
def test_failed_enhancement_reload_stops_finishing(tmp_path: Path, stage: str) -> None:
    """A failed external reload cannot leave the old image as a successful result."""
    path = tmp_path / "image.fit"
    fits.writeto(path, np.full((3, 32, 32), 0.2, dtype=np.float32))
    siril = MagicMock()
    siril.load.return_value = False
    config = Config(
        veralux_revela_enabled=stage == "revela",
        veralux_vectra_enabled=stage == "vectra",
    )
    with pytest.raises(RuntimeError, match=f"{stage.capitalize()} failed"):
        apply_enhancements(siril, path, config, lambda _: None)
