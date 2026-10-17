"""Radiometric regression tests for exposure blending."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from siril_job_runner.hdr import HDRBlender, blend_exposures
from siril_job_runner.models import StackInfo


def test_three_exposures_preserve_core_and_faint_signal() -> None:
    """Recover a known scene even when both longer exposures saturate."""
    scene = np.linspace(0.001, 0.8, 256 * 256).reshape(256, 256)
    short = scene
    medium = np.minimum(scene * 3, 1)
    long = np.minimum(scene * 18, 1)
    combined = blend_exposures(short, medium, 0.7, 0.9)
    combined = blend_exposures(combined, long, 0.7, 0.9)
    np.testing.assert_allclose(combined, scene * 18, rtol=2e-6, atol=1e-6)
    assert combined.max() > 5


def test_matching_removes_exposure_background_offset() -> None:
    """An additive sky difference must not create a step at the blend boundary."""
    scene = np.linspace(0.001, 0.4, 256 * 256).reshape(256, 256)
    short = scene + 0.01
    long = np.minimum(scene * 4 + 0.03, 1)
    result = blend_exposures(short, long, 0.7, 0.9)
    np.testing.assert_allclose(result, scene * 4 + 0.03, atol=1e-6)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_input_fails(bad: float) -> None:
    """A corrupt input must not silently generate a plausible-looking output."""
    short = np.ones((256, 256))
    short[20, 20] = bad
    with pytest.raises(ValueError, match="non-finite"):
        blend_exposures(short, np.ones_like(short), 0.7, 0.9)


def test_unmatchable_input_fails() -> None:
    """A flat image cannot establish an exposure relationship."""
    image = np.full((256, 256), 0.1)
    with pytest.raises(ValueError, match="Too few"):
        blend_exposures(image, image, 0.7, 0.9)


def test_sparse_noisy_nebula_retains_recovered_core() -> None:
    """Background noise must not set the exposure ratio or erase the core."""
    rng = np.random.default_rng(42)
    scene = np.full((512, 512), 0.01)
    scene[224:288, 224:288] = np.linspace(0.01, 0.3, 64 * 64).reshape(64, 64)
    short = scene + rng.normal(0, 0.002, scene.shape)
    long = np.minimum(scene * 18, 1) + rng.normal(0, 0.002, scene.shape)
    result = blend_exposures(short, long, 0.7, 0.9)
    core = scene > 0.1
    assert np.median(result[core]) == pytest.approx(
        np.median(scene[core] * 18), rel=0.03
    )
    np.testing.assert_array_equal(
        result[long < 0.7], long[long < 0.7].astype(np.float32)
    )


def test_background_noise_cannot_determine_exposure_ratio() -> None:
    """Unrelated sky noise is insufficient evidence for an exposure match."""
    rng = np.random.default_rng(42)
    short = rng.normal(0.01, 0.002, (512, 512))
    long = rng.normal(0.18, 0.002, short.shape)
    with pytest.raises(ValueError, match="Too few"):
        blend_exposures(short, long, 0.7, 0.9)


def test_saved_hdr_preserves_relative_brightness(tmp_path: Path) -> None:
    """The saved multi-exposure image preserves the independently known scene."""
    scene = np.linspace(0.001, 0.8, 256 * 256).reshape(256, 256)
    stacks = []
    for exposure in (1, 3, 18):
        path = tmp_path / f"stack_{exposure}.fit"
        fits.writeto(path, np.minimum(scene * exposure, 1).astype(np.float32))
        stacks.append(StackInfo(path, "G", exposure))
    output = HDRBlender().blend_channel("G", stacks, tmp_path / "hdr.fit")
    data = fits.getdata(output)
    np.testing.assert_allclose(data, scene / scene.max(), atol=1e-6)
    assert np.isfinite(data).all()
