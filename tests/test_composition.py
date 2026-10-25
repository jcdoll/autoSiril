"""Composition must use current inputs and stop on external command failures."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from siril_job_runner.composition import compose_and_stretch
from siril_job_runner.config import ColorRemovalMode, Config, SubskyMethod
from siril_job_runner.job_runner import JobRunner
from siril_job_runner.models import ValidationResult
from siril_job_runner.output_manager import (
    ManagedJobOutput,
    archive_composition_outputs,
)


def image_siril(directory: Path) -> MagicMock:
    """Emulate only the external image engine, using real FITS files."""
    siril = MagicMock()
    cwd = directory
    image = np.zeros((8, 8))

    def path(name: str) -> Path:
        """Resolve Siril's optional FITS extension against its working directory."""
        result = Path(name)
        if not result.is_absolute():
            result = cwd / result
        return result if result.suffix else result.with_suffix(".fit")

    def cd(name: str) -> bool:
        """Change the external engine's working directory."""
        nonlocal cwd
        cwd = Path(name)
        return True

    def load(name: str) -> bool:
        """Load actual image pixels, rejecting missing files."""
        nonlocal image
        image = fits.getdata(path(name), memmap=False)
        return True

    def save(name: str) -> bool:
        """Save the external engine's current image."""
        fits.writeto(path(name), image, overwrite=True)
        return True

    def applyreg(name: str, **kwargs: object) -> bool:
        """Produce aligned copies only for members of the supplied sequence."""
        for line in (cwd / f"{name}.seq").read_text().splitlines():
            if line.startswith("I "):
                number = int(line.split()[1])
                source = cwd / f"{name}{number:05d}.fit"
                fits.writeto(
                    cwd / f"r_{source.name}", fits.getdata(source), overwrite=True
                )
        return True

    def rgbcomp(
        r: str | None = None,
        g: str | None = None,
        b: str | None = None,
        lum: str | None = None,
        rgb: str | None = None,
        out: str | None = None,
    ) -> bool:
        """Compose the supplied image files at the external engine boundary."""
        nonlocal image
        if rgb is not None:
            image = fits.getdata(path(rgb), memmap=False)
        else:
            assert r is not None and g is not None and b is not None
            image = np.stack([fits.getdata(path(name)) for name in (r, g, b)])
        assert out is not None
        return save(out)

    def savetif(name: str, **kwargs: object) -> bool:
        """Stand in for the external TIFF encoder."""
        Path(name + ".tif").write_bytes(b"test TIFF encoder")
        return True

    def savejpg(name: str, **kwargs: object) -> bool:
        """Stand in for the external JPEG encoder."""
        Path(name + ".jpg").write_bytes(b"test JPEG encoder")
        return True

    siril.cd.side_effect = cd
    siril.load.side_effect = load
    siril.save.side_effect = save
    siril.seqapplyreg.side_effect = applyreg
    siril.rgbcomp.side_effect = rgbcomp
    siril.savetif.side_effect = savetif
    siril.savejpg.side_effect = savejpg
    return siril


def config(**overrides: object) -> Config:
    """Disable optional expensive operations for boundary regression cases."""
    from siril_job_runner.config import with_overrides

    return with_overrides(
        {
            "starnet_enabled": False,
            "spcc_enabled": False,
            "deconv_enabled": False,
            "diagnostic_previews": False,
            "post_stack_subsky_method": SubskyMethod.NONE,
            "color_removal_mode": ColorRemovalMode.NONE,
            "broadband_neutralization": False,
            "veralux_silentium_enabled": False,
            "veralux_revela_enabled": False,
            "veralux_vectra_enabled": False,
            **overrides,
        }
    )


def write_stacks(output: Path, channels: str, exposure: int = 10) -> list[Path]:
    """Create independent input channels with recognizable values."""
    directory = output / "stacks"
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for index, channel in enumerate(channels, 1):
        path = directory / f"stack_{channel}_{exposure}s.fit"
        fits.writeto(path, np.full((8, 8), 0.1 * index), overwrite=True)
        result.append(path)
    return result


@pytest.mark.parametrize(
    "job_type,channels", [("RGB", "BGR"), ("LRGB", "BGLR"), ("HOO", "HO")]
)
@pytest.mark.parametrize("command", ["register", "seqapplyreg", "load", "rgbcomp"])
def test_composition_failure_cannot_reuse_previous_images(
    tmp_path: Path,
    job_type: str,
    channels: str,
    command: str,
) -> None:
    """Old registered images cannot turn a failed operation into a successful run."""
    inputs = write_stacks(tmp_path, channels)
    registered = tmp_path / "stacks/work/registered"
    registered.mkdir(parents=True)
    for index in range(1, len(channels) + 1):
        fits.writeto(registered / f"r_stack_{index:05d}.fit", np.ones((8, 8)))
    siril = image_siril(tmp_path)
    getattr(siril, command).side_effect = None
    getattr(siril, command).return_value = False
    with pytest.raises(RuntimeError, match=f"{command} failed"):
        compose_and_stretch(siril, tmp_path, job_type, config(palette="HOO"))
    assert not (tmp_path / "rgb_veralux.fit").exists()
    assert all(path.exists() for path in inputs)


def test_missing_hdr_registration_output_never_uses_unaligned_source(
    tmp_path: Path,
) -> None:
    """A successful status must still produce every required aligned stack."""
    write_stacks(tmp_path, "BGR", 10)
    write_stacks(tmp_path, "BGR", 180)
    siril = image_siril(tmp_path)
    siril.seqapplyreg.side_effect = None
    siril.seqapplyreg.return_value = True
    with pytest.raises(FileNotFoundError, match="Missing registered stack"):
        compose_and_stretch(siril, tmp_path, "RGB", config())


@pytest.mark.parametrize("job_type,channels", [("RGB", "BGR"), ("LRGB", "BGLR")])
@pytest.mark.parametrize("command", ["platesolve", "spcc"])
def test_requested_color_calibration_failure_stops_exports(
    tmp_path: Path, job_type: str, channels: str, command: str
) -> None:
    """An enabled calibration cannot silently produce uncalibrated final images."""
    inputs = write_stacks(tmp_path, channels)
    siril = image_siril(tmp_path)
    getattr(siril, command).return_value = False
    with pytest.raises(RuntimeError, match="SPCC"):
        compose_and_stretch(siril, tmp_path, job_type, config(spcc_enabled=True))
    assert not list(tmp_path.glob("*_autostretch*"))
    assert all(path.exists() for path in inputs)


def test_successful_requested_color_calibration_produces_image(tmp_path: Path) -> None:
    """The enabled calibration path still exports a readable calibrated image."""
    write_stacks(tmp_path, "BGLR")
    result = compose_and_stretch(
        image_siril(tmp_path), tmp_path, "LRGB", config(spcc_enabled=True)
    )
    assert result.linear_pcc_path is not None
    assert fits.getdata(result.linear_pcc_path).shape == (3, 8, 8)
    assert result.auto_fit.is_file()


@pytest.mark.parametrize("job_type,channels", [("RGB", "BGR"), ("LRGB", "BGLR")])
def test_full_job_archives_current_images_with_requested_suffix(
    tmp_path: Path,
    job_type: str,
    channels: str,
) -> None:
    """Retired exposure groups cannot enter current, correctly named exports."""
    job_path = tmp_path / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "name": "Test",
                "type": job_type,
                "lights": {"R": ["lights"]},
                "calibration": {
                    "bias": "2026_10_01",
                    "darks": "2026_10_01",
                    "flats": "2026_10_01",
                },
                "output": "processed",
            }
        )
    )
    output = tmp_path / "processed"
    selected = write_stacks(output, channels)
    retired = write_stacks(output, channels, 180)
    # Both exposure groups are valid HDR inputs, with different observable pixels.
    # An accidental HDR run must fail the output assertion, not fixture validation.
    for paths, offset in [(selected, 0.0), (retired, 0.05)]:
        for index, path in enumerate(paths, 1):
            pixels = np.broadcast_to(
                np.linspace(0.1 * index - 0.04, 0.1 * index + 0.04, 32), (32, 32)
            )
            fits.writeto(path, pixels + offset, overwrite=True)
    runner = JobRunner(job_path, tmp_path, siril=image_siril(output))
    runner.config.config = config(output_suffix="review")
    runner.validate = MagicMock(return_value=ValidationResult(True, [], [], [], [], ""))
    runner.run_calibration = MagicMock()
    runner.run_preprocessing = MagicMock(return_value={p.stem: p for p in selected})
    try:
        result = runner.run()
        assert result is not None and result.linear_path is not None
        assert result.auto_fit.name.endswith("_review.fit")
        np.testing.assert_allclose(
            np.median(fits.getdata(result.linear_path), axis=(1, 2)),
            [0.1 * channels.index(c) + 0.1 for c in "RGB"],
        )
        job = ManagedJobOutput(job_path, "Test", job_type, output, tmp_path / "archive")
        _, archived = archive_composition_outputs(job, result)
        np.testing.assert_array_equal(
            fits.getdata(result.auto_fit), fits.getdata(archived.auto_fit)
        )
        assert all(path.exists() for path in retired)
    finally:
        runner.close()


def test_retry_cannot_use_old_siril_star_detections(tmp_path: Path) -> None:
    """Changing an ordinal's source invalidates Siril's persistent star list."""
    write_stacks(tmp_path, "BGR")
    cache = tmp_path / "stacks/work/registered/cache"
    cache.mkdir(parents=True)
    for index in range(1, 4):
        (cache / f"stack_{index:05d}.lst").write_text("previous image coordinates")
    unrelated = cache / "other_sequence.lst"
    unrelated.write_text("keep other sequences")
    siril = image_siril(tmp_path)
    apply_current = siril.seqapplyreg.side_effect

    def apply_cached_alignment(name: str, **kwargs: object) -> bool:
        """Siril reuses cached star coordinates for ordinal filenames on retries."""
        assert callable(apply_current)
        apply_current(name, **kwargs)
        for index in range(1, 4):
            if (cache / f"{name}{index:05d}.lst").exists():
                path = cache.parent / f"r_{name}{index:05d}.fit"
                fits.writeto(path, np.full((8, 8), 0.9), overwrite=True)
        return True

    siril.seqapplyreg.side_effect = apply_cached_alignment
    result = compose_and_stretch(siril, tmp_path, "RGB", config())
    assert result.linear_path is not None
    np.testing.assert_allclose(
        np.median(fits.getdata(result.linear_path), axis=(1, 2)), [0.3, 0.2, 0.1]
    )
    assert unrelated.read_text() == "keep other sequences"
