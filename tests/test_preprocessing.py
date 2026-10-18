"""Calibration validation through the public preprocessing entry point."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from siril_job_runner.config import Config, SubskyMethod
from siril_job_runner.models import FrameInfo
from siril_job_runner.preprocessing import preprocess_with_exposure_groups


@pytest.mark.parametrize("flip", [False, True])
def test_calibration_orientation_preserves_inputs(tmp_path: Path, flip: bool) -> None:
    """Working calibration copies follow the requested orientation."""
    raw = tmp_path / "raw"
    raw.mkdir()
    frames = []
    for index, temperature in enumerate((-10.0, -20.0), 1):
        path = raw / f"light{index}.fit"
        fits.writeto(path, np.full((4, 4), 100.0 * index))
        frames.append(FrameInfo(path, 10.0, temperature, "G", 100))
    bias, flat, warm = [
        tmp_path / name for name in ("bias.fit", "flat.fit", "warm.fit")
    ]
    for path, value in ((bias, 0), (flat, 1), (warm, 1)):
        fits.writeto(path, np.full((4, 4), float(value)) + np.arange(4)[:, None])
    siril = MagicMock()
    cwd = tmp_path

    def cd(path: str) -> bool:
        """Emulate the external process working directory."""
        nonlocal cwd
        cwd = Path(path)
        return True

    def calibrate(name: str, *, dark: str, flat: str) -> bool:
        """Apply each selected frame's supplied dark as Siril would."""
        for line in (cwd / f"{name}.seq").read_text().splitlines():
            if line.startswith("I "):
                _, number, selected = line.split()
                if selected == "1":
                    source = cwd / f"light{int(number):05d}.fit"
                    flat_data = fits.getdata(flat)
                    calibrated = (
                        (fits.getdata(source) - fits.getdata(dark))
                        * np.mean(flat_data)
                        / flat_data
                    )
                    fits.writeto(cwd / f"pp_light{int(number):05d}.fit", calibrated)
        return True

    def stack(*args: object, **kwargs: object) -> bool:
        """Combine all calibrated frames at the external stacking boundary."""
        output = kwargs["out"]
        assert isinstance(output, str)
        values = [fits.getdata(path) for path in sorted(cwd.glob("pp_light*.fit"))]
        fits.writeto(output, np.mean(values, axis=0))
        return True

    def calibration(
        filter_name: str, exposure: float, temperature: float
    ) -> tuple[Path, Path, Path]:
        """Use a common master, as an explicit temperature override does."""
        return bias, warm, flat

    siril.cd.side_effect = cd
    siril.calibrate.side_effect = calibrate
    siril.stack.side_effect = stack
    result = preprocess_with_exposure_groups(
        siril,
        frames,
        tmp_path / "processed",
        calibration,
        Config(
            pre_stack_subsky_method=SubskyMethod.NONE, flip_calibration_vertical=flip
        ),
    )
    assert len(result) == 1
    expected_dark_rows = np.arange(4)
    if flip:
        expected_dark_rows = expected_dark_rows[::-1]
    expected_rows = (149.0 - expected_dark_rows) * 2.5 / (1 + expected_dark_rows)
    expected = np.broadcast_to(expected_rows[:, None], (4, 4))
    np.testing.assert_allclose(fits.getdata(result["stack_G_10s"]), expected)
    np.testing.assert_array_equal(
        fits.getdata(warm), np.ones((4, 4)) + np.arange(4)[:, None]
    )
    assert all(frame.path.exists() for frame in frames)

    np.testing.assert_array_equal(
        fits.getdata(flat), np.ones((4, 4)) + np.arange(4)[:, None]
    )


def test_different_calibration_choices_fail_before_processing(tmp_path: Path) -> None:
    """A mixed group must not silently use only its first frame's calibration."""
    frames = []
    for index, temperature in enumerate((-10.0, -20.0)):
        path = tmp_path / f"input{index}.fit"
        fits.writeto(path, np.full((4, 4), 100.0))
        frames.append(FrameInfo(path, 10.0, temperature, "G", 100))

    def calibration(
        filter_name: str, exposure: float, temperature: float
    ) -> tuple[Path, Path, Path]:
        """Resolve distinct darks for the two temperatures."""
        return (
            tmp_path / "bias.fit",
            tmp_path / f"dark{temperature}.fit",
            tmp_path / "flat.fit",
        )

    output = tmp_path / "processed"
    with pytest.raises(
        ValueError, match="G @ 10s requires different calibration masters"
    ):
        preprocess_with_exposure_groups(MagicMock(), frames, output, calibration)
    assert not output.exists()
    for frame in frames:
        np.testing.assert_array_equal(fits.getdata(frame.path), 100.0)
