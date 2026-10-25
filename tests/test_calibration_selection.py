"""Calibration identity and input validation through public processing interfaces."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from siril_job_runner.calibration import CalibrationManager
from siril_job_runner.frame_analysis import build_requirements_table
from siril_job_runner.job_runner import JobRunner
from siril_job_runner.models import CalibrationDates, FrameInfo, ValidationResult
from siril_job_runner.preprocessing import preprocess_with_exposure_groups


def runner_for(tmp_path: Path, lights: dict[str, list[str]]) -> JobRunner:
    """Create a real job runner pointing only at temporary input paths."""
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "name": "Selection",
                "type": "LRGB",
                "lights": lights,
                "calibration": {
                    "bias": "2026_10_01",
                    "darks": "2026_10_01",
                    "flats": "2026_10_01",
                },
                "output": "processed",
            }
        )
    )
    return JobRunner(job, tmp_path, siril=MagicMock())


def write_light(path: Path, exposure: float = 300) -> None:
    """Write a valid, small input with independently specified metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(
        path,
        np.full((8, 8), 0.1, dtype=np.float32),
        fits.Header({"EXPTIME": exposure, "CCD-TEMP": -10.0, "FILTER": "L"}),
    )


def test_raw_dark_fallback_respects_selected_date(tmp_path: Path) -> None:
    """An old set cannot substitute for the requested unavailable date."""
    dates = CalibrationDates("2026_10_01", "2026_10_01", "2026_10_01")
    manager = CalibrationManager(tmp_path, dates)
    old = tmp_path / "calibration/raw/darks/2024_01_01_-10C/300/dark.fit"
    write_light(old)
    status = manager.check_dark(300, -9)
    assert not status.exists and not status.can_build
    current = manager.get_dark_raw_path(300, -10) / "dark.fit"
    write_light(current)
    status = manager.check_dark(300, -9)
    assert status.can_build and status.raw_path == current.parent
    assert old.is_file()


def test_built_dark_keeps_actual_temperature_for_later_matching(tmp_path: Path) -> None:
    """A tolerance match must not widen tolerance when reused on another run."""
    manager = CalibrationManager(
        tmp_path, CalibrationDates("2026_10_01", "2026_10_01", "2026_10_01")
    )
    source = manager.get_dark_raw_path(300, -10) / "dark.fit"
    write_light(source)
    siril = MagicMock()

    def stack(*args: object, **kwargs: object) -> bool:
        """Write the actual source metadata at the external stacking boundary."""
        target = kwargs["out"]
        assert isinstance(target, str)
        fits.writeto(target, fits.getdata(source), fits.getheader(source))
        return True

    siril.stack.side_effect = stack
    built = manager.build_dark_master(300, -8, siril)
    assert fits.getheader(built)["CCD-TEMP"] == -10
    assert manager.check_dark(300, -8).master_path == built
    outside = manager.check_dark(300, -6)
    assert not outside.exists and not outside.can_build
    assert source.is_file()


def test_runner_reuses_each_resolved_requirement_without_second_tolerance(
    tmp_path: Path,
) -> None:
    """Rounded frame temperatures select their own previously resolved master."""
    runner = runner_for(tmp_path, {"L": ["lights"]})
    frames = [
        FrameInfo(tmp_path / "first.fit", 300, -7.9, "L"),
        FrameInfo(tmp_path / "second.fit", 300, -5.9, "L"),
    ]
    requirements = build_requirements_table(frames)
    resolved = {-8: tmp_path / "actual_-10C.fit", -6: tmp_path / "actual_-6C.fit"}
    manager = MagicMock()
    manager.build_dark_master.side_effect = lambda exp, temp, siril: resolved[temp]
    runner.cal_manager = manager
    try:
        runner.run_calibration(ValidationResult(True, frames, requirements, [], [], ""))
        assert runner.get_calibration("L", 300, -7.9)[1] == resolved[-8]
        assert runner.get_calibration("L", 300, -5.9)[1] == resolved[-6]
        assert runner.get_calibration("L", 300, -4)[1] is None
    finally:
        runner.close()


@pytest.mark.parametrize("exposure", [0.2, 0.8, 0.0, -1.0, float("inf"), float("nan")])
def test_unsupported_exposure_fails_before_preprocessing_writes(
    tmp_path: Path, exposure: float
) -> None:
    """Do not reuse, overwrite or create a stack using a truncated exposure."""
    source = tmp_path / "input.fit"
    source.write_bytes(b"input remains intact")
    frames = [FrameInfo(source, exposure, -10, "L")]
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match="positive whole number"):
        preprocess_with_exposure_groups(MagicMock(), frames, output, MagicMock())
    assert not output.exists() and source.read_bytes() == b"input remains intact"


def test_fractional_dark_cannot_reuse_integer_master(tmp_path: Path) -> None:
    """Calibration path selection obeys the same exposure restriction as lights."""
    manager = CalibrationManager(tmp_path, CalibrationDates("d", "d", "d"))
    with pytest.raises(ValueError, match="positive whole number"):
        manager.check_dark(0.8, -10)


@pytest.mark.parametrize("problem", ["missing", "empty", "fractional"])
def test_job_validation_rejects_incomplete_requested_inputs(
    tmp_path: Path, problem: str
) -> None:
    """One valid folder must not hide another missing, empty or unsupported folder."""
    good = tmp_path / "good/light.fit"
    write_light(good)
    if problem == "empty":
        (tmp_path / "bad").mkdir()
    elif problem == "fractional":
        write_light(tmp_path / "bad/light.fit", 0.8)
    runner = runner_for(tmp_path, {"L": ["good", "bad"]})
    try:
        validation = runner.validate()
        assert not validation.valid
        assert "bad" in validation.message
        if problem == "fractional":
            assert "positive whole number" in validation.message
        else:
            assert "No readable light frames" in validation.message
        assert good.is_file()
    finally:
        runner.close()
