"""Tests for job_config module."""

import json
import tempfile
from pathlib import Path

import pytest

from siril_job_runner.job_config import JobConfig, JobOptions, load_job, validate_job_file


@pytest.fixture
def valid_job_dict():
    """Valid job configuration dict."""
    return {
        "name": "TestJob",
        "type": "LRGB",
        "calibration": {
            "bias": "2024-01-15",
            "darks": "2024-01-15",
            "flats": "2024-01-20",
        },
        "lights": {
            "L": ["M42/L"],
            "R": "M42/R",  # Single string
        },
        "output": "M42/processed",
        "options": {
            "fwhm_filter": 2.0,
            "temp_tolerance": 3,
        },
    }


@pytest.fixture
def job_file(valid_job_dict):
    """Create temporary job file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(valid_job_dict, f)
        return Path(f.name)


def test_job_config_from_dict(valid_job_dict):
    """Test creating JobConfig from dict."""
    config = JobConfig.from_dict(valid_job_dict)

    assert config.name == "TestJob"
    assert config.job_type == "LRGB"
    assert config.calibration_bias == "2024-01-15"
    assert config.calibration_darks == "2024-01-15"
    assert config.calibration_flats == "2024-01-20"
    assert config.output == "M42/processed"


def test_job_config_normalizes_lights(valid_job_dict):
    """Test that single string lights are normalized to lists."""
    config = JobConfig.from_dict(valid_job_dict)

    # Both should be lists
    assert config.lights["L"] == ["M42/L"]
    assert config.lights["R"] == ["M42/R"]


def test_job_config_options(valid_job_dict):
    """Test options parsing."""
    config = JobConfig.from_dict(valid_job_dict)

    assert config.options.fwhm_filter == 2.0
    assert config.options.temp_tolerance == 3
    assert config.options.denoise is False  # Default


def test_job_config_default_options():
    """Test default options when not specified."""
    minimal = {
        "name": "Test",
        "type": "LRGB",
        "calibration": {"bias": "2024-01-01", "darks": "2024-01-01", "flats": "2024-01-01"},
        "lights": {"L": ["L/"]},
        "output": "out",
    }
    config = JobConfig.from_dict(minimal)

    assert config.options.fwhm_filter == 1.8
    assert config.options.temp_tolerance == 2.0
    assert config.options.denoise is False
    assert config.options.palette == "HOO"


def test_load_job_from_file(job_file):
    """Test loading job from file."""
    config = load_job(job_file)
    assert config.name == "TestJob"


def test_validate_job_file_valid(job_file):
    """Test validation of valid job file."""
    is_valid, error = validate_job_file(job_file)
    assert is_valid
    assert error is None


def test_validate_job_file_not_found():
    """Test validation of non-existent file."""
    is_valid, error = validate_job_file(Path("/nonexistent/job.json"))
    assert not is_valid
    assert "not found" in error.lower()


def test_validate_job_file_invalid_json():
    """Test validation of invalid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not valid json")
        path = Path(f.name)

    is_valid, error = validate_job_file(path)
    assert not is_valid
    assert "json" in error.lower()


def test_get_filters(valid_job_dict):
    """Test get_filters method."""
    config = JobConfig.from_dict(valid_job_dict)
    filters = config.get_filters()
    assert set(filters) == {"L", "R"}


def test_get_light_directories(valid_job_dict):
    """Test get_light_directories method."""
    config = JobConfig.from_dict(valid_job_dict)
    dirs = config.get_light_directories("L")
    assert dirs == ["M42/L"]


def test_job_options_defaults():
    """Test JobOptions defaults."""
    opts = JobOptions()
    assert opts.fwhm_filter == 1.8
    assert opts.temp_tolerance == 2.0
    assert opts.denoise is False
    assert opts.palette == "HOO"
