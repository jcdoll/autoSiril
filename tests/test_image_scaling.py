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
        result = hdus[0].data / 65535.0
    assert np.isfinite(result).all()
    assert 0.06 < np.median(result[:, 32:96, 32:96]) < 0.10
