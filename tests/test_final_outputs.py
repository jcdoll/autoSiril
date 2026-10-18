"""Final output failures must be visible to the caller."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from siril_job_runner.stretch_helpers import save_all_formats


@pytest.mark.parametrize("failure", ["save", "load", "savetif", "savejpg"])
def test_failed_export_is_not_reported_as_success(tmp_path: Path, failure: str) -> None:
    """A failed external save or reload must fail the processing operation."""
    siril = MagicMock()
    fits.writeto(tmp_path / "final.fit", np.ones((4, 4)))
    getattr(siril, failure).return_value = False
    with pytest.raises(RuntimeError, match="Failed"):
        save_all_formats(siril, tmp_path, "final", lambda message: None)


def test_nonfinite_final_image_is_rejected(tmp_path: Path) -> None:
    """Successful command status cannot hide corrupt image pixels."""
    fits.writeto(tmp_path / "final.fit", np.full((4, 4), np.nan))
    with pytest.raises(ValueError, match="Invalid final FITS"):
        save_all_formats(MagicMock(), tmp_path, "final", lambda message: None)


def test_missing_export_is_rejected(tmp_path: Path) -> None:
    """Success status without the promised file is still an export failure."""
    fits.writeto(tmp_path / "final.fit", np.ones((4, 4)))
    with pytest.raises(ValueError, match="Missing or empty"):
        save_all_formats(MagicMock(), tmp_path, "final", lambda message: None)
