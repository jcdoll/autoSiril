"""
Calibration path resolution and status checking.

Provides methods for locating calibration files and checking their status.
"""

from pathlib import Path
from typing import Optional

from .config import Config
from .fits_utils import temperatures_match
from .models import CalibrationDates, CalibrationStatus


class CalibrationPathsMixin:
    """Mixin for calibration path resolution and status checking."""

    base_path: Path
    dates: CalibrationDates
    config: Config
    calibration_dir: Path
    masters_dir: Path
    raw_dir: Path

    # Path resolution

    def get_bias_master_path(self) -> Path:
        """Get expected path for bias master."""
        cfg = self.config
        filename = f"{cfg.bias_prefix}{self.dates.bias}{cfg.fit_extension}"
        return self.masters_dir / cfg.bias_subdir / filename

    def get_dark_master_path(self, exposure: float, temp: float) -> Path:
        """Get expected path for dark master."""
        cfg = self.config
        exp_str = f"{int(exposure)}{cfg.exposure_suffix}"
        temp_str = f"{int(round(temp))}{cfg.temp_suffix}"
        filename = f"{cfg.dark_prefix}{exp_str}_{temp_str}_{self.dates.darks}{cfg.fit_extension}"
        return self.masters_dir / cfg.dark_subdir / filename

    def get_flat_master_path(self, filter_name: str) -> Path:
        """Get expected path for flat master."""
        cfg = self.config
        filename = (
            f"{cfg.flat_prefix}{filter_name}_{self.dates.flats}{cfg.fit_extension}"
        )
        return self.masters_dir / cfg.flat_subdir / filename

    def get_bias_raw_path(self) -> Path:
        """Get expected path for raw bias frames."""
        return self.raw_dir / self.config.bias_subdir / self.dates.bias

    def get_dark_raw_path(self, exposure: float, temp: float) -> Path:
        """Get expected path for raw dark frames."""
        cfg = self.config
        exp_str = f"{int(exposure)}"
        temp_str = f"{int(round(temp))}{cfg.temp_suffix}"
        # Structure: darks/{date}_{temp}/{exposure}/
        return (
            self.raw_dir / cfg.dark_subdir / f"{self.dates.darks}_{temp_str}" / exp_str
        )

    def get_flat_raw_path(self, filter_name: str) -> Path:
        """Get expected path for raw flat frames."""
        return self.raw_dir / self.config.flat_subdir / self.dates.flats / filter_name

    # Status checking

    def check_bias(self) -> CalibrationStatus:
        """Check if bias master exists or can be built."""
        master_path = self.get_bias_master_path()
        raw_path = self.get_bias_raw_path()

        if master_path.exists():
            return CalibrationStatus(
                exists=True,
                can_build=True,
                master_path=master_path,
                raw_path=None,
                message="Master exists",
            )

        if raw_path.exists() and any(raw_path.glob(self.config.fit_glob)):
            return CalibrationStatus(
                exists=False,
                can_build=True,
                master_path=master_path,
                raw_path=raw_path,
                message="Can build from raw",
            )

        return CalibrationStatus(
            exists=False,
            can_build=False,
            master_path=master_path,
            raw_path=raw_path,
            message=f"No master or raw frames at {raw_path}",
        )

    def check_dark(self, exposure: float, temp: float) -> CalibrationStatus:
        """Check if dark master exists or can be built (with temperature tolerance)."""
        master_path = self.get_dark_master_path(exposure, temp)
        raw_path = self.get_dark_raw_path(exposure, temp)

        if master_path.exists():
            return CalibrationStatus(
                exists=True,
                can_build=True,
                master_path=master_path,
                raw_path=None,
                message="Master exists",
            )

        if raw_path.exists() and any(raw_path.glob(self.config.fit_glob)):
            return CalibrationStatus(
                exists=False,
                can_build=True,
                master_path=master_path,
                raw_path=raw_path,
                message="Can build from raw",
            )

        # Try temperature tolerance matching for existing masters
        matching_master = self.find_matching_dark(exposure, temp)
        if matching_master:
            return CalibrationStatus(
                exists=True,
                can_build=True,
                master_path=matching_master,
                raw_path=None,
                message=f"Using tolerance-matched master: {matching_master.name}",
            )

        # Try tolerance matching for raw frames
        matching_raw = self._find_matching_dark_raw(exposure, temp)
        if matching_raw:
            return CalibrationStatus(
                exists=False,
                can_build=True,
                master_path=master_path,
                raw_path=matching_raw,
                message=f"Can build from tolerance-matched raw: {matching_raw}",
            )

        return CalibrationStatus(
            exists=False,
            can_build=False,
            master_path=master_path,
            raw_path=raw_path,
            message=f"No master or raw frames at {raw_path}",
        )

    def _find_matching_dark_raw(self, exposure: float, temp: float) -> Optional[Path]:
        """Find raw dark frames within temperature tolerance."""
        cfg = self.config
        darks_raw_dir = self.raw_dir / cfg.dark_subdir
        if not darks_raw_dir.exists():
            return None

        exp_str = f"{int(exposure)}"
        for temp_dir in darks_raw_dir.iterdir():
            if not temp_dir.is_dir():
                continue
            # Parse temp from dir name like "2025_01_23_-10C"
            parts = temp_dir.name.split("_")
            if len(parts) >= 2:
                temp_part = parts[-1]  # Last part should be temp like "-10C"
                try:
                    dir_temp = float(temp_part.replace(cfg.temp_suffix, ""))
                    if temperatures_match(temp, dir_temp, cfg.temp_tolerance):
                        exp_path = temp_dir / exp_str
                        if exp_path.exists() and any(exp_path.glob(cfg.fit_glob)):
                            return exp_path
                except ValueError:
                    continue
        return None

    def check_flat(self, filter_name: str) -> CalibrationStatus:
        """Check if flat master exists or can be built."""
        master_path = self.get_flat_master_path(filter_name)
        raw_path = self.get_flat_raw_path(filter_name)

        if master_path.exists():
            return CalibrationStatus(
                exists=True,
                can_build=True,
                master_path=master_path,
                raw_path=None,
                message="Master exists",
            )

        if raw_path.exists() and any(raw_path.glob(self.config.fit_glob)):
            return CalibrationStatus(
                exists=False,
                can_build=True,
                master_path=master_path,
                raw_path=raw_path,
                message="Can build from raw",
            )

        return CalibrationStatus(
            exists=False,
            can_build=False,
            master_path=master_path,
            raw_path=raw_path,
            message=f"No master or raw frames at {raw_path}",
        )

    def find_matching_dark(self, exposure: float, temp: float) -> Optional[Path]:
        """
        Find a dark master matching exposure and temperature (with tolerance).

        Returns the master path if found, None otherwise.
        """
        master_path = self.get_dark_master_path(exposure, temp)
        if master_path.exists():
            return master_path

        cfg = self.config
        darks_dir = self.masters_dir / cfg.dark_subdir
        if not darks_dir.exists():
            return None

        exp_str = f"{int(exposure)}{cfg.exposure_suffix}"
        glob_pattern = (
            f"{cfg.dark_prefix}{exp_str}_*_{self.dates.darks}{cfg.fit_extension}"
        )
        for master in darks_dir.glob(glob_pattern):
            parts = master.stem.split("_")
            if len(parts) >= 3:
                temp_part = parts[2]  # e.g., "-10C"
                try:
                    master_temp = float(temp_part.replace(cfg.temp_suffix, ""))
                    if temperatures_match(temp, master_temp, cfg.temp_tolerance):
                        return master
                except ValueError:
                    continue

        return None
