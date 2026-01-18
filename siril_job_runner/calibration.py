"""
Calibration file management for Siril job processing.

Handles finding, building, and caching master calibration files.
"""

from pathlib import Path
from typing import Optional

from .calibration_paths import CalibrationPathsMixin
from .config import DEFAULTS, Config
from .logger import JobLogger
from .models import CalibrationDates, CalibrationStatus
from .protocols import SirilInterface

# Re-export for backwards compatibility
__all__ = ["CalibrationManager", "CalibrationDates", "CalibrationStatus"]


class CalibrationManager(CalibrationPathsMixin):
    """Manages calibration file finding and building."""

    def __init__(
        self,
        base_path: Path,
        dates: CalibrationDates,
        config: Config = DEFAULTS,
        logger: Optional[JobLogger] = None,
    ):
        self.base_path = Path(base_path)
        self.dates = dates
        self.config = config
        self.logger = logger
        self.calibration_dir = self.base_path / config.calibration_base_dir
        self.masters_dir = self.calibration_dir / config.calibration_masters_dir
        self.raw_dir = self.calibration_dir / config.calibration_raw_dir

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    # Building masters

    def build_bias_master(self, siril: SirilInterface) -> Path:
        """Build bias master from raw frames."""
        status = self.check_bias()
        if status.exists:
            return status.master_path

        if not status.can_build:
            raise ValueError(status.message)

        self._log(f"Building bias master: {status.master_path.name}")

        status.master_path.parent.mkdir(parents=True, exist_ok=True)

        raw_files = list(status.raw_path.glob(self.config.fit_glob))
        if not raw_files:
            raise FileNotFoundError(f"No bias frames found in {status.raw_path}")

        cfg = self.config
        seq_name = "bias"
        if not siril.cd(str(status.raw_path)):
            raise RuntimeError(f"Failed to cd to bias raw path: {status.raw_path}")
        if not siril.convert(seq_name, out=cfg.process_dir):
            raise RuntimeError(f"Failed to convert bias frames in {status.raw_path}")
        if not siril.cd(str(status.raw_path / "process")):
            raise RuntimeError("Failed to cd to bias process path")
        if not siril.stack(
            seq_name,
            cfg.cal_rejection,
            cfg.cal_sigma,
            cfg.cal_sigma,
            cfg.cal_no_norm,
            out=str(status.master_path),
        ):
            raise RuntimeError("Failed to stack bias frames")

        if not status.master_path.exists():
            raise FileNotFoundError(f"Bias master not created: {status.master_path}")

        return status.master_path

    def build_dark_master(
        self, exposure: float, temp: float, siril: SirilInterface
    ) -> Path:
        """Build dark master from raw frames."""
        status = self.check_dark(exposure, temp)
        if status.exists:
            return status.master_path

        if not status.can_build:
            raise ValueError(status.message)

        self._log(f"Building dark master: {status.master_path.name}")

        status.master_path.parent.mkdir(parents=True, exist_ok=True)

        raw_files = list(status.raw_path.glob(self.config.fit_glob))
        if not raw_files:
            raise FileNotFoundError(f"No dark frames found in {status.raw_path}")

        cfg = self.config
        seq_name = "dark"
        if not siril.cd(str(status.raw_path)):
            raise RuntimeError(f"Failed to cd to dark raw path: {status.raw_path}")
        if not siril.convert(seq_name, out=cfg.process_dir):
            raise RuntimeError(f"Failed to convert dark frames in {status.raw_path}")
        if not siril.cd(str(status.raw_path / "process")):
            raise RuntimeError("Failed to cd to dark process path")
        if not siril.stack(
            seq_name,
            cfg.cal_rejection,
            cfg.cal_sigma,
            cfg.cal_sigma,
            cfg.cal_no_norm,
            out=str(status.master_path),
        ):
            raise RuntimeError("Failed to stack dark frames")

        if not status.master_path.exists():
            raise FileNotFoundError(f"Dark master not created: {status.master_path}")

        return status.master_path

    def build_flat_master(
        self, filter_name: str, bias_path: Path, siril: SirilInterface
    ) -> Path:
        """Build flat master from raw frames (calibrated with bias)."""
        status = self.check_flat(filter_name)
        if status.exists:
            return status.master_path

        if not status.can_build:
            raise ValueError(status.message)

        self._log(f"Building flat master: {status.master_path.name}")

        status.master_path.parent.mkdir(parents=True, exist_ok=True)

        raw_files = list(status.raw_path.glob(self.config.fit_glob))
        if not raw_files:
            raise FileNotFoundError(f"No flat frames found in {status.raw_path}")
        if not bias_path.exists():
            raise FileNotFoundError(f"Bias master not found: {bias_path}")

        cfg = self.config
        if not siril.cd(str(status.raw_path)):
            raise RuntimeError(f"Failed to cd to flat raw path: {status.raw_path}")
        if not siril.convert(filter_name, out=cfg.process_dir):
            raise RuntimeError(f"Failed to convert flat frames in {status.raw_path}")
        if not siril.cd(str(status.raw_path / "process")):
            raise RuntimeError("Failed to cd to flat process path")
        if not siril.calibrate(filter_name, bias=str(bias_path)):
            raise RuntimeError("Failed to calibrate flat frames with bias")
        calibrated_seq = f"{cfg.calibrated_prefix}{filter_name}"
        if not siril.stack(
            calibrated_seq,
            cfg.cal_rejection,
            cfg.cal_sigma,
            cfg.cal_sigma,
            cfg.cal_flat_norm,
            out=str(status.master_path),
        ):
            raise RuntimeError("Failed to stack flat frames")

        if not status.master_path.exists():
            raise FileNotFoundError(f"Flat master not created: {status.master_path}")

        return status.master_path
