"""
Preprocessing module for Siril job processing.

Handles per-channel preprocessing: convert, calibrate, subsky, register, stack.
Supports stacking by exposure for HDR workflows.
"""

import shutil
from pathlib import Path
from typing import Callable, Optional

from .config import DEFAULTS, Config
from .logger import JobLogger
from .models import FrameInfo, StackGroup
from .preprocessing_pipeline import run_pipeline
from .preprocessing_utils import group_frames_by_filter_exposure, link_or_copy
from .protocols import SirilInterface

# Re-export for backwards compatibility
__all__ = [
    "link_or_copy",
    "group_frames_by_filter_exposure",
    "Preprocessor",
    "preprocess_with_exposure_groups",
]


class Preprocessor:
    """Handles preprocessing of light frames."""

    def __init__(
        self,
        siril: SirilInterface,
        config: Config = DEFAULTS,
        logger: Optional[JobLogger] = None,
    ):
        self.siril = siril
        self.config = config
        self.logger = logger

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    def _log_detail(self, message: str) -> None:
        if self.logger:
            self.logger.detail(message)

    def _clean_process_dir(self, process_dir: Path) -> None:
        """Remove old Siril output files from process directory, preserving source/."""
        patterns = ["*.fits", "*.fit", "*.seq", "*.csv"]
        removed = 0
        for pattern in patterns:
            for f in process_dir.glob(pattern):
                if f.parent == process_dir:
                    f.unlink()
                    removed += 1

        for d in process_dir.iterdir():
            if d.is_dir() and d.name != "source":
                shutil.rmtree(d)
                removed += 1

        if removed > 0:
            self._log_detail(
                f"Cleaned {removed} old files/dirs from {process_dir.name}"
            )

    def _is_stack_cached(
        self,
        stack_path: Path,
        frames: list,
        bias_master: Path,
        dark_master: Path,
        flat_master: Path,
    ) -> bool:
        """Check if stack exists and is newer than all sources."""
        if not stack_path.exists():
            return False

        stack_mtime = stack_path.stat().st_mtime

        # Check all source frames
        for frame in frames:
            if frame.path.stat().st_mtime > stack_mtime:
                return False

        # Check calibration masters
        for master in [bias_master, dark_master, flat_master]:
            if master.stat().st_mtime > stack_mtime:
                return False

        return True

    def process_stack_group(
        self,
        group: StackGroup,
        output_dir: Path,
        bias_master: Path,
        dark_master: Path,
        flat_master: Path,
        force: bool = False,
    ) -> Path:
        """
        Process a single stack group (filter + exposure).

        Args:
            force: If True, reprocess even if cached stack exists.

        Returns path to the stacked result.
        """
        stacks_dir = output_dir / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        stack_path = stacks_dir / f"{group.stack_name}.fit"

        # Check cache first
        if not force and self._is_stack_cached(
            stack_path, group.frames, bias_master, dark_master, flat_master
        ):
            if self.logger:
                self.logger.step(
                    f"Using cached {group.filter_name} @ {group.exposure_str} "
                    f"({len(group.frames)} frames)"
                )
            self._log(f"Stack is up-to-date: {stack_path.name}")
            return stack_path

        if self.logger:
            self.logger.step(
                f"Preprocessing {group.filter_name} @ {group.exposure_str} "
                f"({len(group.frames)} frames)"
            )

        process_dir = (
            output_dir / "process" / f"{group.filter_name}_{group.exposure_str}"
        )
        process_dir.mkdir(parents=True, exist_ok=True)

        self._clean_process_dir(process_dir)

        num_frames = self._prepare_frames(group, process_dir)

        stack_path = run_pipeline(
            siril=self.siril,
            num_frames=num_frames,
            process_dir=process_dir,
            stacks_dir=stacks_dir,
            stack_name=group.stack_name,
            bias_master=bias_master,
            dark_master=dark_master,
            flat_master=flat_master,
            config=self.config,
            log_fn=self._log,
            log_detail_fn=self._log_detail,
        )

        return stack_path

    def _prepare_frames(self, group: StackGroup, process_dir: Path) -> int:
        """Link frames to working directory with sequential naming."""
        self._log("Preparing frames...")

        linked_count = 0
        for i, frame in enumerate(group.frames, 1):
            src = frame.path
            if not src.exists():
                raise FileNotFoundError(f"Source frame not found: {src}")

            dest = process_dir / f"light{i:05d}.fit"
            if not dest.exists():
                link_or_copy(src, dest)

            if not dest.exists():
                raise IOError(f"Failed to link/copy frame to: {dest}")
            linked_count += 1

        fit_files = list(process_dir.glob("light*.fit"))
        if not fit_files:
            raise FileNotFoundError(f"No light frames found in {process_dir}")

        self._log_detail(
            f"Prepared {linked_count} frames ({len(fit_files)} files in {process_dir})"
        )
        return linked_count


def preprocess_with_exposure_groups(
    siril: SirilInterface,
    frames: list[FrameInfo],
    output_dir: Path,
    get_calibration: Callable,
    config: Config = DEFAULTS,
    logger: Optional[JobLogger] = None,
) -> dict[str, Path]:
    """
    Preprocess frames, grouping by filter and exposure.

    Args:
        siril: Siril interface
        frames: List of FrameInfo (already scanned)
        output_dir: Output directory
        get_calibration: Callable(filter, exposure, temp) -> (bias, dark, flat) paths
        config: Configuration
        logger: Optional logger

    Returns:
        Dict of stack_name -> stacked result path
        e.g., {"stack_L_180s": Path(...), "stack_L_30s": Path(...)}
    """
    preprocessor = Preprocessor(siril, config, logger)

    groups = group_frames_by_filter_exposure(frames)

    if logger:
        logger.step(f"Processing {len(groups)} stack groups")

    results = {}
    for group in groups:
        temp = group.frames[0].temperature if group.frames else 0.0

        bias, dark, flat = get_calibration(group.filter_name, group.exposure, temp)

        if not all([bias, dark, flat]):
            if logger:
                logger.error(
                    f"Missing calibration for {group.filter_name} @ {group.exposure_str}"
                )
            continue

        stack_path = preprocessor.process_stack_group(
            group=group,
            output_dir=output_dir,
            bias_master=bias,
            dark_master=dark,
            flat_master=flat,
            force=config.force_reprocess,
        )
        results[group.stack_name] = stack_path

    return results
