"""
Preprocessing module for Siril job processing.

Handles per-channel preprocessing: convert, calibrate, subsky, register, stack.
Supports stacking by exposure for HDR workflows.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from collections import defaultdict
import shutil

from .logger import JobLogger
from .fits_utils import FrameInfo
from .protocols import SirilInterface


@dataclass
class StackGroup:
    """A group of frames to stack together (same filter + exposure)."""

    filter_name: str
    exposure: float
    frames: list[FrameInfo]

    @property
    def exposure_str(self) -> str:
        return f"{int(self.exposure)}s"

    @property
    def stack_name(self) -> str:
        """Name for the output stack file."""
        return f"stack_{self.filter_name}_{self.exposure_str}"


def group_frames_by_filter_exposure(frames: list[FrameInfo]) -> list[StackGroup]:
    """
    Group frames by (filter, exposure) for separate stacking.

    Returns list of StackGroup, sorted by filter then exposure.
    """
    groups: dict[tuple[str, float], list[FrameInfo]] = defaultdict(list)

    for frame in frames:
        key = (frame.filter_name, frame.exposure)
        groups[key].append(frame)

    result = []
    for (filter_name, exposure), frame_list in sorted(groups.items()):
        result.append(StackGroup(
            filter_name=filter_name,
            exposure=exposure,
            frames=frame_list,
        ))

    return result


class Preprocessor:
    """Handles preprocessing of light frames."""

    def __init__(
        self,
        siril: SirilInterface,
        logger: Optional[JobLogger] = None,
        fwhm_filter: float = 1.8,
    ):
        self.siril = siril
        self.logger = logger
        self.fwhm_filter = fwhm_filter

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    def _log_detail(self, message: str) -> None:
        if self.logger:
            self.logger.detail(message)

    def process_stack_group(
        self,
        group: StackGroup,
        output_dir: Path,
        bias_master: Path,
        dark_master: Path,
        flat_master: Path,
    ) -> Path:
        """
        Process a single stack group (filter + exposure).

        Returns path to the stacked result.
        """
        if self.logger:
            self.logger.step(
                f"Preprocessing {group.filter_name} @ {group.exposure_str} "
                f"({len(group.frames)} frames)"
            )

        # Create working directories
        process_dir = output_dir / "process" / f"{group.filter_name}_{group.exposure_str}"
        stacks_dir = output_dir / "stacks"
        process_dir.mkdir(parents=True, exist_ok=True)
        stacks_dir.mkdir(parents=True, exist_ok=True)

        # Copy frames to working directory
        source_dir = self._prepare_frames(group, process_dir)

        # Run preprocessing pipeline
        stack_path = self._run_pipeline(
            seq_name=f"{group.filter_name}_{group.exposure_str}",
            source_dir=source_dir,
            process_dir=process_dir,
            stacks_dir=stacks_dir,
            stack_name=group.stack_name,
            bias_master=bias_master,
            dark_master=dark_master,
            flat_master=flat_master,
        )

        return stack_path

    def _prepare_frames(self, group: StackGroup, process_dir: Path) -> Path:
        """Copy frames to working directory with consistent naming."""
        self._log("Preparing frames...")

        source_dir = process_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(group.frames, 1):
            src = frame.path
            # Use sequential naming for Siril compatibility
            suffix = src.suffix
            dest = source_dir / f"light_{i:04d}{suffix}"
            if not dest.exists():
                shutil.copy2(src, dest)

        self._log_detail(f"Prepared {len(group.frames)} frames")
        return source_dir

    def _run_pipeline(
        self,
        seq_name: str,
        source_dir: Path,
        process_dir: Path,
        stacks_dir: Path,
        stack_name: str,
        bias_master: Path,
        dark_master: Path,
        flat_master: Path,
    ) -> Path:
        """Run the preprocessing pipeline."""

        # Step 1: Convert
        self._log("Converting...")
        self.siril.cd(str(source_dir))
        self.siril.convert("light", out=str(process_dir))

        # Step 2: Calibrate
        self._log("Calibrating...")
        self.siril.cd(str(process_dir))
        self.siril.calibrate(
            "light",
            bias=str(bias_master),
            dark=str(dark_master),
            flat=str(flat_master),
        )

        # Step 3: Background extraction
        self._log("Background extraction...")
        self.siril.seqsubsky("pp_light", 1)

        # Step 4: Registration
        self._log("Registering (2-pass)...")
        self.siril.register("bkg_pp_light", twopass=True)

        # Step 5: Apply registration with FWHM filter
        self._log("Applying registration...")
        self.siril.seqapplyreg(
            "bkg_pp_light",
            filter_fwhm=f"{self.fwhm_filter}k",
        )

        # Step 6: Stack
        self._log("Stacking...")
        stack_path = stacks_dir / f"{stack_name}.fit"
        self.siril.stack(
            "r_bkg_pp_light",
            "rej", "w", "3", "3",
            norm="addscale",
            fastnorm=True,
            out=str(stack_path),
        )

        self._log(f"Complete -> {stack_path.name}")
        return stack_path


def preprocess_with_exposure_groups(
    siril: SirilInterface,
    frames: list[FrameInfo],
    output_dir: Path,
    get_calibration: callable,
    logger: Optional[JobLogger] = None,
    fwhm_filter: float = 1.8,
) -> dict[str, Path]:
    """
    Preprocess frames, grouping by filter and exposure.

    Args:
        siril: Siril interface
        frames: List of FrameInfo (already scanned)
        output_dir: Output directory
        get_calibration: Callable(filter, exposure, temp) -> (bias, dark, flat) paths
        logger: Optional logger
        fwhm_filter: FWHM filter factor

    Returns:
        Dict of stack_name -> stacked result path
        e.g., {"stack_L_180s": Path(...), "stack_L_30s": Path(...)}
    """
    preprocessor = Preprocessor(siril, logger, fwhm_filter)

    # Group frames by filter + exposure
    groups = group_frames_by_filter_exposure(frames)

    if logger:
        logger.step(f"Processing {len(groups)} stack groups")

    results = {}
    for group in groups:
        # Get representative temperature (use first frame)
        temp = group.frames[0].temperature if group.frames else 0.0

        # Get calibration paths for this group
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
        )
        results[group.stack_name] = stack_path

    return results


# Keep old function for backwards compatibility but mark deprecated
def preprocess_all_filters(
    siril: SirilInterface,
    filters_config: dict[str, list[Path]],
    output_dir: Path,
    calibration_paths: dict[str, dict[str, Path]],
    logger: Optional[JobLogger] = None,
    fwhm_filter: float = 1.8,
) -> dict[str, Path]:
    """
    DEPRECATED: Use preprocess_with_exposure_groups instead.

    This function doesn't support HDR/multi-exposure workflows.
    """
    if logger:
        logger.warning("Using deprecated preprocess_all_filters - HDR not supported")

    # Import here to avoid circular import
    from .fits_utils import scan_multiple_directories

    # Scan frames
    all_frames = []
    for filter_name, light_dirs in filters_config.items():
        frames = scan_multiple_directories([Path(d) for d in light_dirs])
        all_frames.extend(frames)

    def get_cal(filter_name, exposure, temp):
        cal = calibration_paths.get(filter_name, {})
        return cal.get("bias"), cal.get("dark"), cal.get("flat")

    return preprocess_with_exposure_groups(
        siril=siril,
        frames=all_frames,
        output_dir=output_dir,
        get_calibration=get_cal,
        logger=logger,
        fwhm_filter=fwhm_filter,
    )
