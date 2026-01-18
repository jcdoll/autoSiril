"""
Job validation for Siril job processing.

Handles scanning frames and checking calibration availability.
"""

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .fits_utils import check_clipping, scan_multiple_directories
from .frame_analysis import (
    build_date_summary_table,
    build_requirements_table,
    format_date_summary_table,
    get_unique_filters,
)
from .models import FrameInfo, ValidationResult

if TYPE_CHECKING:
    from .calibration import CalibrationManager
    from .config import Config
    from .logger import JobLogger


def validate_job(
    job_name: str,
    lights: dict[str, list[str]],
    base_path: Path,
    cal_manager: "CalibrationManager",
    config: "Config",
    logger: "JobLogger",
    get_dark_temp: Callable[[float], float],
) -> ValidationResult:
    """
    Validate a job - scan frames and check calibration.

    Args:
        job_name: Name of the job
        lights: Dict of filter_name -> list of directories
        base_path: Base path for resolving relative directories
        cal_manager: CalibrationManager instance
        config: Configuration
        logger: Logger instance
        get_dark_temp: Function to get dark temperature (for override)

    Returns:
        ValidationResult with details
    """
    logger.step(f"Validating job: {job_name}")

    # Scan all light frames
    all_frames: list[FrameInfo] = []
    for filter_name, dirs in lights.items():
        full_dirs = [base_path / d for d in dirs]
        frames = scan_multiple_directories(full_dirs)
        all_frames.extend(frames)
        logger.substep(f"{filter_name}: {len(frames)} frames from {len(dirs)} dirs")

    if not all_frames:
        return ValidationResult(
            valid=False,
            frames=[],
            requirements=[],
            missing_calibration=[],
            buildable_calibration=[],
            message="No light frames found",
        )

    # Check saturation levels by exposure
    by_exposure: dict[float, list[FrameInfo]] = defaultdict(list)
    for frame in all_frames:
        by_exposure[frame.exposure].append(frame)

    logger.step("Clipping check:")
    for exp in sorted(by_exposure.keys(), reverse=True):
        frames_at_exp = by_exposure[exp]
        # Sample frames to check clipping
        sample = frames_at_exp[: min(5, len(frames_at_exp))]
        low_pcts = []
        high_pcts = []
        for frame in sample:
            info = check_clipping(frame.path, config)
            if info:
                low_pcts.append(info.clipped_low_percent)
                high_pcts.append(info.clipped_high_percent)
        if low_pcts and high_pcts:
            avg_low = sum(low_pcts) / len(low_pcts)
            avg_high = sum(high_pcts) / len(high_pcts)
            logger.substep(
                f"{int(exp)}s: {avg_low:.3f}% black, {avg_high:.3f}% white "
                f"({len(frames_at_exp)} frames)"
            )

    # Show date summary table
    date_summary = build_date_summary_table(all_frames)
    all_filters = sorted(get_unique_filters(all_frames))
    # Reorder filters to put common ones first: L, R, G, B, H, S, O
    filter_order = ["L", "R", "G", "B", "H", "S", "O"]
    ordered_filters = [f for f in filter_order if f in all_filters]
    ordered_filters += [f for f in all_filters if f not in filter_order]

    logger.step("Frames by date:")
    for line in format_date_summary_table(date_summary, ordered_filters):
        logger.info(line)

    # Build requirements table
    requirements = build_requirements_table(all_frames)
    logger.step("Requirements:")
    for req in requirements:
        logger.substep(
            f"{req.filter_name}: {req.exposure_str} @ {req.temp_str} ({req.count} frames)"
        )

    # Check calibration availability
    missing = []
    buildable = []

    # Check bias (single, no temperature dependency)
    status = cal_manager.check_bias()
    if not status.exists and not status.can_build:
        missing.append("bias")
    elif not status.exists and status.can_build:
        buildable.append("bias")

    # Check darks for each exposure/temp combo (with override if set)
    dark_combos = {
        (req.exposure, get_dark_temp(req.temperature)) for req in requirements
    }
    for exp, temp in dark_combos:
        status = cal_manager.check_dark(exp, temp)
        if not status.exists and not status.can_build:
            missing.append(f"dark_{int(exp)}s_{int(temp)}C")
        elif not status.exists and status.can_build:
            buildable.append(f"dark_{int(exp)}s_{int(temp)}C")

    # Check flats for each filter
    filters = {req.filter_name for req in requirements}
    for filter_name in filters:
        status = cal_manager.check_flat(filter_name)
        if not status.exists and not status.can_build:
            missing.append(f"flat_{filter_name}")
        elif not status.exists and status.can_build:
            buildable.append(f"flat_{filter_name}")

    # Report status
    logger.step("Calibration status:")
    for name in buildable:
        logger.substep(f"[BUILD] {name}")
    for name in missing:
        logger.substep(f"[MISSING] {name}")

    valid = len(missing) == 0
    message = (
        "Validation passed" if valid else f"Missing {len(missing)} calibration files"
    )

    return ValidationResult(
        valid=valid,
        frames=all_frames,
        requirements=requirements,
        missing_calibration=missing,
        buildable_calibration=buildable,
        message=message,
    )
