"""Combine aligned exposure stacks while retaining unsaturated core detail."""

from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits

from .config import DEFAULTS, Config
from .logger import JobLogger
from .models import StackInfo


def blend_exposures(
    shorter: np.ndarray,
    longer: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Match an unclipped shorter HDR image to a longer exposure and blend.

    Work from shortest to longest. Weights use the unmodified longer image,
    so a recovered core never determines the saturation mask of another image.
    Keep values above one until the final blend is saved.
    """
    if shorter.shape != longer.shape or shorter.ndim != 2:
        raise ValueError("HDR requires aligned, equally sized mono images")
    if not 0 <= low < high <= 1:
        raise ValueError("HDR thresholds must satisfy 0 <= low < high <= 1")
    if not np.isfinite(shorter).all() or not np.isfinite(longer).all():
        raise ValueError("HDR input contains non-finite pixels")
    # Fit short = a * long + b, using the less noisy longer exposure as
    # the reference. Exclude sky noise so it cannot determine the slope.
    stride = max(1, int(np.sqrt(longer.size / 2_000_000)))
    x = shorter[::stride, ::stride].ravel().astype(np.float64)
    y = longer[::stride, ::stride].ravel().astype(np.float64)
    overlap = (x > 0) & (y > 0) & (y < low)
    if np.count_nonzero(overlap) < 100:
        raise ValueError("Too few unsaturated pixels to match HDR exposures")
    background = float(np.median(y[overlap]))
    differences = np.diff(longer[::stride].astype(np.float64), axis=1)[:, ::stride]
    noise = (
        1.4826
        * float(np.median(np.abs(differences - np.median(differences))))
        / np.sqrt(2)
    )
    valid = overlap & (y > background + 5 * noise)
    x, y = x[valid], y[valid]
    if x.size < 100 or np.ptp(y) == 0:
        raise ValueError("Too few unsaturated signal pixels to match HDR exposures")
    inverse_slope, inverse_offset = np.polyfit(y, x, 1)
    if not np.isfinite([inverse_slope, inverse_offset]).all() or inverse_slope <= 0:
        raise ValueError("Invalid HDR exposure match")
    slope, offset = 1 / inverse_slope, -inverse_offset / inverse_slope
    weight = np.clip((longer - low) / (high - low), 0, 1)
    return np.asarray(
        longer * (1 - weight) + (shorter * slope + offset) * weight, dtype=np.float32
    )


class HDRBlender:
    """
    Handles HDR blending of multiple exposures per channel.

    Inputs are already aligned by composition. Blend shortest to longest,
    preserving recovered core values until the final normalization.
    """

    def __init__(
        self,
        config: Config = DEFAULTS,
        logger: Optional[JobLogger] = None,
    ):
        self.config = config
        self.logger = logger

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    def _log_step(self, message: str) -> None:
        if self.logger:
            self.logger.step(message)

    def blend_channel(
        self,
        channel: str,
        stacks: list[StackInfo],
        output_path: Path,
    ) -> Path:
        """
        Blend multiple exposures of a single channel into HDR result.

        Args:
            channel: Channel/filter name (e.g., "L", "R")
            stacks: List of StackInfo for this channel, sorted by exposure
            output_path: Where to save the blended result

        Returns:
            Path to the blended HDR stack
        """
        if len(stacks) < 2:
            if stacks:
                return stacks[0].path
            raise ValueError(f"No stacks provided for channel {channel}")

        self._log_step(f"HDR blending {channel} ({len(stacks)} exposures)")

        sorted_stacks = sorted(stacks, key=lambda s: s.exposure)
        with fits.open(sorted_stacks[0].path) as hdul:
            blended = hdul[0].data.astype(np.float32)
        header = None
        for stack in sorted_stacks[1:]:
            self._log(f"Blending into {stack.exposure}s exposure")
            with fits.open(stack.path) as hdul:
                longer = hdul[0].data.astype(np.float32)
                header = hdul[0].header.copy()
            blended = blend_exposures(
                blended,
                longer,
                self.config.hdr_low_threshold,
                self.config.hdr_high_threshold,
            )
        # Siril's subsequent color operations expect [0, 1]. Normalize once,
        # after every exposure has contributed, preserving recovered highlights.
        blended = np.maximum(blended, 0)
        peak = float(blended.max())
        if peak > 1:
            blended /= peak
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(output_path, blended, header, overwrite=True)

        self._log(f"HDR complete: {output_path.name}")
        return output_path

    def blend_all_channels(
        self,
        stacks_by_channel: dict[str, list[StackInfo]],
        output_dir: Path,
    ) -> dict[str, Path]:
        """
        Blend all channels that have multiple exposures.

        Args:
            stacks_by_channel: Dict mapping channel name to list of StackInfo
            output_dir: Directory for output stacks

        Returns:
            Dict mapping channel name to HDR-blended stack path
            (or original path if only single exposure)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for channel, stacks in stacks_by_channel.items():
            if len(stacks) > 1:
                # Use numbered naming matching LRGB_CHANNEL_INDEX: B=00001, G=00002, L=00003, R=00004
                channel_to_idx = {
                    "B": "00001",
                    "G": "00002",
                    "L": "00003",
                    "R": "00004",
                    "H": "00001",
                    "O": "00002",
                    "S": "00003",
                }  # Also support narrowband
                idx = channel_to_idx.get(channel, channel)
                output_path = output_dir / f"stack_{idx}.fit"
                results[channel] = self.blend_channel(channel, stacks, output_path)
            elif len(stacks) == 1:
                results[channel] = stacks[0].path

        return results
