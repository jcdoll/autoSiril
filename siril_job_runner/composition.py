"""
Composition module for Siril job processing.

Handles LRGB composition, narrowband palette mixing, and stretching.
Based on LRGB_pre.ssf and LRGB_compose.ssf workflows.
"""

from pathlib import Path
from typing import Optional

from .compose_broadband import compose_lrgb, compose_rgb
from .compose_narrowband import compose_narrowband
from .config import DEFAULTS, Config
from .fits_utils import check_color_balance
from .hdr import HDRBlender
from .logger import JobLogger
from .models import CompositionResult, StackInfo
from .protocols import SirilInterface
from .stack_discovery import PALETTES, discover_stacks, is_hdr_mode
from .stretch_pipeline import StretchPipeline

# Re-export for backwards compatibility
__all__ = [
    "PALETTES",
    "discover_stacks",
    "is_hdr_mode",
    "Composer",
    "compose_and_stretch",
]


class Composer:
    """
    Handles image composition and stretching.

    Workflow based on LRGB_pre.ssf:
    1. Register all stacks to each other
    2. Linear match all channels to reference (R)
    3. Optional: Deconvolve L channel
    4. rgbcomp R G B -> rgb
    5. rgbcomp -lum=L rgb -> lrgb (for LRGB)
    6. autostretch + mtf + satu for auto output
    """

    def __init__(
        self,
        siril: SirilInterface,
        output_dir: Path,
        config: Config = DEFAULTS,
        logger: Optional[JobLogger] = None,
    ):
        self.siril = siril
        self.output_dir = Path(output_dir)
        self.config = config
        self.logger = logger
        self.stacks_dir = self.output_dir / "stacks"
        self._stretch_pipeline = StretchPipeline(siril, output_dir, config, self._log)

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    def _log_step(self, message: str) -> None:
        if self.logger:
            self.logger.step(message)

    def _log_color_balance(self, image_path: Path) -> None:
        """Check and log color balance for an RGB image."""
        balance = check_color_balance(image_path)
        if balance is None:
            return

        self._log(
            f"Color balance: R={balance.r_median:.1f}, "
            f"G={balance.g_median:.1f}, B={balance.b_median:.1f}"
        )

        if balance.is_imbalanced:
            self._log(
                f"WARNING: Color imbalance detected! "
                f"{balance.dominant_channel} dominates by {balance.dominance_ratio:.1f}x"
            )

    def compose_lrgb(
        self,
        stacks: dict[str, list[StackInfo]],
    ) -> CompositionResult:
        """Compose LRGB image from stacked channels."""
        return compose_lrgb(
            siril=self.siril,
            stacks=stacks,
            stacks_dir=self.stacks_dir,
            output_dir=self.output_dir,
            config=self.config,
            stretch_pipeline=self._stretch_pipeline,
            log_fn=self._log,
            log_step_fn=self._log_step,
            log_color_balance_fn=self._log_color_balance,
        )

    def compose_rgb(
        self,
        stacks: dict[str, list[StackInfo]],
    ) -> CompositionResult:
        """Compose RGB image (no luminance channel)."""
        return compose_rgb(
            siril=self.siril,
            stacks=stacks,
            stacks_dir=self.stacks_dir,
            output_dir=self.output_dir,
            config=self.config,
            stretch_pipeline=self._stretch_pipeline,
            log_fn=self._log,
            log_step_fn=self._log_step,
            log_color_balance_fn=self._log_color_balance,
        )

    def compose_narrowband(
        self,
        stacks: dict[str, list[StackInfo]],
        palette: str = "HOO",
    ) -> CompositionResult:
        """Compose narrowband image using palette mapping."""
        return compose_narrowband(
            siril=self.siril,
            stacks=stacks,
            stacks_dir=self.stacks_dir,
            output_dir=self.output_dir,
            config=self.config,
            stretch_pipeline=self._stretch_pipeline,
            palette=palette,
            log_fn=self._log,
            log_step_fn=self._log_step,
            log_color_balance_fn=self._log_color_balance,
        )


def compose_and_stretch(
    siril: SirilInterface,
    output_dir: Path,
    job_type: str,
    palette: str = "HOO",
    config: Config = DEFAULTS,
    logger: Optional[JobLogger] = None,
) -> CompositionResult:
    """
    Discover stacks and compose based on job type.

    Automatically handles HDR mode by blending multiple exposures per channel
    before composition.

    Args:
        siril: Siril interface
        output_dir: Output directory (contains stacks/ subdirectory)
        job_type: "LRGB", "RGB", "SHO", or "HOO"
        palette: Narrowband palette (for SHO/HOO)
        config: Configuration with processing parameters
        logger: Optional logger

    Returns:
        CompositionResult with paths to all outputs
    """
    stacks_dir = Path(output_dir) / "stacks"
    stacks = discover_stacks(stacks_dir)

    if not stacks:
        raise FileNotFoundError(f"No stacks found in {stacks_dir}")

    if logger:
        logger.step("Discovered stacks:")
        for _filter_name, stack_list in sorted(stacks.items()):
            for s in stack_list:
                logger.substep(f"{s.name}.fit")

    # Check for HDR mode and blend if needed
    if is_hdr_mode(stacks):
        if logger:
            logger.step("Multiple exposures detected - HDR blending")

        blender = HDRBlender(siril, output_dir, config, logger)

        # Blend all channels
        hdr_stacks_dir = stacks_dir / "hdr"
        hdr_stacks_dir.mkdir(parents=True, exist_ok=True)

        blended_paths = blender.blend_all_channels(stacks, hdr_stacks_dir)

        # Convert back to single-exposure stacks dict for composition
        # Use exposure=0 to indicate HDR blend (not a real exposure time)
        stacks = {}
        for channel, path in blended_paths.items():
            stacks[channel] = [StackInfo(path=path, filter_name=channel, exposure=0)]

        if logger:
            logger.step("HDR blending complete")

    composer = Composer(siril, output_dir, config, logger)

    if job_type == "LRGB":
        return composer.compose_lrgb(stacks)
    elif job_type == "RGB":
        return composer.compose_rgb(stacks)
    elif job_type in ("SHO", "HOO"):
        return composer.compose_narrowband(stacks, palette=palette or job_type)
    else:
        raise ValueError(f"Unknown job type: {job_type}")
