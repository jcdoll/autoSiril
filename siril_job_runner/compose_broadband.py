"""
Broadband composition for LRGB and RGB imaging.

Handles composition of L, R, G, B filter stacks into color images.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from .compose_helpers import (
    apply_color_removal_step,
    apply_rgb_deconvolution,
    apply_spcc_step,
)
from .config import Config
from .models import CompositionResult, StackInfo
from .psf_analysis import analyze_psf, format_psf_stats
from .stretch_pipeline import StretchPipeline

if TYPE_CHECKING:
    from .protocols import SirilInterface


def compose_lrgb(
    siril: "SirilInterface",
    stacks: dict[str, list[StackInfo]],
    stacks_dir: Path,
    output_dir: Path,
    config: Config,
    stretch_pipeline: StretchPipeline,
    log_fn: callable,
    log_step_fn: callable,
    log_color_balance_fn: callable,
) -> CompositionResult:
    """
    Compose LRGB image from stacked channels.

    Args:
        siril: Siril interface
        stacks: Dict from discover_stacks() - filter name to list of StackInfo
        stacks_dir: Directory containing stacks
        output_dir: Output directory
        config: Configuration
        stretch_pipeline: StretchPipeline instance for final stretching
        log_fn: Logging function for substeps
        log_step_fn: Logging function for main steps
        log_color_balance_fn: Function to log color balance

    Returns CompositionResult with paths to all outputs.
    """
    log_step_fn("Composing LRGB")

    # Verify all required channels (single exposure each)
    for ch in ["L", "R", "G", "B"]:
        if ch not in stacks:
            raise ValueError(f"Missing required channel: {ch}")
        if len(stacks[ch]) != 1:
            raise ValueError(
                f"Expected single exposure for {ch}, got {len(stacks[ch])}"
            )

    siril.cd(str(stacks_dir))

    # Step 1: Register stacks to each other
    # After convert, files are numbered alphabetically: B=1, G=2, L=3, R=4
    log_fn("Cross-registering stacks...")
    siril.convert("stack", out="./registered")
    siril.cd(str(stacks_dir / "registered"))
    siril.register("stack", twopass=True)
    siril.seqapplyreg("stack", framing="min")

    # Step 2: Linear match to reference (R)
    # Alphabetical order: B=00001, G=00002, L=00003, R=00004
    log_fn("Linear matching to R reference...")
    siril.load("r_stack_00004")  # R
    siril.save("R")

    cfg = config
    siril.load("r_stack_00001")  # B
    siril.linear_match("R", cfg.linear_match_low, cfg.linear_match_high)
    siril.save("B")

    siril.load("r_stack_00002")  # G
    siril.linear_match("R", cfg.linear_match_low, cfg.linear_match_high)
    siril.save("G")

    siril.load("r_stack_00003")  # L
    siril.linear_match("R", cfg.linear_match_low, cfg.linear_match_high)
    siril.save("L")

    # Step 3: Optional deconvolution on L
    l_name = "L"
    if cfg.deconv_enabled:
        log_fn("Deconvolving L channel...")
        siril.load("L")
        psf_path = str(output_dir / "psf_L.fit") if cfg.deconv_save_psf else None
        if siril.makepsf(
            method=cfg.deconv_psf_method,
            symmetric=True,
            save_psf=psf_path,
        ):
            if psf_path:
                log_fn("PSF saved: psf_L.fit")
                psf_stats = analyze_psf(Path(psf_path))
                if psf_stats:
                    for line in format_psf_stats(psf_stats):
                        log_fn(f"  {line}")
            if siril.rl(
                iters=cfg.deconv_iterations,
                regularization=cfg.deconv_regularization,
                alpha=cfg.deconv_alpha,
            ):
                siril.save("L_deconv")
                l_name = "L_deconv"
            else:
                log_fn("Deconvolution failed, using original L")
        else:
            log_fn("PSF creation failed, skipping deconvolution")

    # Step 4: Compose RGB
    log_fn("Creating RGB composite...")
    siril.rgbcomp(r="R", g="G", b="B", out="rgb")

    # Step 5: Add luminance
    log_fn("Adding luminance channel...")
    siril.rgbcomp(lum=l_name, rgb="rgb", out="lrgb")

    # Save linear (unstretched) result
    linear_path = output_dir / "lrgb_linear.fit"
    siril.load("lrgb")
    siril.save(str(linear_path))
    log_fn(f"Saved linear: {linear_path.name}")
    log_color_balance_fn(linear_path)

    # Step 6: Spectrophotometric Color Calibration (optional)
    linear_pcc_path, stretch_source = apply_spcc_step(
        siril, cfg, output_dir, "lrgb", log_fn
    )

    # Step 7: Color cast removal
    stretch_source = apply_color_removal_step(siril, cfg, stretch_source, log_fn)

    # Step 8: Optional deconvolution on RGB composite
    stretch_source = apply_rgb_deconvolution(
        siril, cfg, output_dir, stretch_source, "lrgb", log_fn
    )

    # Step 9: Auto-stretch and save
    auto_paths = stretch_pipeline.auto_stretch(stretch_source, "lrgb_auto")

    return CompositionResult(
        linear_path=linear_path,
        linear_pcc_path=linear_pcc_path,
        auto_fit=auto_paths["fit"],
        auto_tif=auto_paths["tif"],
        auto_jpg=auto_paths["jpg"],
        stacks_dir=stacks_dir,
    )


def compose_rgb(
    siril: "SirilInterface",
    stacks: dict[str, list[StackInfo]],
    stacks_dir: Path,
    output_dir: Path,
    config: Config,
    stretch_pipeline: StretchPipeline,
    log_fn: callable,
    log_step_fn: callable,
    log_color_balance_fn: callable,
) -> CompositionResult:
    """
    Compose RGB image (no luminance channel).

    For cases where L channel is not available.
    """
    log_step_fn("Composing RGB")

    # Verify required channels
    for ch in ["R", "G", "B"]:
        if ch not in stacks:
            raise ValueError(f"Missing required channel: {ch}")
        if len(stacks[ch]) != 1:
            raise ValueError(
                f"Expected single exposure for {ch}, got {len(stacks[ch])}"
            )

    siril.cd(str(stacks_dir))

    # Register stacks
    # Alphabetical: B=00001, G=00002, R=00003
    log_fn("Cross-registering stacks...")
    siril.convert("stack", out="./registered")
    siril.cd(str(stacks_dir / "registered"))
    siril.register("stack", twopass=True)
    siril.seqapplyreg("stack", framing="min")

    # Linear match to R
    log_fn("Linear matching to R reference...")
    siril.load("r_stack_00003")  # R
    siril.save("R")

    cfg = config
    siril.load("r_stack_00001")  # B
    siril.linear_match("R", cfg.linear_match_low, cfg.linear_match_high)
    siril.save("B")

    siril.load("r_stack_00002")  # G
    siril.linear_match("R", cfg.linear_match_low, cfg.linear_match_high)
    siril.save("G")

    # Compose RGB
    log_fn("Creating RGB composite...")
    siril.rgbcomp(r="R", g="G", b="B", out="rgb")

    # Save linear result
    linear_path = output_dir / "rgb_linear.fit"
    siril.load("rgb")
    siril.save(str(linear_path))
    log_fn(f"Saved linear: {linear_path.name}")
    log_color_balance_fn(linear_path)

    # Spectrophotometric Color Calibration (optional)
    linear_pcc_path, stretch_source = apply_spcc_step(
        siril, cfg, output_dir, "rgb", log_fn
    )

    # Color cast removal
    stretch_source = apply_color_removal_step(siril, cfg, stretch_source, log_fn)

    # Optional deconvolution on RGB composite
    stretch_source = apply_rgb_deconvolution(
        siril, cfg, output_dir, stretch_source, "rgb", log_fn
    )

    # Auto-stretch
    auto_paths = stretch_pipeline.auto_stretch(stretch_source, "rgb_auto")

    return CompositionResult(
        linear_path=linear_path,
        linear_pcc_path=linear_pcc_path,
        auto_fit=auto_paths["fit"],
        auto_tif=auto_paths["tif"],
        auto_jpg=auto_paths["jpg"],
        stacks_dir=stacks_dir,
    )
