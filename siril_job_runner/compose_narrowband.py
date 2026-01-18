"""
Narrowband composition for SHO imaging.

Handles composition of H, S, O filter stacks into palette-mapped images.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from .compose_helpers import apply_color_removal
from .config import Config
from .models import CompositionResult, StackInfo
from .psf_analysis import analyze_psf, format_psf_stats
from .stack_discovery import PALETTES
from .stretch_pipeline import StretchPipeline

if TYPE_CHECKING:
    from .protocols import SirilInterface


def compose_narrowband(
    siril: "SirilInterface",
    stacks: dict[str, list[StackInfo]],
    stacks_dir: Path,
    output_dir: Path,
    config: Config,
    stretch_pipeline: StretchPipeline,
    palette: str,
    log_fn: callable,
    log_step_fn: callable,
    log_color_balance_fn: callable,
) -> CompositionResult:
    """
    Compose narrowband image using palette mapping.

    Palettes:
    - HOO: H->R, O->G, O->B
    - SHO: S->R, H->G, O->B
    """
    log_step_fn(f"Composing narrowband ({palette})")

    if palette not in PALETTES:
        raise ValueError(
            f"Unknown palette: {palette}. Available: {list(PALETTES.keys())}"
        )

    mapping = PALETTES[palette]
    required = set(mapping.values())

    for ch in required:
        if ch not in stacks:
            raise ValueError(f"Missing required channel: {ch}")
        if len(stacks[ch]) != 1:
            raise ValueError(
                f"Expected single exposure for {ch}, got {len(stacks[ch])}"
            )

    siril.cd(str(stacks_dir))

    # Register stacks
    log_fn("Cross-registering stacks...")
    siril.convert("stack", out="./registered")
    siril.cd(str(stacks_dir / "registered"))
    siril.register("stack", twopass=True)
    siril.seqapplyreg("stack", framing="min")

    # For HOO: H=00001, O=00002
    # For SHO: H=00001, O=00002, S=00003
    # Determine numbering based on what channels exist
    channels_sorted = sorted(stacks.keys())
    channel_to_num = {ch: f"{i + 1:05d}" for i, ch in enumerate(channels_sorted)}

    # Note: No linear matching for narrowband
    # Linear matching destroys the relative intensities between channels
    # which are needed for proper palette mapping. Each narrowband filter
    # captures different emission lines with different intrinsic brightness.
    log_fn("Saving registered channels (no linear match for narrowband)...")
    for ch in required:
        siril.load(f"r_stack_{channel_to_num[ch]}")
        siril.save(ch)

    # Map channels according to palette
    log_fn(f"Applying {palette} palette...")
    r_src = mapping["R"]
    g_src = mapping["G"]
    b_src = mapping["B"]

    siril.rgbcomp(r=r_src, g=g_src, b=b_src, out="narrowband")

    # Save linear result
    type_name = palette.lower()
    linear_path = output_dir / f"{type_name}_linear.fit"
    siril.load("narrowband")
    siril.save(str(linear_path))
    log_fn(f"Saved linear: {linear_path.name}")
    log_color_balance_fn(linear_path)

    # Note: PCC not applicable for narrowband - uses synthetic colors
    linear_pcc_path = None

    # Color cast removal - especially useful for narrowband
    cfg = config
    stretch_source = str(stacks_dir / "registered" / "narrowband")
    if cfg.color_removal_mode != "none":
        siril.load("narrowband")
        if apply_color_removal(siril, cfg, log_fn):
            siril.save(str(linear_path))
            stretch_source = str(linear_path)
        else:
            log_fn("Color removal failed, continuing without")

    # Optional deconvolution on narrowband composite
    if cfg.deconv_enabled:
        log_fn("Deconvolving narrowband composite...")
        siril.load("narrowband")
        psf_path = (
            str(output_dir / f"psf_{type_name}.fit") if cfg.deconv_save_psf else None
        )
        if siril.makepsf(
            method=cfg.deconv_psf_method,
            symmetric=True,
            save_psf=psf_path,
        ):
            if psf_path:
                log_fn(f"PSF saved: psf_{type_name}.fit")
                psf_stats = analyze_psf(Path(psf_path))
                if psf_stats:
                    for line in format_psf_stats(psf_stats):
                        log_fn(f"  {line}")
            if siril.rl(
                iters=cfg.deconv_iterations,
                regularization=cfg.deconv_regularization,
                alpha=cfg.deconv_alpha,
            ):
                deconv_path = output_dir / f"{type_name}_deconv.fit"
                siril.save(str(deconv_path))
                stretch_source = str(deconv_path)
                log_fn(f"Saved deconvolved: {deconv_path.name}")
            else:
                log_fn("Narrowband deconvolution failed, using original")
        else:
            log_fn("PSF creation failed, skipping deconvolution")

    # Auto-stretch
    auto_paths = stretch_pipeline.auto_stretch(stretch_source, f"{type_name}_auto")

    return CompositionResult(
        linear_path=linear_path,
        linear_pcc_path=linear_pcc_path,
        auto_fit=auto_paths["fit"],
        auto_tif=auto_paths["tif"],
        auto_jpg=auto_paths["jpg"],
        stacks_dir=stacks_dir,
    )
