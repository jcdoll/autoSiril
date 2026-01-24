"""
Narrowband composition for SHO imaging.

Handles composition of H, S, O filter stacks into palette-mapped images.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .compose_helpers import (
    apply_color_removal,
    apply_deconvolution,
    neutralize_rgb_background,
    save_diagnostic_preview,
)
from .config import Config
from .models import CompositionResult, StackInfo
from .palettes import build_effective_palette, formula_to_pixelmath, get_palette
from .siril_file_ops import link_or_copy
from .stretch_helpers import (
    apply_saturation,
    apply_stretch,
    stretch_and_finalize,
)

if TYPE_CHECKING:
    from .protocols import SirilInterface


def _process_dynamic_palette(
    siril: "SirilInterface",
    cfg: Config,
    palette,
    has_luminance: bool,
    narrowband_channels: list[str],
    registered_dir: Path,
    output_dir: Path,
    type_name: str,
    apply_palette_fn: Callable[[], None],
    log_fn: Callable[[str], None],
) -> dict[str, Path]:
    """
    Process dynamic palette: stretch per-channel, apply scale expressions, then palette.

    Dynamic palettes require non-linear (stretched) data for their formulas to work
    correctly. The workflow is:
    1. Stretch each channel independently
    2. Optionally separate stars (to allow aggressive nebula processing)
    3. Apply channel scale expressions (e.g., O boost for bicolor)
    4. Apply palette formulas to create RGB
    5. Neutralize RGB background
    6. Composite stars back (if separated)
    7. Apply saturation and save

    Returns dict with 'fit', 'tif', 'jpg' paths for the primary output.
    """
    methods = (
        ["autostretch", "veralux"] if cfg.stretch_compare else [cfg.stretch_method]
    )
    primary_paths = None

    # All channels to process (narrowband + L if present)
    all_channels = list(narrowband_channels)
    if has_luminance:
        all_channels.append("L")

    for method in methods:
        suffix = f"_{method}" if cfg.stretch_compare else ""
        log_fn(f"Dynamic palette: processing with {method} stretch...")

        # Step 1: Stretch each channel independently
        for ch in all_channels:
            siril.load(f"{ch}_linear")
            ch_path = registered_dir / f"{ch}_linear.fit"
            apply_stretch(siril, method, ch_path, cfg, log_fn)
            siril.save(ch)
            log_fn(f"  {ch}: stretched ({method})")

        # Step 2: Star separation (optional)
        # Allows aggressive channel scaling on nebula without affecting star colors
        star_source_ch = None
        if cfg.narrowband_star_separation:
            star_source_ch = _separate_stars(
                siril, cfg, has_luminance, all_channels, registered_dir, log_fn
            )

        # Determine channel name prefix (starless_ if star separation enabled)
        use_starless = cfg.narrowband_star_separation
        ch_prefix = "starless_" if use_starless else ""

        # Step 3: Apply channel scale expressions
        # Non-linear scaling (e.g., O + k*O^3) to boost signal while preserving ratios
        _apply_scale_expressions(siril, cfg, narrowband_channels, ch_prefix, log_fn)

        # Step 4: Apply palette formulas
        if use_starless:
            # Copy starless channels to expected names for palette function
            for ch in narrowband_channels:
                siril.load(f"starless_{ch}")
                siril.save(ch)
            if has_luminance:
                siril.load("starless_L")
                siril.save("L")

        apply_palette_fn()

        # Step 5: Neutralize RGB background
        if cfg.post_stack_subsky_method != "none":
            neutralize_rgb_background(siril, "narrowband", cfg, log_fn)

        # Step 6: Composite stars back (if separated)
        if cfg.narrowband_star_separation and star_source_ch:
            _composite_stars(siril, cfg, star_source_ch, log_fn)

        # Step 7: Apply saturation and save outputs
        siril.load("narrowband")
        apply_saturation(siril, cfg)

        base_name = f"{type_name}_auto{suffix}"
        auto_fit = output_dir / f"{base_name}.fit"
        auto_tif = output_dir / f"{base_name}.tif"
        auto_jpg = output_dir / f"{base_name}.jpg"
        siril.save(str(output_dir / base_name))
        siril.load(str(auto_fit))
        siril.savetif(str(output_dir / base_name))
        siril.savejpg(str(output_dir / base_name), quality=90)
        log_fn(f"  Saved: {auto_fit.name}, {auto_tif.name}, {auto_jpg.name}")

        if primary_paths is None:
            primary_paths = {"fit": auto_fit, "tif": auto_tif, "jpg": auto_jpg}

    return primary_paths


def _separate_stars(
    siril: "SirilInterface",
    cfg: Config,
    has_luminance: bool,
    all_channels: list[str],
    registered_dir: Path,
    log_fn: Callable[[str], None],
) -> str | None:
    """
    Separate stars from all channels using StarNet.

    Returns the channel name used as star source (for later compositing),
    or None if star separation failed.
    """
    log_fn("  Star separation enabled, running StarNet...")

    # Determine star source: L if available, else H, or configured channel
    if cfg.narrowband_star_source == "auto":
        star_source_ch = "L" if has_luminance else "H"
    else:
        star_source_ch = cfg.narrowband_star_source

    # Run StarNet on all channels
    for ch in all_channels:
        siril.load(ch)
        if siril.starnet(stretch=False):  # Already stretched
            siril.save(f"starless_{ch}")
            # StarNet creates starmask_<filename>.fit in the working directory
            starmask_path = registered_dir / f"starmask_{ch}.fit"
            if starmask_path.exists() and ch == star_source_ch:
                siril.load(str(starmask_path))
                siril.save("stars_source")
                log_fn(f"    {ch}: starless saved, stars extracted (source)")
            else:
                log_fn(f"    {ch}: starless saved")
        else:
            log_fn(f"    {ch}: StarNet failed, using original")
            siril.load(ch)
            siril.save(f"starless_{ch}")

    return star_source_ch


def _apply_scale_expressions(
    siril: "SirilInterface",
    cfg: Config,
    narrowband_channels: list[str],
    ch_prefix: str,
    log_fn: Callable[[str], None],
) -> None:
    """
    Apply channel scale expressions before palette formulas.

    Scale expressions allow non-linear channel manipulation (e.g., O + k*O^3)
    to boost signal in a way that survives linear background neutralization.
    The non-linear component preserves ratio changes even after linear correction.
    """
    scale_exprs = {
        "H": cfg.palette_h_scale_expr,
        "O": cfg.palette_o_scale_expr,
        "S": cfg.palette_s_scale_expr,
    }
    for ch in narrowband_channels:
        expr = scale_exprs.get(ch)
        if expr:
            ch_name = f"{ch_prefix}{ch}"
            siril.load(ch_name)
            pm_expr = formula_to_pixelmath(expr)
            siril.pm(pm_expr)
            siril.save(ch_name)
            log_fn(f"  {ch}: scaled with {expr}")


def _composite_stars(
    siril: "SirilInterface",
    cfg: Config,
    star_source_ch: str,
    log_fn: Callable[[str], None],
) -> None:
    """
    Composite stars back onto the processed narrowband image.

    Stars are blended using max() for a simple additive-like effect that
    preserves star brightness without over-brightening overlapping regions.
    """
    log_fn("  Compositing stars back...")
    siril.load("narrowband")
    siril.save("narrowband_starless")

    # Prepare stars: mono (grayscale) creates white stars
    siril.load("stars_source")
    if cfg.narrowband_star_color == "mono":
        siril.save("_stars_mono")
        siril.rgbcomp(
            r="_stars_mono",
            g="_stars_mono",
            b="_stars_mono",
            out="_stars_rgb",
        )
        log_fn(f"    Using {star_source_ch} stars as grayscale (white)")
    else:
        siril.save("_stars_rgb")
        log_fn(f"    Using {star_source_ch} stars with native color")

    # Blend using max() - simpler than screen blend, avoids over-brightening
    siril.load("narrowband_starless")
    siril.pm("max($narrowband_starless$, $_stars_rgb$)")
    siril.save("narrowband")
    log_fn("    Stars composited (max blend)")


def compose_narrowband(
    siril: "SirilInterface",
    stacks: dict[str, list[StackInfo]],
    stacks_dir: Path,
    output_dir: Path,
    config: Config,
    job_type: str,
    log_fn: Callable[[str], None],
    log_step_fn: Callable[[str], None],
    log_color_balance_fn: Callable,
) -> CompositionResult:
    """
    Compose narrowband image using palette mapping.

    Job types determine channel requirements:
    - HOO: H, O channels
    - SHO: S, H, O channels
    - LHOO: L, H, O channels (L as luminance)
    - LSHO: L, S, H, O channels (L as luminance)

    Palette (from config) determines color mapping:
    - SHO: S->R, H->G, O->B (direct)
    - SHO_FORAXX: S->R, 0.5*H+0.5*O->G, O->B (blended)
    - etc.
    """
    # Determine if job uses luminance channel
    has_luminance = job_type.startswith("L")

    # Get palette from config and apply any overrides
    palette = get_palette(config.palette)
    palette = build_effective_palette(
        palette,
        r_override=config.palette_r_override,
        g_override=config.palette_g_override,
        b_override=config.palette_b_override,
    )

    log_step_fn(f"Composing narrowband ({job_type}, palette: {palette.name})")

    # Required channels = palette requirements + L if job type includes it
    required = set(palette.required)
    if has_luminance:
        required.add("L")

    for ch in required:
        if ch not in stacks:
            raise ValueError(f"Missing required channel: {ch}")
        if len(stacks[ch]) != 1:
            raise ValueError(
                f"Expected single exposure for {ch}, got {len(stacks[ch])}"
            )

    # Build channel index mapping from sorted required channels
    # This ensures we control the numbering explicitly (like broadband does)
    channels_sorted = sorted(required)
    channel_to_num = {ch: f"{i + 1:05d}" for i, ch in enumerate(channels_sorted)}

    # Create working directory with numbered files for Siril's convert
    work_dir = stacks_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    log_fn("Preparing stacks for registration...")
    for ch, idx in channel_to_num.items():
        src_path = stacks[ch][0].path
        link_path = work_dir / f"stack_{idx}.fit"
        link_or_copy(src_path, link_path)
        log_fn(f"  {ch}: {src_path.name} -> stack_{idx}.fit")

    siril.cd(str(work_dir))

    # Register stacks
    log_fn("Cross-registering stacks...")
    siril.convert("stack", out="./registered")
    siril.cd(str(work_dir / "registered"))

    cfg = config
    if cfg.cross_reg_twopass:
        # 2-pass: Siril auto-selects best reference (ignores setref)
        log_fn("Using 2-pass registration (Siril auto-selects reference)")
        siril.register("stack", twopass=True)
    else:
        # 1-pass: Use our preferred reference
        # L if available (highest SNR), otherwise H (most stars for narrowband)
        ref_ch = "L" if has_luminance else "H"
        ref_idx = int(channel_to_num[ref_ch])
        siril.setref("stack", ref_idx)
        log_fn(
            f"Using 1-pass registration with {ref_ch} as reference (image {ref_idx})"
        )
        siril.register("stack", twopass=False)

    siril.seqapplyreg("stack", framing="min")

    # Note: No linear matching for narrowband
    # Linear matching destroys the relative intensities between channels
    # which are needed for proper palette mapping. Each narrowband filter
    # captures different emission lines with different intrinsic brightness.
    log_fn("Saving registered channels (no linear match for narrowband)...")
    for ch, idx in channel_to_num.items():
        siril.load(f"r_stack_{idx}")
        siril.save(ch)
        log_fn(f"  {ch}: saved")

    # Post-stack background extraction (on registered channel stacks)
    if cfg.post_stack_subsky_method != "none":
        method_desc = (
            "RBF"
            if cfg.post_stack_subsky_method == "rbf"
            else f"polynomial degree {cfg.post_stack_subsky_degree}"
        )
        log_fn(f"Post-stack background extraction ({method_desc})...")
        for ch in required:
            siril.load(ch)
            if siril.subsky(
                rbf=(cfg.post_stack_subsky_method == "rbf"),
                degree=cfg.post_stack_subsky_degree,
                samples=cfg.post_stack_subsky_samples,
                tolerance=cfg.post_stack_subsky_tolerance,
                smooth=cfg.post_stack_subsky_smooth,
            ):
                siril.save(ch)
                log_fn(f"  {ch}: background extracted")
            else:
                log_fn(f"  {ch}: subsky failed, using original")

    # Narrowband channel balancing (linear_match to equalize backgrounds)
    # Matches S and O to H (or configured reference) using low/high bounds
    # to only affect background/mid-tones, preserving nebula emission ratios
    if cfg.narrowband_balance_enabled:
        ref_ch = cfg.narrowband_balance_reference
        if ref_ch in required:
            # Channels to match: all required except reference and L (luminance)
            channels_to_match = [ch for ch in required if ch != ref_ch and ch != "L"]
            if channels_to_match:
                log_fn(
                    f"Balancing channels to {ref_ch} "
                    f"(bounds: {cfg.narrowband_balance_low:.2f}-{cfg.narrowband_balance_high:.2f})..."
                )
                siril.load(ref_ch)
                for ch in channels_to_match:
                    siril.load(ch)
                    if siril.linear_match(
                        ref=ref_ch,
                        low=cfg.narrowband_balance_low,
                        high=cfg.narrowband_balance_high,
                    ):
                        siril.save(ch)
                        log_fn(f"  {ch}: matched to {ref_ch}")
                    else:
                        log_fn(f"  {ch}: linear_match failed, using original")
        else:
            log_fn(
                f"WARNING: Reference channel {ref_ch} not in stacks, skipping balance"
            )

    # Diagnostic previews for individual stacks (linear)
    if cfg.diagnostic_previews:
        log_fn("Saving diagnostic previews...")
        for ch in required:
            save_diagnostic_preview(siril, ch, output_dir, log_fn)

    # ==========================================================================
    # PALETTE APPLICATION
    # Two workflows based on palette.dynamic flag:
    #
    # Static palette (dynamic=False):
    #   Register -> Subsky -> Balance -> PALETTE -> Stretch(combined) -> Saturation
    #   - Palette applied to linear data
    #   - Final stretch on combined RGB
    #
    # Dynamic palette (dynamic=True):
    #   Register -> Subsky -> Balance -> Stretch(per-ch) -> PALETTE -> Saturation
    #   - Each channel stretched independently first
    #   - Palette applied to non-linear data (required for dynamic formulas)
    #   - No final stretch (already done per-channel)
    # ==========================================================================

    narrowband_channels = [ch for ch in required if ch != "L"]
    registered_dir = work_dir / "registered"

    # For dynamic palettes, we need to handle compare mode specially
    # Save linear channels first so we can re-stretch with different methods
    if palette.dynamic:
        log_fn("Dynamic palette: saving linear channels for stretch comparison...")
        for ch in narrowband_channels:
            siril.load(ch)
            siril.save(f"{ch}_linear")

        # LinearFit stronger channels to weakest in linear space
        # This balances peak intensities so the dynamic formula can produce both blue and gold.
        # Reference: https://thecoldestnights.com/2020/06/pixinsight-dynamic-narrowband-combinations-with-pixelmath/
        # See also: https://jonrista.com/the-astrophotographers-guide/pixinsights/narrow-band-combinations-with-pixelmath-hoo/
        if cfg.palette_linearfit_to_weakest:
            # Measure signal strength (max value) for each channel to find weakest
            channel_max = {}
            for ch in narrowband_channels:
                ch_path = registered_dir / f"{ch}_linear.fit"
                stats = siril.get_image_stats(ch_path)
                channel_max[ch] = stats["max"]
                log_fn(f"  {ch}: max={stats['max']:.4f}")

            # Weakest channel has lowest max signal
            ref_ch = min(channel_max, key=channel_max.get)
            channels_to_fit = [ch for ch in narrowband_channels if ch != ref_ch]

            if channels_to_fit:
                log_fn(
                    f"LinearFit to weakest channel ({ref_ch}, max={channel_max[ref_ch]:.4f})..."
                )
                for ch in channels_to_fit:
                    siril.load(f"{ch}_linear")
                    if siril.linear_match(ref=f"{ref_ch}_linear"):
                        siril.save(f"{ch}_linear")
                        log_fn(f"  {ch}: fitted to {ref_ch}")
                    else:
                        log_fn(f"  {ch}: linear_match failed, using original")

    # Helper to apply palette to current channels
    def apply_palette_combination():
        """Apply palette formulas to currently loaded/saved channels."""
        if palette.is_simple():
            siril.rgbcomp(r=palette.r, g=palette.g, b=palette.b, out="narrowband_rgb")
        else:
            r_pm = formula_to_pixelmath(palette.r)
            g_pm = formula_to_pixelmath(palette.g)
            b_pm = formula_to_pixelmath(palette.b)
            siril.pm(r_pm)
            siril.save("_pm_r")
            siril.pm(g_pm)
            siril.save("_pm_g")
            siril.pm(b_pm)
            siril.save("_pm_b")
            siril.rgbcomp(r="_pm_r", g="_pm_g", b="_pm_b", out="narrowband_rgb")

        if has_luminance:
            siril.rgbcomp(lum="L", rgb="narrowband_rgb", out="narrowband")
        else:
            siril.load("narrowband_rgb")
            siril.save("narrowband")

    # Map channels according to palette
    log_fn(
        f"Applying {palette.name} palette: R={palette.r}, G={palette.g}, B={palette.b}"
    )

    if not palette.is_simple():
        r_pm = formula_to_pixelmath(palette.r)
        g_pm = formula_to_pixelmath(palette.g)
        b_pm = formula_to_pixelmath(palette.b)
        log_fn(f"  PixelMath: R={r_pm}, G={g_pm}, B={b_pm}")

    type_name = job_type.lower()

    if not palette.dynamic:
        # Static palette: apply to linear data, stretch later
        apply_palette_combination()
        if has_luminance:
            log_fn("Adding L channel as luminance...")

        # Save linear result (only for static palettes - dynamic palettes are already stretched)
        linear_path = output_dir / f"{type_name}_linear.fit"
        siril.load("narrowband")
        siril.save(str(linear_path))
        log_fn(f"Saved linear: {linear_path.name}")
        log_color_balance_fn(linear_path)

        if cfg.diagnostic_previews:
            save_diagnostic_preview(siril, "narrowband", output_dir, log_fn)

        stretch_source = str(work_dir / "registered" / "narrowband.fit")
    else:
        # Dynamic palette: linear result not meaningful (channels will be stretched first)
        # The actual processing happens in the final stretch section below
        linear_path = output_dir / f"{type_name}_linear.fit"
        # Save linear channels composed without stretch for reference
        apply_palette_combination()
        siril.load("narrowband")
        siril.save(str(linear_path))
        log_fn(f"Saved linear (pre-stretch reference): {linear_path.name}")
        stretch_source = None  # Not used for dynamic palettes

    # Note: PCC not applicable for narrowband - uses synthetic colors
    linear_pcc_path = None
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
        deconv_result = apply_deconvolution(
            siril, "narrowband", "narrowband_deconv", cfg, output_dir, log_fn, type_name
        )
        if deconv_result == "narrowband_deconv":
            deconv_path = output_dir / f"{type_name}_deconv.fit"
            siril.load("narrowband_deconv")
            siril.save(str(deconv_path))
            stretch_source = str(deconv_path)
            log_fn(f"Saved deconvolved: {deconv_path.name}")

    # Final stretch and output
    if palette.dynamic:
        auto_paths = _process_dynamic_palette(
            siril=siril,
            cfg=cfg,
            palette=palette,
            has_luminance=has_luminance,
            narrowband_channels=narrowband_channels,
            registered_dir=registered_dir,
            output_dir=output_dir,
            type_name=type_name,
            apply_palette_fn=apply_palette_combination,
            log_fn=log_fn,
        )
    else:
        # Static palette: stretch combined RGB now
        auto_paths = stretch_and_finalize(
            siril=siril,
            input_path=Path(stretch_source),
            output_dir=output_dir,
            basename=f"{type_name}_auto",
            config=cfg,
            log_fn=log_fn,
        )

    return CompositionResult(
        linear_path=linear_path,
        linear_pcc_path=linear_pcc_path,
        auto_fit=auto_paths["fit"],
        auto_tif=auto_paths["tif"],
        auto_jpg=auto_paths["jpg"],
        stacks_dir=stacks_dir,
    )
