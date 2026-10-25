"""
Narrowband composition for SHO/HOO imaging.

Handles composition of H, S, O filter stacks into palette-mapped images.
Supports optional L channel for LSHO/LHOO variants.

Processing flow:
    1. Cross-register all channel stacks
    2. Post-stack background extraction (subsky)
    3. Channel balancing (linear_match to H)
    4. StarNet on linear data (if enabled) -> {ch}_linear_starless + stars
    5. For each stretch method (autostretch, veralux):
       a. Stretch channels (starless if available)
       b. Apply palette formulas -> narrowband RGB
       c. Add L as luminance (if LSHO/LHOO)
       d. Color removal (SCNR)
       e. Background neutralization
       f. Save starless output (if starnet enabled)
       g. Composite stars back (if starnet enabled)
       h. Apply saturation
       i. Save final output
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .compose_helpers import (
    apply_color_removal,
    apply_deconvolution,
    neutralize_rgb_background,
    register_composition_stacks,
    save_diagnostic_preview,
)
from .config import (
    ColorRemovalMode,
    Config,
    StarSource,
    StretchMethod,
    SubskyMethod,
)
from .models import CompositionResult, StackInfo
from .palettes import build_effective_palette, formula_to_pixelmath, get_palette
from .stretch_helpers import (
    apply_enhancements,
    apply_saturation,
    apply_stretch,
    save_all_formats,
)

if TYPE_CHECKING:
    from .protocols import SirilInterface


def _separate_stars_linear(
    siril: "SirilInterface",
    cfg: Config,
    has_luminance: bool,
    all_channels: list[str],
    registered_dir: Path,
    log_fn: Callable[[str], None],
) -> str:
    """
    Separate stars from all linear channels using StarNet.

    Runs on {ch}_linear files with stretch=True (internal MTF stretch).
    Creates {ch}_linear_starless for each channel and stars_source from the
    designated star channel.

    Returns the channel name used as star source (for later compositing),
    raising if star separation fails.
    """
    log_fn("  Star separation enabled, running StarNet on linear data...")

    # Determine star source: L if available, else H, or configured channel
    if cfg.narrowband_star_source == StarSource.AUTO:
        star_source_ch = "L" if has_luminance else "H"
    else:
        star_source_ch = cfg.narrowband_star_source

    # Run StarNet on all linear channels
    for ch in all_channels:
        if not siril.load(f"{ch}_linear"):
            raise RuntimeError("Siril load failed during composition")
        siril.starnet(stretch=True)  # Linear data, needs internal stretch
        if not siril.save(f"{ch}_linear_starless"):
            raise RuntimeError("Siril save failed during composition")
        # StarNet creates starmask_<filename>.fit in the working directory
        starmask_path = registered_dir / f"starmask_{ch}_linear.fit"
        if starmask_path.exists() and ch == star_source_ch:
            if not siril.load(str(starmask_path)):
                raise RuntimeError("Siril load failed during composition")
            if not siril.save("stars_source"):
                raise RuntimeError("Siril save failed during composition")
            log_fn(f"    {ch}: starless saved, stars extracted (source)")
        else:
            log_fn(f"    {ch}: starless saved")

    return star_source_ch


def _apply_scale_expressions(
    siril: "SirilInterface",
    cfg: Config,
    narrowband_channels: list[str],
    log_fn: Callable[[str], None],
) -> None:
    """
    Apply channel scale expressions before palette formulas.

    Scale expressions allow non-linear channel manipulation (e.g., O + k*O^3)
    to boost signal in a way that survives linear background neutralization.
    Operates on stretched channels saved with their base names (H, O, S).
    """
    scale_exprs = {
        "H": cfg.palette_h_scale_expr,
        "O": cfg.palette_o_scale_expr,
        "S": cfg.palette_s_scale_expr,
    }
    for ch in narrowband_channels:
        expr = scale_exprs.get(ch)
        if expr:
            if not siril.load(ch):
                raise RuntimeError("Siril load failed during composition")
            pm_expr = formula_to_pixelmath(expr)
            if not siril.pm(pm_expr):
                raise RuntimeError("Siril pm failed during composition")
            if not siril.save(ch):
                raise RuntimeError("Siril save failed during composition")
            log_fn(f"  {ch}: scaled with {expr}")


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
    - HOO: H->R, O->G, O->B (direct)
    - SHO: S->R, H->G, O->B (direct)
    - SHO_FORAXX: S->R, 0.5*H+0.5*O->G, O->B (blended)

    When starnet_enabled=True, outputs per stretch method:
    - {type}_stars.fit/tif/jpg (once, before stretch loop)
    - {type}_starless_{method}.fit/tif/jpg
    - {type}_auto_{method}.fit/tif/jpg (with stars composited)

    When starnet_enabled=False:
    - {type}_auto_{method}.fit/tif/jpg only
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
    channels_sorted = sorted(required)
    channel_to_num = {ch: f"{i + 1:05d}" for i, ch in enumerate(channels_sorted)}

    # Working files are separate from the retained source stacks.
    work_dir = stacks_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = config
    ref_ch = "L" if has_luminance else "H"
    log_fn("Cross-registering stacks...")
    register_composition_stacks(
        siril,
        [stacks[ch][0].path for ch in channels_sorted],
        work_dir,
        int(channel_to_num[ref_ch]),
        cfg.cross_reg_twopass,
    )

    # Save registered channels
    log_fn("Saving registered channels...")
    for ch, idx in channel_to_num.items():
        if not siril.load(f"r_stack_{idx}"):
            raise RuntimeError("Siril load failed during composition")
        if not siril.save(ch):
            raise RuntimeError("Siril save failed during composition")
        log_fn(f"  {ch}: saved")

    # Post-stack background extraction
    if cfg.post_stack_subsky_method != SubskyMethod.NONE:
        method_desc = (
            "RBF"
            if cfg.post_stack_subsky_method == SubskyMethod.RBF
            else f"polynomial degree {cfg.post_stack_subsky_degree}"
        )
        log_fn(f"Post-stack background extraction ({method_desc})...")
        for ch in required:
            if not siril.load(ch):
                raise RuntimeError("Siril load failed during composition")
            if siril.subsky(
                rbf=(cfg.post_stack_subsky_method == SubskyMethod.RBF),
                degree=cfg.post_stack_subsky_degree,
                samples=cfg.post_stack_subsky_samples,
                tolerance=cfg.post_stack_subsky_tolerance,
                smooth=cfg.post_stack_subsky_smooth,
            ):
                if not siril.save(ch):
                    raise RuntimeError("Siril save failed during composition")
                log_fn(f"  {ch}: background extracted")
            else:
                log_fn(f"  {ch}: subsky failed, using original")

    # Narrowband channel balancing (linear_match to equalize backgrounds)
    if cfg.narrowband_balance_enabled:
        ref_ch = cfg.narrowband_balance_reference
        if ref_ch in required:
            channels_to_match = [ch for ch in required if ch != ref_ch and ch != "L"]
            if channels_to_match:
                log_fn(
                    f"Balancing channels to {ref_ch} "
                    f"(bounds: {cfg.narrowband_balance_low:.2f}-{cfg.narrowband_balance_high:.2f})..."
                )
                if not siril.load(ref_ch):
                    raise RuntimeError("Siril load failed during composition")
                for ch in channels_to_match:
                    if not siril.load(ch):
                        raise RuntimeError("Siril load failed during composition")
                    if siril.linear_match(
                        ref=ref_ch,
                        low=cfg.narrowband_balance_low,
                        high=cfg.narrowband_balance_high,
                    ):
                        if not siril.save(ch):
                            raise RuntimeError("Siril save failed during composition")
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

    # Save linear channels for stretch comparison and reprocessing
    narrowband_channels = [ch for ch in required if ch != "L"]
    all_channels = list(narrowband_channels)
    if has_luminance:
        all_channels.append("L")
    registered_dir = work_dir / "registered"

    log_fn("Saving linear channels...")
    for ch in all_channels:
        if not siril.load(ch):
            raise RuntimeError("Siril load failed during composition")
        if not siril.save(f"{ch}_linear"):
            raise RuntimeError("Siril save failed during composition")

    # LinearFit to weakest channel (optional, for dynamic palettes)
    if cfg.palette_linearfit_to_weakest:
        channel_max = {}
        for ch in narrowband_channels:
            ch_path = registered_dir / f"{ch}_linear.fit"
            stats = siril.get_image_stats(ch_path)
            channel_max[ch] = stats["max"]
            log_fn(f"  {ch}: max={stats['max']:.4f}")

        ref_ch = min(channel_max, key=channel_max.__getitem__)
        channels_to_fit = [ch for ch in narrowband_channels if ch != ref_ch]

        if channels_to_fit:
            log_fn(
                f"LinearFit to weakest channel ({ref_ch}, max={channel_max[ref_ch]:.4f})..."
            )
            for ch in channels_to_fit:
                if not siril.load(f"{ch}_linear"):
                    raise RuntimeError("Siril load failed during composition")
                if siril.linear_match(ref=f"{ref_ch}_linear"):
                    if not siril.save(f"{ch}_linear"):
                        raise RuntimeError("Siril save failed during composition")
                    log_fn(f"  {ch}: fitted to {ref_ch}")
                else:
                    log_fn(f"  {ch}: linear_match failed, using original")

    # Helper to apply palette to current channels
    def apply_palette_combination():
        """Apply palette formulas to currently loaded/saved channels."""
        if palette.is_simple():
            if not siril.rgbcomp(
                r=palette.r, g=palette.g, b=palette.b, out="narrowband_rgb"
            ):
                raise RuntimeError("Siril rgbcomp failed during composition")
        else:
            r_pm = formula_to_pixelmath(palette.r)
            g_pm = formula_to_pixelmath(palette.g)
            b_pm = formula_to_pixelmath(palette.b)
            if not siril.pm(r_pm):
                raise RuntimeError("Siril pm failed during composition")
            if not siril.save("_pm_r"):
                raise RuntimeError("Siril save failed during composition")
            if not siril.pm(g_pm):
                raise RuntimeError("Siril pm failed during composition")
            if not siril.save("_pm_g"):
                raise RuntimeError("Siril save failed during composition")
            if not siril.pm(b_pm):
                raise RuntimeError("Siril pm failed during composition")
            if not siril.save("_pm_b"):
                raise RuntimeError("Siril save failed during composition")
            if not siril.rgbcomp(r="_pm_r", g="_pm_g", b="_pm_b", out="narrowband_rgb"):
                raise RuntimeError("Siril rgbcomp failed during composition")

        if has_luminance:
            if not siril.rgbcomp(lum="L", rgb="narrowband_rgb", out="narrowband"):
                raise RuntimeError("Siril rgbcomp failed during composition")
        else:
            if not siril.load("narrowband_rgb"):
                raise RuntimeError("Siril load failed during composition")
            if not siril.save("narrowband"):
                raise RuntimeError("Siril save failed during composition")

    # Log palette info
    log_fn(
        f"Applying {palette.name} palette: R={palette.r}, G={palette.g}, B={palette.b}"
    )
    if not palette.is_simple():
        r_pm = formula_to_pixelmath(palette.r)
        g_pm = formula_to_pixelmath(palette.g)
        b_pm = formula_to_pixelmath(palette.b)
        log_fn(f"  PixelMath: R={r_pm}, G={g_pm}, B={b_pm}")

    type_name = job_type.lower()

    # Optional deconvolution on linear channels
    if cfg.deconv_enabled:
        log_fn("Deconvolving linear channels...")
        for ch in narrowband_channels:
            deconv_result = apply_deconvolution(
                siril,
                f"{ch}_linear",
                f"{ch}_linear_deconv",
                cfg,
                output_dir,
                log_fn,
                ch,
            )
            if deconv_result == f"{ch}_linear_deconv":
                if not siril.load(f"{ch}_linear_deconv"):
                    raise RuntimeError("Siril load failed during composition")
                if not siril.save(f"{ch}_linear"):
                    raise RuntimeError("Siril save failed during composition")
                log_fn(f"  {ch}: deconvolved")

    # ==========================================================================
    # STAR SEPARATION (on linear data, before stretch)
    # ==========================================================================
    star_source_ch = None
    if cfg.starnet_enabled:
        star_source_ch = _separate_stars_linear(
            siril, cfg, has_luminance, all_channels, registered_dir, log_fn
        )
        # Save stars output
        if star_source_ch:
            if not siril.load("stars_source"):
                raise RuntimeError("Siril load failed during composition")
            stars_name = f"{type_name}_stars"
            save_all_formats(siril, output_dir, stars_name, log_fn)

    # ==========================================================================
    # UNIFIED PROCESSING LOOP
    # For each stretch method: stretch -> palette -> neutralize -> saturation
    # ==========================================================================
    methods = (
        [StretchMethod.AUTOSTRETCH, StretchMethod.VERALUX]
        if cfg.stretch_compare
        else [cfg.stretch_method]
    )
    if cfg.stretch_compare:
        log_fn("Comparing stretch methods (autostretch vs veralux)...")

    primary_paths = None
    palette_path = output_dir / f"{type_name}_palette_{methods[0]}.fit"

    for method in methods:
        suffix = f"_{method}" if cfg.stretch_compare else ""
        log_fn(f"Processing with {method} stretch...")

        # Step 1: Stretch each channel (use starless if available)
        for ch in all_channels:
            if cfg.starnet_enabled and star_source_ch:
                if not siril.load(f"{ch}_linear_starless"):
                    raise RuntimeError("Siril load failed during composition")
                ch_path = registered_dir / f"{ch}_linear_starless.fit"
            else:
                if not siril.load(f"{ch}_linear"):
                    raise RuntimeError("Siril load failed during composition")
                ch_path = registered_dir / f"{ch}_linear.fit"
            if not apply_stretch(siril, method, ch_path, cfg, log_fn):
                raise RuntimeError("apply_stretch failed during composition")
            if not siril.save(ch):
                raise RuntimeError("Siril save failed during composition")
            log_fn(f"  {ch}: stretched ({method})")

        # Step 2: Apply channel scale expressions
        _apply_scale_expressions(siril, cfg, narrowband_channels, log_fn)

        # Step 3: Apply palette formulas
        apply_palette_combination()

        # Save the first stretched palette before color adjustments
        if primary_paths is None:
            if not siril.load("narrowband"):
                raise RuntimeError("Siril load failed during composition")
            if not siril.save(str(palette_path)):
                raise RuntimeError("Siril save failed during composition")
            log_fn(f"Saved stretched palette: {palette_path.name}")
            log_color_balance_fn(palette_path)

        # Step 4: Color removal (SCNR)
        if cfg.color_removal_mode != ColorRemovalMode.NONE:
            if not siril.load("narrowband"):
                raise RuntimeError("Siril load failed during composition")
            if not apply_color_removal(siril, cfg, log_fn):
                raise RuntimeError("apply_color_removal failed during composition")
            if not siril.save("narrowband"):
                raise RuntimeError("Siril save failed during composition")

        # Step 5: Neutralize RGB background
        if cfg.narrowband_neutralization and not neutralize_rgb_background(
            siril, "narrowband", cfg, log_fn
        ):
            raise RuntimeError("neutralize_rgb_background failed during composition")

        # Step 6: Save starless output (if star separation enabled)
        if cfg.starnet_enabled and star_source_ch:
            if not siril.load("narrowband"):
                raise RuntimeError("Siril load failed during composition")
            starless_name = f"{type_name}_starless{suffix}"
            save_all_formats(siril, output_dir, starless_name, log_fn, cfg)

        # Step 7: Composite stars back (if separated)
        if cfg.starnet_enabled and star_source_ch:
            from .veralux_starcomposer import apply_starcomposer

            starless_path = registered_dir / "narrowband.fit"
            success, _ = apply_starcomposer(
                siril, starless_path, registered_dir / "stars_source.fit", cfg, log_fn
            )
            if not success:
                raise RuntimeError("StarComposer failed to restore narrowband stars")
            if not siril.save("narrowband"):
                raise RuntimeError("Siril save failed during composition")

        # Step 8: Apply saturation
        if not siril.load("narrowband"):
            raise RuntimeError("Siril load failed during composition")
        apply_saturation(siril, cfg)

        # Step 9: Apply enhancements (Silentium/Revela/Vectra) if enabled
        base_name = f"{type_name}_auto{suffix}"
        if not siril.save(str(output_dir / base_name)):
            raise RuntimeError("Siril save failed during composition")
        image_path = output_dir / f"{base_name}.fit"
        apply_enhancements(siril, image_path, cfg, log_fn)

        # Step 10: Save outputs
        paths = save_all_formats(siril, output_dir, base_name, log_fn, cfg)

        if primary_paths is None:
            primary_paths = paths

    if primary_paths is None:
        raise ValueError("No narrowband palette produced an image")
    return CompositionResult(
        linear_path=None,
        linear_pcc_path=None,  # No PCC for narrowband
        auto_fit=primary_paths["fit"],
        auto_tif=primary_paths["tif"],
        auto_jpg=primary_paths["jpg"],
        stacks_dir=stacks_dir,
    )
