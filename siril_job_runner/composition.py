"""
Composition module for Siril job processing.

Handles LRGB composition, narrowband palette mixing, and stretching.
Based on LRGB_pre.ssf and LRGB_compose.ssf workflows.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .logger import JobLogger
from .protocols import SirilInterface


# Narrowband palette definitions (channel mappings)
PALETTES = {
    "HOO": {"R": "H", "G": "O", "B": "O"},
    "SHO": {"R": "S", "G": "H", "B": "O"},
}


@dataclass
class CompositionResult:
    """Result of composition stage."""

    linear_path: Path  # Unstretched composed image (for VeraLux)
    auto_fit: Path  # Auto-stretched .fit
    auto_tif: Path  # Auto-stretched .tif
    auto_jpg: Path  # Auto-stretched .jpg
    stacks_dir: Path  # Directory containing linear stacks


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
        logger: Optional[JobLogger] = None,
    ):
        self.siril = siril
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.stacks_dir = self.output_dir / "stacks"

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    def _log_step(self, message: str) -> None:
        if self.logger:
            self.logger.step(message)

    def compose_lrgb(self, deconvolve_l: bool = True) -> CompositionResult:
        """
        Compose LRGB image from stacked channels.

        Expects stacks: stack_L.fit, stack_R.fit, stack_G.fit, stack_B.fit
        in self.stacks_dir.

        Returns CompositionResult with paths to all outputs.
        """
        self._log_step("Composing LRGB")

        # Verify all stacks exist
        for ch in ["L", "R", "G", "B"]:
            stack_path = self.stacks_dir / f"stack_{ch}.fit"
            if not stack_path.exists():
                # Try with exposure suffix
                matches = list(self.stacks_dir.glob(f"stack_{ch}_*.fit"))
                if not matches:
                    raise FileNotFoundError(f"Missing stack: {stack_path}")

        self.siril.cd(str(self.stacks_dir))

        # Step 1: Register stacks to each other
        self._log("Cross-registering stacks...")
        self.siril.convert("stack", out="./registered")
        self.siril.cd(str(self.stacks_dir / "registered"))
        self.siril.register("stack", twopass=True)
        self.siril.seqapplyreg("stack", framing="min")

        # Step 2: Linear match to reference (R)
        # Stacks are in alphabetical order: B=1, G=2, L=3, R=4
        self._log("Linear matching to R reference...")
        self.siril.load("r_stack_00004")  # R
        self.siril.save("R")

        self.siril.load("r_stack_00001")  # B
        self.siril.linear_match("R", 0, 0.92)
        self.siril.save("B")

        self.siril.load("r_stack_00002")  # G
        self.siril.linear_match("R", 0, 0.92)
        self.siril.save("G")

        self.siril.load("r_stack_00003")  # L
        self.siril.linear_match("R", 0, 0.92)
        self.siril.save("L")

        # Step 3: Optional deconvolution on L
        l_name = "L"
        if deconvolve_l:
            self._log("Deconvolving L channel...")
            self.siril.load("L")
            self.siril.makepsf("blind")
            self.siril.rl()
            self.siril.save("L_deconv")
            l_name = "L_deconv"

        # Step 4: Compose RGB
        self._log("Creating RGB composite...")
        self.siril.rgbcomp(r="R", g="G", b="B", out="rgb")

        # Step 5: Add luminance
        self._log("Adding luminance channel...")
        self.siril.rgbcomp(lum=l_name, rgb="rgb", out="lrgb")

        # Save linear (unstretched) result
        linear_path = self.output_dir / "lrgb_linear.fit"
        self.siril.load("lrgb")
        self.siril.save(str(linear_path))
        self._log(f"Saved linear: {linear_path.name}")

        # Step 6: Auto-stretch and save
        auto_paths = self._auto_stretch("lrgb", "lrgb_auto")

        return CompositionResult(
            linear_path=linear_path,
            auto_fit=auto_paths["fit"],
            auto_tif=auto_paths["tif"],
            auto_jpg=auto_paths["jpg"],
            stacks_dir=self.stacks_dir,
        )

    def compose_rgb(self) -> CompositionResult:
        """
        Compose RGB image (no luminance channel).

        For cases where L channel is not available.
        """
        self._log_step("Composing RGB")

        self.siril.cd(str(self.stacks_dir))

        # Register stacks
        self._log("Cross-registering stacks...")
        self.siril.convert("stack", out="./registered")
        self.siril.cd(str(self.stacks_dir / "registered"))
        self.siril.register("stack", twopass=True)
        self.siril.seqapplyreg("stack", framing="min")

        # Linear match to R
        self._log("Linear matching to R reference...")
        self.siril.load("r_stack_R")
        self.siril.save("R")

        for ch in ["G", "B"]:
            self.siril.load(f"r_stack_{ch}")
            self.siril.linear_match("R", 0, 0.92)
            self.siril.save(ch)

        # Compose RGB
        self._log("Creating RGB composite...")
        self.siril.rgbcomp(r="R", g="G", b="B", out="rgb")

        # Save linear result
        linear_path = self.output_dir / "rgb_linear.fit"
        self.siril.load("rgb")
        self.siril.save(str(linear_path))

        # Auto-stretch
        auto_paths = self._auto_stretch("rgb", "rgb_auto")

        return CompositionResult(
            linear_path=linear_path,
            auto_fit=auto_paths["fit"],
            auto_tif=auto_paths["tif"],
            auto_jpg=auto_paths["jpg"],
            stacks_dir=self.stacks_dir,
        )

    def compose_narrowband(self, palette: str = "HOO") -> CompositionResult:
        """
        Compose narrowband image using palette mapping.

        Palettes:
        - HOO: H->R, O->G, O->B
        - SHO: S->R, H->G, O->B
        """
        self._log_step(f"Composing narrowband ({palette})")

        if palette not in PALETTES:
            raise ValueError(f"Unknown palette: {palette}. Available: {list(PALETTES.keys())}")

        mapping = PALETTES[palette]
        self.siril.cd(str(self.stacks_dir))

        # Register stacks
        self._log("Cross-registering stacks...")
        self.siril.convert("stack", out="./registered")
        self.siril.cd(str(self.stacks_dir / "registered"))
        self.siril.register("stack", twopass=True)
        self.siril.seqapplyreg("stack", framing="min")

        # Determine reference channel (H for both palettes)
        self._log("Linear matching to H reference...")
        self.siril.load("r_stack_H")
        self.siril.save("H")

        for ch in ["O", "S"]:
            try:
                self.siril.load(f"r_stack_{ch}")
                self.siril.linear_match("H", 0, 0.92)
                self.siril.save(ch)
            except Exception:
                pass  # S may not exist for HOO

        # Map channels according to palette
        self._log(f"Applying {palette} palette...")
        r_src = mapping["R"]
        g_src = mapping["G"]
        b_src = mapping["B"]

        self.siril.rgbcomp(r=r_src, g=g_src, b=b_src, out="narrowband")

        # Save linear result
        type_name = palette.lower()
        linear_path = self.output_dir / f"{type_name}_linear.fit"
        self.siril.load("narrowband")
        self.siril.save(str(linear_path))

        # Auto-stretch
        auto_paths = self._auto_stretch("narrowband", f"{type_name}_auto")

        return CompositionResult(
            linear_path=linear_path,
            auto_fit=auto_paths["fit"],
            auto_tif=auto_paths["tif"],
            auto_jpg=auto_paths["jpg"],
            stacks_dir=self.stacks_dir,
        )

    def _auto_stretch(self, input_name: str, output_name: str) -> dict[str, Path]:
        """
        Apply auto-stretch pipeline and save in multiple formats.

        Pipeline from LRGB_compose.ssf:
        - autostretch
        - mtf 0.20 0.5 1.0
        - satu 1 0
        """
        self._log("Auto-stretching...")

        self.siril.load(input_name)
        self.siril.autostretch(linked=True)
        self.siril.mtf(0.20, 0.5, 1.0)
        self.siril.satu(1, 0)

        # Save in multiple formats
        self.siril.cd(str(self.output_dir))

        fit_path = self.output_dir / f"{output_name}.fit"
        tif_path = self.output_dir / f"{output_name}.tif"
        jpg_path = self.output_dir / f"{output_name}.jpg"

        self.siril.save(str(fit_path))
        self.siril.savetif(str(tif_path), astro=True, deflate=True)
        self.siril.savejpg(str(jpg_path), 90)

        self._log(f"Saved: {fit_path.name}, {tif_path.name}, {jpg_path.name}")

        return {"fit": fit_path, "tif": tif_path, "jpg": jpg_path}


def compose_and_stretch(
    siril: SirilInterface,
    stacks: dict[str, Path],
    output_dir: Path,
    job_type: str,
    palette: str = "HOO",
    deconvolve_l: bool = True,
    logger: Optional[JobLogger] = None,
) -> CompositionResult:
    """
    Compose and stretch based on job type.

    Args:
        siril: Siril interface
        stacks: Dict of stack_name -> path (from preprocessing)
        output_dir: Output directory
        job_type: "LRGB", "RGB", "SHO", or "HOO"
        palette: Narrowband palette (for SHO/HOO)
        deconvolve_l: Whether to deconvolve L channel (LRGB only)
        logger: Optional logger

    Returns:
        CompositionResult with paths to all outputs
    """
    composer = Composer(siril, output_dir, logger)

    if job_type == "LRGB":
        return composer.compose_lrgb(deconvolve_l=deconvolve_l)
    elif job_type == "RGB":
        return composer.compose_rgb()
    elif job_type in ("SHO", "HOO"):
        return composer.compose_narrowband(palette=palette or job_type)
    else:
        raise ValueError(f"Unknown job type: {job_type}")
