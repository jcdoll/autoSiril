"""
Composition module for Siril job processing.

Handles RGB composition, LRGB combination, and narrowband palette mixing.
"""

from pathlib import Path
from typing import Protocol, Optional

from .logger import JobLogger


class SirilInterface(Protocol):
    """Protocol for Siril interface."""

    def cd(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def save(self, path: str) -> None: ...
    def convert(self, name: str, **kwargs) -> None: ...
    def register(self, name: str, **kwargs) -> None: ...
    def seqapplyreg(self, name: str, **kwargs) -> None: ...
    def linear_match(self, ref: str, low: float, high: float) -> None: ...
    def rgbcomp(self, *args, **kwargs) -> None: ...
    def pm(self, expression: str) -> None: ...
    def autostretch(self) -> None: ...
    def mtf(self, low: float, mid: float, high: float) -> None: ...
    def satu(self, amount: float, threshold: float) -> None: ...
    def savetif(self, path: str, **kwargs) -> None: ...
    def savejpg(self, path: str, quality: int) -> None: ...


# Narrowband palette definitions (pixelmath expressions)
PALETTES = {
    "HOO": {
        "R": "$H$",
        "G": "$O$",
        "B": "$O$",
    },
    "HOO_natural": {
        "R": "$H$",
        "G": "0.2*$H$ + 0.8*$O$",
        "B": "0.15*$H$ + 0.85*$O$",
    },
    "SHO_blue_red": {
        "R": "0.76*$H$ + 0.24*$S$",
        "G": "$O$",
        "B": "0.85*$O$ + 0.15*$H$",
    },
    "SHO_blue_gold": {
        "R": "0.7*$S$ + 0.3*$H$",
        "G": "0.3*$S$ + 0.3*$H$ + 0.4*$O$",
        "B": "$O$",
    },
}


class Composer:
    """Handles image composition and stretching."""

    def __init__(self, siril: SirilInterface, logger: Optional[JobLogger] = None):
        self.siril = siril
        self.logger = logger

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    def compose_lrgb(
        self,
        stacks: dict[str, Path],
        output_dir: Path,
    ) -> Path:
        """
        Compose LRGB image from stacked channels.

        Args:
            stacks: Dict with L, R, G, B paths
            output_dir: Output directory

        Returns:
            Path to unstretched LRGB result
        """
        self._log("Composing LRGB...")

        # Ensure all required channels exist
        required = ["L", "R", "G", "B"]
        for ch in required:
            if ch not in stacks:
                raise ValueError(f"Missing required channel: {ch}")

        stacks_dir = output_dir / "stacks"
        self.siril.cd(str(stacks_dir))

        # Register stacks to each other
        self._log("Cross-registering stacks...")
        self.siril.convert("stacks", out=str(stacks_dir / "reg"))
        self.siril.cd(str(stacks_dir / "reg"))
        self.siril.register("stacks", twopass=True)
        self.siril.seqapplyreg("stacks", framing="min")

        # Linear match to L channel
        self._log("Linear matching channels...")
        self.siril.load("r_stacks_L")
        self.siril.save("L_matched")

        for ch in ["R", "G", "B"]:
            self.siril.load(f"r_stacks_{ch}")
            self.siril.linear_match("L_matched", 0, 0.92)
            self.siril.save(f"{ch}_matched")

        # Compose RGB
        self._log("Creating RGB composite...")
        self.siril.rgbcomp("R_matched", "G_matched", "B_matched", out="rgb")

        # Add luminance
        self._log("Adding luminance...")
        self.siril.rgbcomp(lum="L_matched", rgb="rgb", out="lrgb")

        # Save unstretched result
        unstretched_path = output_dir / "unstretched_lrgb.fit"
        self.siril.load("lrgb")
        self.siril.save(str(unstretched_path))

        return unstretched_path

    def compose_narrowband(
        self,
        stacks: dict[str, Path],
        output_dir: Path,
        palette: str = "HOO",
    ) -> Path:
        """
        Compose narrowband image using palette.

        Args:
            stacks: Dict with H, O (and optionally S) paths
            output_dir: Output directory
            palette: Palette name (HOO, SHO_blue_gold, etc.)

        Returns:
            Path to unstretched result
        """
        self._log(f"Composing narrowband ({palette})...")

        if palette not in PALETTES:
            raise ValueError(f"Unknown palette: {palette}. Available: {list(PALETTES.keys())}")

        palette_def = PALETTES[palette]

        stacks_dir = output_dir / "stacks"
        self.siril.cd(str(stacks_dir))

        # Register stacks
        self._log("Cross-registering stacks...")
        self.siril.convert("stacks", out=str(stacks_dir / "reg"))
        self.siril.cd(str(stacks_dir / "reg"))
        self.siril.register("stacks", twopass=True)
        self.siril.seqapplyreg("stacks", framing="min")

        # Linear match to H channel
        self._log("Linear matching channels...")
        self.siril.load("r_stacks_H")
        self.siril.save("H")

        for ch in ["O", "S"]:
            if ch in stacks or f"r_stacks_{ch}" in str(stacks_dir):
                try:
                    self.siril.load(f"r_stacks_{ch}")
                    self.siril.linear_match("H", 0, 0.92)
                    self.siril.save(ch)
                except Exception:
                    pass  # Channel may not exist for HOO

        # Apply palette using pixelmath
        self._log("Applying palette...")
        for channel, expression in palette_def.items():
            self.siril.pm(expression)
            self.siril.save(channel)

        # Compose RGB
        self.siril.rgbcomp("R", "G", "B", out="narrowband")

        # Save unstretched result
        type_name = "sho" if "S" in stacks else "hoo"
        unstretched_path = output_dir / f"unstretched_{type_name}.fit"
        self.siril.load("narrowband")
        self.siril.save(str(unstretched_path))

        return unstretched_path

    def stretch(
        self,
        input_path: Path,
        output_dir: Path,
        name: str = "output",
    ) -> dict[str, Path]:
        """
        Apply stretch to image and save in multiple formats.

        Args:
            input_path: Path to unstretched image
            output_dir: Output directory
            name: Base name for output files

        Returns:
            Dict with paths to fit, tif, jpg outputs
        """
        self._log("Stretching...")

        self.siril.cd(str(output_dir))
        self.siril.load(str(input_path))

        # Apply stretch
        self.siril.autostretch()
        self.siril.mtf(0.20, 0.5, 1.0)
        self.siril.satu(1, 0)

        # Save in multiple formats
        fit_path = output_dir / f"{name}.fit"
        tif_path = output_dir / f"{name}.tif"
        jpg_path = output_dir / f"{name}.jpg"

        self.siril.save(str(fit_path))
        self.siril.savetif(str(tif_path), astro=True, deflate=True)
        self.siril.savejpg(str(jpg_path), 90)

        self._log(f"Saved: {fit_path.name}, {tif_path.name}, {jpg_path.name}")

        # Placeholder for future VeraLux CLI integration
        # TODO: VeraLux CLI integration (when available)
        # subprocess.run(["python", "VeraLux_HyperMetric_Stretch.py",
        #                 "--input", str(input_path),
        #                 "--output", str(output_dir / f"{name}_veralux.fit"),
        #                 "--profile", "ready-to-use"])

        return {
            "fit": fit_path,
            "tif": tif_path,
            "jpg": jpg_path,
        }


def compose_and_stretch(
    siril: SirilInterface,
    stacks: dict[str, Path],
    output_dir: Path,
    job_type: str,
    palette: str = "HOO",
    logger: Optional[JobLogger] = None,
) -> dict[str, Path]:
    """
    Compose and stretch based on job type.

    Returns dict with unstretched and stretched output paths.
    """
    composer = Composer(siril, logger)

    if job_type == "LRGB":
        unstretched = composer.compose_lrgb(stacks, output_dir)
        name = "lrgb"
    else:  # SHO or HOO
        unstretched = composer.compose_narrowband(stacks, output_dir, palette)
        name = "sho" if job_type == "SHO" else "hoo"

    stretched = composer.stretch(unstretched, output_dir, name)

    return {
        "unstretched": unstretched,
        **stretched,
    }
