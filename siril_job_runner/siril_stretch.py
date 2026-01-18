"""
Siril stretching commands.

Provides mixin class for image stretching operations.
"""

from typing import Optional


class SirilStretchMixin:
    """Mixin for Siril stretching operations."""

    def execute(self, command: str) -> bool:
        """Execute a raw Siril command string. Must be implemented by subclass."""
        raise NotImplementedError

    # Deconvolution

    def makepsf(
        self,
        method: str = "stars",
        symmetric: bool = True,
        save_psf: Optional[str] = None,
    ) -> bool:
        """
        Create PSF for deconvolution.

        Args:
            method: "stars" (from image stars), "blind" (estimate from image)
            symmetric: For stars method, force symmetric PSF
            save_psf: Optional path to save PSF image
        """
        cmd = f"makepsf {method}"
        if method == "stars" and symmetric:
            cmd += " -sym"
        if save_psf:
            cmd += f" -savepsf={save_psf.replace(chr(92), '/')}"
        return self.execute(cmd)

    def rl(
        self,
        iters: int = 10,
        regularization: str = "tv",
        alpha: float = 0.001,
    ) -> bool:
        """
        Richardson-Lucy deconvolution.

        Args:
            iters: Number of iterations (default 10)
            regularization: "tv" (Total Variation) or "fh" (Frobenius Hessian)
            alpha: Regularization strength (higher = smoother, default 0.001)
        """
        cmd = f"rl -iters={iters} -{regularization} -alpha={alpha}"
        return self.execute(cmd)

    # Stretching

    def autostretch(
        self,
        linked: bool = True,
        shadowclip: float = -2.8,
        targetbg: float = 0.10,
    ) -> bool:
        """
        Auto-stretch image.

        Args:
            linked: Use linked stretch (same for all channels)
            shadowclip: Shadows clipping point in sigma from histogram peak
            targetbg: Target background brightness [0,1], lower = darker
        """
        cmd = "autostretch"
        if linked:
            cmd += " -linked"
        cmd += f" {shadowclip} {targetbg}"
        return self.execute(cmd)

    def mtf(self, low: float, mid: float, high: float) -> bool:
        """Midtone transfer function."""
        return self.execute(f"mtf {low} {mid} {high}")

    def linstretch(self, bp: float) -> bool:
        """
        Linear stretch to set black point.

        Stretches the image linearly so that bp becomes the new black point.
        Applies to all channels by default (preserves color).

        Args:
            bp: Black point value (0-1) - pixels at this value become 0
        """
        return self.execute(f"linstretch -BP={bp}")

    def modasinh(
        self,
        D: float,
        LP: float = 0.0,
        SP: float = 0.0,
        HP: float = 1.0,
    ) -> bool:
        """
        Modified arcsinh stretch.

        More stable than GHT for automation, similar parameters.

        Args:
            D: Stretch strength (0-10)
            LP: Shadow protection - linear range from 0 to SP
            SP: Symmetry point (0-1) - where stretch intensity peaks
            HP: Highlight protection - linear range from HP to 1
        """
        cmd = f"modasinh -D={D}"
        if LP != 0.0:
            cmd += f" -LP={LP}"
        if SP != 0.0:
            cmd += f" -SP={SP}"
        if HP != 1.0:
            cmd += f" -HP={HP}"
        return self.execute(cmd)

    def ght(
        self,
        D: float,
        B: float = 0.0,
        LP: float = 0.0,
        SP: float = 0.0,
        HP: float = 1.0,
    ) -> bool:
        """
        Generalized Hyperbolic Stretch.

        Full control stretch with focal point parameter.

        Args:
            D: Stretch strength (0-10)
            B: Focal point / local stretch intensity (-5 to 15)
            LP: Shadow protection - linear range from 0 to SP
            SP: Symmetry point (0-1) - where stretch intensity peaks
            HP: Highlight protection - linear range from HP to 1
        """
        cmd = f"ght -D={D}"
        if B != 0.0:
            cmd += f" -B={B}"
        if LP != 0.0:
            cmd += f" -LP={LP}"
        if SP != 0.0:
            cmd += f" -SP={SP}"
        if HP != 1.0:
            cmd += f" -HP={HP}"
        return self.execute(cmd)

    def autoghs(
        self,
        shadowsclip: float = 0.0,
        D: float = 3.0,
        B: float = 0.0,
        LP: float = 0.0,
        HP: float = 1.0,
        linked: bool = True,
    ) -> bool:
        """
        Automatic Generalized Hyperbolic Stretch.

        Calculates SP automatically from image statistics (k*sigma from median).
        More robust for automation than fixed SP values.

        Args:
            shadowsclip: k value for SP calculation (sigma from median, can be negative)
            D: Stretch strength (0-10)
            B: Focal point / local stretch intensity (-5 to 15)
            LP: Shadow protection - linear range from 0 to SP
            HP: Highlight protection - linear range from HP to 1
            linked: Calculate SP as mean across channels (True) or per-channel (False)
        """
        cmd = f"autoghs {shadowsclip} {D}"
        if linked:
            cmd = f"autoghs -linked {shadowsclip} {D}"
        if B != 0.0:
            cmd += f" -b={B}"
        if LP != 0.0:
            cmd += f" -lp={LP}"
        if HP != 1.0:
            cmd += f" -hp={HP}"
        return self.execute(cmd)
