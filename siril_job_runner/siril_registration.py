"""
Siril registration and stacking commands.

Provides mixin class for calibration, registration, and stacking operations.
"""

from typing import Optional


class SirilRegistrationMixin:
    """Mixin for Siril calibration, registration, and stacking operations."""

    def execute(self, command: str) -> bool:
        """Execute a raw Siril command string. Must be implemented by subclass."""
        raise NotImplementedError

    # Calibration

    def calibrate(
        self,
        name: str,
        bias: Optional[str] = None,
        dark: Optional[str] = None,
        flat: Optional[str] = None,
    ) -> bool:
        """Calibrate a sequence."""
        cmd = f"calibrate {name}"
        if bias:
            cmd += f" -bias={bias.replace(chr(92), '/')}"
        if dark:
            cmd += f" -dark={dark.replace(chr(92), '/')}"
        if flat:
            cmd += f" -flat={flat.replace(chr(92), '/')}"
        return self.execute(cmd)

    # Background extraction

    def subsky(
        self,
        rbf: bool = True,
        degree: int = 1,
        samples: int = 20,
        tolerance: float = 1.0,
        smooth: float = 0.5,
    ) -> bool:
        """Background extraction on loaded image."""
        method = "-rbf" if rbf else str(degree)
        cmd = f"subsky {method}"
        cmd += f" -samples={samples} -tolerance={tolerance} -smooth={smooth}"
        return self.execute(cmd)

    # Registration

    def register(self, name: str, twopass: bool = False) -> bool:
        """Register a sequence."""
        cmd = f"register {name}"
        if twopass:
            cmd += " -2pass"
        return self.execute(cmd)

    def setref(self, name: str, index: int) -> bool:
        """Set reference image for a sequence (1-based index)."""
        return self.execute(f"setref {name} {index}")

    def seqapplyreg(
        self,
        name: str,
        framing: Optional[str] = None,
        filter_fwhm: Optional[float] = None,
    ) -> bool:
        """
        Apply registration to sequence.

        Args:
            name: Sequence name
            framing: Framing mode (e.g., "min", "max", "current")
            filter_fwhm: Absolute FWHM threshold in pixels (filters wFWHM)
        """
        cmd = f"seqapplyreg {name}"
        if framing:
            cmd += f" -framing={framing}"
        if filter_fwhm is not None:
            cmd += f" -filter-wfwhm={filter_fwhm:.2f}"
        return self.execute(cmd)

    # Stacking

    def stack(
        self,
        name: str,
        rejection: str = "rej",
        weight: str = "w",
        sigma_low: str = "3",
        sigma_high: str = "3",
        norm: Optional[str] = None,
        fastnorm: bool = False,
        out: Optional[str] = None,
    ) -> bool:
        """Stack a sequence."""
        cmd = f"stack {name} {rejection} {weight} {sigma_low} {sigma_high}"
        if norm:
            cmd += f" -norm={norm}"
        if fastnorm:
            cmd += " -fastnorm"
        if out:
            out = out.replace("\\", "/")
            cmd += f" -out={out}"
        return self.execute(cmd)

    # Linear matching

    def linear_match(self, ref: str, low: float = 0, high: float = 0.92) -> bool:
        """Linear match current image to reference."""
        return self.execute(f"linear_match {ref} {low} {high}")

    # Composition

    def rgbcomp(
        self,
        r: Optional[str] = None,
        g: Optional[str] = None,
        b: Optional[str] = None,
        lum: Optional[str] = None,
        rgb: Optional[str] = None,
        out: Optional[str] = None,
    ) -> bool:
        """
        RGB composition.

        Either provide r, g, b for RGB composition,
        or lum + rgb to add luminance to existing RGB.
        """
        if lum and rgb:
            # Adding luminance to RGB
            cmd = f"rgbcomp -lum={lum} {rgb}"
        elif r and g and b:
            # Creating RGB from channels
            cmd = f"rgbcomp {r} {g} {b}"
        else:
            raise ValueError("Either provide r,g,b or lum+rgb")

        if out:
            cmd += f" -out={out}"
        return self.execute(cmd)
