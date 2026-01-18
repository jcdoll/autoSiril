"""
Wrapper for pysiril to provide a convenient method-based interface.

pysiril uses Execute() for all commands. This wrapper provides
typed methods that build command strings internally.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits

from .siril_color import SirilColorMixin
from .siril_file_ops import SirilFileOpsMixin
from .siril_registration import SirilRegistrationMixin
from .siril_stretch import SirilStretchMixin


class SirilWrapper(
    SirilFileOpsMixin,
    SirilRegistrationMixin,
    SirilStretchMixin,
    SirilColorMixin,
):
    """
    Wrapper around pysiril's Execute() method.

    Provides convenient methods for common Siril commands.
    All methods internally call Execute() with the appropriate command string.
    """

    def __init__(self, siril):
        """
        Initialize wrapper with a pysiril Siril instance.

        Args:
            siril: pysiril.siril.Siril instance (already Open()'d)
        """
        self._siril = siril

    def execute(self, command: str) -> bool:
        """Execute a raw Siril command string."""
        return self._siril.Execute(command)

    # Statistics

    def get_image_stats(self, filepath: Path) -> dict[str, float]:
        """
        Get image statistics from a FITS file using astropy.

        Args:
            filepath: Path to FITS file

        Returns:
            Dict with median, mean, std, min, max for the image.
            For RGB images, returns stats for combined luminance.
        """
        with fits.open(filepath) as hdul:
            data = hdul[0].data.astype(np.float64)

        # Handle different data layouts
        if data.ndim == 3:
            # RGB image - compute luminance (simple average)
            if data.shape[0] == 3:
                # (3, H, W) format
                luminance = np.mean(data, axis=0)
            else:
                # (H, W, 3) format
                luminance = np.mean(data, axis=2)
            data = luminance

        # Normalize to 0-1 if needed (16-bit data)
        if data.max() > 1.5:
            data = data / 65535.0

        return {
            "median": float(np.median(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }
