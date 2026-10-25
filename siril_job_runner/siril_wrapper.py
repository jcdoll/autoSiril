"""
Wrapper for pysiril to provide a convenient method-based interface.

pysiril uses Execute() for all commands. This wrapper provides
typed methods that build command strings internally.
"""

import re
from pathlib import Path

import numpy as np
from astropy.io import fits

from .siril_color import SirilColorMixin
from .siril_file_ops import SirilFileOpsMixin
from .siril_registration import SirilRegistrationMixin
from .siril_stretch import SirilStretchMixin
from .veralux_core import normalized_image


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
        # Siril can omit a newline before an error status (notably makepsf).
        # pysiril otherwise treats that status as log text and waits forever.
        original_log = siril.tr.retoursiril

        def receive_log(line: str) -> None:
            """Recover a command status appended to a Siril log message."""
            match = re.search(r"status: (error|success)(?: .*)?$", line)
            if match is not None:
                original_log(line[: match.start()])
                siril.qStatus.put(f"status: {match.group(1)}")
            else:
                original_log(line)

        siril.tr.retoursiril = receive_log

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
        with fits.open(filepath, memmap=False) as hdul:
            data = normalized_image(hdul[0].data)

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

        return {
            "median": float(np.median(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }
