"""
FITS header reading utilities for Siril job processing.
"""

from pathlib import Path
from typing import Optional

from .config import DEFAULTS, Config
from .models import ClippingInfo, FrameInfo

try:
    from astropy.io import fits
except ImportError:
    fits = None


# Common FITS keyword variations
EXPOSURE_KEYWORDS = ["EXPTIME", "EXPOSURE", "EXP_TIME", "EXPOTIME"]
TEMPERATURE_KEYWORDS = ["CCD-TEMP", "CCD_TEMP", "CCDTEMP", "TEMP", "SENSOR-TEMP"]
FILTER_KEYWORDS = ["FILTER", "FILTER1", "FILTNAM", "FWHEEL"]
GAIN_KEYWORDS = ["GAIN", "CCDGAIN", "EGAIN"]


def _get_header_value(header, keywords: list[str], default=None):
    """Try multiple keywords and return first found value."""
    for kw in keywords:
        if kw in header:
            return header[kw]
    return default


def read_fits_header(path: Path) -> Optional[FrameInfo]:
    """
    Read relevant info from a FITS file header.

    Returns None if file cannot be read or required info is missing.
    """
    if fits is None:
        raise ImportError(
            "astropy is required for FITS reading. Install with: pip install astropy"
        )

    path = Path(path)
    if not path.exists():
        return None

    try:
        with fits.open(path) as hdul:
            header = hdul[0].header

            exposure = _get_header_value(header, EXPOSURE_KEYWORDS)
            temperature = _get_header_value(header, TEMPERATURE_KEYWORDS)
            filter_name = _get_header_value(header, FILTER_KEYWORDS, "Unknown")
            gain = _get_header_value(header, GAIN_KEYWORDS)

            if exposure is None:
                return None

            if temperature is None:
                temperature = DEFAULTS.default_temperature

            return FrameInfo(
                path=path,
                exposure=float(exposure),
                temperature=float(temperature),
                filter_name=str(filter_name).strip(),
                gain=int(gain) if gain is not None else None,
            )

    except Exception:
        return None


def scan_directory(directory: Path, pattern: str = "*.fit") -> list[FrameInfo]:
    """
    Scan a directory for FITS files and read their headers.

    Args:
        directory: Directory to scan
        pattern: Glob pattern for FITS files (default: *.fit)

    Returns:
        List of FrameInfo for successfully read files
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    frames = []
    for path in directory.glob(pattern):
        if path.is_file():
            info = read_fits_header(path)
            if info is not None:
                frames.append(info)

    return frames


def scan_multiple_directories(
    directories: list[Path], pattern: str = "*.fit"
) -> list[FrameInfo]:
    """Scan multiple directories and combine results."""
    frames = []
    for directory in directories:
        frames.extend(scan_directory(directory, pattern))
    return frames


def temperatures_match(temp1: float, temp2: float, tolerance: float = 2.0) -> bool:
    """Check if two temperatures match within tolerance."""
    return abs(temp1 - temp2) <= tolerance


def check_clipping(path: Path, config: Config = DEFAULTS) -> Optional[ClippingInfo]:
    """
    Check a FITS file for clipped pixels (both black and white).

    Args:
        path: Path to FITS file
        config: Configuration with clipping thresholds

    Returns:
        ClippingInfo with clipping statistics, or None if file cannot be read
    """
    if fits is None:
        raise ImportError("astropy is required for FITS reading")

    path = Path(path)
    if not path.exists():
        return None

    try:
        with fits.open(path) as hdul:
            data = hdul[0].data
            if data is None:
                return None

            import numpy as np

            # Determine thresholds based on dtype
            if data.dtype == np.uint16:
                bit_depth = 16
                low_threshold = config.clipping_low_16bit
                high_threshold = config.clipping_high_16bit
            elif data.dtype == np.uint8:
                bit_depth = 8
                low_threshold = config.clipping_low_8bit
                high_threshold = config.clipping_high_8bit
            elif data.dtype in (np.float32, np.float64):
                bit_depth = 16
                max_val = float(data.max())
                if max_val <= config.float_normalized_threshold:  # Normalized data
                    low_threshold = config.clipping_low_float
                    high_threshold = config.clipping_high_float
                else:
                    low_threshold = config.clipping_low_16bit
                    high_threshold = config.clipping_high_16bit
            else:
                bit_depth = 16
                low_threshold = config.clipping_low_16bit
                high_threshold = config.clipping_high_16bit

            total_pixels = data.size
            clipped_low = int((data <= low_threshold).sum())
            clipped_high = int((data >= high_threshold).sum())

            return ClippingInfo(
                path=path,
                total_pixels=total_pixels,
                clipped_low=clipped_low,
                clipped_high=clipped_high,
                bit_depth=bit_depth,
            )

    except Exception:
        return None
