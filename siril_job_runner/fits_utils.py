"""
FITS header reading utilities for Siril job processing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from collections import defaultdict

try:
    from astropy.io import fits
except ImportError:
    fits = None


@dataclass
class FrameInfo:
    """Information extracted from a FITS frame header."""

    path: Path
    exposure: float  # seconds
    temperature: float  # Celsius
    filter_name: str
    gain: Optional[int] = None

    @property
    def exposure_str(self) -> str:
        """Exposure as string for matching (e.g., '300s')."""
        return f"{int(self.exposure)}s"

    @property
    def temp_str(self) -> str:
        """Temperature as string for matching (e.g., '-10C')."""
        return f"{int(round(self.temperature))}C"


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
        raise ImportError("astropy is required for FITS reading. Install with: pip install astropy")

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

            # Default temperature if not found (some cameras don't report)
            if temperature is None:
                temperature = 0.0

            return FrameInfo(
                path=path,
                exposure=float(exposure),
                temperature=float(temperature),
                filter_name=str(filter_name).strip(),
                gain=int(gain) if gain is not None else None,
            )

    except Exception:
        return None


def scan_directory(directory: Path, pattern: str = "*.fit*") -> list[FrameInfo]:
    """
    Scan a directory for FITS files and read their headers.

    Args:
        directory: Directory to scan
        pattern: Glob pattern for FITS files (default: *.fit*)

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


def scan_multiple_directories(directories: list[Path], pattern: str = "*.fit*") -> list[FrameInfo]:
    """Scan multiple directories and combine results."""
    frames = []
    for directory in directories:
        frames.extend(scan_directory(directory, pattern))
    return frames


@dataclass
class RequirementsEntry:
    """Entry in the requirements table."""

    filter_name: str
    exposure: float
    temperature: float
    count: int

    @property
    def exposure_str(self) -> str:
        return f"{int(self.exposure)}s"

    @property
    def temp_str(self) -> str:
        return f"{int(round(self.temperature))}C"


def build_requirements_table(frames: list[FrameInfo]) -> list[RequirementsEntry]:
    """
    Build a requirements table from a list of frames.

    Groups frames by (filter, exposure, temperature) and counts them.
    """
    counts: dict[tuple[str, float, float], int] = defaultdict(int)

    for frame in frames:
        # Round temperature to nearest integer for grouping
        rounded_temp = round(frame.temperature)
        key = (frame.filter_name, frame.exposure, rounded_temp)
        counts[key] += 1

    entries = []
    for (filter_name, exposure, temp), count in sorted(counts.items()):
        entries.append(RequirementsEntry(
            filter_name=filter_name,
            exposure=exposure,
            temperature=temp,
            count=count,
        ))

    return entries


def get_unique_exposures(frames: list[FrameInfo]) -> set[float]:
    """Get unique exposure times from frames."""
    return {frame.exposure for frame in frames}


def get_unique_temperatures(frames: list[FrameInfo]) -> set[float]:
    """Get unique temperatures from frames (rounded to int)."""
    return {round(frame.temperature) for frame in frames}


def get_unique_filters(frames: list[FrameInfo]) -> set[str]:
    """Get unique filter names from frames."""
    return {frame.filter_name for frame in frames}


def temperatures_match(temp1: float, temp2: float, tolerance: float = 2.0) -> bool:
    """Check if two temperatures match within tolerance."""
    return abs(temp1 - temp2) <= tolerance
