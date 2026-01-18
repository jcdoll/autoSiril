"""
Registration statistics dataclass for sequence analysis.

Contains FWHM statistics and bimodality analysis results.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RegistrationStats:
    """FWHM statistics from a registered sequence."""

    n_images: int
    fwhm_values: np.ndarray
    wfwhm_values: np.ndarray
    roundness_values: np.ndarray
    metric_values: np.ndarray  # Registration quality score (higher = better)
    image_indices: np.ndarray  # 1-based indices matching wfwhm_values

    # Reference image info
    reference_index: int  # 1-based index of current reference
    reference_wfwhm: float  # wFWHM of reference image (0 if not in data)

    # Basic statistics
    median: float
    mean: float
    std: float
    cv: float  # coefficient of variation (std/mean)
    skewness: float  # distribution skewness (>0 = right tail)
    mad: float  # median absolute deviation
    q1: float
    q3: float

    # Bimodality analysis
    is_bimodal: bool = False
    delta_bic: float = 0.0
    dip_pvalue: float = 1.0
    gmm_means: Optional[np.ndarray] = None
    gmm_stds: Optional[np.ndarray] = None
    gmm_weights: Optional[np.ndarray] = None

    # Computed threshold
    threshold: Optional[float] = None
    threshold_reason: str = ""
    filter_case: str = ""  # bimodal, skewed, broad, tight
    n_rejected: int = 0

    # Histogram data
    hist_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    hist_counts: np.ndarray = field(default_factory=lambda: np.array([]))
