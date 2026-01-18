"""
Sequence file analysis for adaptive FWHM filtering.

Parses Siril .seq files after registration to extract FWHM statistics
and compute adaptive thresholds based on distribution shape.

Adaptive Filtering Decision Tree:
    1. Bimodal (GMM+dip test): Two distinct populations of images.
       -> Threshold at midpoint between modes to reject the worse mode.

    2. Skewed (long tail): Single population with outliers on high end.
       -> Threshold at median + k*MAD (aggressive tail cutting).

    3. Broad symmetric (high CV, low skew): Wide but symmetric distribution.
       -> Threshold at high percentile (permissive, just cut extremes).

    4. Tight symmetric (low CV): All images similar quality.
       -> Keep all images, no filtering needed.
"""

from typing import Optional

import numpy as np

from .sequence_parse import parse_sequence_file
from .sequence_stats import RegistrationStats
from .sequence_threshold import compute_adaptive_threshold, detect_bimodality

# Re-export for backwards compatibility
__all__ = [
    "RegistrationStats",
    "parse_sequence_file",
    "detect_bimodality",
    "compute_adaptive_threshold",
    "format_histogram",
    "format_stats_log",
    "find_valid_reference",
]


def format_histogram(stats: RegistrationStats, width: int = 30) -> list[str]:
    """
    Format FWHM distribution as text histogram.

    Args:
        stats: RegistrationStats with histogram data
        width: Maximum bar width in characters

    Returns:
        List of histogram lines
    """
    if len(stats.hist_counts) == 0:
        return []

    lines = ["wFWHM distribution:"]
    max_count = max(stats.hist_counts) if len(stats.hist_counts) > 0 else 1

    for i, count in enumerate(stats.hist_counts):
        if count == 0:
            continue

        bin_start = stats.hist_bins[i]
        bin_end = stats.hist_bins[i + 1]
        pct = 100 * count / stats.n_images

        # Scale bar to width
        bar_len = int(width * count / max_count) if max_count > 0 else 0
        bar = "#" * bar_len

        # Mark threshold position
        threshold_marker = ""
        if stats.threshold is not None:
            if bin_start <= stats.threshold < bin_end:
                threshold_marker = " <-- threshold"
            elif bin_start >= stats.threshold and i > 0:
                if stats.hist_bins[i - 1] < stats.threshold:
                    threshold_marker = " [rejected]"
            elif bin_start >= stats.threshold:
                threshold_marker = " [rejected]"

        lines.append(
            f"  {bin_start:4.0f}-{bin_end:4.0f}px: {bar:<{width}} "
            f"{count:3d} ({pct:5.1f}%){threshold_marker}"
        )

    return lines


def format_stats_log(stats: RegistrationStats) -> list[str]:
    """
    Format registration stats for logging.

    Returns list of log lines including histogram and threshold decision.
    """
    lines = []

    # Histogram first for visual overview
    lines.extend(format_histogram(stats))
    lines.append("")  # blank line

    # Basic statistics
    lines.append(
        f"Stats: N={stats.n_images}, median={stats.median:.2f}px, "
        f"std={stats.std:.2f}, CV={stats.cv:.1%}, skew={stats.skewness:.2f}"
    )

    # Registration quality metrics
    metric_min = np.min(stats.metric_values)
    metric_max = np.max(stats.metric_values)
    metric_median = np.median(stats.metric_values)
    low_metric_count = np.sum(stats.metric_values < 500)
    lines.append(
        f"Metric: min={metric_min:.0f}, median={metric_median:.0f}, "
        f"max={metric_max:.0f}, low(<500)={low_metric_count}"
    )

    # Bimodality test results (if we have enough images)
    if stats.n_images >= 10:
        bimodal_str = "BIMODAL" if stats.is_bimodal else "unimodal"
        lines.append(
            f"Bimodality: dBIC={stats.delta_bic:.1f}, "
            f"dip_p={stats.dip_pvalue:.3f} -> {bimodal_str}"
        )

    # GMM mode details if bimodal
    if stats.is_bimodal and stats.gmm_means is not None:
        sorted_idx = np.argsort(stats.gmm_means)
        mode_strs = [
            f"{stats.gmm_means[i]:.2f}px (w={stats.gmm_weights[i]:.2f})"
            for i in sorted_idx
        ]
        lines.append(f"GMM modes: {', '.join(mode_strs)}")

    # Threshold decision
    case_descriptions = {
        "bimodal": "Two populations detected",
        "skewed": "Long tail detected",
        "broad": "Broad symmetric distribution",
        "tight": "Tight distribution, keeping all",
        "insufficient": "Too few images for filtering",
    }
    case_desc = case_descriptions.get(stats.filter_case, stats.filter_case)

    if stats.threshold is not None:
        lines.append(
            f"Decision: {case_desc} -> threshold={stats.threshold:.2f}px "
            f"({stats.threshold_reason})"
        )
        lines.append(f"Result: Rejecting {stats.n_rejected}/{stats.n_images} images")
    else:
        lines.append(
            f"Decision: {case_desc} -> no filtering ({stats.threshold_reason})"
        )

    return lines


def find_valid_reference(stats: RegistrationStats) -> Optional[int]:
    """
    Find the best reference image that passes the threshold filter.

    If the current reference would be filtered out, returns the index of
    the image with the lowest wFWHM that passes the threshold.

    Args:
        stats: RegistrationStats with threshold computed

    Returns:
        1-based image index for new reference, or None if no change needed
    """
    if stats.threshold is None:
        return None

    # Check if current reference passes the filter
    if stats.reference_wfwhm > 0 and stats.reference_wfwhm <= stats.threshold:
        return None

    # Find images that pass the filter
    passing_mask = stats.wfwhm_values <= stats.threshold
    if not np.any(passing_mask):
        return None

    # Pick the one with lowest wFWHM (best quality)
    passing_wfwhm = stats.wfwhm_values[passing_mask]
    passing_indices = stats.image_indices[passing_mask]
    best_idx = np.argmin(passing_wfwhm)

    return int(passing_indices[best_idx])
