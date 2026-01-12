"""
Sequence file analysis for adaptive FWHM filtering.

Parses Siril .seq files after registration to extract FWHM statistics
and compute adaptive thresholds using GMM + dip test for bimodality detection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from diptest import diptest
from sklearn.mixture import GaussianMixture

from .config import DEFAULTS, Config


@dataclass
class RegistrationStats:
    """FWHM statistics from a registered sequence."""

    n_images: int
    fwhm_values: np.ndarray
    wfwhm_values: np.ndarray
    roundness_values: np.ndarray

    # Computed statistics
    median: float
    mean: float
    std: float
    cv: float  # coefficient of variation
    q1: float
    q3: float
    p90: float

    # Bimodality analysis
    is_bimodal: bool
    delta_bic: float
    dip_pvalue: float
    gmm_means: Optional[np.ndarray] = None
    gmm_stds: Optional[np.ndarray] = None
    gmm_weights: Optional[np.ndarray] = None

    # Computed threshold
    threshold: Optional[float] = None
    threshold_reason: str = ""
    n_rejected: int = 0


def parse_sequence_file(seq_path: Path) -> Optional[RegistrationStats]:
    """
    Parse a Siril .seq file and extract registration data.

    The .seq file contains R0 lines with format:
    R0 FWHM wFWHM roundness quality metric n_stars transform_type ...

    Args:
        seq_path: Path to .seq file

    Returns:
        RegistrationStats or None if parsing fails
    """
    if not seq_path.exists():
        return None

    fwhm_list = []
    wfwhm_list = []
    roundness_list = []

    with open(seq_path) as f:
        for line in f:
            if line.startswith("R0 "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        fwhm = float(parts[1])
                        wfwhm = float(parts[2])
                        roundness = float(parts[3])

                        # Skip reference image (has 0 0 nan) and invalid entries
                        if fwhm > 0 and not np.isnan(roundness):
                            fwhm_list.append(fwhm)
                            wfwhm_list.append(wfwhm)
                            roundness_list.append(roundness)
                    except (ValueError, IndexError):
                        continue

    if len(fwhm_list) == 0:
        return None

    fwhm = np.array(fwhm_list)
    wfwhm = np.array(wfwhm_list)
    roundness = np.array(roundness_list)

    return RegistrationStats(
        n_images=len(fwhm),
        fwhm_values=fwhm,
        wfwhm_values=wfwhm,
        roundness_values=roundness,
        median=float(np.median(fwhm)),
        mean=float(np.mean(fwhm)),
        std=float(np.std(fwhm)),
        cv=float(np.std(fwhm) / np.mean(fwhm)) if np.mean(fwhm) > 0 else 0.0,
        q1=float(np.percentile(fwhm, 25)),
        q3=float(np.percentile(fwhm, 75)),
        p90=float(np.percentile(fwhm, 90)),
        is_bimodal=False,
        delta_bic=0.0,
        dip_pvalue=1.0,
    )


def detect_bimodality(
    fwhm: np.ndarray,
    bic_threshold: float = 10.0,
    dip_alpha: float = 0.05,
) -> tuple[bool, float, float, Optional[GaussianMixture]]:
    """
    Detect bimodality using GMM + BIC with dip test confirmation.

    Args:
        fwhm: Array of FWHM values
        bic_threshold: delta-BIC threshold (>10 = very strong evidence)
        dip_alpha: significance level for dip test

    Returns:
        (is_bimodal, delta_bic, dip_pvalue, gmm2_model)
    """
    if len(fwhm) < 10:
        return False, 0.0, 1.0, None

    fwhm_reshaped = fwhm.reshape(-1, 1)

    # Fit 1-component and 2-component GMMs
    gmm1 = GaussianMixture(n_components=1, random_state=42)
    gmm2 = GaussianMixture(n_components=2, random_state=42)

    gmm1.fit(fwhm_reshaped)
    gmm2.fit(fwhm_reshaped)

    # Compare BIC (lower is better, so positive delta means 2-component is better)
    delta_bic = gmm1.bic(fwhm_reshaped) - gmm2.bic(fwhm_reshaped)

    # Hartigan's dip test for unimodality
    dip_stat, dip_pvalue = diptest(fwhm)

    # Bimodal if both tests agree
    is_bimodal = (delta_bic > bic_threshold) and (dip_pvalue < dip_alpha)

    return is_bimodal, delta_bic, dip_pvalue, gmm2 if is_bimodal else None


def compute_adaptive_threshold(
    stats: RegistrationStats,
    config: Config = DEFAULTS,
) -> RegistrationStats:
    """
    Compute adaptive FWHM threshold based on distribution analysis.

    Strategy:
    1. If bimodal: threshold = lower_mode_mean + bimodal_sigma * lower_mode_std
    2. If high CV (>cv_threshold): threshold = percentile
    3. If low CV: no filtering (keep all)

    Args:
        stats: RegistrationStats from parse_sequence_file
        config: Configuration with threshold parameters

    Returns:
        Updated RegistrationStats with threshold and analysis
    """
    fwhm = stats.fwhm_values

    # Not enough images for statistical filtering
    if stats.n_images < config.fwhm_min_images:
        stats.threshold = None
        stats.threshold_reason = (
            f"Too few images ({stats.n_images} < {config.fwhm_min_images})"
        )
        return stats

    # Detect bimodality
    is_bimodal, delta_bic, dip_pvalue, gmm2 = detect_bimodality(
        fwhm,
        bic_threshold=config.fwhm_bic_threshold,
        dip_alpha=config.fwhm_dip_alpha,
    )

    stats.is_bimodal = is_bimodal
    stats.delta_bic = delta_bic
    stats.dip_pvalue = dip_pvalue

    if is_bimodal and gmm2 is not None:
        # Extract GMM parameters
        means = gmm2.means_.flatten()
        stds = np.sqrt(gmm2.covariances_.flatten())
        weights = gmm2.weights_

        stats.gmm_means = means
        stats.gmm_stds = stds
        stats.gmm_weights = weights

        # Find lower and upper modes
        lower_idx = np.argmin(means)
        upper_idx = np.argmax(means)
        lower_mean = means[lower_idx]
        lower_std = stds[lower_idx]
        upper_mean = means[upper_idx]

        # Threshold = min of (lower_mean + k*sigma) or midpoint between modes
        # This ensures we don't include images from the upper mode
        sigma_threshold = lower_mean + config.fwhm_bimodal_sigma * lower_std
        midpoint_threshold = (lower_mean + upper_mean) / 2

        if sigma_threshold < midpoint_threshold:
            threshold = sigma_threshold
            stats.threshold_reason = (
                f"Bimodal (dBIC={delta_bic:.1f}, dip_p={dip_pvalue:.3f}): "
                f"{lower_mean:.2f} + {config.fwhm_bimodal_sigma}*{lower_std:.2f} = {threshold:.2f}"
            )
        else:
            threshold = midpoint_threshold
            stats.threshold_reason = (
                f"Bimodal (dBIC={delta_bic:.1f}, dip_p={dip_pvalue:.3f}): "
                f"midpoint({lower_mean:.2f}, {upper_mean:.2f}) = {threshold:.2f}"
            )
        stats.threshold = float(threshold)

    elif stats.cv > config.fwhm_cv_threshold:
        # High variance unimodal - use percentile
        threshold = np.percentile(fwhm, config.fwhm_unimodal_percentile)
        stats.threshold = float(threshold)
        stats.threshold_reason = (
            f"High variance (CV={stats.cv:.1%} > {config.fwhm_cv_threshold:.0%}): "
            f"P{config.fwhm_unimodal_percentile:.0f}={threshold:.2f}"
        )

    else:
        # Low variance - keep all images
        stats.threshold = None
        stats.threshold_reason = f"Low variance (CV={stats.cv:.1%}), keeping all"

    # Count how many would be rejected
    if stats.threshold is not None:
        stats.n_rejected = int(np.sum(fwhm > stats.threshold))

    return stats


def format_stats_log(stats: RegistrationStats) -> list[str]:
    """
    Format registration stats for logging.

    Returns list of log lines.
    """
    lines = [
        f"FWHM stats: N={stats.n_images}, "
        f"median={stats.median:.2f}, std={stats.std:.2f}, CV={stats.cv:.1%}",
        f"Quartiles: Q1={stats.q1:.2f}, Q3={stats.q3:.2f}, P90={stats.p90:.2f}",
    ]

    # Always log bimodality test results if we have enough images
    if stats.n_images >= 10:
        bimodal_str = "BIMODAL" if stats.is_bimodal else "unimodal"
        lines.append(
            f"Bimodality test: dBIC={stats.delta_bic:.1f}, dip_p={stats.dip_pvalue:.3f} -> {bimodal_str}"
        )

    if stats.is_bimodal and stats.gmm_means is not None:
        sorted_idx = np.argsort(stats.gmm_means)
        mode_strs = [
            f"{stats.gmm_means[i]:.2f}px (w={stats.gmm_weights[i]:.2f})"
            for i in sorted_idx
        ]
        lines.append(f"GMM modes: {', '.join(mode_strs)}")

    if stats.threshold is not None:
        lines.append(
            f"Threshold: {stats.threshold:.2f}px, "
            f"rejecting {stats.n_rejected}/{stats.n_images} images"
        )
        lines.append(f"Reason: {stats.threshold_reason}")
    else:
        lines.append(f"No filtering: {stats.threshold_reason}")

    return lines
