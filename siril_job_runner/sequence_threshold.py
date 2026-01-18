"""
Adaptive threshold computation for sequence filtering.

Uses bimodality detection and distribution analysis to determine
optimal FWHM filtering thresholds.
"""

from typing import Optional

import numpy as np
from diptest import diptest
from sklearn.mixture import GaussianMixture

from .config import DEFAULTS, Config
from .sequence_stats import RegistrationStats


def detect_bimodality(
    fwhm: np.ndarray,
    bic_threshold: float = 10.0,
    dip_alpha: float = 0.05,
) -> tuple[bool, float, float, Optional[GaussianMixture]]:
    """
    Detect bimodality using GMM + BIC with dip test confirmation.

    Uses two statistical tests that must both pass:
    1. BIC comparison: 2-component GMM must fit significantly better than 1-component
    2. Dip test: Distribution must have a significant "dip" indicating two modes

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
    _dip_stat, dip_pvalue = diptest(fwhm)

    # Bimodal if both tests agree
    is_bimodal = (delta_bic > bic_threshold) and (dip_pvalue < dip_alpha)

    return is_bimodal, delta_bic, dip_pvalue, gmm2 if is_bimodal else None


def compute_adaptive_threshold(
    stats: RegistrationStats,
    config: Config = DEFAULTS,
) -> RegistrationStats:
    """
    Compute adaptive FWHM threshold based on distribution shape.

    Decision tree:
        1. Bimodal (GMM+dip): threshold at midpoint between modes
           - Cleanly separates two distinct image quality populations

        2. Skewed (skewness > threshold): threshold at median + k*MAD
           - Aggressively cuts the long tail of poor quality images

        3. Broad symmetric (high CV, low skew): threshold at high percentile
           - Permissive filtering, only removes extreme outliers

        4. Tight symmetric (low CV): keep all images
           - All images are similar quality, no filtering needed

    Args:
        stats: RegistrationStats from parse_sequence_file
        config: Configuration with threshold parameters

    Returns:
        Updated RegistrationStats with threshold and analysis
    """
    # Use wFWHM since that's what Siril's seqapplyreg filters on
    fwhm = stats.wfwhm_values

    # Case 0: Not enough images for statistical filtering
    if stats.n_images < config.fwhm_min_images:
        stats.threshold = None
        stats.filter_case = "insufficient"
        stats.threshold_reason = (
            f"Too few images ({stats.n_images} < {config.fwhm_min_images})"
        )
        return stats

    # Check for bimodality
    is_bimodal, delta_bic, dip_pvalue, gmm2 = detect_bimodality(
        fwhm,
        bic_threshold=config.fwhm_bic_threshold,
        dip_alpha=config.fwhm_dip_alpha,
    )

    stats.is_bimodal = is_bimodal
    stats.delta_bic = delta_bic
    stats.dip_pvalue = dip_pvalue

    # Case 1: Bimodal distribution - two distinct populations
    if is_bimodal and gmm2 is not None:
        means = gmm2.means_.flatten()
        stds = np.sqrt(gmm2.covariances_.flatten())
        weights = gmm2.weights_

        stats.gmm_means = means
        stats.gmm_stds = stds
        stats.gmm_weights = weights

        # Find lower and upper modes
        lower_idx = np.argmin(means)
        lower_mean = means[lower_idx]
        lower_std = stds[lower_idx]
        upper_mean = means[np.argmax(means)]

        # Threshold = min of (lower_mean + k*sigma) or midpoint
        # This ensures we cleanly separate the two modes
        sigma_threshold = lower_mean + config.fwhm_bimodal_sigma * lower_std
        midpoint_threshold = (lower_mean + upper_mean) / 2

        if sigma_threshold < midpoint_threshold:
            threshold = sigma_threshold
            stats.threshold_reason = (
                f"{lower_mean:.2f} + {config.fwhm_bimodal_sigma}*{lower_std:.2f}"
            )
        else:
            threshold = midpoint_threshold
            stats.threshold_reason = f"midpoint({lower_mean:.2f}, {upper_mean:.2f})"

        stats.threshold = float(threshold)
        stats.filter_case = "bimodal"

    # Case 2: Skewed distribution - long tail on right
    elif stats.skewness > config.fwhm_skew_threshold:
        # Use median + k*MAD for robust tail cutting
        threshold = stats.median + config.fwhm_skew_mad_factor * stats.mad
        stats.threshold = float(threshold)
        stats.filter_case = "skewed"
        stats.threshold_reason = (
            f"median({stats.median:.2f}) + "
            f"{config.fwhm_skew_mad_factor}*MAD({stats.mad:.2f})"
        )

    # Case 3: Broad symmetric distribution - high variance but not skewed
    elif stats.cv > config.fwhm_cv_threshold:
        threshold = np.percentile(fwhm, config.fwhm_broad_percentile)
        stats.threshold = float(threshold)
        stats.filter_case = "broad"
        stats.threshold_reason = f"P{config.fwhm_broad_percentile:.0f}"

    # Case 4: Tight symmetric distribution - keep all
    else:
        stats.threshold = None
        stats.filter_case = "tight"
        stats.threshold_reason = f"CV={stats.cv:.1%} < {config.fwhm_cv_threshold:.0%}"

    # Count how many would be rejected
    if stats.threshold is not None:
        stats.n_rejected = int(np.sum(fwhm > stats.threshold))

    return stats
