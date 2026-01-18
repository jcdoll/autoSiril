"""
Sequence file parsing for FWHM analysis.

Parses Siril .seq files after registration to extract FWHM statistics.
"""

import contextlib
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import median_abs_deviation, skew

from .sequence_stats import RegistrationStats


def parse_sequence_file(seq_path: Path) -> Optional[RegistrationStats]:
    """
    Parse a Siril .seq file and extract registration data.

    The .seq file contains:
    - S line: 'name' start nb_images nb_selected fixed_len reference_image version
    - R0 lines: FWHM wFWHM roundness quality metric n_stars transform_type ...

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
    metric_list = []
    index_list = []
    reference_index = -1
    current_image_index = 0

    with open(seq_path) as f:
        for line in f:
            # Parse S line for reference index
            # Format: S 'name' start nb_images nb_selected fixed_len reference_image version
            if line.startswith("S "):
                parts = line.split()
                if len(parts) >= 7:
                    with contextlib.suppress(ValueError, IndexError):
                        # reference_image is 1-based, -1 means auto
                        reference_index = int(parts[6])

            # Parse R0 lines for FWHM data
            # Format: R0 FWHM wFWHM roundness quality metric n_stars ...
            if line.startswith("R0 "):
                current_image_index += 1
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        fwhm = float(parts[1])
                        wfwhm = float(parts[2])
                        roundness = float(parts[3])
                        metric = float(parts[5])

                        # Skip reference image (has 0 0 nan) and invalid entries
                        if fwhm > 0 and not np.isnan(roundness):
                            fwhm_list.append(fwhm)
                            wfwhm_list.append(wfwhm)
                            roundness_list.append(roundness)
                            metric_list.append(metric)
                            index_list.append(current_image_index)
                    except (ValueError, IndexError):
                        continue

    if len(fwhm_list) == 0:
        return None

    fwhm = np.array(fwhm_list)
    wfwhm = np.array(wfwhm_list)
    roundness = np.array(roundness_list)
    metric = np.array(metric_list)
    indices = np.array(index_list)

    # Find reference wFWHM (0 if reference not in parsed data, e.g. if it was skipped)
    ref_wfwhm = 0.0
    if reference_index > 0:
        ref_mask = indices == reference_index
        if np.any(ref_mask):
            ref_wfwhm = float(wfwhm[ref_mask][0])

    # Use wFWHM for all analysis since Siril's seqapplyreg filters by wFWHM
    # Compute histogram bins (1px wide, from floor to ceil)
    bin_min = max(0, int(np.floor(wfwhm.min())))
    bin_max = int(np.ceil(wfwhm.max())) + 1
    hist_counts, hist_bins = np.histogram(wfwhm, bins=range(bin_min, bin_max + 1))

    return RegistrationStats(
        n_images=len(wfwhm),
        fwhm_values=fwhm,
        wfwhm_values=wfwhm,
        roundness_values=roundness,
        metric_values=metric,
        image_indices=indices,
        reference_index=reference_index,
        reference_wfwhm=ref_wfwhm,
        median=float(np.median(wfwhm)),
        mean=float(np.mean(wfwhm)),
        std=float(np.std(wfwhm)),
        cv=float(np.std(wfwhm) / np.mean(wfwhm)) if np.mean(wfwhm) > 0 else 0.0,
        skewness=float(skew(wfwhm)),
        mad=float(median_abs_deviation(wfwhm)),
        q1=float(np.percentile(wfwhm, 25)),
        q3=float(np.percentile(wfwhm, 75)),
        hist_bins=hist_bins,
        hist_counts=hist_counts,
    )
