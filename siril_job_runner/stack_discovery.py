"""
Stack discovery utilities for composition.

Parses stacked image filenames to extract filter and exposure information.
"""

import re
from pathlib import Path

from .models import StackInfo

# Narrowband palette definitions (channel mappings)
PALETTES = {
    "HOO": {"R": "H", "G": "O", "B": "O"},
    "SHO": {"R": "S", "G": "H", "B": "O"},
}


def discover_stacks(stacks_dir: Path) -> dict[str, list[StackInfo]]:
    """
    Discover stacks in the stacks directory.

    Parses filenames like `stack_L_180s.fit` to extract filter and exposure.

    Returns:
        Dict mapping filter name to list of StackInfo (multiple if HDR)
    """
    pattern = re.compile(r"^stack_([A-Z]+)_(\d+)s\.fit$")
    result: dict[str, list[StackInfo]] = {}

    for path in stacks_dir.glob("stack_*_*s.fit"):
        match = pattern.match(path.name)
        if match:
            filter_name = match.group(1)
            exposure = int(match.group(2))
            info = StackInfo(path=path, filter_name=filter_name, exposure=exposure)

            if filter_name not in result:
                result[filter_name] = []
            result[filter_name].append(info)

    # Sort each filter's stacks by exposure
    for filter_name in result:
        result[filter_name].sort(key=lambda s: s.exposure)

    return result


def is_hdr_mode(stacks: dict[str, list[StackInfo]]) -> bool:
    """Check if any filter has multiple exposures (HDR mode)."""
    return any(len(stack_list) > 1 for stack_list in stacks.values())
