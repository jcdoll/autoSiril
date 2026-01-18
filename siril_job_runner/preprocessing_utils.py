"""
Preprocessing utility functions.

Provides file operations and frame grouping for preprocessing.
"""

import os
import shutil
from collections import defaultdict
from pathlib import Path

from .models import FrameInfo, StackGroup


def link_or_copy(src: Path, dest: Path) -> None:
    """Hard link if possible, otherwise copy."""
    try:
        os.link(src, dest)
    except OSError:
        # Cross-device link or unsupported filesystem, fall back to copy
        shutil.copy2(src, dest)


def create_sequence_file(seq_path: Path, num_images: int, seq_name: str) -> None:
    """
    Create a Siril .seq file directly.

    Format matches pysiril's CreateSeqFile output:
    - Header comments
    - S line: sequence metadata (fixed_len=5 for 5-digit numbering)
    - L line: layer count (-1 = auto)
    - I lines: one per image (index, included flag)
    """
    with open(seq_path, "w", newline="") as f:
        f.write(
            "#Siril sequence file. "
            "Contains list of files (images), selection, and registration data\n"
        )
        f.write(
            "#S 'sequence_name' start_index nb_images nb_selected "
            "fixed_len reference_image version\n"
        )
        # S 'name' start nb_images nb_selected fixed_len ref_image version
        # fixed_len=5 means 5-digit numbering (00001, 00002, etc.)
        f.write(f"S '{seq_name}' 1 {num_images} {num_images} 5 -1 1\n")
        f.write("L -1\n")
        for i in range(1, num_images + 1):
            f.write(f"I {i} 1\n")


def group_frames_by_filter_exposure(frames: list[FrameInfo]) -> list[StackGroup]:
    """
    Group frames by (filter, exposure) for separate stacking.

    Returns list of StackGroup, sorted by filter then exposure.
    """
    groups: dict[tuple[str, float], list[FrameInfo]] = defaultdict(list)

    for frame in frames:
        key = (frame.filter_name, frame.exposure)
        groups[key].append(frame)

    result = []
    for (filter_name, exposure), frame_list in sorted(groups.items()):
        result.append(
            StackGroup(
                filter_name=filter_name,
                exposure=exposure,
                frames=frame_list,
            )
        )

    return result
