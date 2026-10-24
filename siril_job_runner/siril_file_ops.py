"""
Siril file operation commands.

Provides mixin class for file and directory operations.
"""

import shutil
import sys
from pathlib import Path
from typing import Optional


def quote_siril_arg(value: str) -> str:
    """Quote a complete Siril path argument, including any option prefix."""
    if any(char in value for char in ('"', "\n", "\r")):
        raise ValueError(
            "Siril path arguments cannot contain double quotes or newlines"
        )
    return '"' + value.replace("\\", "/") + '"'


def link_or_copy(src: Path, dst: Path) -> None:
    """
    Create symlink or copy file if symlinks unavailable (Windows without privileges).

    Args:
        src: Source file path
        dst: Destination path for link/copy
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if sys.platform == "win32":
        # Windows symlinks require admin or developer mode; use copy instead
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


class SirilFileOpsMixin:
    """Mixin for Siril file and directory operations."""

    def execute(self, command: str) -> bool:
        """Execute a raw Siril command string. Must be implemented by subclass."""
        raise NotImplementedError

    # Directory operations

    def cd(self, path: str) -> bool:
        """Change working directory."""
        # Normalize path separators for Siril
        path = path.replace("\\", "/")
        return self.execute(f"cd {quote_siril_arg(path)}")

    # File operations

    def load(self, path: str) -> bool:
        """Load an image file."""
        path = path.replace("\\", "/")
        return self.execute(f"load {quote_siril_arg(path)}")

    def save(self, path: str) -> bool:
        """Save current image."""
        path = path.replace("\\", "/")
        return self.execute(f"save {quote_siril_arg(path)}")

    def savetif(self, path: str, astro: bool = False, deflate: bool = False) -> bool:
        """Save as TIFF."""
        path = path.replace("\\", "/")
        opts = []
        if astro:
            opts.append("-astro")
        if deflate:
            opts.append("-deflate")
        opts_str = " ".join(opts)
        return self.execute(f"savetif {quote_siril_arg(path)} {opts_str}".strip())

    def savejpg(self, path: str, quality: int = 90) -> bool:
        """Save as JPEG."""
        path = path.replace("\\", "/")
        return self.execute(f"savejpg {quote_siril_arg(path)} {quality}")

    def close(self) -> bool:
        """Close current image."""
        return self.execute("close")

    # Conversion and sequences

    def convert(self, name: str, out: Optional[str] = None) -> bool:
        """Convert files to Siril sequence."""
        cmd = f"convert {quote_siril_arg(name)}"
        if out:
            out = out.replace("\\", "/")
            cmd += f" {quote_siril_arg('-out=' + out)}"
        return self.execute(cmd)

    def split(self, r_file: str, g_file: str, b_file: str) -> bool:
        """Split loaded RGB image into three mono channel files."""
        r_file = r_file.replace("\\", "/")
        g_file = g_file.replace("\\", "/")
        b_file = b_file.replace("\\", "/")
        return self.execute(
            f"split {quote_siril_arg(r_file)} {quote_siril_arg(g_file)} {quote_siril_arg(b_file)}"
        )
