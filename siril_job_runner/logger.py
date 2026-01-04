"""
Logging and progress reporting for Siril job processing.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, TextIO


class JobLogger:
    """Logger with elapsed time tracking and file output."""

    def __init__(self, output_dir: Optional[Path] = None, job_name: str = "job"):
        self.start_time = time.time()
        self.log_file: Optional[TextIO] = None
        self.output_dir = output_dir
        self.job_name = job_name

        if output_dir:
            self._init_log_file(output_dir, job_name)

    def _init_log_file(self, output_dir: Path, job_name: str) -> None:
        """Initialize log file in output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = output_dir / f"job_log_{job_name}_{timestamp}.txt"
        self.log_file = open(log_path, "w", encoding="utf-8")
        self._write_to_file(f"Job started: {datetime.now().isoformat()}")
        self._write_to_file(f"Job name: {job_name}")
        self._write_to_file("-" * 60)

    def _elapsed(self) -> str:
        """Get elapsed time as [MM:SS] string."""
        elapsed = int(time.time() - self.start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        return f"[{minutes:02d}:{seconds:02d}]"

    def _write_to_file(self, message: str) -> None:
        """Write message to log file if open."""
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()

    def _output(self, message: str, indent: int = 0) -> None:
        """Output to console and log file."""
        prefix = "  " * indent
        timestamped = f"{self._elapsed()} {prefix}{message}"
        print(timestamped)
        self._write_to_file(timestamped)

    def info(self, message: str) -> None:
        """Log info message."""
        self._output(message)

    def step(self, message: str) -> None:
        """Log a major step."""
        self._output(message)

    def substep(self, message: str) -> None:
        """Log a substep (indented)."""
        self._output(message, indent=1)

    def detail(self, message: str) -> None:
        """Log a detail (double indented)."""
        self._output(message, indent=2)

    def warning(self, message: str) -> None:
        """Log a warning."""
        self._output(f"WARNING: {message}")

    def error(self, message: str) -> None:
        """Log an error."""
        self._output(f"ERROR: {message}")

    def success(self, message: str) -> None:
        """Log a success message."""
        self._output(f"OK: {message}")

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        """Log a simple table."""
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        # Format header
        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        separator = "-+-".join("-" * w for w in widths)

        self._output(header_line)
        self._output(separator)
        for row in rows:
            row_line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            self._output(row_line)

    @contextmanager
    def timed_operation(self, name: str):
        """Context manager for timing an operation."""
        self.step(f"{name}...")
        op_start = time.time()
        try:
            yield
        finally:
            op_elapsed = time.time() - op_start
            self.substep(f"completed in {op_elapsed:.1f}s")

    def close(self) -> None:
        """Close the log file."""
        if self.log_file:
            self._write_to_file("-" * 60)
            self._write_to_file(f"Job completed: {datetime.now().isoformat()}")
            total_elapsed = time.time() - self.start_time
            self._write_to_file(f"Total time: {total_elapsed:.1f}s")
            self.log_file.close()
            self.log_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.error(f"Job failed: {exc_val}")
        self.close()
        return False


# Convenience function for simple logging without file output
def create_logger(output_dir: Optional[Path] = None, job_name: str = "job") -> JobLogger:
    """Create a new JobLogger instance."""
    return JobLogger(output_dir, job_name)
