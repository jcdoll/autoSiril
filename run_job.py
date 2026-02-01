#!/usr/bin/env python3
"""
CLI entry point for Siril job processing.

Usage:
    python run_job.py jobs/M42.json                # Single job
    python run_job.py jobs/M42.json jobs/M31.json  # Multiple jobs
    python run_job.py jobs/                        # All jobs in directory
    python run_job.py jobs/ --pattern "SH2*"       # Filtered by pattern
    python run_job.py jobs/M42.json --validate     # Validate only
    python run_job.py jobs/M42.json --dry-run      # Show what would happen
    python run_job.py jobs/M42.json --stage preprocess
    python run_job.py jobs/M42.json --seq-stats    # View registration stats
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for pysiril output (contains Greek letters like σ)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class TeeWriter:
    """Write to both stdout and a file."""

    def __init__(self, file_path: Path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from siril_job_runner.job_config import load_job, load_settings, validate_job_file
from siril_job_runner.job_runner import JobRunner
from siril_job_runner.sequence_analysis import (
    compute_adaptive_threshold,
    format_stats_log,
    parse_sequence_file,
)


def resolve_jobs(paths: list[Path], pattern: str | None) -> list[Path]:
    """
    Expand directories and apply pattern filter.

    Args:
        paths: List of job files or directories
        pattern: Glob pattern to filter jobs (e.g., "SH2*")

    Returns:
        Sorted list of job file paths
    """
    jobs = []
    for p in paths:
        if p.is_dir():
            glob_pattern = f"{pattern}.json" if pattern else "*.json"
            jobs.extend(p.glob(glob_pattern))
        elif pattern is None or p.match(f"*{pattern}*"):
            jobs.append(p)
    return sorted(set(jobs))


def print_summary(results: list[tuple[Path, str, str | None]]) -> None:
    """Print summary table of job results."""
    if len(results) <= 1:
        return

    success = sum(1 for _, status, _ in results if status == "success")
    failed = sum(1 for _, status, _ in results if status == "failed")
    skipped = sum(1 for _, status, _ in results if status == "skipped")

    print()
    print("=" * 60)
    print(f"BATCH SUMMARY: {success} success, {failed} failed, {skipped} skipped")
    print("-" * 60)
    for path, status, err in results:
        status_str = status.upper()
        if err:
            print(f"  {path.stem}: {status_str} - {err}")
        else:
            print(f"  {path.stem}: {status_str}")
    print("=" * 60)


def print_seq_stats(job_path: Path, base_path: Path) -> None:
    """
    Print registration stats from existing .seq files for a job.

    Discovers all process directories and prints stats from pp_light.seq files.
    Shows a summary comparison table followed by detailed per-filter stats.
    """
    import numpy as np

    config = load_job(job_path)
    output_dir = base_path / config.output
    process_dir = output_dir / "process"

    if not process_dir.exists():
        print(f"No process directory found at: {process_dir}")
        print("Run the job first to generate registration data.")
        return

    # Find all filter directories
    filter_dirs = sorted([d for d in process_dir.iterdir() if d.is_dir()])
    if not filter_dirs:
        print(f"No filter directories found in: {process_dir}")
        return

    # Collect all stats first
    all_stats = []
    for filter_dir in filter_dirs:
        seq_path = filter_dir / "pp_light.seq"
        if not seq_path.exists():
            continue

        stats = parse_sequence_file(seq_path)
        if stats is None:
            continue

        stats = compute_adaptive_threshold(stats, config.config)
        all_stats.append((filter_dir.name, stats))

    if not all_stats:
        print(f"No registration data found in: {process_dir}")
        print("Run the job first to generate registration data.")
        return

    print(f"Registration stats for job: {config.name}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Print summary comparison table
    print("\nSummary (min/med/max):")
    header = (
        f"|{'Filter':<10}|{'Frames':>7}|{'wFWHM (px)':<17}|{'FWHM (px)':<17}"
        f"|{'Roundness':<15}|{'Stars':<15}|{'Threshold':<17}|"
    )
    sep = (
        f"|{'-'*10}|{'-'*7}|{'-'*17}|{'-'*17}"
        f"|{'-'*15}|{'-'*15}|{'-'*17}|"
    )
    print(sep)
    print(header)
    print(sep)

    for name, stats in all_stats:
        # Format min/med/max strings
        wfwhm_str = (
            f"{np.min(stats.wfwhm_values):.2f}/"
            f"{np.median(stats.wfwhm_values):.2f}/"
            f"{np.max(stats.wfwhm_values):.2f}"
        )
        fwhm_str = (
            f"{np.min(stats.fwhm_values):.2f}/"
            f"{np.median(stats.fwhm_values):.2f}/"
            f"{np.max(stats.fwhm_values):.2f}"
        )
        round_str = (
            f"{np.min(stats.roundness_values):.2f}/"
            f"{np.median(stats.roundness_values):.2f}/"
            f"{np.max(stats.roundness_values):.2f}"
        )
        stars_str = (
            f"{int(np.min(stats.star_count_values))}/"
            f"{int(np.median(stats.star_count_values))}/"
            f"{int(np.max(stats.star_count_values))}"
        )

        # Format threshold
        if stats.threshold is not None:
            thresh_str = f"{stats.threshold:.2f} (-{stats.n_rejected})"
        else:
            thresh_str = "none"

        row = (
            f"|{name:<10}|{stats.n_images:>7}|{wfwhm_str:<17}|{fwhm_str:<17}"
            f"|{round_str:<15}|{stars_str:<15}|{thresh_str:<17}|"
        )
        print(row)

    print(sep)
    print("\nValues shown as min/med/max. Threshold column shows wFWHM cutoff and frames rejected.")

    # Print detailed per-filter stats
    print("\n" + "=" * 80)
    print("Detailed stats per filter:")

    for name, stats in all_stats:
        print(f"\n{name}:")
        print("-" * 40)
        for line in format_stats_log(stats):
            print(f"  {line}")


def get_siril_interface():
    """
    Get Siril interface.

    Returns SirilWrapper around pysiril if available, otherwise None.
    """
    try:
        from pysiril.siril import Siril

        from siril_job_runner.siril_wrapper import SirilWrapper

        siril = Siril()
        siril.Open()
        return SirilWrapper(siril), siril  # Return wrapper and raw for cleanup
    except ImportError:
        print("WARNING: pysiril not available.")
        print("Install with: uv pip install git+https://gitlab.com/free-astro/pysiril.git")
        print("Running in validation-only mode.")
        return None, None
    except Exception as e:
        print(f"WARNING: Could not connect to Siril: {e}")
        print("Make sure Siril is installed and accessible.")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Run Siril image processing jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_job.py jobs/M42.json              # Run single job
    python run_job.py jobs/M42.json jobs/M31.json # Run multiple jobs
    python run_job.py jobs/                      # Run all jobs in directory
    python run_job.py jobs/ --pattern "SH2*"    # Run jobs matching pattern
    python run_job.py jobs/M42.json --validate   # Validate only
    python run_job.py jobs/M42.json --dry-run    # Show what would happen
    python run_job.py jobs/M42.json --stage calibrate
    python run_job.py jobs/M42.json --seq-stats  # View registration stats
        """,
    )

    parser.add_argument(
        "job_paths",
        type=Path,
        nargs="+",
        help="Job file(s) or directory containing job files",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        help="Filter jobs by glob pattern (e.g., 'SH2*', 'M*')",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining jobs after a failure",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate job file and check calibration, then exit",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing",
    )

    parser.add_argument(
        "--stage",
        choices=["calibrate", "preprocess", "compose"],
        help="Run only a specific stage",
    )

    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Base path for data (default: parent of job file)",
    )

    parser.add_argument(
        "--log",
        action="store_true",
        help="Write output to timestamped log file in logs/ folder",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if cached stacks exist",
    )

    parser.add_argument(
        "--seq-stats",
        action="store_true",
        help="Print registration stats from existing .seq files and exit",
    )

    args = parser.parse_args()

    # Resolve job files from paths
    job_files = resolve_jobs(args.job_paths, args.pattern)
    if not job_files:
        print("ERROR: No job files found")
        if args.pattern:
            print(f"  Pattern: {args.pattern}")
        print(f"  Paths: {[str(p) for p in args.job_paths]}")
        sys.exit(1)

    is_batch = len(job_files) > 1
    if is_batch:
        print(f"Found {len(job_files)} jobs to process")
        for jf in job_files:
            print(f"  - {jf.stem}")
        print()

    # Set up logging to file if requested
    tee = None
    if args.log:
        logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = "batch" if is_batch else job_files[0].stem
        log_path = logs_dir / f"{log_name}_{timestamp}.log"
        tee = TeeWriter(log_path)
        sys.stdout = tee
        sys.stderr = tee
        print(f"Logging to: {log_path}")

    # Validate all job files exist
    for job_file in job_files:
        if not job_file.exists():
            print(f"ERROR: Job file not found: {job_file}")
            sys.exit(1)

    # Determine base path
    repo_root = Path(__file__).parent
    if args.base_path:
        base_path = args.base_path
    else:
        # Try to load from settings.json
        settings = load_settings(repo_root)
        if "base_path" in settings:
            base_path = Path(settings["base_path"])
        else:
            print("ERROR: No base_path specified.")
            print("Either:")
            print("  1. Use --base-path argument")
            print("  2. Create settings.json with base_path (copy from settings.template.json)")
            sys.exit(1)

    # Handle --seq-stats (no Siril needed, only supports single job)
    if args.seq_stats:
        if is_batch:
            print("ERROR: --seq-stats only supports single job")
            sys.exit(1)
        print_seq_stats(job_files[0], base_path)
        sys.exit(0)

    # Get Siril interface (shared across all jobs)
    siril = None
    siril_raw = None
    if not args.validate and not args.dry_run:
        siril, siril_raw = get_siril_interface()
        if siril is None and not args.validate:
            print("Cannot run without Siril connection. Use --validate or --dry-run.")
            sys.exit(1)

    # Process jobs
    results: list[tuple[Path, str, str | None]] = []

    try:
        for i, job_file in enumerate(job_files):
            if is_batch:
                print()
                print("=" * 60)
                print(f"[{i + 1}/{len(job_files)}] Processing: {job_file.stem}")
                print("=" * 60)

            # Quick validation check
            is_valid, error = validate_job_file(job_file)
            if not is_valid:
                print(f"ERROR: Invalid job file: {error}")
                results.append((job_file, "failed", f"Invalid: {error}"))
                if not args.continue_on_error:
                    break
                continue

            runner = None
            try:
                runner = JobRunner(
                    job_path=job_file,
                    base_path=base_path,
                    siril=siril,
                    dry_run=args.dry_run,
                    force=args.force,
                )

                if args.validate:
                    # Validation only
                    result = runner.validate()
                    print()
                    if result.valid:
                        print("Validation PASSED")
                        print(f"  {len(result.frames)} light frames found")
                        print(f"  {len(result.buildable_calibration)} calibration masters to build")
                        results.append((job_file, "success", None))
                    else:
                        print("Validation FAILED")
                        print(f"  Missing: {', '.join(result.missing_calibration)}")
                        results.append((job_file, "failed", "Validation failed"))
                        if not args.continue_on_error:
                            break

                elif args.stage:
                    # Run specific stage
                    validation = runner.validate()
                    if not validation.valid:
                        print(f"ERROR: {validation.message}")
                        results.append((job_file, "failed", validation.message))
                        if not args.continue_on_error:
                            break
                        continue

                    if args.stage == "calibrate":
                        runner.run_calibration(validation)
                    elif args.stage == "preprocess":
                        runner.run_calibration(validation)
                        runner.run_preprocessing(validation.frames)
                    elif args.stage == "compose":
                        result = runner.run_composition()
                        if result:
                            print()
                            print("Outputs:")
                            print(f"  Linear: {result.linear_path}")
                            if result.linear_pcc_path:
                                print(f"  Linear (PCC): {result.linear_pcc_path}")
                            print(f"  Auto FIT: {result.auto_fit}")
                            print(f"  Auto TIF: {result.auto_tif}")
                            print(f"  Auto JPG: {result.auto_jpg}")
                            print(f"  Stacks: {result.stacks_dir}")

                    results.append((job_file, "success", None))

                else:
                    # Full pipeline
                    result = runner.run()
                    if result:
                        print()
                        print("Outputs:")
                        print(f"  Linear: {result.linear_path}")
                        if result.linear_pcc_path:
                            print(f"  Linear (PCC): {result.linear_pcc_path}")
                        print(f"  Auto FIT: {result.auto_fit}")
                        print(f"  Auto TIF: {result.auto_tif}")
                        print(f"  Auto JPG: {result.auto_jpg}")
                        print(f"  Stacks: {result.stacks_dir}")
                    results.append((job_file, "success", None))

            except Exception as e:
                print(f"ERROR: {e}")
                results.append((job_file, "failed", str(e)))
                if not args.continue_on_error:
                    break

            finally:
                if runner is not None:
                    runner.close()

        # Print batch summary
        print_summary(results)

        # Exit with error if any jobs failed
        if any(status == "failed" for _, status, _ in results):
            sys.exit(1)

    finally:
        # Close Siril connection (use raw pysiril object)
        if siril_raw is not None:
            try:
                siril_raw.Close()
            except Exception:
                pass

        # Close log file
        if tee is not None:
            sys.stdout = tee.stdout
            sys.stderr = tee.stdout
            tee.close()


if __name__ == "__main__":
    main()
