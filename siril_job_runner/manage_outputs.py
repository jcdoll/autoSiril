"""
CLI for archiving and cleaning Siril processed outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .output_manager import (
    ArchiveResult,
    CleanResult,
    ManagedJobOutput,
    archive_outputs,
    clean_processed_dirs,
    discover_job_files,
    load_repo_settings,
    resolve_base_path,
    resolve_job_output,
    resolve_job_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        description="Archive final outputs and clean processed folders.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_cmd = subparsers.add_parser(
        "list", help="List all jobs with resolved processed and archive paths"
    )
    list_cmd.set_defaults(func=run_list)

    archive = subparsers.add_parser("archive", help="Copy final outputs to outputs/")
    archive.add_argument("job_paths", type=Path, nargs="+", help="Job JSON files")
    archive.set_defaults(func=run_archive)

    clean = subparsers.add_parser("clean", help="Remove processed folders")
    clean.add_argument("job_paths", type=Path, nargs="*", help="Job JSON files")
    clean.add_argument(
        "--all",
        action="store_true",
        help="Clean processed folders resolved from all job files in jobs/",
    )
    clean.set_defaults(func=run_clean)

    return parser


def _context() -> tuple[Path, dict, Path]:
    """Load shared CLI context."""
    repo_root = Path.cwd().resolve()
    base_path = resolve_base_path(repo_root)
    settings = load_repo_settings(repo_root)
    return base_path, settings, repo_root


def _print_job_output(output: ManagedJobOutput) -> None:
    """Print resolved paths for one job."""
    status = "exists" if output.processed_dir.is_dir() else "missing"
    print(f"Job: {output.job_path}")
    print(f"  target: {output.target_name}")
    print(f"  type: {output.job_type}")
    print(f"  processed: {output.processed_dir} [{status}]")
    print(f"  archive: {output.archive_dir}")


def _print_archive_result(result: ArchiveResult) -> None:
    """Print archive result details."""
    _print_job_output(result.output)
    if not result.deliverables:
        print("  no deliverables found")
        return

    for source, destination in result.copied:
        print(f"  copied: {source} -> {destination}")


def _print_clean_result(result: CleanResult) -> None:
    """Print cleanup result details."""
    for target in result.targets:
        if target in result.skipped_missing:
            print(f"Missing: {target}")
        else:
            print(f"Removed: {target}")


def run_list(args: argparse.Namespace) -> int:
    """Run the list subcommand."""
    base_path, settings, repo_root = _context()
    job_paths = discover_job_files(repo_root / "jobs")
    for output in resolve_job_outputs(job_paths, base_path, settings):
        _print_job_output(output)

    return 0


def run_archive(args: argparse.Namespace) -> int:
    """Run the archive subcommand."""
    base_path, settings, _repo_root = _context()
    for job_path in args.job_paths:
        output = resolve_job_output(job_path, base_path, settings)
        result = archive_outputs(output)
        _print_archive_result(result)

    return 0


def run_clean(args: argparse.Namespace) -> int:
    """Run the clean subcommand."""
    base_path, settings, repo_root = _context()
    if not args.all and not args.job_paths:
        raise ValueError("clean requires job paths or --all")
    if args.all and args.job_paths:
        raise ValueError("clean accepts job paths or --all, not both")

    job_paths = discover_job_files(repo_root / "jobs") if args.all else args.job_paths

    outputs = resolve_job_outputs(job_paths, base_path, settings)
    result = clean_processed_dirs(
        [output.processed_dir for output in outputs], base_path
    )
    _print_clean_result(result)

    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the manage-outputs CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
