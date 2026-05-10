"""
Reusable output archiving and cleanup helpers.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .job_config import JobConfig, load_job, load_settings
from .models import CompositionResult

FINAL_OUTPUT_SUFFIXES = frozenset(
    {".fit", ".fits", ".tif", ".tiff", ".jpg", ".jpeg", ".png"}
)
NON_DELIVERABLE_PREFIXES = ("diag_", "psf_", "starmask_")


@dataclass(frozen=True)
class ManagedJobOutput:
    """Resolved output paths for one job."""

    job_path: Path
    target_name: str
    job_type: str
    processed_dir: Path
    archive_dir: Path


@dataclass(frozen=True)
class ArchiveResult:
    """Result of copying final deliverables to an archive directory."""

    output: ManagedJobOutput
    deliverables: list[Path]
    copied: list[tuple[Path, Path]] = field(default_factory=list)


@dataclass(frozen=True)
class CleanResult:
    """Result of removing processed directories."""

    targets: list[Path]
    removed: list[Path] = field(default_factory=list)
    skipped_missing: list[Path] = field(default_factory=list)


def resolve_base_path(repo_root: Path) -> Path:
    """Resolve the image data base path from settings.json."""
    settings = load_settings(repo_root)
    if "base_path" not in settings:
        raise ValueError("No base_path specified. Create settings.json.")
    return Path(settings["base_path"]).expanduser()


def load_repo_settings(repo_root: Path) -> dict:
    """Load settings.json from the repository root."""
    return load_settings(repo_root)


def get_target_name(config: JobConfig) -> str:
    """Derive a target name from job output or first light path."""
    output_parts = Path(config.output).parts
    if output_parts and output_parts[0] not in (".", ""):
        return output_parts[0]

    for paths in config.lights.values():
        if paths:
            light_parts = paths[0].replace("\\", "/").split("/")
            if light_parts and light_parts[0]:
                return light_parts[0]

    raise ValueError(f"Cannot determine target name for job: {config.name}")


def get_archive_name(config: JobConfig) -> str:
    """Build the shared output directory name for a job."""
    target_name = get_target_name(config)
    return f"{target_name}_{config.job_type.lower()}_output"


def resolve_job_output(
    job_path: Path,
    base_path: Path,
    settings: Optional[dict] = None,
) -> ManagedJobOutput:
    """Resolve processed and archive paths for a job file."""
    config = load_job(job_path, settings)
    return ManagedJobOutput(
        job_path=Path(job_path),
        target_name=get_target_name(config),
        job_type=config.job_type,
        processed_dir=base_path / config.output,
        archive_dir=base_path / "outputs" / get_archive_name(config),
    )


def discover_job_files(jobs_dir: Path) -> list[Path]:
    """Find job JSON files in a jobs directory."""
    jobs_dir = Path(jobs_dir)
    if not jobs_dir.is_dir():
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")
    return sorted(jobs_dir.glob("*.json"))


def resolve_job_outputs(
    job_paths: list[Path],
    base_path: Path,
    settings: Optional[dict] = None,
) -> list[ManagedJobOutput]:
    """Resolve processed and archive paths for multiple job files."""
    return [resolve_job_output(job_path, base_path, settings) for job_path in job_paths]


def is_processed_dir_name(name: str) -> bool:
    """Return whether a directory name is an autoSiril processed output folder."""
    return name == "processed" or name.startswith("processed_")


def is_deliverable_file(path: Path) -> bool:
    """Return whether a top-level processed file should be archived."""
    if not path.is_file():
        return False
    if path.name.startswith(NON_DELIVERABLE_PREFIXES):
        return False
    if path.stem.endswith("_temp"):
        return False
    if path.suffix.lower() in FINAL_OUTPUT_SUFFIXES:
        return True
    return path.name.startswith("job_log_") and path.suffix.lower() == ".txt"


def discover_deliverables(processed_dir: Path) -> list[Path]:
    """Find final deliverables and logs in a processed directory."""
    processed_dir = Path(processed_dir)
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
    return sorted(path for path in processed_dir.iterdir() if is_deliverable_file(path))


def archive_outputs(output: ManagedJobOutput) -> ArchiveResult:
    """Copy final deliverables and logs to the shared archive directory."""
    deliverables = discover_deliverables(output.processed_dir)
    copied: list[tuple[Path, Path]] = []

    output.archive_dir.mkdir(parents=True, exist_ok=True)
    for source in deliverables:
        destination = output.archive_dir / source.name
        shutil.copy2(source, destination)
        if destination.stat().st_size != source.stat().st_size:
            raise OSError(f"Copy verification failed: {source} -> {destination}")
        copied.append((source, destination))

    return ArchiveResult(
        output=output,
        deliverables=deliverables,
        copied=copied,
    )


def _archived_path(path: Path, archive_result: ArchiveResult) -> Path:
    """Resolve a processed output path to its archived copy."""
    source_path = Path(path)
    copied_by_source = {
        source.resolve(): destination for source, destination in archive_result.copied
    }
    copied_by_name = {
        source.name: destination for source, destination in archive_result.copied
    }

    archived = copied_by_source.get(source_path.resolve())
    if archived is not None:
        return archived

    archived = copied_by_name.get(source_path.name)
    if archived is not None:
        return archived

    raise FileNotFoundError(f"Expected output was not archived: {source_path}")


def _archived_optional_path(
    path: Optional[Path], archive_result: ArchiveResult
) -> Optional[Path]:
    """Resolve an optional processed output path to its archived copy."""
    if path is None:
        return None
    return _archived_path(path, archive_result)


def map_composition_result_to_archive(
    result: CompositionResult, archive_result: ArchiveResult
) -> CompositionResult:
    """Return a composition result whose final output paths point at the archive."""
    return CompositionResult(
        linear_path=_archived_path(result.linear_path, archive_result),
        linear_pcc_path=_archived_optional_path(result.linear_pcc_path, archive_result),
        auto_fit=_archived_path(result.auto_fit, archive_result),
        auto_tif=_archived_path(result.auto_tif, archive_result),
        auto_jpg=_archived_path(result.auto_jpg, archive_result),
        stacks_dir=result.stacks_dir,
    )


def archive_composition_outputs(
    output: ManagedJobOutput, result: CompositionResult
) -> tuple[ArchiveResult, CompositionResult]:
    """Archive final job outputs and return result paths pointing at the archive."""
    archive_result = archive_outputs(output)
    archived_result = map_composition_result_to_archive(result, archive_result)
    return archive_result, archived_result


def validate_clean_target(path: Path, base_path: Path) -> None:
    """Validate that a cleanup target is a processed folder under base_path."""
    if not is_processed_dir_name(path.name):
        raise ValueError(f"Refusing to clean non-processed directory: {path}")
    if path.is_symlink():
        raise ValueError(f"Refusing to clean symlink: {path}")

    base_resolved = base_path.resolve()
    path_resolved = path.resolve()
    try:
        path_resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"Refusing to clean outside base_path: {path}") from exc

    relative_parts = path_resolved.relative_to(base_resolved).parts
    if len(relative_parts) != 2:
        raise ValueError(f"Refusing to clean non-target-level directory: {path}")


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Deduplicate paths while preserving order."""
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def clean_processed_dirs(paths: list[Path], base_path: Path) -> CleanResult:
    """Remove processed directories resolved from job files."""
    targets = _unique_paths([Path(path) for path in paths])
    removed: list[Path] = []
    skipped_missing: list[Path] = []

    for target in targets:
        if not target.exists():
            skipped_missing.append(target)
            continue
        validate_clean_target(target, base_path)
        shutil.rmtree(target)
        removed.append(target)

    return CleanResult(
        targets=targets,
        removed=removed,
        skipped_missing=skipped_missing,
    )
