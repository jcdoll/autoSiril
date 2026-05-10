"""Tests for output archive and cleanup helpers."""

import json
from pathlib import Path

import pytest

from siril_job_runner.output_manager import (
    archive_outputs,
    clean_processed_dirs,
    discover_deliverables,
    discover_job_files,
    resolve_job_output,
    validate_clean_target,
)


def write_settings(repo_root: Path, base_path: Path) -> None:
    """Write a local settings file for CLI tests."""
    settings = {"base_path": str(base_path)}
    (repo_root / "settings.json").write_text(json.dumps(settings), encoding="utf-8")


def write_job(
    jobs_dir: Path,
    name: str = "M42",
    job_type: str = "LRGB",
    output: str = "M42/processed_lrgb",
) -> Path:
    """Write a minimal job file and return its path."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "name": f"{name}_Test",
        "type": job_type,
        "calibration": {
            "bias": "2024_01_15",
            "darks": "2024_01_15",
            "flats": "2024_01_20",
        },
        "lights": {"L": [f"{name}/2024_01_15/L180"]},
        "output": output,
    }
    path = jobs_dir / f"{name}.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    return path


def test_resolve_job_output_uses_job_type_archive_name(tmp_path: Path) -> None:
    """Resolved archive directory includes target and job type."""
    job_path = write_job(tmp_path)
    base_path = tmp_path / "astro"

    output = resolve_job_output(job_path, base_path, settings={})

    assert output.target_name == "M42"
    assert output.job_type == "LRGB"
    assert output.processed_dir == base_path / "M42" / "processed_lrgb"
    assert output.archive_dir == base_path / "outputs" / "M42_lrgb_output"


def test_discover_deliverables_excludes_intermediate_files(tmp_path: Path) -> None:
    """Only top-level final files and job logs are deliverables."""
    processed = tmp_path / "M42" / "processed_lrgb"
    process_dir = processed / "process"
    process_dir.mkdir(parents=True)
    (processed / "lrgb_veralux.fit").write_text("fit", encoding="utf-8")
    (processed / "lrgb_veralux.tif").write_text("tif", encoding="utf-8")
    (processed / "lrgb_veralux.jpg").write_text("jpg", encoding="utf-8")
    (processed / "job_log_M42_20260101.txt").write_text("log", encoding="utf-8")
    (processed / "diag_rgb.jpg").write_text("diagnostic", encoding="utf-8")
    (processed / "notes.txt").write_text("notes", encoding="utf-8")
    (processed / "psf_rgb.fit").write_text("psf", encoding="utf-8")
    (processed / "rgb_starless_temp.fit").write_text("temp", encoding="utf-8")
    (process_dir / "light00001.fit").write_text("intermediate", encoding="utf-8")

    deliverables = discover_deliverables(processed)

    assert [path.name for path in deliverables] == [
        "job_log_M42_20260101.txt",
        "lrgb_veralux.fit",
        "lrgb_veralux.jpg",
        "lrgb_veralux.tif",
    ]


def test_archive_outputs_copies_and_verifies_files(tmp_path: Path) -> None:
    """Archive copies deliverables to outputs."""
    job_path = write_job(tmp_path)
    base_path = tmp_path / "astro"
    processed = base_path / "M42" / "processed_lrgb"
    processed.mkdir(parents=True)
    (processed / "lrgb_veralux.jpg").write_text("jpg", encoding="utf-8")
    output = resolve_job_output(job_path, base_path, settings={})

    result = archive_outputs(output)

    assert [(src.name, dst.name) for src, dst in result.copied] == [
        ("lrgb_veralux.jpg", "lrgb_veralux.jpg")
    ]
    assert (output.archive_dir / "lrgb_veralux.jpg").read_text(
        encoding="utf-8"
    ) == "jpg"


def test_discover_job_files_only_returns_json(tmp_path: Path) -> None:
    """Job discovery returns sorted JSON job files."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "B.json").write_text("{}", encoding="utf-8")
    (jobs_dir / "A.json").write_text("{}", encoding="utf-8")
    (jobs_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    paths = discover_job_files(jobs_dir)

    assert [path.name for path in paths] == ["A.json", "B.json"]


def test_clean_processed_dirs_removes_existing_and_skips_missing(
    tmp_path: Path,
) -> None:
    """Cleanup removes existing job outputs and reports missing ones."""
    base_path = tmp_path / "astro"
    existing = base_path / "M42" / "processed_lrgb"
    missing = base_path / "M31" / "processed_lrgb"
    existing.mkdir(parents=True)

    result = clean_processed_dirs([existing, missing], base_path)

    assert result.targets == [existing, missing]
    assert result.removed == [existing]
    assert result.skipped_missing == [missing]
    assert not existing.exists()


def test_validate_clean_target_rejects_nested_processed_dir(tmp_path: Path) -> None:
    """Cleanup refuses processed folders below the target level."""
    base_path = tmp_path / "astro"
    target = base_path / "M42" / "night" / "processed_lrgb"
    target.mkdir(parents=True)

    with pytest.raises(ValueError, match="non-target-level"):
        validate_clean_target(target, base_path)


def test_cli_list_reports_all_jobs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The list command reports job-derived paths without deleting anything."""
    from siril_job_runner.manage_outputs import main

    repo_root = tmp_path
    base_path = tmp_path / "astro"
    monkeypatch.chdir(repo_root)
    write_settings(repo_root, base_path)
    write_job(repo_root / "jobs", name="M42", output="M42/processed_lrgb")

    exit_code = main(["list"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Job:" in captured.out
    assert "M42/processed_lrgb" in captured.out
    assert "[missing]" in captured.out


def test_cli_archive_copies_job_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The archive command copies outputs for a job file."""
    from siril_job_runner.manage_outputs import main

    repo_root = tmp_path
    base_path = tmp_path / "astro"
    monkeypatch.chdir(repo_root)
    write_settings(repo_root, base_path)
    job_path = write_job(repo_root / "jobs", name="M42", output="M42/processed_lrgb")
    processed = base_path / "M42" / "processed_lrgb"
    processed.mkdir(parents=True)
    (processed / "lrgb_veralux.jpg").write_text("jpg", encoding="utf-8")

    exit_code = main(["archive", str(job_path)])

    archived = base_path / "outputs" / "M42_lrgb_output" / "lrgb_veralux.jpg"
    assert exit_code == 0
    assert archived.read_text(encoding="utf-8") == "jpg"


def test_cli_clean_single_job_removes_only_that_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clean command removes only the processed folder for one job file."""
    from siril_job_runner.manage_outputs import main

    repo_root = tmp_path
    base_path = tmp_path / "astro"
    monkeypatch.chdir(repo_root)
    write_settings(repo_root, base_path)
    m42_job = write_job(repo_root / "jobs", name="M42", output="M42/processed_lrgb")
    write_job(repo_root / "jobs", name="M31", output="M31/processed_lrgb")
    m42_output = base_path / "M42" / "processed_lrgb"
    m31_output = base_path / "M31" / "processed_lrgb"
    m42_output.mkdir(parents=True)
    m31_output.mkdir(parents=True)

    exit_code = main(["clean", str(m42_job)])

    assert exit_code == 0
    assert not m42_output.exists()
    assert m31_output.exists()


def test_cli_clean_all_only_uses_job_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clean --all command only removes processed folders resolved from jobs."""
    from siril_job_runner.manage_outputs import main

    repo_root = tmp_path
    base_path = tmp_path / "astro"
    monkeypatch.chdir(repo_root)
    write_settings(repo_root, base_path)
    write_job(repo_root / "jobs", name="M42", output="M42/processed_lrgb")
    job_output = base_path / "M42" / "processed_lrgb"
    unmanaged_output = base_path / "M42" / "processed"
    job_output.mkdir(parents=True)
    unmanaged_output.mkdir()

    exit_code = main(["clean", "--all"])

    assert exit_code == 0
    assert not job_output.exists()
    assert unmanaged_output.exists()
