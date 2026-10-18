"""Tests for output archive and cleanup helpers."""

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from siril_job_runner.models import CompositionResult
from siril_job_runner.output_manager import (
    archive_composition_outputs,
    archive_outputs,
    clean_job_temporary_files,
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
    timestamp = datetime(2026, 10, 17, 23, 59).timestamp()
    os.utime(processed / "lrgb_veralux.jpg", (timestamp, timestamp))
    output = resolve_job_output(job_path, base_path, settings={})

    result = archive_outputs(output)

    assert [(src.name, dst.name) for src, dst in result.copied] == [
        ("lrgb_veralux.jpg", "M42_20261017_lrgb_veralux.jpg")
    ]
    assert (output.archive_dir / "M42_20261017_lrgb_veralux.jpg").read_text(
        encoding="utf-8"
    ) == "jpg"


def test_archive_composition_outputs_returns_archive_paths(tmp_path: Path) -> None:
    """Archived composition results point final files at outputs."""
    job_path = write_job(tmp_path)
    base_path = tmp_path / "astro"
    processed = base_path / "M42" / "processed_lrgb"
    stacks = processed / "stacks"
    stacks.mkdir(parents=True)
    for name in [
        "lrgb_autostretch.fit",
        "rgb_linear_spcc.fit",
        "lrgb_veralux.fit",
        "lrgb_veralux.tif",
        "lrgb_veralux.jpg",
        "job_log_M42_20260101.txt",
    ]:
        (processed / name).write_text(name, encoding="utf-8")
        timestamp = datetime(2026, 10, 17, 23, 59).timestamp()
        os.utime(processed / name, (timestamp, timestamp))
    output = resolve_job_output(job_path, base_path, settings={})
    result = CompositionResult(
        linear_path=processed / "lrgb_autostretch.fit",
        linear_pcc_path=processed / "rgb_linear_spcc.fit",
        auto_fit=processed / "lrgb_veralux.fit",
        auto_tif=processed / "lrgb_veralux.tif",
        auto_jpg=processed / "lrgb_veralux.jpg",
        stacks_dir=stacks,
    )

    archive_result, archived = archive_composition_outputs(output, result)

    assert (
        archive_result.output.archive_dir == base_path / "outputs" / "M42_lrgb_output"
    )
    assert (
        archived.linear_path == output.archive_dir / "M42_20261017_lrgb_autostretch.fit"
    )
    assert (
        archived.linear_pcc_path
        == output.archive_dir / "M42_20261017_rgb_linear_spcc.fit"
    )
    assert archived.auto_fit == output.archive_dir / "M42_20261017_lrgb_veralux.fit"
    assert archived.auto_tif == output.archive_dir / "M42_20261017_lrgb_veralux.tif"
    assert archived.auto_jpg == output.archive_dir / "M42_20261017_lrgb_veralux.jpg"
    assert archived.stacks_dir == stacks
    assert (output.archive_dir / "job_log_M42_20260101.txt").exists()


def test_archive_composition_outputs_requires_expected_files(tmp_path: Path) -> None:
    """Archive fails if a reported composition output was not copied."""
    job_path = write_job(tmp_path)
    base_path = tmp_path / "astro"
    processed = base_path / "M42" / "processed_lrgb"
    stacks = processed / "stacks"
    stacks.mkdir(parents=True)
    (processed / "lrgb_veralux.jpg").write_text("jpg", encoding="utf-8")
    output = resolve_job_output(job_path, base_path, settings={})
    result = CompositionResult(
        linear_path=processed / "missing_linear.fit",
        linear_pcc_path=None,
        auto_fit=processed / "missing_veralux.fit",
        auto_tif=processed / "missing_veralux.tif",
        auto_jpg=processed / "lrgb_veralux.jpg",
        stacks_dir=stacks,
    )

    with pytest.raises(FileNotFoundError, match="Expected output was not archived"):
        archive_composition_outputs(output, result)


def test_discover_job_files_only_returns_json(tmp_path: Path) -> None:
    """Job discovery returns sorted JSON job files."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "B.json").write_text("{}", encoding="utf-8")
    (jobs_dir / "A.json").write_text("{}", encoding="utf-8")
    (jobs_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    paths = discover_job_files(jobs_dir)

    assert [path.name for path in paths] == ["A.json", "B.json"]


def test_clean_keeps_inputs_stacks_and_final_images(tmp_path: Path) -> None:
    """Only disposable directories are removed; source symlinks are not followed."""
    job = write_job(tmp_path)
    base = tmp_path / "astro"
    source = base / "M42/2024_01_15/L180/light.fit"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    processed = base / "M42/processed_lrgb"
    scratch = processed / "process/L_180s"
    scratch.mkdir(parents=True)
    (scratch / "light00001.fit").symlink_to(source)
    (scratch / "pp_light00001.fit").write_bytes(b"temporary")
    stack = processed / "stacks/stack_L_180s.fit"
    stack.parent.mkdir()
    stack.write_bytes(b"combined")
    final = processed / "final.fit"
    final.write_bytes(b"final")
    clean_job_temporary_files(job, base)
    assert not scratch.exists()
    assert source.read_bytes() == b"original"
    assert stack.read_bytes() == b"combined"
    assert final.read_bytes() == b"final"
    assert clean_job_temporary_files(job, base) == []


@pytest.mark.parametrize("symlink", [False, True])
def test_clean_rejects_input_overlap_before_any_deletion(
    tmp_path: Path, symlink: bool
) -> None:
    """A bad job path or directory link cannot turn cleanup into input deletion."""
    job = write_job(tmp_path)
    base = tmp_path / "astro"
    processed = base / "M42/processed_lrgb"
    scratch = processed / "process"
    scratch.mkdir(parents=True)
    marker = scratch / "keep.fit"
    marker.write_bytes(b"keep")
    if symlink:
        source = base / "M42/2024_01_15/L180"
        source.mkdir(parents=True)
        (processed / "hdr").symlink_to(source, target_is_directory=True)
    else:
        data = json.loads(job.read_text())
        data["lights"]["L"] = ["M42/processed_lrgb/process"]
        job.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        clean_job_temporary_files(job, base)
    assert marker.read_bytes() == b"keep"


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

    timestamp = datetime(2026, 10, 17, 23, 59).timestamp()
    os.utime(processed / "lrgb_veralux.jpg", (timestamp, timestamp))

    exit_code = main(["archive", str(job_path)])

    archived = (
        base_path / "outputs" / "M42_lrgb_output" / "M42_20261017_lrgb_veralux.jpg"
    )
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
    (m42_output / "process").mkdir(parents=True)
    m31_output.mkdir(parents=True)

    exit_code = main(["clean", str(m42_job)])

    assert exit_code == 0
    assert m42_output.exists()
    assert not (m42_output / "process").exists()
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
    (job_output / "process").mkdir(parents=True)
    unmanaged_output.mkdir()

    exit_code = main(["clean", "--all"])

    assert exit_code == 0
    assert job_output.exists()
    assert not (job_output / "process").exists()
    assert unmanaged_output.exists()


def test_archive_keeps_export_date_and_distinct_variants(tmp_path: Path) -> None:
    """Formats spanning midnight share a date, even when archived again later."""
    job = write_job(tmp_path)
    output = resolve_job_output(job, tmp_path / "astro", settings={})
    output.processed_dir.mkdir(parents=True)
    for name, moment in [
        ("lrgb_autostretch.fit", datetime(2026, 10, 17, 23, 59)),
        ("lrgb_autostretch.jpg", datetime(2026, 10, 18, 0, 1)),
        ("lrgb_autostretch_starless.jpg", datetime(2026, 10, 18, 0, 2)),
    ]:
        source = output.processed_dir / name
        source.write_text(name)
        timestamp = moment.timestamp()
        os.utime(source, (timestamp, timestamp))

    first = archive_outputs(output)
    second = archive_outputs(output)

    expected = {
        "M42_20261017_lrgb_autostretch.fit": "lrgb_autostretch.fit",
        "M42_20261017_lrgb_autostretch.jpg": "lrgb_autostretch.jpg",
        "M42_20261018_lrgb_autostretch_starless.jpg": "lrgb_autostretch_starless.jpg",
    }
    assert {p.name: p.read_text() for _, p in first.copied} == expected
    assert first.copied == second.copied
    assert {p.name for p in output.archive_dir.iterdir()} == set(expected)
