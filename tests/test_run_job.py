"""Tests for run_job CLI helpers."""

import pytest

import run_job


def test_find_siril_executable_prefers_siril(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PATH lookup prefers the standard siril executable."""

    def fake_which(executable: str) -> str | None:
        paths = {
            "siril": "/opt/homebrew/bin/siril",
            "siril-cli": "/opt/homebrew/bin/siril-cli",
        }
        return paths.get(executable)

    monkeypatch.setattr(run_job.shutil, "which", fake_which)

    assert run_job.find_siril_executable() == "/opt/homebrew/bin/siril"


def test_find_siril_executable_uses_siril_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PATH lookup uses siril-cli if siril is not available."""

    def fake_which(executable: str) -> str | None:
        if executable == "siril-cli":
            return "/usr/local/bin/siril-cli"
        return None

    monkeypatch.setattr(run_job.shutil, "which", fake_which)

    assert run_job.find_siril_executable() == "/usr/local/bin/siril-cli"


def test_find_siril_executable_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PATH lookup returns None if no Siril executable is available."""

    def fake_which(_executable: str) -> None:
        return None

    monkeypatch.setattr(run_job.shutil, "which", fake_which)

    assert run_job.find_siril_executable() is None
