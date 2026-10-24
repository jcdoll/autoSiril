"""Command completion when Siril joins a status onto its log message."""

from queue import Queue
from unittest.mock import MagicMock

import pytest

from siril_job_runner.siril_wrapper import SirilWrapper


@pytest.mark.parametrize("status,expected", [("error", False), ("success", True)])
def test_appended_command_status_returns(status: str, expected: bool) -> None:
    """Reproduce the makepsf wire response rather than waiting indefinitely."""
    connection = MagicMock()
    connection.qStatus = Queue()
    wrapper = SirilWrapper(connection)

    def execute(command: str) -> bool:
        """Simulate the external reader routing a joined line to its logger."""
        connection.tr.retoursiril(
            f"log: No suitable stars detectable. Aborting...status: {status} makepsf"
        )
        return connection.qStatus.get(timeout=0.1) == "status: success"

    connection.Execute.side_effect = execute
    assert wrapper.makepsf() is expected


def test_plate_solve_refreshes_coordinates_without_flipping() -> None:
    """Reject the stale-WCS shortcut and keep RGB aligned with luminance."""
    connection = MagicMock()

    def solve(command: str) -> bool:
        """Model Siril's existing-solution shortcut and orientation control."""
        arguments = command.split()
        return arguments[0] == "platesolve" and {"-force", "-noflip"} <= set(arguments)

    connection.Execute.side_effect = solve
    assert SirilWrapper(connection).platesolve()


def test_starnet_failure_does_not_report_success() -> None:
    """An enabled star-removal step must fail the job when Siril rejects it."""
    connection = MagicMock()
    connection.Execute.return_value = False
    with pytest.raises(
        RuntimeError, match="StarNet failed.*starless outputs are incomplete"
    ):
        SirilWrapper(connection).starnet()


def test_starnet_success() -> None:
    """A successful external star-removal command completes normally."""
    connection = MagicMock()
    connection.Execute.return_value = True
    SirilWrapper(connection).starnet()


@pytest.mark.parametrize(
    "operation",
    [
        "cd",
        "load",
        "save",
        "savetif",
        "savejpg",
        "convert",
        "calibrate",
        "stack",
        "rgbcomp",
        "makepsf",
    ],
)
def test_paths_with_spaces_reach_siril_as_single_arguments(operation: str) -> None:
    """Preserve paths and complete option values at the external parser boundary."""
    import shlex

    location = "C:/Astro Data/M42 frame"
    connection = MagicMock()
    parsed: list[str] = []

    def execute(command: str) -> bool:
        """Apply the documented whitespace/quote parsing contract."""
        parsed.extend(shlex.split(command))
        return True

    connection.Execute.side_effect = execute
    wrapper = SirilWrapper(connection)
    if operation == "calibrate":
        wrapper.calibrate("light", dark=location, flat=location)
        assert "-dark=" + location in parsed and "-flat=" + location in parsed
    elif operation == "rgbcomp":
        wrapper.rgbcomp(lum=location, rgb=location, out=location)
        assert (
            location in parsed
            and "-lum=" + location in parsed
            and "-out=" + location in parsed
        )
    elif operation == "makepsf":
        wrapper.makepsf(save_psf=location)
        assert "-savepsf=" + location in parsed
    elif operation in ("convert", "stack"):
        getattr(wrapper, operation)("light", out=location)
        assert "-out=" + location in parsed
    else:
        getattr(wrapper, operation)(location.replace("/", "\\"))
        assert location in parsed
