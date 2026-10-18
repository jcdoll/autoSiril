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
