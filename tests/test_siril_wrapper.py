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
