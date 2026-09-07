import pytest

from operation_dispatcher import OperationDispatchError
from workflow import run_workflow


def test_workflow_runs_bounded_steps():
    result = run_workflow([{"operation": "system_info", "request": {}}])
    assert result["status"] == "succeeded"


def test_workflow_rejects_too_many_steps():
    with pytest.raises(OperationDispatchError):
        run_workflow([{"operation": "system_info"}] * 6)
