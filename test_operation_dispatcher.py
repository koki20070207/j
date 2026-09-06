import pytest

from operation_dispatcher import OperationDispatchError, dispatch_operation


def test_dispatcher_rejects_unknown_operation():
    with pytest.raises(OperationDispatchError):
        dispatch_operation("run_shell", {})


def test_dispatcher_dispatches_system_info():
    result = dispatch_operation("system_info")
    assert result["platform"]


def test_dispatcher_rejects_extra_arguments():
    with pytest.raises(OperationDispatchError):
        dispatch_operation("system_info", {"command": "dir"})
