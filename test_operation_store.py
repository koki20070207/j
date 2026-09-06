"""操作履歴・確認・スケジューラのテスト。"""

import config
import db
from operation_store import (
    claim_confirmation,
    cancel_confirmation,
    create_confirmation,
    create_operation,
    finish_operation,
    list_operations,
    resolve_confirmation,
)
from scheduler import list_due_tasks, schedule_task


def test_operation_lifecycle_and_confirmation(tmp_path, monkeypatch):
    database_path = str(tmp_path / "operations.db")
    monkeypatch.setattr(config, "DB_PATH", database_path)
    monkeypatch.setattr(db, "DB_PATH", database_path)
    db.init_db()

    operation_id = create_operation("test", {"value": "x"}, "running")
    finish_operation(operation_id, "succeeded", {"ok": True})
    assert list_operations()[0]["operation_id"] == operation_id
    assert list_operations()[0]["status"] == "succeeded"

    confirmation_id = create_confirmation("test", {"value": "x"})
    assert resolve_confirmation(confirmation_id, True) is True
    assert claim_confirmation(confirmation_id) == ("test", {"value": "x"})
    assert claim_confirmation(confirmation_id) is None
    assert resolve_confirmation(confirmation_id, False) is False


def test_scheduler_returns_due_task_once(tmp_path, monkeypatch):
    database_path = str(tmp_path / "scheduler.db")
    monkeypatch.setattr(config, "DB_PATH", database_path)
    monkeypatch.setattr(db, "DB_PATH", database_path)
    db.init_db()

    schedule_task("test", "system_info", 60)

    assert len(list_due_tasks()) == 1
    assert list_due_tasks() == []


def test_confirmation_can_be_cancelled_before_execution(tmp_path, monkeypatch):
    database_path = str(tmp_path / "cancel.db")
    monkeypatch.setattr(config, "DB_PATH", database_path)
    monkeypatch.setattr(db, "DB_PATH", database_path)
    db.init_db()

    confirmation_id = create_confirmation("system_info", {})
    assert cancel_confirmation(confirmation_id) is True
    assert claim_confirmation(confirmation_id) is None
    assert cancel_confirmation(confirmation_id) is False
