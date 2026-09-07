"""Jarvis Coreの単一起動ロックのテスト。"""

import pytest

import jarvis_core


def test_acquire_single_instance_replaces_stale_pid_file(tmp_path, monkeypatch):
    pid_file = tmp_path / "jarvis_core.pid"
    pid_file.write_text("999999", encoding="utf-8")
    monkeypatch.setattr(jarvis_core, "CORE_PID_FILE", str(pid_file))
    monkeypatch.setattr(jarvis_core, "_pid_is_running", lambda pid: False)

    jarvis_core._acquire_single_instance_lock()

    assert pid_file.read_text(encoding="utf-8") == str(jarvis_core.os.getpid())
    jarvis_core._release_single_instance_lock()
    assert not pid_file.exists()


def test_acquire_single_instance_rejects_running_pid(tmp_path, monkeypatch):
    pid_file = tmp_path / "jarvis_core.pid"
    pid_file.write_text("1234", encoding="utf-8")
    monkeypatch.setattr(jarvis_core, "CORE_PID_FILE", str(pid_file))
    monkeypatch.setattr(jarvis_core, "_pid_is_running", lambda pid: True)

    with pytest.raises(SystemExit):
        jarvis_core._acquire_single_instance_lock()

    assert pid_file.read_text(encoding="utf-8") == "1234"
