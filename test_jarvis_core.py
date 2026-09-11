"""Jarvis Coreの単一起動ロックのテスト。"""

import pytest

import jarvis_core


def test_wait_for_api_server_returns_after_health_check(monkeypatch):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(jarvis_core.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    thread = jarvis_core.threading.Thread()
    monkeypatch.setattr(thread, "is_alive", lambda: True)

    jarvis_core._wait_for_api_server(thread, [])


def test_wait_for_api_server_raises_original_error():
    original_error = OSError("port is already in use")

    with pytest.raises(RuntimeError) as error:
        jarvis_core._wait_for_api_server(jarvis_core.threading.Thread(), [original_error])

    assert error.value.__cause__ is original_error


def test_shutdown_signal_sets_event():
    jarvis_core._shutdown_event.clear()

    jarvis_core._handle_shutdown_signal(2, None)

    assert jarvis_core._shutdown_event.is_set()
    jarvis_core._shutdown_event.clear()


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
