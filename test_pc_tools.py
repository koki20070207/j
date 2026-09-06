"""安全なPC操作基盤のテスト。"""

import pytest

import pc_tools
from pc_tools import PCOperationError, get_system_info, list_directory, move_file, open_url, search_files, show_notification


def test_list_directory_is_read_only_and_structured(tmp_path, monkeypatch):
    (tmp_path / "folder").mkdir()
    (tmp_path / "note.txt").write_text("test", encoding="utf-8")
    monkeypatch.setattr(pc_tools.Path, "home", staticmethod(lambda: tmp_path))

    entries = list_directory()

    assert {entry["name"] for entry in entries} == {"folder", "note.txt"}
    assert not (tmp_path / "note.txt").read_text(encoding="utf-8") == ""


def test_search_files_finds_only_files(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.txt").write_text("test", encoding="utf-8")
    monkeypatch.setattr(pc_tools.Path, "home", staticmethod(lambda: tmp_path))

    assert search_files("*.txt") == [r"docs\report.txt"]


def test_pc_tools_reject_path_outside_user_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(pc_tools.Path, "home", staticmethod(lambda: tmp_path))

    with pytest.raises(PCOperationError):
        list_directory(str(tmp_path.parent))


def test_search_files_rejects_empty_pattern(tmp_path, monkeypatch):
    monkeypatch.setattr(pc_tools.Path, "home", staticmethod(lambda: tmp_path))

    with pytest.raises(PCOperationError):
        search_files("")


def test_open_url_rejects_non_http_scheme():
    with pytest.raises(PCOperationError):
        open_url("file:///secret.txt")


def test_open_url_uses_browser(monkeypatch):
    monkeypatch.setattr(pc_tools.webbrowser, "open", lambda url, new: True)

    assert open_url("https://github.com")["status"] == "opened"


def test_open_url_rejects_unlisted_site(monkeypatch):
    monkeypatch.setattr(pc_tools.webbrowser, "open", lambda url, new: True)

    with pytest.raises(PCOperationError):
        open_url("https://example.com")


def test_show_notification_validates_and_returns_operation_result(monkeypatch):
    monkeypatch.setattr(pc_tools, "_send_windows_notification", lambda title, message: None)
    result = show_notification("Jarvis", "処理が完了しました")

    assert result["status"] == "shown"


def test_get_system_info_has_runtime_fields():
    result = get_system_info()

    assert result["platform"]
    assert result["python"]


def test_move_file_creates_backup_before_move(tmp_path, monkeypatch):
    monkeypatch.setattr(pc_tools.Path, "home", staticmethod(lambda: tmp_path))
    source = tmp_path / "source.txt"
    destination = tmp_path / "archive" / "source.txt"
    source.write_text("safe", encoding="utf-8")

    result = move_file(str(source), str(destination))

    assert result["status"] == "moved"
    assert destination.read_text(encoding="utf-8") == "safe"
    assert (tmp_path / "source.txt.jarvis-backup").read_text(encoding="utf-8") == "safe"
