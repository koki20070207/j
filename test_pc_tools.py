"""安全なPC操作基盤のテスト。"""

import pytest

import pc_tools
from pc_tools import PCOperationError, list_directory, search_files


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
