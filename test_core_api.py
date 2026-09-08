"""Core APIの通常メモ操作を検証するテスト。"""

import pytest
from fastapi import HTTPException

import config
import db
from core_api import (
    delete_all_normal_memos,
    delete_normal_memo,
    get_memos,
    move_pc_file,
    read_pc_url,
)
from tools import add_memo


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """テストごとに一時SQLiteを使い、開発用DBを変更しない。"""
    database_path = str(tmp_path / "core_api.db")
    monkeypatch.setattr(config, "DB_PATH", database_path)
    monkeypatch.setattr(db, "DB_PATH", database_path)
    db.init_db()
    return database_path


def test_get_memos_returns_structured_records(isolated_db):
    add_memo("APIから確認するメモ")

    response = get_memos()

    assert response["memos"][0]["id"].startswith("memo_")
    assert response["memos"][0]["text"] == "APIから確認するメモ"
    assert response["memos"][0]["done"] is False


def test_delete_normal_memo_returns_not_found_for_missing_id(isolated_db):
    with pytest.raises(HTTPException) as error:
        delete_normal_memo(999999)

    assert error.value.status_code == 404


def test_delete_all_normal_memos_returns_deleted_count(isolated_db):
    add_memo("削除対象1")
    add_memo("削除対象2")

    response = delete_all_normal_memos()

    assert response == {"deleted_count": 2}
    assert get_memos() == {"memos": []}


def test_read_pc_url_exposes_read_only_operation(monkeypatch):
    monkeypatch.setattr(
        "core_api.read_url",
        lambda url: {"url": url, "content": "ok", "status": "read"},
    )

    response = read_pc_url("https://docs.python.org/")

    assert response["operation"] == "read_url"
    assert response["result"]["status"] == "read"


def test_move_pc_file_records_successful_operation(isolated_db, tmp_path, monkeypatch):
    source = str(tmp_path / "source.txt")
    destination = str(tmp_path / "destination.txt")
    monkeypatch.setattr(
        "core_api.move_file",
        lambda actual_source, actual_destination: {
            "source": actual_source,
            "destination": actual_destination,
            "status": "moved",
        },
    )

    response = move_pc_file(source, destination)

    assert response["operation"] == "move_file"
    assert response["result"]["status"] == "moved"
    assert response["operation_id"]
