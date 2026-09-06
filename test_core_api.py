"""Core APIの通常メモ操作を検証するテスト。"""

import pytest
from fastapi import HTTPException

import config
import db
from core_api import delete_all_normal_memos, delete_normal_memo, get_memos
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
