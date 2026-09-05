"""Edge-case coverage for the memo tools."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

from db import get_connection, init_db
from tools import (
    _is_likely_time_query,
    add_memo,
    list_memos,
    mark_memo_done,
    search_memos,
)


@pytest.fixture(autouse=True)
def initialized_database() -> None:
    init_db()


@pytest.mark.parametrize(
    "value",
    ["", " ", "\t\n", "a", "a" * 5001, "これ" * 2500],
)
def test_rejects_empty_short_and_oversized_memos(value: str) -> None:
    result = add_memo(value)
    assert isinstance(result, str)
    assert "保存しました" not in result


@pytest.mark.parametrize(
    "value",
    [
        "予定 🗓️ 15:00",
        "日本語と English mixed",
        r"quotes ' \" backslash \\ percent % underscore _",
        "改行\nを含むメモ",
    ],
)
def test_accepts_unicode_and_special_text(value: str) -> None:
    result = add_memo(value)
    assert "保存しました" in result


@pytest.mark.parametrize(
    "value",
    ["何時", "今何時", "時刻", "昨日", "明日"],
)
def test_detects_short_time_queries(value: str) -> None:
    assert _is_likely_time_query(value) is True
    assert "保存していません" in add_memo(value)


@pytest.mark.parametrize(
    "value",
    [
        "2026年8月28日15時に歯医者",
        "明日の会議を15時に予約",
        "時間がないので買い物をする",
        "今日の予定は時間がある",
        "2026-08-28T15:00:00+09:00",
    ],
)
def test_preserves_detailed_time_memos(value: str) -> None:
    assert _is_likely_time_query(value) is False
    assert "保存しました" in add_memo(value)


@pytest.mark.parametrize(
    "query",
    ["'; DROP TABLE memos; --", "UNION SELECT", "%", "_", "\\x00"],
)
def test_search_treats_security_strings_as_data(query: str) -> None:
    result = search_memos(query)
    assert isinstance(result, str)
    with get_connection() as connection:
        connection.execute("SELECT COUNT(*) FROM memos").fetchone()


def test_search_partial_and_empty_results() -> None:
    add_memo("Project Aurora planning")
    assert "Project Aurora planning" in search_memos("Aurora")
    assert "該当するメモはありません" in search_memos("missing-keyword")


def test_list_and_search_return_bounded_strings() -> None:
    for index in range(120):
        add_memo(f"bulk memo {index} " + "x" * 400)
    assert len(list_memos()) < 50000
    assert len(search_memos("bulk")) < 50000


@pytest.mark.parametrize("memo_id", [-1, 0, 2**63, "1", None])
def test_mark_done_rejects_invalid_ids(memo_id: object) -> None:
    result = mark_memo_done(memo_id)  # type: ignore[arg-type]
    assert "無効" in result or "見つかりません" in result or "失敗" in result


def test_mark_done_state_is_persisted() -> None:
    add_memo("state transition")
    assert "完了状態" in mark_memo_done(1)
    assert "✅" in list_memos()


def test_created_at_is_jst_iso_timestamp() -> None:
    add_memo("timestamp check")
    with get_connection() as connection:
        value = connection.execute(
            "SELECT created_at FROM memos ORDER BY id DESC LIMIT 1"
        ).fetchone()["created_at"]
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 9 * 60 * 60


def test_rapid_concurrent_adds_remain_consistent() -> None:
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(add_memo, [f"concurrent {i}" for i in range(40)]))
    assert all("保存しました" in result for result in results)
    with get_connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM memos WHERE text LIKE 'concurrent %'"
        ).fetchone()["count"]
    assert count == 40


def test_transaction_rolls_back_failed_write() -> None:
    with pytest.raises(Exception):
        with get_connection() as connection:
            connection.execute(
                "INSERT INTO memos (text, created_at, done) VALUES (?, ?, ?)",
                ("rollback-check", "not-a-problem", 0),
            )
            connection.execute("INSERT INTO missing_table VALUES (1)")
    assert "rollback-check" not in list_memos()
