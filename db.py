"""
SQLite統合データ層。

【この変更の背景】
chat_sessions.json / answer_cache.json / memos.json と、JSONファイルが
機能ごとに散らばっていた。今後さらに機能（アクティビティログ等）が増える
たびにファイルが増えるのを避けるため、1つのSQLiteファイル（jarvis.db）に集約する。

【WALモードを使う理由（重要な設計判断）】
これまでは「Streamlit UIプロセスが1つだけ、必要な時にJSONを読み書きする」
という前提で問題なかった。しかしJarvis Core導入後は、Core（常駐プロセス、
随時バックグラウンドで書き込みうる）とUI（随時読み書き）が別プロセスとして
同時に動く。SQLiteのデフォルト（ロールバックジャーナル）モードは書き込み中
に読み込みも含めてDB全体をロックするため、これでは頻繁な待ち・
「database is locked」エラーに直結する。
WAL（Write-Ahead Logging）モードにすると書き込み中でも読み込みはブロック
されなくなり、複数プロセスからの同時アクセスに大幅に強くなる。ローカル
ディスクのみで使う前提なので、WALのデメリット（ネットワークドライブ非対応等）
は今回は問題にならない。

【接続の持ち方について】
GenerativeModelインスタンスの使い回し（llm_client.py）とは異なり、DB接続は
呼び出しのたびに開閉する方式にしている。SQLite接続はスレッドをまたいだ共有に
注意が必要で、Streamlitの実行モデル（リクエストごとにスレッドが変わりうる）
を踏まえると、都度接続する方が安全かつ単純と判断した。SQLiteは接続コストが
低いため、この規模の用途では性能上の問題にならない。
"""

import contextlib
import sqlite3
from typing import Iterator

from config import DB_PATH
from logging_setup import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
    cache_key   TEXT PRIMARY KEY,
    answer      TEXT NOT NULL,
    tag         TEXT NOT NULL DEFAULT '未分類',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    done        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, seq);
"""


@contextlib.contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """SQLite接続を1つ開き、with節を抜ける際に自動でcommit/closeする。

    正常終了時はcommit、例外発生時はrollbackしてから例外を再送出する。
    """
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")  # ロック競合時は最大5秒リトライ待機
        conn.execute("PRAGMA foreign_keys=ON")    # chat_messagesのON DELETE CASCADEに必要
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """テーブルが存在しなければ作成する。起動のたびに呼んでよい冪等な処理。"""
    with get_connection() as conn:
        conn.executescript(_SCHEMA)
    logger.info("SQLiteデータベースを初期化しました: %s", DB_PATH)


def table_row_count(table_name: str) -> int:
    """指定テーブルの件数を返す（移行スクリプトの冪等性チェック等に使用）。
     
    Args:
        table_name: テーブル名。既知の値（answer_cache, memos, chat_sessions, chat_messages）のみ許可。
     
    Returns:
        テーブルの行数。
     
    Raises:
        ValueError: テーブル名が許可リストに含まれていない場合。
    """
    allowed_tables = {"answer_cache", "memos", "chat_sessions", "chat_messages"}
    if table_name not in allowed_tables:
         raise ValueError(f"不正なテーブル名: {table_name}。許可されているテーブルは {allowed_tables}")
     
    with get_connection() as conn:
         # テーブル名は既知の値に限定されているため、f-stringの使用は安全
         row = conn.execute(f"SELECT COUNT(*) AS c FROM {table_name}").fetchone()
    return row["c"]
