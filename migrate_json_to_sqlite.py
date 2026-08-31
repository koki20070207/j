"""
既存のJSONファイル（chat_sessions.json / answer_cache.json / memos.json）から
SQLite（jarvis.db）へ1回だけデータを移行するスクリプト。

【安全設計】
・元のJSONファイルは自動削除しない。移行結果を確認してから、
  必要であれば手動でバックアップ・削除してください。
・冪等性は「テーブルごとに、既にレコードが1件でもあれば、そのテーブルの
  移行はスキップする」という単純な方式で担保している（テーブル単位）。
  同じJSONを指して2回実行しても二重登録されない。逆に言うと、JSON側を
  更新してから再実行しても反映されないので、その場合はjarvis.dbを
  削除してから再実行してください。

使い方（appディレクトリと同じ場所で実行）:
    python migrate_json_to_sqlite.py
"""

import os

from config import ANSWER_CACHE_FILE, MEMO_FILE, SESSION_FILE
from db import get_connection, init_db, table_row_count
from json_store import load_json
from logging_setup import get_logger

logger = get_logger(__name__)


def _migrate_answer_cache() -> None:
    if table_row_count("answer_cache") > 0:
        print("・answer_cache: 既にDBにデータがあるためスキップします。")
        return
    if not os.path.exists(ANSWER_CACHE_FILE):
        print("・answer_cache: 元のJSONファイルが見つからないためスキップします。")
        return

    data = load_json(ANSWER_CACHE_FILE)
    with get_connection() as conn:
        for cache_key, entry in data.items():
            conn.execute(
                "INSERT OR IGNORE INTO answer_cache (cache_key, answer, tag) VALUES (?, ?, ?)",
                (cache_key, entry.get("answer", ""), entry.get("tag", "未分類")),
            )
    print(f"・answer_cache: {len(data)}件を移行しました。")


def _migrate_memos() -> None:
    if table_row_count("memos") > 0:
        print("・memos: 既にDBにデータがあるためスキップします。")
        return
    if not os.path.exists(MEMO_FILE):
        print("・memos: 元のJSONファイルが見つからないためスキップします。")
        return

    data = load_json(MEMO_FILE)
    items = data.get("items", [])
    with get_connection() as conn:
        for item in items:
            conn.execute(
                "INSERT INTO memos (text, created_at, done) VALUES (?, ?, ?)",
                (item.get("text", ""), item.get("created_at", ""), 1 if item.get("done") else 0),
            )
    print(f"・memos: {len(items)}件を移行しました（旧IDの文字列'memo_1'等は引き継がず、新しい連番になります）。")


def _migrate_chat_sessions() -> None:
    if table_row_count("chat_sessions") > 0:
        print("・chat_sessions: 既にDBにデータがあるためスキップします。")
        return
    if not os.path.exists(SESSION_FILE):
        print("・chat_sessions: 元のJSONファイルが見つからないためスキップします。")
        return

    data = load_json(SESSION_FILE)
    msg_count = 0
    with get_connection() as conn:
        for session_id, session_data in data.items():
            conn.execute(
                "INSERT OR IGNORE INTO chat_sessions (session_id, title) VALUES (?, ?)",
                (session_id, session_data.get("title", "")),
            )
            for seq, msg in enumerate(session_data.get("messages", [])):
                conn.execute(
                    "INSERT INTO chat_messages (session_id, seq, role, content) VALUES (?, ?, ?, ?)",
                    (session_id, seq, msg.get("role", ""), msg.get("content", "")),
                )
                msg_count += 1
    print(f"・chat_sessions: {len(data)}スレッド／{msg_count}メッセージを移行しました。")


if __name__ == "__main__":
    init_db()
    print("移行を開始します...")
    _migrate_answer_cache()
    _migrate_memos()
    _migrate_chat_sessions()
    print("完了しました。中身を確認できたら、元のJSONファイルは手動で削除・バックアップしてください。")
