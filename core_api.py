"""
Jarvis Core が公開するローカルAPI。UI（Streamlit）はここを叩いてCoreの状態や
データを取得する。127.0.0.1のみで待ち受け、外部ネットワークには公開しない。

【Step 3時点でのスコープ】
「Core⇔UIの最小限の連携」として、まずは①Coreの生死確認、②既存データ
（メモ）を1つAPI経由で取得できることの2点を実証する。自律タスクの投入や
アクティビティログの取得はStep 4・6でエンドポイントを追加していく
（UI側の呼び出し方のパターンは、ここで作るものを踏襲すればよい）。

このファイル単体では起動しない。jarvis_core.py の run_forever() が
バックグラウンドスレッドとしてuvicornごと起動する。
"""

import time

from fastapi import FastAPI

from db import get_connection
from logging_setup import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Jarvis Core API")

_start_time = time.time()


@app.get("/health")
def health() -> dict:
    """Coreが生きているかどうかをUIが確認するためのエンドポイント。"""
    return {
        "status": "ok",
        "uptime_sec": round(time.time() - _start_time, 1),
    }


@app.get("/memos")
def get_memos() -> dict:
    """保存済みメモの一覧を構造化データで返す（tools.list_memosはLLM向けの
    整形済み文字列を返すため、UI表示用にはこちらを使う）。"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, text, created_at, done FROM memos ORDER BY id DESC"
        ).fetchall()
    return {
        "memos": [
            {"id": f"memo_{r['id']}", "text": r["text"], "created_at": r["created_at"], "done": bool(r["done"])}
            for r in rows
        ]
    }
