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

import chromadb
from fastapi import FastAPI, HTTPException

from config import CHAT_COLLECTION_NAME, CHROMA_DB_PATH
from db import get_connection
from logging_setup import get_logger
from memory_store import reset_chat_memory
from pc_tools import PCOperationError, list_directory, search_files
from tools import reset_memos

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


@app.get("/pc/files")
def list_pc_files(root: str | None = None) -> dict:
    """ユーザープロファイル配下のディレクトリを一覧する。"""
    try:
        return {"root": root, "entries": list_directory(root)}
    except PCOperationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/pc/search")
def search_pc_files(pattern: str, root: str | None = None) -> dict:
    """ユーザープロファイル配下のファイル名を検索する。"""
    try:
        return {"pattern": pattern, "root": root, "files": search_files(pattern, root)}
    except PCOperationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


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


@app.delete("/memos/{memo_id}")
def delete_normal_memo(memo_id: int) -> dict:
    """SQLiteの通常メモを1件削除する。"""
    if memo_id <= 0:
        raise HTTPException(status_code=400, detail="memo_idは正の整数で指定してください。")

    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM memos WHERE id = ?", (memo_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="指定されたメモが見つかりません。")

    return {"deleted_count": 1, "id": f"memo_{memo_id}"}


@app.delete("/memos")
def delete_all_normal_memos() -> dict:
    """SQLiteの通常メモをすべて削除する。"""
    deleted_count = reset_memos()
    return {"deleted_count": deleted_count}


@app.post("/memory/reset")
def reset_memory() -> dict:
    """Webクライアントから長期記憶を全削除するためのローカルAPI。"""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(CHAT_COLLECTION_NAME)
    except ValueError:
        return {"deleted_count": 0, "message": "長期記憶コレクションは存在しません。"}

    deleted_count = reset_chat_memory(collection)
    return {"deleted_count": deleted_count, "message": "長期記憶をリセットしました。"}
