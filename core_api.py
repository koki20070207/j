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
from operation_store import (
    claim_confirmation,
    cancel_confirmation,
    create_confirmation,
    create_operation,
    finish_operation,
    finish_confirmation,
    list_operations,
    resolve_confirmation,
)
from scheduler import delete_task, schedule_task, set_task_enabled
from pc_tools import (
    PCOperationError,
    get_system_info,
    launch_app,
    list_directory,
    open_url,
    search_files,
    show_notification,
)
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


@app.post("/pc/apps/{app_name}")
def start_pc_app(app_name: str) -> dict:
    """許可リストにあるアプリを起動する。"""
    try:
        operation_id = create_operation("launch_app", {"app_name": app_name}, "running")
        result = launch_app(app_name)
        finish_operation(operation_id, "succeeded", result)
        return {"operation_id": operation_id, "operation": "launch_app", "result": result}
    except PCOperationError as error:
        if "operation_id" in locals():
            finish_operation(operation_id, "failed", error=str(error))
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/pc/url")
def open_pc_url(url: str) -> dict:
    """安全なHTTP(S) URLを既定ブラウザで開く。"""
    try:
        return {"operation": "open_url", "result": open_url(url)}
    except PCOperationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/pc/notification")
def notify_pc(title: str, message: str) -> dict:
    """ローカル通知を記録する。"""
    try:
        return {"operation": "notification", "result": show_notification(title, message)}
    except PCOperationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/pc/system")
def get_pc_system_info() -> dict:
    """基本的なシステム情報を返す。"""
    return {"operation": "system_info", "result": get_system_info()}


@app.get("/operations")
def get_operations(limit: int = 50) -> dict:
    return {"operations": list_operations(limit)}


@app.post("/confirmations")
def request_confirmation(operation: str, payload: dict) -> dict:
    confirmation_id = create_confirmation(operation, payload)
    return {"confirmation_id": confirmation_id, "status": "pending"}


@app.post("/confirmations/{confirmation_id}")
def resolve_confirmation_request(confirmation_id: str, approved: bool) -> dict:
    if not resolve_confirmation(confirmation_id, approved):
        raise HTTPException(status_code=409, detail="確認要求が存在しないか期限切れです。")
    return {"confirmation_id": confirmation_id, "status": "approved" if approved else "rejected"}


@app.post("/confirmations/{confirmation_id}/execute")
def execute_confirmed_operation(confirmation_id: str, max_attempts: int = 1) -> dict:
    if not 1 <= max_attempts <= 3:
        raise HTTPException(status_code=400, detail="再試行回数は1〜3回で指定してください。")
    claimed = claim_confirmation(confirmation_id)
    if claimed is None:
        raise HTTPException(status_code=409, detail="承認済みで実行可能な確認要求がありません。")
    operation, payload = claimed
    operation_id = create_operation(operation, payload, "running")
    from operation_dispatcher import OperationDispatchError, dispatch_operation

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = dispatch_operation(operation, payload)
        except OperationDispatchError as error:
            last_error = error
            if attempt == max_attempts:
                finish_operation(operation_id, "failed", error=str(error))
                finish_confirmation(confirmation_id, False)
                raise HTTPException(status_code=400, detail=str(error)) from error
        else:
            result["attempts"] = attempt
            finish_operation(operation_id, "succeeded", result)
            finish_confirmation(confirmation_id, True)
            return {
                "confirmation_id": confirmation_id,
                "operation_id": operation_id,
                "status": "succeeded",
                "result": result,
            }
    raise HTTPException(status_code=400, detail=str(last_error))


@app.post("/confirmations/{confirmation_id}/cancel")
def cancel_confirmation_request(confirmation_id: str) -> dict:
    if not cancel_confirmation(confirmation_id):
        raise HTTPException(status_code=409, detail="キャンセル可能な確認要求がありません。")
    return {"confirmation_id": confirmation_id, "status": "cancelled"}


@app.post("/tasks")
def create_scheduled_task(name: str, operation: str, interval_sec: int, payload: dict | None = None) -> dict:
    try:
        return {
            "task_id": schedule_task(name, operation, interval_sec, payload),
            "status": "scheduled",
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/tasks/{task_id}/stop")
def stop_scheduled_task(task_id: str) -> dict:
    if not set_task_enabled(task_id, False):
        raise HTTPException(status_code=404, detail="タスクが見つかりません。")
    return {"task_id": task_id, "status": "stopped"}


@app.post("/tasks/{task_id}/resume")
def resume_scheduled_task(task_id: str) -> dict:
    if not set_task_enabled(task_id, True):
        raise HTTPException(status_code=404, detail="タスクが見つかりません。")
    return {"task_id": task_id, "status": "resumed"}


@app.delete("/tasks/{task_id}")
def remove_scheduled_task(task_id: str) -> dict:
    if not delete_task(task_id):
        raise HTTPException(status_code=404, detail="タスクが見つかりません。")
    return {"task_id": task_id, "status": "deleted"}


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
