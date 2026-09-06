"""操作履歴・確認要求・スケジュールのSQLite永続化。"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from db import get_connection


def create_operation(operation: str, request: Dict[str, Any], status: str = "pending") -> str:
    operation_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO operations (operation_id, operation, status, request_json) VALUES (?, ?, ?, ?)",
            (operation_id, operation, status, json.dumps(request, ensure_ascii=False)),
        )
    return operation_id


def finish_operation(operation_id: str, status: str, result: Dict[str, Any] | None = None, error: str | None = None) -> None:
    if status not in {"succeeded", "failed", "cancelled"}:
        raise ValueError("不正な操作状態です。")
    with get_connection() as conn:
        conn.execute(
            "UPDATE operations SET status = ?, result_json = ?, error = ?, "
            "started_at = COALESCE(started_at, created_at), finished_at = datetime('now') "
            "WHERE operation_id = ?",
            (status, json.dumps(result or {}, ensure_ascii=False), error, operation_id),
        )


def list_operations(limit: int = 50) -> List[dict]:
    limit = max(1, min(limit, 200))
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT operation_id, operation, status, result_json, error, created_at, started_at, finished_at "
            "FROM operations ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def create_confirmation(operation: str, request: Dict[str, Any], ttl_sec: int = 300) -> str:
    confirmation_id = str(uuid.uuid4())
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_sec)).isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO confirmations (confirmation_id, operation, request_json, expires_at) VALUES (?, ?, ?, ?)",
            (confirmation_id, operation, json.dumps(request, ensure_ascii=False), expires_at),
        )
    return confirmation_id


def resolve_confirmation(confirmation_id: str, approved: bool) -> bool:
    status = "approved" if approved else "rejected"
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE confirmations SET status = ?, resolved_at = datetime('now') "
            "WHERE confirmation_id = ? AND status = 'pending' AND expires_at > ?",
            (status, confirmation_id, datetime.now(timezone.utc).isoformat()),
        )
    return cursor.rowcount == 1
