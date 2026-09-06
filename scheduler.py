"""常駐Coreから呼び出す安全な軽量スケジューラ。"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from db import get_connection
from operation_dispatcher import SUPPORTED_OPERATIONS


def schedule_task(
    name: str,
    operation: str,
    interval_sec: int,
    request: Dict[str, Any] | None = None,
) -> str:
    if not name.strip() or interval_sec < 1:
        raise ValueError("タスク名と正の実行間隔が必要です。")
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError("許可されていない操作はスケジュールできません。")
    task_id = str(uuid.uuid4())
    next_run = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO scheduled_tasks "
            "(task_id, name, operation, request_json, interval_sec, next_run_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, name.strip(), operation, json.dumps(request or {}, ensure_ascii=False), interval_sec, next_run),
        )
    return task_id


def list_due_tasks() -> List[dict]:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT task_id, name, operation, request_json, interval_sec, next_run_at FROM scheduled_tasks "
            "WHERE enabled = 1 AND next_run_at <= ? ORDER BY next_run_at",
            (now,),
        ).fetchall()
        for row in rows:
            next_run = datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + row["interval_sec"],
                timezone.utc,
            ).isoformat()
            conn.execute(
                "UPDATE scheduled_tasks SET next_run_at = ? "
                "WHERE task_id = ? AND next_run_at <= ?",
                (next_run, row["task_id"], now),
            )
    tasks = []
    for row in rows:
        item = dict(row)
        item["request"] = json.loads(item.pop("request_json") or "{}")
        tasks.append(item)
    return tasks


def set_task_enabled(task_id: str, enabled: bool) -> bool:
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE scheduled_tasks SET enabled = ? WHERE task_id = ?",
            (1 if enabled else 0, task_id),
        )
    return cursor.rowcount == 1


def delete_task(task_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM scheduled_tasks WHERE task_id = ?", (task_id,))
    return cursor.rowcount == 1
