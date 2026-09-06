"""常駐Coreから呼び出す安全な軽量スケジューラ。"""

import uuid
from datetime import datetime, timezone
from typing import List

from db import get_connection


def schedule_task(name: str, operation: str, interval_sec: int) -> str:
    if not name.strip() or interval_sec < 1:
        raise ValueError("タスク名と正の実行間隔が必要です。")
    task_id = str(uuid.uuid4())
    next_run = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO scheduled_tasks (task_id, name, operation, interval_sec, next_run_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, name.strip(), operation, interval_sec, next_run),
        )
    return task_id


def list_due_tasks() -> List[dict]:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT task_id, name, operation, interval_sec FROM scheduled_tasks "
            "WHERE enabled = 1 AND next_run_at <= ? ORDER BY next_run_at",
            (now,),
        ).fetchall()
        for row in rows:
            next_run = datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + row["interval_sec"],
                timezone.utc,
            ).isoformat()
            conn.execute(
                "UPDATE scheduled_tasks SET next_run_at = ? WHERE task_id = ?",
                (next_run, row["task_id"]),
            )
    return [dict(row) for row in rows]
