"""
Geminiの関数呼び出し（Function Calling）で使う「ツール」を定義するモジュール。

ここに定義したPython関数は、そのままGemini API（google-genai SDK）の
config.tools に渡すことで、SDKの「自動関数呼び出し（Automatic Function Calling）」
機能により以下がすべて自動で行われる:
  1. 関数のシグネチャ（型ヒント）とdocstringから呼び出しスキーマが自動生成される
  2. モデルが必要と判断したときに、モデル自身がこの関数を呼び出す
  3. 戻り値（文字列）が自動でモデルに返され、それを踏まえて最終回答が生成される

つまり、Jarvisに新しい「できること」を1つ増やしたいときは、
  (1) この下に型ヒント＋日本語docstring付きの関数を書く
  (2) ファイル末尾の AVAILABLE_TOOLS リストに追加する
の2手順だけでよく、llm_client.py 側の変更は不要。

【注意】ここに置くのは「実行しても安全な」ツールに限定すること。
PC操作・ファイル削除など取り返しのつかない操作を行うツールを追加する場合は、
実行前にユーザーへ確認を挟む仕組み（automatic_function_calling を無効化して
手動でハンドリングする等）を別途検討してから追加してください。
"""

from datetime import datetime
from typing import List
from zoneinfo import ZoneInfo

from db import get_connection
from logging_setup import get_logger

logger = get_logger(__name__)

JST = ZoneInfo("Asia/Tokyo")
_WEEKDAY_JA = ["月", "火", "水", "木", "金", "土", "日"]


# ------------------------------------------------------------------
# ツール1: 現在日時の取得
# ------------------------------------------------------------------
def get_current_datetime() -> str:
    """現在の日付と時刻（日本時間）を返す。

    「今日」「明日」「3日後」のような相対的な日時が関わる質問に答えるときや、
    メモ・リマインダーに具体的な日時を記録する前に、まずこれを呼び出して
    現在時刻を確認すること。
    """
    now = datetime.now(JST)
    return now.strftime(f"%Y年%m月%d日（{_WEEKDAY_JA[now.weekday()]}）%H:%M")


# ------------------------------------------------------------------
# ツール2・3: メモ／ToDo／リマインダーの保存と一覧
# 【Step 2で変更】memos.json（JSON全体を毎回読み書き）からSQLite（db.py）へ移行。
# ------------------------------------------------------------------
def add_memo(text: str) -> str:
    """メモ・ToDo・リマインダーを1件保存する。

    ユーザーが「〇〇をメモして」「覚えておいて」「リマインダーを追加して」の
    ように、後で思い出したい・記録しておきたい内容を伝えてきたときに呼び出す。
    保存した内容は list_memos で一覧取得できる。

    Args:
        text: 保存する内容。日時が関係する場合は、get_current_datetimeで
            確認した具体的な日時を含めること（例: 「8/28(金) 15:00 歯医者」）。
    """
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO memos (text, created_at, done) VALUES (?, ?, 0)",
                (text, datetime.now(JST).isoformat()),
            )
            entry_id = cursor.lastrowid
    except Exception as e:
        logger.warning("メモの保存に失敗しました: %s", e)
        return "メモの保存に失敗しました（データベースエラー）。"

    logger.info("メモを追加しました（memo_%d）: %s", entry_id, text)
    return f"メモを保存しました（ID: memo_{entry_id}）: {text}"


def list_memos() -> str:
    """保存されているメモ・ToDo・リマインダーを一覧で返す。

    ユーザーが「メモ一覧を見せて」「リマインダーある？」「ToDo確認して」の
    ように、保存済みの内容を確認したいときに呼び出す。
    """
    with get_connection() as conn:
        rows = conn.execute("SELECT id, text, done FROM memos ORDER BY id").fetchall()

    if not rows:
        return "保存されているメモはまだありません。"

    lines = [f"{'✅' if row['done'] else '・'} [memo_{row['id']}] {row['text']}" for row in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------
# 利用可能なツール一覧
# 新しいツール関数を追加したら、必ずここにも追加すること（追加を忘れると
# 関数を定義しただけではモデルから呼び出せない）。
# ------------------------------------------------------------------
AVAILABLE_TOOLS: List = [
    get_current_datetime,
    add_memo,
    list_memos,
]
