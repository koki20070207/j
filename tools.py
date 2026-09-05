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
from typing import List, Optional
from zoneinfo import ZoneInfo
import re

from db import get_connection
from guardrails import check_input_safety
from logging_setup import get_logger

logger = get_logger(__name__)

JST = ZoneInfo("Asia/Tokyo")
_WEEKDAY_JA = ["月", "火", "水", "木", "金", "土", "日"]


# ------------------------------------------------------------------
# 入力バリデーション用ヘルパー関数
# ------------------------------------------------------------------
def _validate_text_input(text: str, min_length: int = 1, max_length: int = 10000, 
                       field_name: str = "テキスト") -> Optional[str]:
    """テキスト入力に対する共通バリデーション。
    
    Args:
       text: バリデーション対象のテキスト
       min_length: 最小文字数
       max_length: 最大文字数
       field_name: エラーメッセージ用のフィールド名
    
    Returns:
       バリデーション成功時はNone、失敗時はエラーメッセージ文字列。
    """
    if not text:
       return f"{field_name}が空です。内容を指定してください。"
    
    text_stripped = text.strip()
    
    if len(text_stripped) < min_length:
       return f"{field_name}が短すぎます（最小{min_length}文字）: {repr(text_stripped)}"
    
    if len(text_stripped) > max_length:
       return f"{field_name}が長すぎます（最大{max_length}文字）"
    
    # プロンプトインジェクション検知
    safety_warning = check_input_safety(text_stripped)
    if safety_warning:
       logger.warning("入力にセキュリティ上の懸念が検出されました: %s", repr(text_stripped))
       return safety_warning
    
    return None


def _is_likely_time_query(text: str) -> bool:
    """テキストが時刻に関するクエリかどうかを判定。
    
    メモとして不適切な「時刻に関する質問」と実際の「時刻付きメモ」を区別する。
    - メモとして不適切: 「何時？」「今何時」（短い、情報価値なし）
    - メモとして有効: 「8/28(金) 15:00 歯医者」（長い、日時情報がある）
    """
    if not text:
       return False
    
    time_related_phrases = ["時間", "何時", "時刻", "日時", "今", "明日", "昨日"]
    is_time_phrase = any(phrase in text for phrase in time_related_phrases)
    
    # 短い時刻関連クエリは、メモとして保存するには不適切
    return is_time_phrase and len(text) < 10


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
     
    Returns:
       メモが正常に保存された場合の確認メッセージ、またはエラーメッセージ。
     
    Raises:
       ValueError: テキストが空またはNoneの場合。
    """
    # 1. 基本的なテキストバリデーション
    validation_error = _validate_text_input(text, min_length=2, max_length=5000, field_name="メモのテキスト")
    if validation_error:
       logger.warning("add_memo: バリデーション失敗: %s", validation_error)
       return validation_error
    
    text_stripped = text.strip()
    
    # 2. 時刻関連クエリの検出（メモとして不適切な問い合わせ）
    if _is_likely_time_query(text_stripped):
       logger.warning(
           "⚠️ 時刻関連クエリがメモとして渡された可能性があります。正常な呼び出しか確認してください: %s",
           repr(text_stripped),
       )
       return "時刻情報はメモとして保存していません。get_current_datetimeで確認後、詳細な情報と共にメモしてください。"
    
    # 3. SQL文脈でのリスク検出（簡易チェック）
    if any(char in text_stripped for char in ["';", "-- ", "/*", "*/"]):
       logger.warning("メモに疑わしい文字列パターンが含まれています: %s", repr(text_stripped))
    
    # 4. データベースに保存
    try:
       with get_connection() as conn:
           cursor = conn.execute(
               "INSERT INTO memos (text, created_at, done) VALUES (?, ?, 0)",
               (text_stripped, datetime.now(JST).isoformat()),
           )
           entry_id = cursor.lastrowid
    except Exception as e:
       logger.warning("メモの保存に失敗しました: %s", e)
       return "メモの保存に失敗しました（データベースエラー）。"

    logger.info("メモを追加しました（memo_%d）: %s", entry_id, text_stripped)
    return f"メモを保存しました（ID: memo_{entry_id}）: {text_stripped}"


def list_memos() -> str:
    """保存されているメモ・ToDo・リマインダーを一覧で返す。

    ユーザーが「メモ一覧を見せて」「リマインダーある？」「ToDo確認して」の
    ように、保存済みの内容を確認したいときに呼び出す。
    
    Returns:
       メモ一覧の文字列。メモがない場合はその旨を伝えるメッセージ。
    """
    try:
       with get_connection() as conn:
           rows = conn.execute("SELECT id, text, done FROM memos ORDER BY id").fetchall()
    except Exception as e:
       logger.warning("メモ一覧の取得に失敗しました: %s", e)
       return "メモ一覧の取得に失敗しました（データベースエラー）。"

    if not rows:
       return "保存されているメモはまだありません。"

    # 出力の安全性チェック（返すテキストが特に長すぎないか確認）
    lines = [f"{'✅' if row['done'] else '・'} [memo_{row['id']}] {row['text']}" for row in rows]
    result = "\n".join(lines)
    
    if len(result) > 50000:
       logger.warning("メモ一覧が非常に大きいです: %d文字", len(result))
       # 最初の50件に制限
       limited_lines = lines[:50]
       if len(lines) > 50:
           limited_lines.append(f"... 他 {len(lines) - 50} 件のメモがあります")
       result = "\n".join(limited_lines)
    
    logger.info("メモ一覧を取得しました（%d件）", len(rows))
    return result


def search_memos(query: str) -> str:
    """保存されているメモを検索する。
    
    ユーザーが「会議に関するメモを検索して」「8月のメモを見て」のように
    特定のキーワードでメモを検索したいときに呼び出す。
    
    Args:
       query: 検索キーワード。
    
    Returns:
       検索結果のメモ一覧。マッチするメモがない場合はその旨を伝えるメッセージ。
    """
    # 1. クエリのバリデーション
    validation_error = _validate_text_input(query, min_length=1, max_length=500, field_name="検索キーワード")
    if validation_error:
       logger.warning("search_memos: バリデーション失敗: %s", validation_error)
       return validation_error
    
    query_stripped = query.strip()
    
    # 2. SQLインジェクション対策（正規表現でメタ文字をチェック）
    # LIKEでの特殊文字は % と _ だけなので、それ以外の疑わしい文字は検出
    if re.search(r"[';\"\\-]", query_stripped):
       logger.warning("検索キーワードに疑わしい文字が含まれています: %s", repr(query_stripped))
    
    # 3. データベース検索
    try:
       with get_connection() as conn:
           # LIKEで大文字小文字を区別しない検索を実施
           rows = conn.execute(
               "SELECT id, text, done FROM memos WHERE text LIKE ? ORDER BY id",
               (f"%{query_stripped}%",),
           ).fetchall()
    except Exception as e:
       logger.warning("メモの検索に失敗しました: %s", e)
       return "メモの検索に失敗しました（データベースエラー）。"

    if not rows:
       return f"「{query_stripped}」に該当するメモはありません。"

    # 4. 結果の安全性チェック
    lines = [f"{'✅' if row['done'] else '・'} [memo_{row['id']}] {row['text']}" for row in rows]
    result = "\n".join(lines)
    
    if len(result) > 50000:
       logger.warning("検索結果が非常に大きいです: %d文字", len(result))
       limited_lines = lines[:50]
       if len(lines) > 50:
           limited_lines.append(f"... 他 {len(lines) - 50} 件の検索結果があります")
       result = "\n".join(limited_lines)
    
    logger.info("メモを検索しました（キーワード: %s、結果: %d件）", query_stripped, len(rows))
    return result


def mark_memo_done(memo_id: int) -> str:
    """指定されたメモを完了状態にする。
    
    ユーザーが「このメモを完了にして」「やることをチェック」のように
    メモを完了状態にしたいときに呼び出す。
    
    Args:
       memo_id: 完了にするメモのID。
    
    Returns:
       完了状態になった旨のメッセージ、またはエラーメッセージ。
    """
    # メモIDの妥当性チェック
    if not isinstance(memo_id, int) or memo_id <= 0 or memo_id > 2**63 - 1:
       return f"無効なメモIDです: {memo_id}。正の整数を指定してください。"
    
    try:
       with get_connection() as conn:
           cursor = conn.execute(
               "UPDATE memos SET done = 1 WHERE id = ?",
               (memo_id,),
           )
           if cursor.rowcount == 0:
               return f"メモID {memo_id} は見つかりません。"
    except Exception as e:
       logger.warning("メモの完了状態の更新に失敗しました: %s", e)
       return "メモの更新に失敗しました（データベースエラー）。"
    
    logger.info("メモを完了状態にしました（memo_%d）", memo_id)
    return f"メモID {memo_id} を完了状態にしました。"


# ------------------------------------------------------------------
# 利用可能なツール一覧
# 新しいツール関数を追加したら、必ずここにも追加すること（追加を忘れると
# 関数を定義しただけではモデルから呼び出せない）。
# ------------------------------------------------------------------
AVAILABLE_TOOLS: List = [
    get_current_datetime,
    add_memo,
    list_memos,
    search_memos,
    mark_memo_done,
]
