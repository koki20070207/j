"""
Gemini API呼び出しまわりを1箇所に集約したモジュール。

【2026-08 変更】google-generativeai（旧SDK）が非推奨になったため、
統合SDKである google-genai に移行し、あわせてGeminiの関数呼び出し
（Function Calling / tools.pyのAVAILABLE_TOOLS）に対応した。

このモジュールで対応している改善項目:
- genai.Clientインスタンスを使い回す（毎回生成しない）
- 429（レート制限）エラー時の自動リトライ（指数バックオフ）
- 完全一致する過去の質問に対するキャッシュ（API呼び出し自体をスキップしてコスト削減）
- プロンプトのテンプレート化（.txtファイルを読み込み、プレースホルダーを置換）
- Gemini関数呼び出し（tools.py）によるツール利用
- Pydanticによる出力スキーマ検証（マルチモーダル抽出のみ。理由は下記コメント参照）

【回答＋タグ生成の方式について】
旧バージョンは response_mime_type="application/json" で {"answer":.., "tag":..}
を強制出力させていたが、これを tools（関数呼び出し）と併用できるかは公式ドキュメント
上で明言されていない。確実に動く構成を優先し、
  1. 回答は自由なテキストとして生成させる（tools を使わせるため）
  2. 分類タグは、回答の最後に "TAG: ..." という1行を出力させ、正規表現で分離する
という方式に変更している。プロンプト側（prompts/answer_system.txt）もこれに合わせて
更新が必要（本パッケージでは再構成版を同梱している）。
"""

import json
import os
import random
import re
import time
from typing import Any, List, Optional, Tuple

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from config import (
    GEMINI_MODEL_NAME,
    PROMPTS_DIR,
    RETRY_BASE_DELAY_SEC,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_DELAY_SEC,
    TOOL_CALL_MAX_ITERATIONS,
)
from db import get_connection
from logging_setup import get_logger
from tools import AVAILABLE_TOOLS

logger = get_logger(__name__)

# ------------------------------------------------------------------
# 回答テキスト末尾の "TAG: ..." 行を取り出す正規表現
# ------------------------------------------------------------------
_TAG_LINE_RE = re.compile(r"^[ \t]*TAG[ \t]*[:：][ \t]*(.+?)[ \t]*$", re.MULTILINE)


# ------------------------------------------------------------------
# Pydanticスキーマ（マルチモーダル抽出の出力検証。回答+タグは自由記述に変更したため対象外）
# ------------------------------------------------------------------
class PageTextItem(BaseModel):
    text: str = ""


# ------------------------------------------------------------------
# genai.Clientインスタンスの使い回し
# ------------------------------------------------------------------
_client_instance: Optional[genai.Client] = None


def get_gemini_client() -> genai.Client:
    """genai.Clientのインスタンスを使い回す。呼び出しのたびに新規生成していた無駄を解消。

    旧SDK（google.generativeai）はモジュールレベルの暗黙的な設定（genai.configure）を
    使う設計だったが、新SDK（google-genai）はClientオブジェクトを介した設計になっている。
    ここで遅延初期化・使い回しをすることで、旧SDKの「モデルインスタンス使い回し」と
    同じ効果を保ちつつ、app.py側の初期化コードへの変更を最小限にしている
    （環境変数 GEMINI_API_KEY は genai.Client() が自動で読み込むため、
    明示的な api_key の受け渡しは不要）。
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = genai.Client()
    return _client_instance


# ------------------------------------------------------------------
# プロンプトテンプレートの読み込み
# ------------------------------------------------------------------
def load_prompt(template_name: str, **replacements: str) -> str:
    """prompts/ ディレクトリからテンプレートを読み込み、<<KEY>> をreplacementsで置換する。

    str.format() を使わないのは、プロンプト中にJSON例（{"answer": ...}）が
    含まれておりカーリーブレースが競合するため。<<KEY>> 形式なら安全に置換できる。
    """
    path = os.path.join(PROMPTS_DIR, template_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"プロンプトファイルが見つかりません: {path}\n"
            f"→ '{PROMPTS_DIR}' フォルダの中に '{template_name}' を用意してください。"
        )
    with open(path, "r", encoding="utf-8") as f:
        template = f.read()

    for key, value in replacements.items():
        template = template.replace(f"<<{key.upper()}>>", value)

    return template


# ------------------------------------------------------------------
# リトライ付きのGemini呼び出し
# ------------------------------------------------------------------
def _is_retryable_error(error: Exception) -> bool:
    """レート制限・一時的なサーバエラーらしきものだけリトライ対象にする。

    新SDK（google-genai）が投げる例外の型は旧SDKと異なる可能性があるため、
    型ではなく文字列（HTTPステータスコードやエラー用語）で判定する方式を維持している。
    こちらのほうがSDKのバージョン差異に対して頑健。
    """
    message = str(error).lower()
    retryable_signals = ["429", "resource_exhausted", "resource exhausted", "rate limit", "503", "unavailable", "deadline exceeded"]
    return any(signal in message for signal in retryable_signals)


def _call_with_retry(contents: Any, config: "types.GenerateContentConfig") -> Tuple[Optional[Any], Optional[str]]:
    """Gemini APIを呼び出し、レート制限系エラーのときだけ指数バックオフで再試行する。

    戻り値: (レスポンス または None, エラーメッセージ または None)
    """
    client = get_gemini_client()
    last_error: Optional[str] = None

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=contents,
                config=config,
            )
            return response, None
        except Exception as e:
            last_error = str(e)

            if not _is_retryable_error(e) or attempt == RETRY_MAX_ATTEMPTS:
                logger.error("Gemini API呼び出しに失敗しました（リトライ対象外 or 上限到達）: %s", last_error)
                return None, last_error

            # 指数バックオフ + ジッター（同時に複数リクエストが一斉リトライして
            # レート制限を再度踏むのを避けるため、少しランダムにずらす）
            delay = min(RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)), RETRY_MAX_DELAY_SEC)
            delay += random.uniform(0, 1.0)
            logger.warning("Gemini APIでリトライ可能なエラー（%d/%d回目）。%.1f秒待機します: %s",
                           attempt, RETRY_MAX_ATTEMPTS, delay, last_error)
            time.sleep(delay)

    return None, last_error


def _gemini_generate_json(contents: Any) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """Gemini APIを呼び出し、JSON形式の構造化出力として解析する共通ヘルパー。

    tools（関数呼び出し）は使わない呼び出し専用（マルチモーダル抽出用）。
    response_mime_type="application/json" と tools の併用は未検証のため、
    このヘルパーではtoolsを渡さない構成に限定している。

    戻り値: (parsed_json または None, レスポンスの生テキスト または None, エラーメッセージ または None)
    """
    config = types.GenerateContentConfig(response_mime_type="application/json")
    response, error = _call_with_retry(contents, config)
    if error:
        return None, None, error

    # response.text へのアクセス自体が失敗するケース（安全フィルタでブロックされ
    # candidatesが空、finish_reasonがSAFETY/RECITATION等）がある。ここを
    # AttributeError/TypeError しか見ていないと ValueError が素通りしてクラッシュするため、
    # 「テキストが取れない」ことそのものをエラーとして扱う。
    try:
        raw_text = response.text
    except (ValueError, AttributeError) as e:
        logger.warning("Geminiの応答からテキストを取得できませんでした（安全フィルタ等でブロックされた可能性があります）: %s", e)
        return None, None, "AIの応答を取得できませんでした（安全フィルタ等でブロックされた可能性があります）"

    try:
        return json.loads(raw_text), raw_text, None
    except (json.JSONDecodeError, TypeError):
        # JSONとしては壊れているが、テキスト自体は届いているので呼び出し側で
        # 生テキストにフォールバックできるようにエラーは立てない
        return None, raw_text, None


# ------------------------------------------------------------------
# 完全一致キャッシュ（同じ質問文に対してAPIを叩かない）
# ------------------------------------------------------------------
def _normalize_cache_key(prompt: str) -> str:
    """キャッシュキーを正規化（前後の空白を除去）。
     
    Args:
        prompt: 正規化対象のプロンプト文字列。
     
    Returns:
        正規化されたキャッシュキー。
     
    Raises:
        ValueError: プロンプトが空またはNoneの場合。
    """
    if not prompt:
         raise ValueError("キャッシュキーとなるプロンプトは空にできません")
    return prompt.strip()


def get_cached_answer(cache_key: str) -> Optional[Tuple[str, str]]:
    """完全一致するキャッシュキーがあれば (answer, tag) を返す。なければNone。

    あくまで「一字一句同じ質問（＋検索条件）」のときだけヒットする単純なキャッシュ。
    【Step 2で変更】answer_cache.json（JSON全体を毎回読み書き）からSQLiteへ移行。
     
    Args:
        cache_key: 検索対象のキャッシュキー。
     
    Returns:
        (answer, tag) のタプル、またはキャッシュがない場合はNone。
     
    Raises:
        ValueError: cache_keyが空またはNoneの場合。
    """
    if not cache_key:
         logger.warning("空のキャッシュキーでの検索要求")
         return None
     
    with get_connection() as conn:
        row = conn.execute(
            "SELECT answer, tag FROM answer_cache WHERE cache_key = ?",
            (_normalize_cache_key(cache_key),),
        ).fetchone()
    if row:
        return row["answer"], row["tag"]
    return None


def set_cached_answer(cache_key: str, answer: str, tag: str) -> None:
    """キャッシュに (cache_key, answer, tag) を保存する。既存キーは上書き。
     
    Args:
        cache_key: キャッシュキー。
        answer: 回答テキスト。
        tag: 分類タグ。
     
    Raises:
        ValueError: 必須パラメータが空またはNoneの場合。
    """
    if not cache_key:
         raise ValueError("キャッシュキーは空にできません")
    if not answer:
         raise ValueError("回答テキストは空にできません")
    if not tag:
         raise ValueError("タグは空にできません")
     
    normalized_key = _normalize_cache_key(cache_key)
    if not normalized_key:
         raise ValueError("正規化後のキャッシュキーが空になりました")
     
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO answer_cache (cache_key, answer, tag) VALUES (?, ?, ?) "
            "ON CONFLICT(cache_key) DO UPDATE SET answer = excluded.answer, tag = excluded.tag",
            (normalized_key, answer, tag),
        )


# ------------------------------------------------------------------
# 回答テキストと分類タグの分離
# ------------------------------------------------------------------
def _split_answer_and_tag(raw_text: str) -> Tuple[str, str]:
    """回答テキスト中の "TAG: ..." 行を取り出し、本文とタグに分離する。

    複数行にTAGらしき記述があっても、最後に出現したものを採用する
    （モデルが本文中で"TAG"という単語に言及した場合の誤爆リスクを下げるため）。
    見つからなかった場合はタグを「未分類」とし、本文はそのまま返す。
    """
    matches = list(_TAG_LINE_RE.finditer(raw_text))
    if not matches:
        return raw_text.strip(), "未分類"

    last_match = matches[-1]
    tag = last_match.group(1).strip() or "未分類"
    answer = (raw_text[: last_match.start()] + raw_text[last_match.end():]).strip()
    return answer or "（回答が空でした）", tag


# ------------------------------------------------------------------
# 回答生成（本文＋分類タグ＋必要に応じてツール呼び出し）
# ------------------------------------------------------------------
def _execute_tool_calls(function_calls: List[Any]) -> List[Any]:
    """モデルが要求した関数呼び出しを実行し、Geminiに返すためのPartのリストを作る。

    自動関数呼び出し（automatic function calling）ではなく、あえて手動ループに
    している理由:
    1. 「今の時間は？」でメモが保存される、という実機での誤動作を調査した際、
       自動関数呼び出しは呼ばれたツールの履歴をレスポンスオブジェクトから
       素直に取り出す手段がなく、何が呼ばれたのか外から検証できなかった。
       手動ループにすることで、呼び出されたツール名・引数・結果を
       必ずログへ残せるようにしている。
    2. Step 5（要確認が必要な操作の判定）では、ツールを実行する"前"に
       判定を挟む必要がある。自動関数呼び出しはSDK内部でブラックボックスに
       実行してしまうため、いずれにせよ手動ループへの移行が必要だった。
    """
    tool_lookup = {fn.__name__: fn for fn in AVAILABLE_TOOLS}
    response_parts = []

    for fc in function_calls:
        fn = tool_lookup.get(fc.name)
        if fn is None:
            result_text = f"エラー: 未知のツール '{fc.name}' が要求されました。"
            logger.warning(result_text)
        else:
            try:
                result_text = fn(**(fc.args or {}))
            except Exception as e:
                result_text = f"エラー: ツール実行に失敗しました（{e}）"
                logger.warning("ツール実行エラー: %s(%s): %s", fc.name, fc.args, e)

        logger.info("ツール呼び出し: %s(%s) → %s", fc.name, fc.args, result_text)
        response_parts.append(
            types.Part.from_function_response(name=fc.name, response={"result": result_text})
        )

    return response_parts


def generate_answer_with_tag(
    full_prompt: str, cache_key: Optional[str] = None, use_cache: bool = True
) -> Tuple[str, str, bool, List[str]]:
    """回答本文と自動分類タグを取得する。tools.AVAILABLE_TOOLSのツールを必要に応じて使う。

    cache_key: キャッシュの照合・保存に使うキー。省略時は full_prompt をそのまま使う。

    戻り値: (answer, tag, was_cached, tools_used)

    tools_used: このターンで実際に呼ばれたツール名のリスト（無ければ空リスト）。
    ツールを使った回答（例: 現在時刻）は状態依存・一時的な内容であることが多く、
    キャッシュや長期記憶に残すと後で古い情報が誤って再利用される
    （実際に「今の時間は？」への誤った回答が長期記憶に残り、後の類似質問で
    再び引き出されて誤答を繰り返す、という事例が実機で発生した）。
    そのため呼び出し側（app.py）はtools_usedが空でない場合、キャッシュ保存・
    長期記憶保存の両方をスキップする想定。
    """
    effective_key = cache_key if cache_key is not None else full_prompt

    if use_cache:
        cached = get_cached_answer(effective_key)
        if cached:
            logger.info("完全一致キャッシュにヒットしたため、API呼び出しをスキップしました。")
            return cached[0], cached[1], True, []

    config = types.GenerateContentConfig(
        tools=AVAILABLE_TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents: List[Any] = [types.Content(role="user", parts=[types.Part(text=full_prompt)])]
    tools_used: List[str] = []

    response = None
    for iteration in range(1, TOOL_CALL_MAX_ITERATIONS + 1):
        response, error = _call_with_retry(contents, config)
        if error:
            return "（エラーによりAIの応答を取得できませんでした。しばらく待ってから再度お試しください）", "未分類", False, tools_used

        function_calls = response.function_calls
        if not function_calls:
            break

        tools_used.extend(fc.name for fc in function_calls)
        contents.append(response.candidates[0].content)
        contents.append(types.Content(role="user", parts=_execute_tool_calls(function_calls)))
    else:
        logger.warning("ツール呼び出しの反復回数が上限（%d回）に達しました。", TOOL_CALL_MAX_ITERATIONS)

    try:
        raw_text = response.text
    except (ValueError, AttributeError) as e:
        logger.warning("Geminiの応答からテキストを取得できませんでした（安全フィルタ等でブロックされた可能性があります）: %s", e)
        return "（AIの応答を取得できませんでした。安全フィルタ等でブロックされた可能性があります）", "未分類", False, tools_used

    answer, tag = _split_answer_and_tag(raw_text or "")

    if use_cache and not tools_used:
        set_cached_answer(effective_key, answer, tag)
    return answer, tag, False, tools_used


# ------------------------------------------------------------------
# マルチモーダル抽出（複数ページ画像 → テキスト/表）
# ------------------------------------------------------------------
def extract_pages_multimodal(instruction: str, images: List[Any], expected_count: int) -> Tuple[Optional[List[str]], Optional[str]]:
    """複数ページ画像をまとめて1回のAPI呼び出しで解析する。

    戻り値: (各ページのテキストのリスト または None, エラーメッセージ または None)
    """
    parsed, _, error = _gemini_generate_json([instruction, *images])

    if error:
        return None, error

    if not isinstance(parsed, list) or len(parsed) != expected_count:
        return None, "画像解析結果の枚数が一致しませんでした。"

    texts = []
    for item in parsed:
        try:
            validated = PageTextItem.model_validate(item) if isinstance(item, dict) else PageTextItem()
        except ValidationError:
            validated = PageTextItem()
        texts.append(validated.text.strip())

    return texts, None