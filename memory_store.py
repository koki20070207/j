"""
長期記憶（過去の会話）とチャットスレッドの保存・検索まわり。

このモジュールで対応している改善項目:
- チャット履歴を「固定件数」ではなく「概算トークン数」で管理する
- 長期記憶の保存件数を1件→複数件（MEMORY_SEARCH_N_RESULTS）に増やして再現性を上げる
- 完全に同一の内容は保存しない／文字数上限で切る（軽量なノイズ対策。
  「毎ターンLLMで要約する」方式は追加のAPI課金が発生するため、デモ段階では見送っている）
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHAT_HISTORY_MAX_MESSAGES,
    CHAT_HISTORY_MAX_TOKENS,
    MEMORY_CONTEXT_MAX_CHARS,
    MEMORY_SEARCH_N_RESULTS,
)
from db import get_connection
from embeddings import embed_passage, embed_query
from logging_setup import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# チャットスレッド（履歴）の保存・読み込み
# 【Step 2で変更】chat_sessions.json（JSON全体を毎回読み書き）からSQLiteへ移行。
#
# app.py側は「全スレッドの辞書を1回読み込み、更新のたびに辞書全体を書き戻す」
# という使い方をしているため、その呼び出し方自体は変えず、内部の保存先だけを
# SQLiteに差し替える形にしている（app.py側の変更を最小限にするため）。
#
# 【気づき・今後の課題】この「読み込みは1回、更新は毎回全体を書き戻す」方式は、
# Jarvis Core（別プロセス）が将来チャット類似の履歴を書き込むようになった場合、
# UIが古い状態を丸ごと書き戻してCore側の更新を上書きしてしまうリスクがある。
# 今のところチャットスレッドを書くのはUIだけなので実害はないが、Step 3
# （Core⇔UI連携）で本格的に見直すのが良い。
# ------------------------------------------------------------------
def load_chat_sessions() -> Dict[str, Any]:
    with get_connection() as conn:
        sessions_rows = conn.execute(
            "SELECT session_id, title FROM chat_sessions ORDER BY created_at"
        ).fetchall()
        result: Dict[str, Any] = {}
        for s in sessions_rows:
            messages = conn.execute(
                "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY seq",
                (s["session_id"],),
            ).fetchall()
            result[s["session_id"]] = {
                "title": s["title"],
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            }
    return result


def save_chat_sessions(sessions: Dict[str, Any]) -> bool:
    """渡された辞書の内容で、chat_sessions/chat_messagesテーブルを丸ごと置き換える。

    1トランザクション内でDELETE→INSERTするため、途中でエラーが起きても
    中途半端な状態でファイルが壊れる（旧JSON方式で起きうった問題）ことはない。
    """
    try:
        with get_connection() as conn:
            conn.execute("DELETE FROM chat_sessions")  # ON DELETE CASCADEでchat_messagesも消える
            for session_id, data in sessions.items():
                conn.execute(
                    "INSERT INTO chat_sessions (session_id, title) VALUES (?, ?)",
                    (session_id, data.get("title", "")),
                )
                for seq, msg in enumerate(data.get("messages", [])):
                    conn.execute(
                        "INSERT INTO chat_messages (session_id, seq, role, content) VALUES (?, ?, ?, ?)",
                        (session_id, seq, msg.get("role", ""), msg.get("content", "")),
                    )
        return True
    except Exception as e:
        logger.warning("チャットスレッドの保存に失敗しました: %s", e)
        return False


# ------------------------------------------------------------------
# トークン基準の履歴管理
# ------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """概算のトークン数を見積もる簡易ヒューリスティック。

    Geminiの正式なトークナイザーは配布されていないため厳密な値ではないが、
    日本語混じりのテキストでは概ね「文字数の半分〜同程度」がトークン数の目安になる。
    ここでは安全側（やや多め）に見積もる。
    """
    return max(1, len(text) // 2)


def build_history_text(chat_history: List[Dict[str, str]], max_tokens: int = CHAT_HISTORY_MAX_TOKENS) -> str:
    """直近の会話履歴を、概算トークン数の上限に収まるように新しい方から積み上げて組み立てる。

    以前は「直近6件」という固定件数だったため、短いやり取りが続くと情報を無駄に削り、
    逆に長いやり取りが続くとプロンプトが肥大化してコストが上がる問題があった。
    """
    selected: List[str] = []
    total_tokens = 0

    for msg in reversed(chat_history[-CHAT_HISTORY_MAX_MESSAGES:]):
        role = "ユーザー" if msg["role"] == "user" else "AI"
        line = f"{role}: {msg['content']}\n"
        line_tokens = estimate_tokens(line)

        if total_tokens + line_tokens > max_tokens and selected:
            break

        selected.append(line)
        total_tokens += line_tokens

    return "".join(reversed(selected))


# ------------------------------------------------------------------
# 長期記憶
# ------------------------------------------------------------------
def register_chat_memory(
    prompt: str,
    answer: str,
    model: SentenceTransformer,
    chat_collection: chromadb.Collection,
    auto_tag: str,
) -> None:
    """会話履歴を保存する。

    - 文字数上限で切って、1件が肥大化しすぎないようにする
    - 完全に同一内容（同じ質問文）がすでにある場合は保存しない（ノイズの重複を防ぐ）

    保存する埋め込みは常に embed_passage()（"passage: " プレフィックス）で計算する。
    以前は呼び出し元がすでに持っている検索用埋め込み（embed_query()、"query: " プレフィックス）を
    使い回すオプションがあったが、e5系モデルは「検索される側（保存文書）はpassage、
    検索する側（クエリ）はquery」という非対称な学習をしているため、
    保存すべきデータをquery用ベクトルで登録すると、後の長期記憶検索
    （search_memory_context = embed_queryしたベクトルで検索）の精度が本来より落ちてしまう。
    再計算のコストよりも検索精度を優先し、常にここでembed_passageし直す。
    """
    conversation_context = f"ユーザー: {prompt}\nAI: {answer}"[:MEMORY_CONTEXT_MAX_CHARS]

    try:
        existing = chat_collection.get(where={"parent_text": conversation_context}, limit=1)
    except Exception:
        existing = None
    if existing and existing.get("ids"):
        logger.info("同一内容の記憶がすでに存在するため、保存をスキップしました。")
        return

    q_embedding = embed_passage(model, prompt)
    q_id = f"chat_q_{int(time.time_ns())}"
    chat_collection.upsert(
        embeddings=[q_embedding],
        documents=[prompt],
        metadatas=[{"parent_text": conversation_context, "type": "question", "tag": auto_tag}],
        ids=[q_id],
    )

    a_embedding = embed_passage(model, answer)
    a_id = f"chat_a_{int(time.time_ns())}"
    chat_collection.upsert(
        embeddings=[a_embedding],
        documents=[answer],
        metadatas=[{"parent_text": conversation_context, "type": "answer", "tag": auto_tag}],
        ids=[a_id],
    )


def delete_memory(collection: chromadb.Collection, parent_text: str) -> None:
    """指定された親テキストを持つ記憶（QとAの子データ両方）をまとめて削除する"""
    collection.delete(where={"parent_text": parent_text})


def reset_chat_memory(collection: chromadb.Collection) -> int:
    """チャット長期記憶を全削除し、削除したレコード数を返す。

    コレクション自体は削除しない。呼び出し元が保持しているChromaDBの
    コレクションハンドルをそのまま使い続けられるようにするためである。
    """
    records = collection.get()
    ids = records.get("ids", [])
    if not ids:
        return 0

    # ChromaDBのバッチ上限を超える場合に備えて分割する。
    for start in range(0, len(ids), 5000):
        collection.delete(ids=ids[start:start + 5000])
    logger.info("チャット長期記憶をリセットしました: %d件", len(ids))
    return len(ids)


def get_grouped_memories(
    collection: chromadb.Collection,
    search_word: Optional[str] = None,
    model: Optional[SentenceTransformer] = None,
) -> Dict[str, List[str]]:
    """記憶を取得し、タグごとにグループ化する（検索対応）"""
    if search_word and model:
        vector = embed_query(model, search_word)
        results = collection.query(query_embeddings=[vector], n_results=10)
        meta_list = results["metadatas"][0] if results["metadatas"] else []
    else:
        results = collection.get()
        meta_list = results["metadatas"] if results["metadatas"] else []

    grouped_data: Dict[str, set] = {}
    for meta in meta_list:
        if not meta:
            continue
        text = meta.get("parent_text")
        tag = meta.get("tag", "未分類")
        if text:
            grouped_data.setdefault(tag, set()).add(text)

    return {k: list(v) for k, v in grouped_data.items()}


def search_memory_context(query_vector: List[float], chat_collection: chromadb.Collection, threshold: float) -> Tuple[str, float]:
    """長期記憶棚から関連する会話を複数件検索し、まとめて返す。

    以前は1件しか拾っていなかったが、関連する過去の会話が複数ある場合に
    取りこぼしていたため、MEMORY_SEARCH_N_RESULTS件まで拾って結合するようにした。
    """
    try:
        results = chat_collection.query(query_embeddings=[query_vector], n_results=MEMORY_SEARCH_N_RESULTS)
    except Exception as e:
        logger.warning("長期記憶データベースの検索に失敗しました: %s", e)
        return "なし", 999.0

    distances = results.get("distances") or []
    if not distances or not distances[0]:
        return "なし", 999.0

    best_score = distances[0][0]
    metadatas = results.get("metadatas") or []

    seen_texts = set()
    memory_chunks = []
    for i, score in enumerate(distances[0]):
        if score >= threshold:
            continue
        meta = metadatas[0][i] if metadatas and len(metadatas[0]) > i else None
        if not meta:
            continue
        text = meta.get("parent_text", "")
        if text and text not in seen_texts:
            seen_texts.add(text)
            memory_chunks.append(text)

    memory_context = "\n---\n".join(memory_chunks) if memory_chunks else "なし"
    return memory_context, best_score