"""
PDF AIチャット（モジュール分割版）

PDFを取り込み、階層的（親子）チャンクに分割してベクトルDB（ChromaDB）に保存し、
質問に対して「PDF検索（CrossEncoderで再ランキング）」「長期記憶（過去の会話）検索」
「（任意で）Web検索」の結果をGeminiに渡して回答を生成するStreamlitアプリ。

このファイルはUIの組み立てのみを担当し、実処理は各モジュールに委譲している。
- rag_engine.py    : PDF登録・検索・ハイライト画像生成
- memory_store.py  : 長期記憶・チャットスレッドの保存/検索
- llm_client.py    : Gemini呼び出し（リトライ・キャッシュ・Pydantic検証込み）
- guardrails.py    : 簡易プロンプトインジェクション対策
- embeddings.py    : e5モデル用のprefix付き埋め込み
- web_search.py    : DuckDuckGo検索
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import (
    CHAT_COLLECTION_NAME,
    CHAT_HISTORY_MAX_TOKENS,
    ANSWER_PROMPT_VERSION,
    CHROMA_DB_PATH,
    CHROMA_DISTANCE_METRIC,
    CORE_API_HOST,
    CORE_API_PORT,
    CORE_API_TIMEOUT_SEC,
    DEFAULT_THRESHOLD_LABEL,
    EMBEDDING_MODEL_NAME,
    PDF_CHILD_COLLECTION_NAME,
    PDF_PARENT_COLLECTION_NAME,
    THRESHOLD_OPTIONS,
)
from db import init_db
from embeddings import embed_query
from env_validation import validate_environment
from guardrails import check_input_safety
from llm_client import generate_answer_with_tag, is_time_only_query, load_prompt
from logging_setup import get_logger
from memory_store import (
    build_history_text,
    delete_memory,
    get_grouped_memories,
    load_chat_sessions,
    register_chat_memory,
    reset_chat_memory,
    save_chat_sessions,
    search_memory_context,
)
from tools import get_current_datetime
from tools import delete_memo, reset_memos
from rag_engine import (
    generate_highlighted_images,
    get_registered_documents,
    get_reranker,
    pdf_path_for_hash,
    register_pdf,
    search_pdf_context,
)
from web_search import search_web_ddg

logger = get_logger(__name__)


# ------------------------------------------------------------------
# 初期化
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="モデルとデータベースを初期化しています...")
def init_system():
    """アプリ起動時に一度だけ実行される初期化処理。"""
    validate_environment()  # 必須環境変数をチェック（不足していれば早期にエラー終了）

    init_db()  # answer_cache / memos / chat_sessions用のSQLiteテーブルを用意（Step 2）

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # e5モデルはコサイン類似度前提のため、ベクトル検索する2つのコレクションは
    # hnsw:space=cosine で作成する（親チャンク用はID参照のみなので必須ではないが統一しておく）
    cosine_meta = {"hnsw:space": CHROMA_DISTANCE_METRIC}
    child_collection = chroma_client.get_or_create_collection(name=PDF_CHILD_COLLECTION_NAME, metadata=cosine_meta)
    parent_collection = chroma_client.get_or_create_collection(name=PDF_PARENT_COLLECTION_NAME, metadata=cosine_meta)
    chat_collection = chroma_client.get_or_create_collection(name=CHAT_COLLECTION_NAME, metadata=cosine_meta)

    # リランキング用CrossEncoderもここで読み込んでおく（初回質問時の待ち時間を減らすため）
    get_reranker()

    return embedding_model, child_collection, parent_collection, chat_collection


# ------------------------------------------------------------------
# セッション状態
# ------------------------------------------------------------------
def _new_session_id() -> str:
    return f"session_{int(time.time_ns())}"


def _init_session_state() -> None:
    if "sessions" not in st.session_state:
        st.session_state.sessions = load_chat_sessions()
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = _new_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _save_current_thread() -> None:
    clean_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    title = (clean_messages[0]["content"][:15] + "...") if clean_messages else "新規チャット"
    st.session_state.sessions[st.session_state.current_session_id] = {
        "title": title,
        "messages": clean_messages,
    }
    save_chat_sessions(st.session_state.sessions)


# ------------------------------------------------------------------
# UI: サイドバー — チャットスレッド
# ------------------------------------------------------------------
def check_core_status() -> dict | None:
    """Jarvis CoreのローカルAPIに生死確認を行う。

    Coreが未起動なのは異常事態ではなく普通に起こりうる状態
    （register_startup.pyでの登録前、単に落ちている等）なので、
    接続失敗は例外を投げっぱなしにせずNoneを返して呼び出し側に委ねる。
    タイムアウトを短く（config.CORE_API_TIMEOUT_SEC）していることで、
    Core未起動時にUIの表示が遅くなるのを防いでいる。
    """
    url = f"http://{CORE_API_HOST}:{CORE_API_PORT}/health"
    try:
        with urllib.request.urlopen(url, timeout=CORE_API_TIMEOUT_SEC) as res:
            return json.loads(res.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def render_thread_sidebar() -> None:
    if st.button("➕ 新しいチャット", type="primary", use_container_width=True):
        st.session_state.current_session_id = _new_session_id()
        st.session_state.messages = []
        st.rerun()

    st.subheader("💬 チャット履歴")
    if not st.session_state.sessions:
        st.caption("履歴はありません")
        return

    for s_id, s_data in reversed(list(st.session_state.sessions.items())):
        title = s_data.get("title", "無題のチャット")
        is_current = s_id == st.session_state.current_session_id
        btn_type = "primary" if is_current else "secondary"

        col_select, col_delete = st.columns([5, 1])
        with col_select:
            if st.button(f"🗨️ {title}", key=f"btn_{s_id}", type=btn_type, use_container_width=True):
                st.session_state.current_session_id = s_id
                st.session_state.messages = s_data.get("messages", [])
                st.rerun()
        with col_delete:
            if st.button("🗑️", key=f"del_{s_id}", help="このスレッドを削除"):
                del st.session_state.sessions[s_id]
                save_chat_sessions(st.session_state.sessions)
                if is_current:
                    st.session_state.current_session_id = _new_session_id()
                    st.session_state.messages = []
                st.rerun()


# ------------------------------------------------------------------
# UI: サイドバー — 資料追加・検索設定
# ------------------------------------------------------------------
def _render_search_scope(registered_docs):
    """検索対象PDFのスコープ選択UI。戻り値は source_hash のリスト、または全体検索ならNone。"""
    if not registered_docs:
        return None

    st.caption("🔎 検索範囲")
    scope_mode = st.radio(
        "検索範囲",
        ["登録済みの全PDFから検索", "PDFを指定して検索"],
        index=0,
        label_visibility="collapsed",
    )
    if scope_mode == "登録済みの全PDFから検索":
        return None

    doc_labels = [d["name"] for d in registered_docs]
    doc_hashes = [d["source_hash"] for d in registered_docs]
    selected_labels = st.multiselect(
        "対象PDF（未選択の場合は全体検索になります）",
        options=doc_labels,
        default=doc_labels[:1],
    )
    if not selected_labels:
        return None
    return [doc_hashes[doc_labels.index(label)] for label in selected_labels]


def render_settings(model, child_collection, parent_collection):
    """Web検索トグル・資料追加・検索設定のUIを描画し、設定値を返す"""
    st.subheader("🌐 Web検索機能")
    use_web_search = st.toggle("Web検索を有効にする", value=False)

    st.header("📁 資料追加")
    uploaded_file = st.file_uploader("PDFを選択", type=["pdf"])
    if uploaded_file and st.button("記憶させる"):
        register_pdf(uploaded_file, model, child_collection, parent_collection)
        st.success(f"「{uploaded_file.name}」の処理が完了しました。")
        st.rerun()

    registered_docs = get_registered_documents(parent_collection)

    if registered_docs:
        doc_labels = [f"{d['name']}（{d['max_page'] + 1}ページ）" for d in registered_docs]
        doc_hashes = [d["source_hash"] for d in registered_docs]
        current_hash = st.session_state.get("active_pdf_hash")
        default_idx = doc_hashes.index(current_hash) if current_hash in doc_hashes else 0
        chosen_idx = st.selectbox(
            "📄 プレビュー表示するPDF",
            options=range(len(doc_labels)),
            format_func=lambda i: doc_labels[i],
            index=default_idx,
        )
        st.session_state["active_pdf_hash"] = doc_hashes[chosen_idx]

    search_scope_hashes = _render_search_scope(registered_docs)

    st.divider()
    st.header("⚙️ 設定")
    threshold_label = st.segmented_control(
        "🔍 検索の厳しさ",
        options=list(THRESHOLD_OPTIONS.keys()),
        selection_mode="single",
        default=DEFAULT_THRESHOLD_LABEL,
    )
    threshold = THRESHOLD_OPTIONS[threshold_label or DEFAULT_THRESHOLD_LABEL]
    st.caption("※距離指標をコサインに変更したばかりのため、閾値は実際の挙動を見ながら調整してください。")

    preview_pos = st.segmented_control(
        "📄 PDFの表示位置",
        options=["左", "非表示", "右"],
        default="非表示",
    )
    return use_web_search, threshold, (preview_pos or "非表示"), search_scope_hashes


# ------------------------------------------------------------------
# UI: サイドバー — 記憶の管理
# ------------------------------------------------------------------
def render_memory_manager(chat_collection, model) -> None:
    st.warning(
        "長期記憶をリセットすると、過去の会話から保存された記憶がすべて削除されます。"
        "チャット履歴やPDFデータは削除されません。"
    )
    reset_confirmation_version = st.session_state.get(
        "reset_chat_memory_confirmation_version",
        0,
    )
    reset_confirmation_key = (
        f"reset_chat_memory_confirmed_{reset_confirmation_version}"
    )
    reset_confirmed = st.checkbox(
        "長期記憶をすべて削除することを確認しました",
        key=reset_confirmation_key,
    )
    if st.button(
        "🧹 長期記憶をリセット",
        disabled=not reset_confirmed,
        type="secondary",
        use_container_width=True,
    ):
        deleted_count = reset_chat_memory(chat_collection)
        st.session_state["show_undo"] = False
        st.session_state["deleted_backup"] = None
        st.session_state["reset_chat_memory_confirmation_version"] = (
            reset_confirmation_version + 1
        )
        st.success(f"長期記憶をリセットしました（{deleted_count}件削除）。")
        time.sleep(1)
        st.rerun()

    search_query = st.text_input("検索（キーワードを入力）", placeholder="例: Python, 料理")
    grouped_memories = get_grouped_memories(chat_collection, search_query, model)

    items_to_delete = []

    if not grouped_memories:
        st.info("該当する記憶がありません。")
    else:
        st.write("▼ カテゴリごとに確認・削除できます")
        for tag, memories in grouped_memories.items():
            with st.expander(f"📁 {tag} ({len(memories)}件)"):
                selected = st.multiselect(
                    "削除したい記憶を選んでください",
                    options=memories,
                    format_func=lambda x: f"💬 {x[:30]}...",
                    key=f"ms_{tag}",
                )
                if selected:
                    items_to_delete.extend(selected)
                    st.write("**【削除プレビュー】**")
                    for m in selected:
                        st.caption(m)

    if items_to_delete:
        if st.button(f"🗑️ 選択した {len(items_to_delete)} 件の記憶を一括消去", type="primary"):
            backup_data = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
            for m in items_to_delete:
                results = chat_collection.get(
                    where={"parent_text": m},
                    include=["embeddings", "documents", "metadatas"],
                )
                if results and results["ids"]:
                    backup_data["ids"].extend(results["ids"])
                    backup_data["embeddings"].extend(results["embeddings"])
                    backup_data["documents"].extend(results["documents"])
                    backup_data["metadatas"].extend(results["metadatas"])

            st.session_state["deleted_backup"] = backup_data
            st.session_state["show_undo"] = True

            for m in items_to_delete:
                delete_memory(chat_collection, m)

            st.rerun()

    if st.session_state.get("show_undo", False):
        st.success("記憶をゴミ箱に移動しました。")
        if st.button("↩️ 元に戻す"):
            backup = st.session_state["deleted_backup"]
            if backup and backup["ids"]:
                chat_collection.add(
                    ids=backup["ids"],
                    embeddings=backup["embeddings"],
                    documents=backup["documents"],
                    metadatas=backup["metadatas"],
                )
            st.session_state["show_undo"] = False
            st.session_state["deleted_backup"] = None
            st.toast("復元しました！", icon="🔙")
            time.sleep(2)
            st.rerun()


def render_memo_manager() -> None:
    """SQLiteに保存された通常メモを表示・個別削除・全削除するUI。"""
    from db import get_connection

    st.caption("通常メモ（AI長期記憶とは別の保存データ）")
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, text, created_at, done FROM memos ORDER BY id DESC"
        ).fetchall()

    if not rows:
        st.info("保存されている通常メモはありません。")
        return

    options = {
        f"memo_{row['id']} | {'完了' if row['done'] else '未完了'} | {row['text']}": row["id"]
        for row in rows
    }
    selection_version = st.session_state.get("normal_memo_selection_version", 0)
    selection_key = f"selected_normal_memos_{selection_version}"
    selected_labels = st.multiselect(
        "削除する通常メモを選択",
        options=list(options),
        key=selection_key,
    )
    if selected_labels and st.button(
        f"選択した {len(selected_labels)} 件を削除",
        type="secondary",
        use_container_width=True,
    ):
        for label in selected_labels:
            delete_memo(options[label])
        st.session_state["normal_memo_selection_version"] = selection_version + 1
        st.success(f"{len(selected_labels)}件の通常メモを削除しました。")
        st.rerun()

    clear_version = st.session_state.get("reset_normal_memos_version", 0)
    clear_key = f"reset_normal_memos_confirmed_{clear_version}"
    confirmed = st.checkbox(
        "通常メモをすべて削除することを確認しました",
        key=clear_key,
    )
    if st.button(
        "通常メモをすべて削除",
        disabled=not confirmed,
        type="secondary",
        use_container_width=True,
    ):
        deleted_count = reset_memos()
        st.session_state["reset_normal_memos_version"] = clear_version + 1
        st.success(f"通常メモを{deleted_count}件削除しました。")
        st.rerun()


# ------------------------------------------------------------------
# UI: メインエリア
# ------------------------------------------------------------------
def render_pdf_preview(preview_pos: str):
    chat_container = st.container()

    active_hash = st.session_state.get("active_pdf_hash")
    active_path = pdf_path_for_hash(active_hash) if active_hash else None

    if preview_pos != "非表示" and active_path and os.path.exists(active_path):
        col1, col2 = st.columns([1, 1])
        preview_col, chat_container = (col2, col1) if preview_pos == "右" else (col1, col2)

        with preview_col:
            st.markdown("**プレビュー**")
            with open(active_path, "rb") as f:
                bytes_data = f.read()
            base64_pdf = base64.b64encode(bytes_data).decode("utf-8")
            pdf_display = (
                f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                f'width="100%" height="700" type="application/pdf"></iframe>'
            )
            st.markdown(pdf_display, unsafe_allow_html=True)

    return chat_container


def render_reference_images(images) -> None:
    if not images:
        return
    st.write("---")
    st.write("**🔍 参考ページと該当箇所**")
    cols = st.columns(len(images))
    for idx, img_info in enumerate(images):
        with cols[idx]:
            source = img_info.get("source")
            caption = f"{source} P.{img_info['page_num'] + 1}" if source else f"ページ {img_info['page_num'] + 1}"
            st.caption(caption)
            st.image(img_info["img_bytes"], use_container_width=True)


def render_debug_panel(mem_context, mem_score, threshold, web_context, hit_pages, pdf_context, auto_tag, was_cached, tools_used=None) -> None:
    with st.expander("内部データを確認"):
        if auto_tag:
            st.write(f"**自動タグ:** {auto_tag}")
        if was_cached:
            st.write("**⚡ 完全一致キャッシュから返答しました（API呼び出しなし）**")
        if tools_used:
            st.write(f"**🔧 使用したツール:** {', '.join(tools_used)}（長期記憶・キャッシュへの保存はスキップ）")
        st.write(f"**長期記憶スコア (距離: {mem_score:.2f} / {threshold}未満で採用):**")
        st.write(mem_context)
        st.write("---")
        st.write("**【Web検索結果】**")
        st.write(web_context)
        st.write("---")
        st.write("**【PDFからの抽出データ（リランキング後）】**")
        if hit_pages:
            for idx, p_info in enumerate(hit_pages):
                rerank_score = p_info.get("rerank_score")
                score_text = f" / rerankスコア: {rerank_score:.2f}" if rerank_score is not None else ""
                st.write(f"**📄 ヒット {idx + 1} (ページ {p_info['page_num'] + 1}, 距離: {p_info['distance']:.2f}{score_text}):**")
                st.write(f"**子データ:** {p_info['child_text']}")
        else:
            st.write("PDFからのヒットなし")
        st.write("---")
        st.write("**AIへ渡した親データ（結合版）:**")
        st.write(pdf_context)


def _typewriter_markdown(placeholder, text: str, target_duration: float = 1.2) -> None:
    """回答テキストを少しずつ表示する疑似ストリーミング。

    Geminiの応答自体は既に1回のAPI呼び出しで全文取得済み（本物のトークン
    ストリーミングにすると「回答＋タグ」を1回のJSON呼び出しで済ませている
    設計を崩し、API呼び出し回数が増えてコストが上がるため、見た目だけの
    演出としてクライアント側で文字を少しずつ表示している）。
    """
    if not text:
        placeholder.markdown(text)
        return

    steps = max(1, min(50, len(text) // 6))
    chunk_size = max(1, -(-len(text) // steps))  # 切り上げ
    delay = target_duration / steps

    displayed = ""
    for i in range(0, len(text), chunk_size):
        displayed += text[i: i + chunk_size]
        placeholder.markdown(displayed)
        time.sleep(delay)
    placeholder.markdown(text)


# ------------------------------------------------------------------
# 回答生成パイプライン
# ------------------------------------------------------------------
def _build_cache_key(prompt: str, use_web_search: bool, search_scope_hashes) -> str:
    """回答キャッシュのキーを作る。

    full_prompt（履歴・長期記憶・Web検索結果まで含む最終プロンプト全文）を
    そのままキーにすると、会話が1往復進むだけで履歴部分が変わってしまい
    「同じ質問文ならキャッシュを再利用する」という意図がほぼ機能しなくなる。
    ここでは「同じ質問文・同じWeb検索設定・同じ検索範囲」という、ターンをまたいでも
    変わりにくい単位でキーを作ることで、完全一致キャッシュを名前どおりに機能させる。
    """
    key_obj = {
        "prompt_version": ANSWER_PROMPT_VERSION,
        "q": prompt.strip(),
        "web": bool(use_web_search),
        "scope": sorted(search_scope_hashes) if search_scope_hashes else None,
    }
    return json.dumps(key_obj, ensure_ascii=False, sort_keys=True)


def generate_rag_response(prompt, chat_history, model, child_collection, parent_collection, chat_collection, threshold, use_web_search, search_scope_hashes):
    # 時刻だけを尋ねる質問は検索・LLM・メモ保存を通さず、決定的に処理する。
    # これにより、検索結果や会話履歴に含まれる買い物メモなどが
    # add_memo の判断へ影響する経路自体をなくす。
    if is_time_only_query(prompt):
        return (
            get_current_datetime(),
            [],
            "なし",
            "なし",
            999.0,
            "なし",
            "未分類",
            False,
            False,
            ["get_current_datetime"],
        )

    query_vector = embed_query(model, prompt)

    pdf_context, hit_pages = search_pdf_context(
        prompt, query_vector, child_collection, parent_collection, threshold, source_hash_filter=search_scope_hashes
    )
    memory_context, mem_score = search_memory_context(query_vector, chat_collection, threshold)
    history_text = build_history_text(chat_history, max_tokens=CHAT_HISTORY_MAX_TOKENS)

    # ------------------------------------------------------------------
    # Web検索の自動フォールバックとエラー制御
    # ------------------------------------------------------------------
    # 手動トグルがON、または「PDFで該当情報が見つからなかった（hit_pagesが空）」場合に自動実行。
    # did_search_web は「実際にWeb検索を行ったか」を呼び出し側（UI）に伝えるためのフラグ。
    # トグルOFFでも自動フォールバックで検索することがあるため、UI側で
    # 「検索した事実」を必ず表示できるように分離して返す。
    did_search_web = use_web_search or (not hit_pages)

    if did_search_web:
        try:
            web_context = search_web_ddg(prompt)
        except Exception as e:
            logger.error(f"Web検索エラー: {e}")
            web_context = "※Web検索処理でエラーが発生したため、内部知識ベースのみで回答します。"
    else:
        web_context = "なし"

    full_prompt = load_prompt(
        "answer_system.txt",
        history=history_text if history_text else "なし",
        memory=memory_context,
        pdf_context=pdf_context,
        web_context=web_context,
        question=prompt,
    )
    cache_key = _build_cache_key(prompt, use_web_search, search_scope_hashes)
    answer, auto_tag, was_cached, tools_used = generate_answer_with_tag(
        full_prompt,
        cache_key=cache_key,
        user_query=prompt,
    )

    return (
        answer, hit_pages, pdf_context, memory_context, mem_score, web_context,
        auto_tag, was_cached, did_search_web, tools_used,
    )


def handle_chat_input(prompt, model, child_collection, parent_collection, chat_collection, threshold, use_web_search, search_scope_hashes):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 簡易ガードレール（追加のAPI呼び出しなしで検知できる範囲のみ）
    safety_warning = check_input_safety(prompt)
    if safety_warning:
        with st.chat_message("assistant"):
            st.warning(safety_warning)
        st.session_state.messages.append({"role": "assistant", "content": safety_warning, "images": []})
        _save_current_thread()
        return

    with st.chat_message("assistant"):
        with st.status("検索中...", expanded=True) as status:
            st.write("回答を作成しています...")

            history_for_ai = st.session_state.messages[:-1]
            (
                answer, hit_pages, pdf_context, mem_context, mem_score,
                web_context, auto_tag, was_cached, did_search_web, tools_used,
            ) = generate_rag_response(
                prompt, history_for_ai, model, child_collection, parent_collection,
                chat_collection, threshold, use_web_search, search_scope_hashes,
            )

            # did_search_web は「トグルON」と「PDFにヒットなしでの自動フォールバック」の
            # 両方を含む。どちらの理由でも、実際にWeb検索した事実は必ずユーザーに見せる
            # （トグルOFFのまま裏で検索されているように見えるのを防ぐ）。
            if did_search_web:
                reason = "" if use_web_search else "（PDFに該当情報が見つからなかったため自動実行）"
                st.write(f"🌐 Webも検索しました{reason}")

            status.update(label="完了", state="complete", expanded=False)

        answer_placeholder = st.empty()
        _typewriter_markdown(answer_placeholder, answer)

        generated_images = generate_highlighted_images(hit_pages)
        render_reference_images(generated_images)
        render_debug_panel(mem_context, mem_score, threshold, web_context, hit_pages, pdf_context, auto_tag, was_cached, tools_used)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "images": generated_images,
    })

    _save_current_thread()

    # ツールを使った回答（例: 現在時刻）は状態依存の一時的な内容であることが多く、
    # 長期記憶に残すと後の類似質問で古い・的外れな内容が誤って再利用される
    # （実機で「現在時刻」の回答が記憶され、後の類似質問で誤って呼び出され
    # 同じ誤答を繰り返す事例が発生したための対策）。
    if tools_used:
        logger.info("ツール使用（%s）のため、長期記憶への保存をスキップします。", tools_used)
    else:
        register_chat_memory(prompt, answer, model, chat_collection, auto_tag=auto_tag)


def render_chat(model, child_collection, parent_collection, chat_collection, threshold, use_web_search, search_scope_hashes, chat_container) -> None:
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                render_reference_images(message.get("images"))

        if prompt := st.chat_input("質問を入力してください", key="user_chat_input"):
            handle_chat_input(prompt, model, child_collection, parent_collection, chat_collection, threshold, use_web_search, search_scope_hashes)


def main() -> None:
    st.set_page_config(page_title="PDF AIチャット", layout="wide")
    st.title("PDF AIチャット")

    model, child_collection, parent_collection, chat_collection = init_system()
    _init_session_state()

    with st.sidebar:
        core_status = check_core_status()
        if core_status:
            uptime_min = round(core_status.get("uptime_sec", 0) / 60, 1)
            st.success(f"🟢 Jarvis Core 起動中（稼働 {uptime_min}分）", icon="✅")
        else:
            st.warning("🔴 Jarvis Core 停止中（自律タスク・PC操作等は利用できません）", icon="⚠️")
        render_thread_sidebar()
        st.divider()
        use_web_search, threshold, preview_pos, search_scope_hashes = render_settings(model, child_collection, parent_collection)
        st.divider()
        st.subheader("📝 記憶の管理")
        render_memo_manager()
        st.divider()
        render_memory_manager(chat_collection, model)

    chat_container = render_pdf_preview(preview_pos)
    render_chat(model, child_collection, parent_collection, chat_collection, threshold, use_web_search, search_scope_hashes, chat_container)


if __name__ == "__main__":
    main()