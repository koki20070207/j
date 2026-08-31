"""
RAGのコア処理。

このモジュールで対応している改善項目:
- 子チャンクへのオーバーラップ導入
- ChromaDBのParent/Child分離（重複データ削減）
- 検索結果のリランキング（CrossEncoder、ローカル実行なので追加課金なし）
- 検索スコープの切り替え（全体 / 指定PDF）
- 図表ページのマルチモーダル解析のバッチ並列化（Gemini呼び出し部分のみ）
- 再アップロード時のゴーストデータ削除
- 複数PDFに対応したハイライト画像生成（以前のバグ修正を引き継ぎ）

PyMuPDFのページレンダリング自体は複数スレッドで同時に触ると不安定になりやすく、
かつこちらで動作検証できる実行環境がないため、そこは意図的に並列化していない
（Gemini呼び出し部分だけ、ネットワークI/O待ちなので安全に並列化できる）。
"""

import concurrent.futures
import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import pymupdf as fitz  # PyMuPDF。`import fitz` 単体は非推奨化されたため新API名からエイリアスする
import streamlit as st
from PIL import Image
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    MIN_CHUNK_LENGTH,
    MULTIMODAL_BATCH_SIZE,
    MULTIMODAL_MAX_PARALLEL_BATCHES,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PDF_DIR,
    PDF_SEARCH_MAX_PARENTS,
    PDF_SEARCH_OVERFETCH,
    RERANK_MODEL_NAME,
    TEXT_SUFFICIENCY_THRESHOLD_CHARS,
)
from embeddings import embed_passages
from llm_client import extract_pages_multimodal, load_prompt
from logging_setup import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# 共通ユーティリティ
# ------------------------------------------------------------------
def sanitize_filename(filename: str) -> str:
    """アップロードされたファイル名から、パスとして危険な文字を除去する。"""
    name = os.path.basename(filename)
    name = re.sub(r"[^\w\-.ぁ-んァ-ヶー一-龠]", "_", name)
    return name or "document.pdf"


def pdf_content_hash(file_bytes: bytes) -> str:
    """PDFの内容からハッシュ値を作る。同一内容の再アップロード検出用。"""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def pdf_path_for_hash(content_hash: str) -> str:
    return os.path.join(PDF_DIR, f"{content_hash}.pdf")


def image_path_for_hash(content_hash: str, page_num: int) -> str:
    return os.path.join(PDF_DIR, f"{content_hash}_page_{page_num}.png")


# ------------------------------------------------------------------
# リランキングモデル（CrossEncoder）の使い回し
# ------------------------------------------------------------------
_reranker_instance: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """CrossEncoderをキャッシュして使い回す（ローカル実行モデルなのでAPI課金は発生しない）"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker_instance


# ------------------------------------------------------------------
# チャンク分割（親子構造・オーバーラップ対応）
# ------------------------------------------------------------------
def create_hierarchical_chunks(
    text: str,
    parent_size: int = PARENT_CHUNK_SIZE,
    child_size: int = CHILD_CHUNK_SIZE,
    parent_overlap: int = PARENT_CHUNK_OVERLAP,
    child_overlap: int = CHILD_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """テキストを親チャンク→子チャンクへ分割する。

    戻り値: [{"parent": str, "children": [str, ...]}, ...]
    Parent/Child分離のDB構造に合わせ、1つの親チャンクにつき1レコードを想定した形にしている
    （以前は子チャンクの数だけ親テキストが重複して返っていた）。
    """
    hierarchical_data = []
    parent_step = max(parent_size - parent_overlap, 1)
    child_step = max(child_size - child_overlap, 1)

    for i in range(0, len(text), parent_step):
        parent_text = text[i: i + parent_size]
        if not parent_text:
            continue

        children = []
        for j in range(0, len(parent_text), child_step):
            child_text = parent_text[j: j + child_size]
            if len(child_text) >= MIN_CHUNK_LENGTH:
                children.append(child_text)
            if j + child_size >= len(parent_text):
                break

        if children:
            hierarchical_data.append({"parent": parent_text, "children": children})

        if i + parent_size >= len(text):
            break

    return hierarchical_data


# ------------------------------------------------------------------
# 図表ページのマルチモーダル解析（バッチ並列化）
# ------------------------------------------------------------------
def _extract_one_batch(batch: List[Tuple[int, Image.Image]]) -> Dict[int, Optional[str]]:
    """1バッチ分（最大MULTIMODAL_BATCH_SIZEページ）を解析する"""
    page_nums = [p for p, _ in batch]
    images = [img for _, img in batch]

    instruction = load_prompt("multimodal_extract.txt", page_count=str(len(images)))
    texts, error = extract_pages_multimodal(instruction, images, expected_count=len(batch))

    if error:
        logger.warning("ページ画像のバッチ解析に失敗したため、該当ページは通常のテキスト抽出にフォールバックします: %s", error)
        return {p: None for p in page_nums}

    return {p: (t or None) for p, t in zip(page_nums, texts)}


def extract_pages_multimodal_batch(pages: List[Tuple[int, Image.Image]]) -> Dict[int, Optional[str]]:
    """図表を含む複数ページの画像を、バッチ単位でGeminiに投げて解析する。

    バッチ同士は互いに独立したAPI呼び出しのため、ThreadPoolExecutorで並列に投げる。
    （呼び出し回数自体は変わらないのでAPIコストは増えない。待ち時間だけ短縮される）
    """
    batches = [pages[i: i + MULTIMODAL_BATCH_SIZE] for i in range(0, len(pages), MULTIMODAL_BATCH_SIZE)]
    results: Dict[int, Optional[str]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MULTIMODAL_MAX_PARALLEL_BATCHES) as executor:
        for batch_result in executor.map(_extract_one_batch, batches):
            results.update(batch_result)

    return results


# ------------------------------------------------------------------
# 再アップロード時のゴーストデータ削除
# ------------------------------------------------------------------
def _cleanup_stale_versions(
    safe_name: str,
    keep_hash: str,
    child_collection: chromadb.Collection,
    parent_collection: chromadb.Collection,
) -> None:
    """同じファイル名で、内容（ハッシュ）が異なる過去バージョンのデータを削除する。

    「同じファイル名のPDFを別の中身で再アップロードしたのに、古いチャンクが
    ゴーストとしてDBに残り続ける」問題への対処。ファイル名が同じでもハッシュが
    異なる＝中身が変わったとみなし、古い方は掃除する。
    """
    try:
        stale = child_collection.get(where={"source": safe_name}, include=["metadatas"])
    except Exception as e:
        logger.warning("旧バージョンの確認に失敗しました: %s", e)
        return

    stale_hashes = {
        meta.get("source_hash")
        for meta in (stale.get("metadatas") or [])
        if meta and meta.get("source_hash") and meta.get("source_hash") != keep_hash
    }

    for stale_hash in stale_hashes:
        try:
            child_collection.delete(where={"source_hash": stale_hash})
            parent_collection.delete(where={"source_hash": stale_hash})
        except Exception as e:
            logger.warning("旧バージョン（%s）のDB削除に失敗しました: %s", stale_hash, e)
            continue

        # 画像・PDF本体も削除
        stale_pdf_path = pdf_path_for_hash(stale_hash)
        if os.path.exists(stale_pdf_path):
            os.remove(stale_pdf_path)
        for fname in os.listdir(PDF_DIR):
            if fname.startswith(f"{stale_hash}_page_"):
                os.remove(os.path.join(PDF_DIR, fname))

        logger.info("「%s」の旧バージョン（hash=%s）を削除しました。", safe_name, stale_hash)


# ------------------------------------------------------------------
# PDF登録
# ------------------------------------------------------------------
def register_pdf(
    uploaded_file: Any,
    model: SentenceTransformer,
    child_collection: chromadb.Collection,
    parent_collection: chromadb.Collection,
) -> None:
    """PDFを取り込み、画像化・テキスト抽出・チャンク化・ベクトル化・DB登録を行う。"""
    os.makedirs(PDF_DIR, exist_ok=True)
    safe_name = sanitize_filename(uploaded_file.name)
    file_bytes = uploaded_file.getvalue()
    content_hash = pdf_content_hash(file_bytes)
    pdf_path = pdf_path_for_hash(content_hash)

    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

    st.session_state["active_pdf_hash"] = content_hash

    try:
        existing = child_collection.get(where={"source_hash": content_hash}, limit=1)
    except Exception:
        existing = None
    if existing and existing.get("ids"):
        st.info(f"「{uploaded_file.name}」は内容が変わっていないため、登録済みのデータを再利用します（API呼び出しなし）。")
        return

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        st.error(f"PDFを開けませんでした。ファイルが破損している可能性があります: {e}")
        return

    if len(doc) == 0:
        doc.close()
        st.warning("ページ数が0のPDFのため、処理をスキップしました。")
        return

    page_text_map: Dict[int, str] = {}
    multimodal_targets: List[Tuple[int, Image.Image, str]] = []
    progress_bar = st.progress(0.0, text="ページを解析しています...")

    try:
        total_pages = len(doc)
        for page_num in range(total_pages):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)

            images = page.get_images()
            drawings = page.get_drawings()
            extracted_text = page.get_text("text")

            # 画像や図形要素があっても、通常抽出だけで十分な文字量が取れているページは
            # （ロゴやアイコン程度の埋め込み画像である可能性が高いため）Gemini Visionには回さない。
            # これによりVision呼び出し＝APIコストを必要なページだけに絞る。
            needs_multimodal = (
                (len(images) > 0 or len(drawings) > 10)
                and len(extracted_text.strip()) < TEXT_SUFFICIENCY_THRESHOLD_CHARS
            )

            if needs_multimodal:
                # PNGへの保存とPIL Imageへの変換は、実際にマルチモーダル解析へ回す
                # ページだけに限定する。以前は全ページ無条件でディスクに保存していたが、
                # 保存した画像を読んでいたのはこの分岐だけだったため、テキストのみの
                # ページでも毎回ディスクI/Oが発生し、ファイルが使われないまま残り続けていた。
                # pix.samplesから直接PIL Imageへ変換すれば、保存→再読込という
                # 無駄な往復も避けられる（ピクセル単位で結果は同一であることを確認済み）。
                img_path = image_path_for_hash(content_hash, page_num)
                pix.save(img_path)
                pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                multimodal_targets.append((page_num, pil_image, extracted_text))
            else:
                page_text_map[page_num] = extracted_text

            progress_bar.progress((page_num + 1) / total_pages, text=f"ページを解析しています... ({page_num + 1}/{total_pages})")

        if multimodal_targets:
            progress_bar.progress(1.0, text=f"図表を含む{len(multimodal_targets)}ページをAIで解析しています...")
            st.toast(f"図表を含む{len(multimodal_targets)}ページをマルチモーダルで解析中...", icon="👀")
            batch_results = extract_pages_multimodal_batch([(p, img) for p, img, _ in multimodal_targets])
            for page_num, _, fallback_text in multimodal_targets:
                extracted = batch_results.get(page_num)
                page_text_map[page_num] = extracted if extracted else fallback_text

        # Parent/Child分離してDB保存準備
        parent_ids, parent_docs, parent_metas = [], [], []
        child_ids, child_docs, child_metas = [], [], []

        for page_num in sorted(page_text_map.keys()):
            page_text = page_text_map[page_num]
            if not page_text or not page_text.strip():
                continue

            for p_idx, block in enumerate(create_hierarchical_chunks(page_text)):
                parent_id = f"{content_hash}_p{page_num}_parent{p_idx}"
                parent_ids.append(parent_id)
                parent_docs.append(block["parent"])
                parent_metas.append({
                    "source": safe_name,
                    "source_hash": content_hash,
                    "page_num": page_num,
                })

                for c_idx, child_text in enumerate(block["children"]):
                    child_ids.append(f"{parent_id}_child{c_idx}")
                    child_docs.append(child_text)
                    child_metas.append({
                        "parent_id": parent_id,
                        "source": safe_name,
                        "source_hash": content_hash,
                        "page_num": page_num,
                        # 注意: 実際にファイルが存在するのはマルチモーダル解析を行ったページのみ
                        # （通常テキスト抽出だけで済んだページはPNGを保存していない）。
                        "image_path": image_path_for_hash(content_hash, page_num),
                    })
    finally:
        doc.close()
        progress_bar.empty()

    if not child_docs:
        st.warning("このPDFからテキストを抽出できませんでした（画像のみのPDFなどの可能性があります）。")
        return

    with st.spinner("ベクトル化してデータベースに保存しています..."):
        child_embeddings = embed_passages(model, child_docs)
        parent_embeddings = embed_passages(model, parent_docs)
        child_collection.upsert(embeddings=child_embeddings, documents=child_docs, metadatas=child_metas, ids=child_ids)
        parent_collection.upsert(embeddings=parent_embeddings, documents=parent_docs, metadatas=parent_metas, ids=parent_ids)

    # 同じファイル名の古いバージョンがあれば掃除する
    _cleanup_stale_versions(safe_name, content_hash, child_collection, parent_collection)


def get_registered_documents(parent_collection: chromadb.Collection) -> List[Dict[str, Any]]:
    """登録済みの文書一覧を、内容ハッシュごとに重複なく取得する（UI用）"""
    try:
        results = parent_collection.get(include=["metadatas"])
    except Exception:
        return []

    docs: Dict[str, Dict[str, Any]] = {}
    for meta in results.get("metadatas") or []:
        if not meta:
            continue
        source_hash = meta.get("source_hash")
        if not source_hash:
            continue
        entry = docs.setdefault(source_hash, {"source_hash": source_hash, "name": meta.get("source", "不明"), "max_page": 0})
        entry["max_page"] = max(entry["max_page"], meta.get("page_num", 0))

    return sorted(docs.values(), key=lambda d: d["name"])


# ------------------------------------------------------------------
# 検索（リランキング・スコープ絞り込み対応）
# ------------------------------------------------------------------
def search_pdf_context(
    query_text: str,
    query_vector: List[float],
    child_collection: chromadb.Collection,
    parent_collection: chromadb.Collection,
    threshold: float,
    source_hash_filter: Optional[List[str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """PDFを検索し、CrossEncoderで再ランキングしたうえで上位の親チャンクをまとめて返す。

    source_hash_filter: 指定すると、そのハッシュ群のPDFだけを検索対象にする（Noneなら全体検索）。
    """
    where_clause = {"source_hash": {"$in": source_hash_filter}} if source_hash_filter else None

    try:
        results = child_collection.query(
            query_embeddings=[query_vector],
            n_results=PDF_SEARCH_OVERFETCH,
            where=where_clause,
        )
    except Exception as e:
        st.warning(f"PDFデータベースの検索に失敗しました: {e}")
        return "なし", []

    distances = results.get("distances") or []
    if not distances or not distances[0]:
        return "なし", []

    # 1. 距離の閾値でまず粗く絞る
    candidates = []
    for i in range(len(distances[0])):
        if distances[0][i] >= threshold:
            continue
        meta = results["metadatas"][0][i] or {}
        candidates.append({
            "child_text": results["documents"][0][i],
            "distance": distances[0][i],
            "parent_id": meta.get("parent_id"),
            "page_num": meta.get("page_num", 0),
            "source": meta.get("source", "不明"),
            "source_hash": meta.get("source_hash", ""),
        })

    if not candidates:
        return "なし", []

    # 2. CrossEncoderでクエリとの関連度を再評価し、並べ替える
    try:
        reranker = get_reranker()
        pairs = [[query_text, c["child_text"]] for c in candidates]
        scores = reranker.predict(pairs)
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    except Exception as e:
        logger.warning("リランキングに失敗したため、ベクトル距離順のまま使用します: %s", e)
        candidates.sort(key=lambda c: c["distance"])

    # 3. 親チャンク単位で重複排除し、上位N件を採用
    unique_parent_ids: List[str] = []
    hit_pages: List[Dict[str, Any]] = []
    for c in candidates:
        parent_id = c["parent_id"]
        if not parent_id or parent_id in unique_parent_ids:
            continue
        unique_parent_ids.append(parent_id)
        hit_pages.append(c)
        if len(unique_parent_ids) >= PDF_SEARCH_MAX_PARENTS:
            break

    if not unique_parent_ids:
        return "なし", []

    # 4. 選ばれた親チャンクの本文をまとめて取得
    try:
        parent_result = parent_collection.get(ids=unique_parent_ids, include=["documents"])
        parent_text_map = dict(zip(parent_result["ids"], parent_result["documents"]))
    except Exception as e:
        st.warning(f"親チャンクの取得に失敗しました: {e}")
        parent_text_map = {}

    pdf_context = ""
    for hit in hit_pages:
        parent_text = parent_text_map.get(hit["parent_id"], hit["child_text"])
        pdf_context += f"【資料「{hit['source']}」 P.{hit['page_num'] + 1}】\n{parent_text}\n\n"

    return pdf_context, hit_pages


# ------------------------------------------------------------------
# ハイライト画像生成
# ------------------------------------------------------------------
def generate_highlighted_images(hit_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ヒットしたチャンクに対応する箇所をハイライトしたページ画像を生成する（APIは使わない）。

    ヒットは複数の異なる文書にまたがる可能性があるため、source_hashごとに
    正しいPDFファイルを開いて処理する。
    """
    generated_images = []
    if not hit_pages:
        return generated_images

    hits_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for page_info in hit_pages:
        source_hash = page_info.get("source_hash") or ""
        hits_by_source.setdefault(source_hash, []).append(page_info)

    for source_hash, hits in hits_by_source.items():
        pdf_path = pdf_path_for_hash(source_hash) if source_hash else None
        if not pdf_path or not os.path.exists(pdf_path):
            continue

        try:
            doc_view = fitz.open(pdf_path)
        except Exception:
            continue

        try:
            seen_pages = set()
            for page_info in hits:
                p_num = page_info["page_num"]
                if p_num >= len(doc_view) or p_num in seen_pages:
                    continue
                seen_pages.add(p_num)

                page = doc_view[p_num]
                raw_text = page_info["child_text"]
                lines = [line.strip() for line in raw_text.split("\n") if len(line.strip()) >= 3]

                for line in lines:
                    if not re.search(r"[一-龠ぁ-ゔァ-ヴーa-zA-Z]", line):
                        continue
                    for inst in page.search_for(line):
                        annot = page.add_highlight_annot(inst)
                        annot.update()

                pix = page.get_pixmap(dpi=120)
                generated_images.append({
                    "page_num": p_num,
                    "img_bytes": pix.tobytes("png"),
                    "source": page_info.get("source", "不明"),
                })
        finally:
            doc_view.close()

    return generated_images