"""
埋め込み生成の共通ヘルパー。

multilingual-e5系モデルは、検索対象のテキスト（保存するドキュメント側）には
"passage: "、検索クエリ側には "query: " というプレフィックスを付けることが
公式に推奨されている（付けなくても動くが、検索精度が本来の性能より落ちる）。
これを毎回書き忘れないよう、埋め込み生成はすべてこのモジュール経由にする。
"""

from typing import List

from config import EMBEDDING_PASSAGE_PREFIX, EMBEDDING_QUERY_PREFIX


def embed_passages(model, texts: List[str]) -> List[List[float]]:
    """保存対象のテキスト（PDFチャンク・過去の会話など）をバッチで埋め込む"""
    prefixed = [f"{EMBEDDING_PASSAGE_PREFIX}{t}" for t in texts]
    return model.encode(prefixed, show_progress_bar=False).tolist()


def embed_passage(model, text: str) -> List[float]:
    """保存対象のテキストを1件だけ埋め込む"""
    return model.encode(f"{EMBEDDING_PASSAGE_PREFIX}{text}").tolist()


def embed_query(model, text: str) -> List[float]:
    """検索クエリ（ユーザーの質問など）を埋め込む"""
    return model.encode(f"{EMBEDDING_QUERY_PREFIX}{text}").tolist()
