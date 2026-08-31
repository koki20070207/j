"""Web検索（Gemini APIは使わないので、ここではコストは発生しない）

検索バックエンドは ddgs パッケージ（旧 duckduckgo_search。2024年9月に改名され、
旧パッケージ名のままだと実行のたびにRuntimeWarningが出るようになった）を使用する。
DDGSクラス自体のインターフェース（.text(query, region=..., max_results=...)）は
互換性があるため、このモジュール内の呼び出し方は変更していない。
"""

from ddgs import DDGS

from logging_setup import get_logger

logger = get_logger(__name__)


def search_web_ddg(query: str, max_results: int = 3) -> str:
    """DuckDuckGoでWeb検索を行い、結果のテキストを返す"""
    try:
        results = DDGS().text(query, region="jp-jp", max_results=max_results)
        if not results:
            return "なし"

        web_text = ""
        for i, r in enumerate(results):
            web_text += (
                f"【Web {i + 1}】 {r.get('title', '')}\n"
                f"URL: {r.get('href', '')}\n"
                f"内容: {r.get('body', '')}\n\n"
            )
        return web_text
    except Exception as e:
        logger.warning("Web検索に失敗しました: %s", e)
        return f"なし (検索エラー: {e})"