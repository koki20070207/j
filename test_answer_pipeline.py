"""回答生成パイプラインの副作用と異常レスポンスを検証する。"""

from types import SimpleNamespace

import app
import llm_client


def test_time_question_bypasses_search_and_llm(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("time-only query should bypass retrieval and LLM")

    monkeypatch.setattr(app, "embed_query", fail_if_called)
    monkeypatch.setattr(app, "search_pdf_context", fail_if_called)
    monkeypatch.setattr(app, "search_memory_context", fail_if_called)
    monkeypatch.setattr(app, "search_web_ddg", fail_if_called)
    monkeypatch.setattr(app, "generate_answer_with_tag", fail_if_called)

    result = app.generate_rag_response(
        "今、何時？",
        [],
        object(),
        object(),
        object(),
        object(),
        0.3,
        True,
        None,
    )

    assert result[1] == []
    assert result[2] == "なし"
    assert result[8] is False
    assert result[9] == ["get_current_datetime"]


def test_model_response_text_is_parsed_without_crashing(monkeypatch):
    monkeypatch.setattr(llm_client, "get_cached_answer", lambda _key: None)
    monkeypatch.setattr(
        llm_client,
        "_call_with_retry",
        lambda *_args, **_kwargs: (SimpleNamespace(function_calls=[], text="回答\nTAG: 未分類"), None),
    )

    answer, tag, cached, tools = llm_client.generate_answer_with_tag(
        "質問",
        cache_key="test-empty-response",
        use_cache=False,
    )

    assert answer == "回答"
    assert tag == "未分類"
    assert cached is False
    assert tools == []


def test_missing_model_response_is_reported_without_crashing(monkeypatch):
    monkeypatch.setattr(llm_client, "get_cached_answer", lambda _key: None)
    monkeypatch.setattr(
        llm_client,
        "_call_with_retry",
        lambda *_args, **_kwargs: (None, None),
    )

    answer, tag, cached, tools = llm_client.generate_answer_with_tag(
        "質問",
        cache_key="test-missing-response",
        use_cache=False,
    )

    assert "有効な応答" in answer
    assert tag == "未分類"
    assert cached is False
    assert tools == []


def test_whitespace_cache_key_is_rejected():
    try:
        llm_client._normalize_cache_key("   ")
    except ValueError:
        return
    raise AssertionError("whitespace-only cache keys must be rejected")
