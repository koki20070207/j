"""時刻質問がメモ保存へ到達しないことを検証する回帰テスト。"""

from types import SimpleNamespace

import llm_client


def test_time_query_variants_are_classified_as_time_only():
    queries = [
        "いまなんじ",
        "今、何時？",
        "日本時間は？",
        "日本の現在時刻を教えてください。",
        "  ｲﾏ､ﾅﾝｼﾞ?  ",
    ]

    assert all(llm_client.is_time_only_query(query) for query in queries)


def test_explicit_save_requests_are_not_classified_as_time_only():
    queries = [
        "いまなんじかメモして",
        "日本時間を覚えておいて",
        "今何時か保存して",
    ]

    assert not any(llm_client.is_time_only_query(query) for query in queries)


def test_unrelated_memo_text_is_not_classified_as_time_only():
    assert not llm_client.is_time_only_query("卵を買うとメモして")
    assert not llm_client.is_time_only_query("卵を買う")


def test_add_memo_is_rejected_before_function_execution(monkeypatch):
    called = False

    def unexpected_add_memo(**_kwargs):
        nonlocal called
        called = True
        return "保存しました"

    unexpected_add_memo.__name__ = "add_memo"
    monkeypatch.setattr(llm_client, "AVAILABLE_TOOLS", [unexpected_add_memo])
    function_call = SimpleNamespace(name="add_memo", args={"text": "卵を買う"})

    llm_client._execute_tool_calls([function_call], user_query="今、何時？")

    assert called is False
