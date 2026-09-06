"""起動時設定と必須プロンプトの検証テスト。"""

from pathlib import Path

from env_validation import find_missing_prompt_files


def test_find_missing_prompt_files_reports_missing_and_empty_files(tmp_path: Path):
    (tmp_path / "answer_system.txt").write_text("valid prompt", encoding="utf-8")

    missing = find_missing_prompt_files(tmp_path)

    assert missing == [str(tmp_path / "multimodal_extract.txt")]


def test_find_missing_prompt_files_accepts_all_non_empty_files(tmp_path: Path):
    for filename in ("answer_system.txt", "multimodal_extract.txt"):
        (tmp_path / filename).write_text("valid prompt", encoding="utf-8")

    assert find_missing_prompt_files(tmp_path) == []
