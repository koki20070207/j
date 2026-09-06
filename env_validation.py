"""
環境変数のバリデーション。

起動時に必須の環境変数が設定されているか確認する。設定漏れは早期エラーで
プロセス起動を止めることで、実行中の後発エラーを防ぐ。
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from logging_setup import get_logger

logger = get_logger(__name__)

# 必須環境変数のリスト
REQUIRED_ENV_VARS = ["GEMINI_API_KEY"]

# 推奨環境変数のリスト（なくても動くが、あるとより良い）
RECOMMENDED_ENV_VARS: List[str] = []
REQUIRED_PROMPT_FILES = ("answer_system.txt", "multimodal_extract.txt")


def find_missing_prompt_files(prompt_dir: Optional[Path] = None) -> List[str]:
    """必須プロンプトの欠落・空ファイルを返す。"""
    directory = prompt_dir or Path(__file__).resolve().parent / "prompts"
    missing = []
    for filename in REQUIRED_PROMPT_FILES:
        path = directory / filename
        try:
            is_invalid = not path.is_file() or not path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            is_invalid = True
        if is_invalid:
            missing.append(str(path))
    return missing


def validate_environment() -> None:
    """必須環境変数をチェックする。不足しているものがあればエラー終了する。
    
    Raises:
        SystemExit: 必須環境変数が不足している場合。
    """
    load_dotenv()
    
    missing_vars = []
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        error_msg = (
            f"必須環境変数が不足しています: {', '.join(missing_vars)}\n"
            f"これらを .env ファイルに設定するか、環境変数として定義してください。"
        )
        logger.error(error_msg)
        print(error_msg, file=sys.stderr)
        sys.exit(1)

    missing_prompts = find_missing_prompt_files()
    if missing_prompts:
        error_msg = (
            "必須プロンプトが見つからないか空です:\n"
            + "\n".join(f"- {path}" for path in missing_prompts)
        )
        logger.error(error_msg)
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    
    logger.info("環境変数チェック完了: すべての必須変数が設定されています。")
    
    # 推奨変数の確認（警告のみ）
    missing_recommended = []
    for var in RECOMMENDED_ENV_VARS:
        if not os.getenv(var):
            missing_recommended.append(var)
    
    if missing_recommended:
        logger.warning(
            "推奨環境変数が設定されていません（必須ではありません）: %s",
            ", ".join(missing_recommended),
        )
