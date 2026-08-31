"""
環境変数のバリデーション。

起動時に必須の環境変数が設定されているか確認する。設定漏れは早期エラーで
プロセス起動を止めることで、実行中の後発エラーを防ぐ。
"""

import os
import sys
from typing import List

from dotenv import load_dotenv
from logging_setup import get_logger

logger = get_logger(__name__)

# 必須環境変数のリスト
REQUIRED_ENV_VARS = ["GEMINI_API_KEY"]

# 推奨環境変数のリスト（なくても動くが、あるとより良い）
RECOMMENDED_ENV_VARS: List[str] = []


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
