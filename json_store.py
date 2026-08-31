"""JSONファイルの読み書き共通処理（チャットスレッド保存・回答キャッシュ保存で共用）"""

import json
import os
from typing import Any, Dict

from logging_setup import get_logger

logger = get_logger(__name__)


def load_json(path: str) -> Dict[str, Any]:
    """JSONファイルを読み込む。存在しない・壊れている場合は空dictを返す。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("JSONファイルの読み込みに失敗しました (%s): %s", path, e)
        return {}


def save_json(path: str, data: Dict[str, Any]) -> bool:
    """JSONファイルへアトミックに書き込む（一時ファイル経由でos.replaceし、途中破損を防ぐ）"""
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return True
    except OSError as e:
        logger.warning("JSONファイルの保存に失敗しました (%s): %s", path, e)
        return False
