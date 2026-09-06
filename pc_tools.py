"""安全なPC操作の基盤。

最初の段階では読み取り専用のファイル一覧・検索だけを提供する。
任意のコマンド実行や削除はこのモジュールに追加しない。
"""

from pathlib import Path
from typing import List, Optional

from logging_setup import get_logger

logger = get_logger(__name__)

MAX_RESULTS = 100
MAX_DEPTH = 6


class PCOperationError(ValueError):
    """入力されたPC操作が許可範囲外の場合に発生する。"""


def _allowed_root(root: Optional[str] = None) -> Path:
    """操作対象をユーザープロファイル配下に限定する。"""
    base = Path.home().resolve()
    candidate = (Path(root).expanduser() if root else base).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise PCOperationError("ユーザープロファイル配下のパスだけ操作できます。") from error
    if not candidate.is_dir():
        raise PCOperationError("対象ディレクトリが存在しません。")
    return candidate


def list_directory(root: Optional[str] = None) -> List[dict]:
    """指定ディレクトリ直下の項目を一覧する。"""
    directory = _allowed_root(root)
    entries = []
    for path in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        entries.append({"name": path.name, "is_dir": path.is_dir()})
    return entries[:MAX_RESULTS]


def search_files(pattern: str, root: Optional[str] = None) -> List[str]:
    """許可されたディレクトリ配下からファイル名を検索する。"""
    if not pattern or len(pattern) > 100:
        raise PCOperationError("検索パターンは1〜100文字で指定してください。")

    directory = _allowed_root(root)
    results = []
    for path in directory.rglob(pattern):
        try:
            relative = path.relative_to(directory)
        except ValueError:
            continue
        if len(relative.parts) > MAX_DEPTH:
            continue
        if path.is_file():
            results.append(str(relative))
        if len(results) >= MAX_RESULTS:
            break
    return results
