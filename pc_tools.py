"""安全なPC操作の基盤。

操作は明示的な許可リストと入力検証を通し、任意コマンドは実行しない。
"""

import platform
import shutil
import subprocess
import urllib.parse
import urllib.request
import webbrowser
import shutil as _shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import ALLOWED_BROWSER_SITES, CORE_API_TIMEOUT_SEC
from logging_setup import get_logger

logger = get_logger(__name__)

MAX_RESULTS = 100
MAX_DEPTH = 6
ALLOWED_APPS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
}


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


def launch_app(app_name: str) -> Dict[str, str]:
    """許可リストにあるWindowsアプリを起動する。"""
    executable = ALLOWED_APPS.get(app_name.strip().lower())
    if executable is None:
        raise PCOperationError("許可されていないアプリです。")
    executable_path = shutil.which(executable)
    if executable_path is None:
        raise PCOperationError("指定アプリがこのPCに見つかりません。")
    try:
        process = subprocess.Popen(
            [executable_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
    except OSError as error:
        raise PCOperationError("アプリの起動に失敗しました。") from error
    logger.info("許可リストのアプリを起動しました: %s (pid=%s)", app_name, process.pid)
    return {"app": app_name.strip().lower(), "status": "started"}


def open_url(url: str) -> Dict[str, str]:
    """許可リスト内のHTTP(S) URLを既定ブラウザで開く。"""
    clean_url = url.strip()
    parsed = urllib.parse.urlparse(clean_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PCOperationError("httpまたはhttpsのURLだけ開けます。")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    request_path = parsed.path or "/"
    allowed_paths = ALLOWED_BROWSER_SITES.get(hostname)
    if allowed_paths is None or not any(request_path.startswith(prefix) for prefix in allowed_paths):
        raise PCOperationError("許可リストにないサイトまたはパスです。")
    if not webbrowser.open(clean_url, new=2):
        raise PCOperationError("既定ブラウザでURLを開けませんでした。")
    return {"url": clean_url, "status": "opened"}


def read_url(url: str) -> Dict[str, str]:
    """許可リスト内のページを読み取り専用で取得する。"""
    clean_url = url.strip()
    parsed = urllib.parse.urlparse(clean_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PCOperationError("httpまたはhttpsのURLだけ読み取れます。")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    request_path = parsed.path or "/"
    allowed_paths = ALLOWED_BROWSER_SITES.get(hostname)
    if allowed_paths is None or not any(request_path.startswith(prefix) for prefix in allowed_paths):
        raise PCOperationError("許可リストにないサイトまたはパスです。")
    request = urllib.request.Request(clean_url, headers={"User-Agent": "JarvisCore/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=CORE_API_TIMEOUT_SEC) as response:
            body = response.read(100_000)
            charset = response.headers.get_content_charset() or "utf-8"
    except (OSError, ValueError) as error:
        raise PCOperationError("許可サイトの読み取りに失敗しました。") from error
    return {"url": clean_url, "content": body.decode(charset, errors="replace"), "status": "read"}


def _send_windows_notification(title: str, message: str) -> None:
    """winotifyが利用可能なWindows環境でトースト通知を表示する。"""
    try:
        from winotify import Notification
    except ImportError as error:
        raise PCOperationError(
            "Windows通知にはwinotifyが必要です。requirements.txtの依存関係をインストールしてください。"
        ) from error
    toast = Notification(app_id="Jarvis Core", title=title, msg=message)
    toast.show()


def show_notification(title: str, message: str) -> Dict[str, str]:
    """WindowsのOS通知を表示する。"""
    clean_title = title.strip()
    clean_message = message.strip()
    if not clean_title or not clean_message or len(clean_title) > 100 or len(clean_message) > 500:
        raise PCOperationError("通知のタイトルと本文を適切に指定してください。")
    _send_windows_notification(clean_title, clean_message)
    logger.info("Windows通知を表示しました: %s", clean_title)
    return {"title": clean_title, "message": clean_message, "status": "shown"}


def get_system_info() -> Dict[str, str]:
    """外部通信なしで基本的なシステム情報を返す。"""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "timestamp": datetime.now().astimezone().isoformat(),
        "timeout_sec": str(CORE_API_TIMEOUT_SEC),
    }


def _safe_file(path: str) -> Path:
    target = Path(path).expanduser().resolve()
    base = Path.home().resolve()
    try:
        target.relative_to(base)
    except ValueError as error:
        raise PCOperationError("ユーザープロファイル配下のファイルだけ操作できます。") from error
    return target


def backup_file(path: str) -> Dict[str, str]:
    target = _safe_file(path)
    if not target.is_file():
        raise PCOperationError("バックアップ対象のファイルが存在しません。")
    backup = target.with_name(f"{target.name}.jarvis-backup")
    shutil.copy2(target, backup)
    return {"path": str(target), "backup": str(backup), "status": "backed_up"}


def move_file(source: str, destination: str) -> Dict[str, str]:
    src = _safe_file(source)
    dst = _safe_file(destination)
    if not src.is_file():
        raise PCOperationError("移動元ファイルが存在しません。")
    if dst.exists():
        raise PCOperationError("移動先は既に存在します。")
    backup = backup_file(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    _shutil.move(str(src), str(dst))
    return {"source": str(src), "destination": str(dst), "backup": backup["backup"], "status": "moved"}
