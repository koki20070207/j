"""許可された低リスク操作だけを実行する共通ディスパッチャ。"""

from typing import Any, Dict

from pc_tools import (
    PCOperationError,
    get_system_info,
    launch_app,
    list_directory,
    open_url,
    search_files,
    show_notification,
)


class OperationDispatchError(ValueError):
    """操作名または操作引数が許可範囲外の場合に発生する。"""


SUPPORTED_OPERATIONS = frozenset(
    {"system_info", "list_files", "search_files", "launch_app", "open_url", "notification"}
)


def dispatch_operation(operation: str, request: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """操作名と引数を許可リストに照合して実行する。"""
    if operation not in SUPPORTED_OPERATIONS:
        raise OperationDispatchError(f"許可されていない操作です: {operation}")
    payload = request or {}
    try:
        if operation == "system_info":
            _require_no_extra_keys(payload)
            return get_system_info()
        if operation == "list_files":
            _require_keys(payload, {"root"})
            return {"entries": list_directory(payload.get("root"))}
        if operation == "search_files":
            _require_keys(payload, {"pattern", "root"})
            return {"files": search_files(payload["pattern"], payload.get("root"))}
        if operation == "launch_app":
            _require_keys(payload, {"app_name"})
            return launch_app(payload["app_name"])
        if operation == "open_url":
            _require_keys(payload, {"url"})
            return open_url(payload["url"])
        if operation == "notification":
            _require_keys(payload, {"title", "message"})
            return show_notification(payload["title"], payload["message"])
    except (KeyError, TypeError, PCOperationError) as error:
        raise OperationDispatchError(str(error)) from error

def _require_keys(payload: Dict[str, Any], allowed: set[str]) -> None:
    if set(payload) - allowed:
        raise OperationDispatchError("操作引数に許可されていない項目があります。")


def _require_no_extra_keys(payload: Dict[str, Any]) -> None:
    if payload:
        raise OperationDispatchError("この操作は引数を受け取りません。")
