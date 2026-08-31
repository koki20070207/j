"""
Jarvis Core を「Windowsログイン時に自動起動」させるための登録スクリプト。

【正式なWindowsサービスではなくタスクスケジューラを使う理由】
README_CHANGES.md / 企画書にも記載の通り、正式なWindowsサービス（SCM経由）は
Session 0分離によりデスクトップ操作ができない。そのため、タスクスケジューラの
「ログオン時」トリガーで、ユーザー本人のセッション内にプロセスを常駐させる。

【使い方】（Windows環境で実行してください。このスクリプト自体はWindows専用です）
    python register_startup.py           # 登録（既存タスクがあれば上書き）
    python register_startup.py --status  # 登録状況の確認
    python register_startup.py --remove  # 登録解除
    python register_startup.py --run-now # 今すぐ起動して動作確認（開発時用）

【重要：このスクリプトはサンドボックス上では動作検証できていません】
schtasksコマンドの構文は資料に基づいて作成していますが、実際にWindows環境で
`python register_startup.py --status` を実行し、正しく登録されたことを
確認してから運用に入ってください。
"""

import argparse
import os
import subprocess
import sys

from config import CORE_TASK_NAME
from logging_setup import get_logger

logger = get_logger(__name__)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_SCRIPT_PATH = os.path.join(THIS_DIR, "jarvis_core.py")


def _is_admin() -> bool:
    """管理者権限で実行されているかを確認する（Windows専用）。

    schtasks /create は、/rl limited（実行時は標準権限）を指定していても、
    登録という操作自体をWindows側が管理者権限で要求してくることがある
    （実機検証で「アクセスが拒否されました」というエラーとして確認済み）。
    生のエラーメッセージだけだと原因が分かりにくいため、事前にチェックして
    分かりやすい案内を出す。
    """
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _find_pythonw() -> str:
    """コンソールウィンドウを出さない pythonw.exe のパスを探す。

    見つからない場合は sys.executable（python.exe）にフォールバックする
    （その場合、起動時に一瞬コンソールが出る可能性がある）。
    """
    candidate = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if os.path.exists(candidate):
        return candidate
    logger.warning("pythonw.exe が見つかりませんでした。python.exe を使用します（コンソールが表示される場合があります）。")
    return sys.executable


def _build_command() -> str:
    """schtasksに渡す実行コマンドを組み立てる。

    【気づいたバグと修正】
    以前は "pythonw.exeのパス" "jarvis_core.pyのパス" という、ダブルクオートを
    2組含む1つの文字列をそのまま /tr に渡していた。これをPythonのsubprocess
    （list形式）経由でschtasksに渡すと、Windows側の引数エスケープが二重にかかり、
    schtasksが正しく解釈できず「アクセスが拒否されました」という誤解を招く
    エラーになることを実機検証で確認した（実際には権限の問題ではなかった）。

    対策として、pythonw.exeの呼び出しをバッチファイル（1つの.batファイル）に
    閉じ込め、schtasksの/trにはそのバッチファイル1つのパスだけを渡すように
    変更した。/trの引数がクオート1組だけになるため、この二重エスケープ問題が
    起きなくなる。
    """
    pythonw = _find_pythonw()
    launcher_path = os.path.join(THIS_DIR, "_jarvis_core_launcher.bat")
    with open(launcher_path, "w", encoding="shift_jis") as f:
        f.write("@echo off\r\n")
        f.write(f'"{pythonw}" "{CORE_SCRIPT_PATH}"\r\n')
    return launcher_path


def register() -> bool:
    if os.name != "nt":
        logger.error("このスクリプトはWindows専用です（現在のOS: %s）。", os.name)
        return False

    if not _is_admin():
        print("❌ 管理者権限がありません。")
        print("   schtasksでのタスク登録は、Windows環境によって管理者権限を要求されます。")
        print("   PowerShellを「管理者として実行」で開き直し、もう一度実行してください。")
        print("   （登録されるタスク自体は、これまで通り標準権限で動作します）")
        return False

    command = _build_command()
    args = [
        "schtasks", "/create",
        "/tn", CORE_TASK_NAME,
        "/tr", command,
        "/sc", "onlogon",
        "/rl", "limited",   # 標準権限で実行（デスクトップ操作にはユーザーセッション内での実行が必要）
        "/f",               # 既存タスクがあれば上書き
    ]

    logger.info("タスクスケジューラに登録します: %s", " ".join(args))
    result = subprocess.run(args, capture_output=True, text=True)

    if result.returncode == 0:
        logger.info("登録に成功しました。次回ログイン時からJarvis Coreが自動起動します。")
        print(f"✅ 登録完了: タスク名「{CORE_TASK_NAME}」")
        print(f"   実行コマンド: {command}")
        print("   動作確認は次のいずれかで行ってください:")
        print(f"   ・python register_startup.py --run-now  （今すぐ試験起動）")
        print(f"   ・python register_startup.py --status   （登録状況の確認）")
        return True
    else:
        logger.error("登録に失敗しました: %s", result.stderr.strip())
        print(f"❌ 登録に失敗しました: {result.stderr.strip()}")
        return False


def status() -> None:
    result = subprocess.run(
        ["schtasks", "/query", "/tn", CORE_TASK_NAME, "/fo", "LIST", "/v"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"タスク「{CORE_TASK_NAME}」は登録されていないようです。")
        print(result.stderr.strip())


def remove() -> bool:
    result = subprocess.run(
        ["schtasks", "/delete", "/tn", CORE_TASK_NAME, "/f"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"✅ タスク「{CORE_TASK_NAME}」の登録を解除しました。")
        return True
    print(f"❌ 解除に失敗しました: {result.stderr.strip()}")
    return False


def run_now() -> None:
    """タスクスケジューラ経由ではなく、動作確認のためにその場で起動する（フォアグラウンド）。"""
    print("Jarvis Core をフォアグラウンドで起動します（Ctrl+Cで停止）。")
    subprocess.run([sys.executable, CORE_SCRIPT_PATH])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jarvis Coreの自動起動をWindowsタスクスケジューラに登録する")
    parser.add_argument("--status", action="store_true", help="登録状況を確認する")
    parser.add_argument("--remove", action="store_true", help="登録を解除する")
    parser.add_argument("--run-now", action="store_true", help="今すぐフォアグラウンドで試験起動する")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.remove:
        sys.exit(0 if remove() else 1)
    elif args.run_now:
        run_now()
    else:
        sys.exit(0 if register() else 1)
