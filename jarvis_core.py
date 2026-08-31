"""
Jarvis Core — 常駐プロセスのエントリーポイント。

【これは何か】
Streamlit UI（app.py）とは別の、独立したプロセスとして動く「本体」。
Windowsログイン時に自動起動し、UI（ブラウザ）を閉じても動き続ける前提で作る。

【なぜ別プロセスにしているか】
自律タスクの実行やPC操作（Computer Use）は、ユーザーがチャット画面を
開いている間だけ動けばよいものではない。「指示を出したら、あとは裏で
進めておいてくれる」ためには、UIの有無に関係なく動き続けるプロセスが要る。

【なぜWindowsサービス（SCM経由）ではないか】
正式なWindowsサービスはSession 0上で動作し、デスクトップ操作
（マウス・キーボード制御、画面操作）ができない。そのため、
タスクスケジューラの「ログオン時」トリガーで起動する、ユーザーの
セッション内に常駐する通常のプロセスとして実装する
（登録は register_startup.py を参照）。

【現時点でできること】
・起動・二重起動防止・グレースフルシャットダウン・ハートビートログ（Step 1）。
・SQLiteデータ層の初期化（Step 2）。
・UIから状態確認・データ取得ができるローカルAPI（Step 3、core_api.py）。
・スケジューラ／自律タスク実行ループ（Step 4）、ツール実行の安全階層判定
  ＋Discord通知（Step 5）は未実装。run_forever() のループ内に今後追加していく。
"""

import os
import signal
import sys
import threading
import time

from config import CORE_API_HOST, CORE_API_PORT, CORE_HEARTBEAT_INTERVAL_SEC, CORE_LOG_FILE, CORE_PID_FILE
from db import init_db
from env_validation import validate_environment
from logging_setup import get_logger

# 起動方法（タスクスケジューラ／手動ダブルクリック等）によらず、常に
# このファイルがあるディレクトリを基準に相対パス（memos.json等）を解決する。
# working directoryに依存すると、タスクスケジューラ経由の起動時に
# ファイルが見つからない不具合が起きやすいため。
os.chdir(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__, log_file=CORE_LOG_FILE)

_shutdown_requested = False


# ------------------------------------------------------------------
# 二重起動防止
# ------------------------------------------------------------------
def _pid_is_running(pid: int) -> bool:
    """指定PIDのプロセスが現在も生きているかを確認する（Windows/Unix両対応）。
     
    Args:
        pid: チェック対象のプロセスID。
     
    Returns:
        プロセスが実行中の場合True。確認に失敗した場合も安全側（True）を返す。
    """
    if os.name == "nt":
        # Windowsにはos.kill(pid, 0)の生存確認相当がないため、tasklistで確認する
        import subprocess
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, timeout=5,
            )
            is_running = str(pid) in result.stdout
            logger.debug("Windows tasklist確認完了: PID %d は%s", pid, "実行中" if is_running else "停止中")
            return is_running
        except subprocess.TimeoutExpired:
            logger.warning("Windows tasklist確認がタイムアウトしました（PID: %d）。安全側に倒して起動を止めます。", pid)
            return True
        except FileNotFoundError:
            logger.warning("tasklist コマンドが見つかりません。プロセス存在チェックをスキップします。")
            return False
        except Exception as e:
            logger.warning("プロセス確認中にエラーが発生しました（PID: %d）: %s。安全側に倒して起動を止めます。", pid, e)
            return True
    else:
        try:
            os.kill(pid, 0)
            logger.debug("Unix シグナルチェック完了: PID %d は実行中", pid)
            return True
        except ProcessLookupError:
            logger.debug("PID %d は停止中です", pid)
            return False
        except OSError as e:
            logger.warning("プロセス確認中にエラーが発生しました（PID: %d）: %s。安全側に倒して起動を止めます。", pid, e)
            return True


def _acquire_single_instance_lock() -> None:
    """既に起動中のjarvis_coreがいれば、ログを残してこのプロセスは終了する。

    PIDファイルが残っていても、そのPIDのプロセスが実際には存在しない
    （前回異常終了した等）場合は、古いPIDファイルとみなして上書きする。
    """
    if os.path.exists(CORE_PID_FILE):
        try:
            with open(CORE_PID_FILE, "r", encoding="utf-8") as f:
                old_pid = int(f.read().strip())
        except (ValueError, OSError):
            old_pid = None

        if old_pid is not None and _pid_is_running(old_pid):
            logger.error("Jarvis Coreは既にPID %d で起動中です。二重起動を中止します。", old_pid)
            sys.exit(1)
        else:
            logger.warning("古いPIDファイルを検出しました（PID %s は実行されていません）。上書きします。", old_pid)

    with open(CORE_PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def _release_single_instance_lock() -> None:
    try:
        if os.path.exists(CORE_PID_FILE):
            os.remove(CORE_PID_FILE)
    except OSError as e:
        logger.warning("PIDファイルの削除に失敗しました: %s", e)


# ------------------------------------------------------------------
# シャットダウン処理
# ------------------------------------------------------------------
def _handle_shutdown_signal(signum, frame) -> None:
    global _shutdown_requested
    logger.info("シャットダウン要求を受信しました（signal=%s）。次のループで安全に終了します。", signum)
    _shutdown_requested = True


def _register_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    # Windowsでコンソールを閉じた時に飛んでくるシグナル（存在する場合のみ登録）
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_shutdown_signal)


# ------------------------------------------------------------------
# ローカルAPI（UIとの連携。Step 3）
# ------------------------------------------------------------------
def _start_api_server() -> None:
    """uvicornを別スレッドで起動する。

    【スレッドで動かしている理由・既知の制限】
    run_forever()側の「シグナルで安全に止める」既存の仕組み（Step 1）を
    崩したくなかったため、メインスレッドはそのままハートビートループに残し、
    APIサーバーだけデーモンスレッドとして同居させる、最小変更の構成にした。
    そのぶん、プロセス終了時にuvicorn側は正常なシャットダウン処理
    （in-flightリクエストの処理完了待ち等）を経ずに終了する。ローカル専用の
    軽いGETエンドポイントしかない現時点では実害はないが、Step 4以降で
    「実行中の自律タスクの状態をAPI経由でやり取りする」ような重い処理が
    増えてきたら、asyncio主体の構成に組み直すことを検討する。
    """
    import uvicorn
    from core_api import app as api_app

    logger.info("Core APIを起動します: http://%s:%d", CORE_API_HOST, CORE_API_PORT)
    uvicorn.run(api_app, host=CORE_API_HOST, port=CORE_API_PORT, log_level="warning")


# ------------------------------------------------------------------
# メインループ
# ------------------------------------------------------------------
def run_forever() -> None:
    logger.info("=" * 60)
    logger.info("Jarvis Core を起動します（PID: %d）", os.getpid())
    logger.info("=" * 60)

    validate_environment()  # 必須環境変数をチェック
    _acquire_single_instance_lock()
    _register_signal_handlers()
    init_db()  # answer_cache / memos / chat_sessions用のSQLiteテーブルを用意（UI側と共有）

    api_thread = threading.Thread(target=_start_api_server, daemon=True)
    api_thread.start()

    try:
        tick = 0
        while not _shutdown_requested:
            tick += 1
            # TODO(Step 4): ここでスケジューラのtick処理・自律タスクの進行チェックを行う
            logger.info("生存確認（heartbeat #%d）。まだスケジューラは接続されていません。", tick)

            # sleepを短い間隔に分けて回すことで、シャットダウン要求から
            # 実際に停止するまでの遅延を短く保つ（最大1秒）
            for _ in range(CORE_HEARTBEAT_INTERVAL_SEC):
                if _shutdown_requested:
                    break
                time.sleep(1)
    finally:
        logger.info("Jarvis Core を終了します。")
        _release_single_instance_lock()


if __name__ == "__main__":
    run_forever()
