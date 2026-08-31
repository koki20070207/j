"""
ロギング設定。

「エラーや処理の足跡を後から追える」ようにするための最低限の仕組み。
本格的なLLMトレースツール（LangSmith等）は課金・追加インフラが要るため、
デモ段階では標準の logging モジュール + ローテーティングファイルで十分と判断した。
"""

import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from config import LOG_FILE


def get_logger(name: str = "pdf_ai_chat", log_file: Optional[str] = None) -> logging.Logger:
    """アプリ共通のロガーを返す（名前ごとのロガーに、初回呼び出し時にのみハンドラを設定する）。

    各モジュールは get_logger(__name__) と呼ぶため、モジュールごとに異なる名前の
    ロガーが作られる（"app" "rag_engine" "llm_client" など）。
    以前はモジュール単位ではなく「アプリ全体で1回だけ」を表す単一のグローバルフラグで
    ガードしていたため、最初に呼ばれたモジュールのロガーにしかハンドラが付かず、
    残り全モジュールのログ（Gemini APIのリトライ警告やPDF検索失敗など）が
    app.log に一切書き出されないという不具合があった。
    ここではロガーごとに「そのロガー自身にすでにハンドラが付いているか」を見て
    判定することで、モジュールごとに正しくハンドラを設定する。

    log_file: 省略時はStreamlit UI用のconfig.LOG_FILEに出力する。
        Jarvis Core（常駐プロセス）側はUIとログが混ざると追いにくいため、
        get_logger(__name__, log_file=CORE_LOG_FILE) のように別ファイルを指定する。
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # ファイル出力（1MBごとにローテーション、直近3世代まで保持）
        file_handler = RotatingFileHandler(log_file or LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # コンソールにも出す（streamlit run実行時のターミナルで確認できる）
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # ルートロガーへの伝播を止める。他のライブラリがルートロガーに
        # ハンドラを追加した場合に、同じログが二重に出力されるのを防ぐため。
        logger.propagate = False

    return logger