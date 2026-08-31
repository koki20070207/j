"""
パスユーティリティ。

相対パスと絶対パスを統一的に扱うため、pathlib.Path を使用する。
このモジュールをインポートしている各ファイルではパス文字列を操作する際に
常にこのモジュールを経由することで、クロスプラットフォーム対応と
パス解決の一貫性を実現する。
"""

from pathlib import Path
from typing import Optional

# 起動スクリプトのあるディレクトリを基準に相対パスを解決
BASE_DIR = Path(__file__).parent.resolve()


def get_base_dir() -> Path:
     """アプリケーションのベースディレクトリ（このファイルの親）を返す。"""
     return BASE_DIR


def get_data_dir() -> Path:
     """データディレクトリを返す（ベースディレクトリ内）。"""
     return BASE_DIR


def get_pdf_dir() -> Path:
     """PDF画像保存ディレクトリのパスを返す。必要に応じて作成する。"""
     pdf_dir = BASE_DIR / "pdf_images"
     pdf_dir.mkdir(exist_ok=True)
     return pdf_dir


def get_chroma_db_path() -> Path:
     """ChromaDB保存ディレクトリのパスを返す。必要に応じて作成する。"""
     chroma_db_path = BASE_DIR / "chroma_db"
     chroma_db_path.mkdir(exist_ok=True)
     return chroma_db_path


def get_prompts_dir() -> Path:
     """プロンプトテンプレート保存ディレクトリのパスを返す。"""
     return BASE_DIR / "prompts"


def get_db_path() -> Path:
     """SQLiteデータベースファイルのパスを返す。"""
     return BASE_DIR / "jarvis.db"


def get_log_file_path() -> Path:
     """Streamlit UI用のログファイルパスを返す。"""
     return BASE_DIR / "app.log"


def get_core_log_file_path() -> Path:
     """Jarvis Core用のログファイルパスを返す。"""
     return BASE_DIR / "jarvis_core.log"


def get_pid_file_path() -> Path:
     """PIDファイルのパスを返す（二重起動防止用）。"""
     return BASE_DIR / "jarvis_core.pid"


def ensure_parent_dir(path: Path) -> Path:
     """ファイルパスの親ディレクトリを作成し、パスを返す。"""
     path.parent.mkdir(parents=True, exist_ok=True)
     return path


def resolve_path(relative_or_absolute: Optional[str]) -> Optional[Path]:
     """相対パスまたは絶対パス文字列を Path オブジェクトに変換する。
     
     Args:
         relative_or_absolute: パス文字列またはNone。
     
     Returns:
         Path オブジェクト、またはNone（入力がNoneの場合）。
     """
     if relative_or_absolute is None:
          return None
     path = Path(relative_or_absolute)
     if not path.is_absolute():
          path = BASE_DIR / path
     return path.resolve()
