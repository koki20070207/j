"""
長期記憶（チャット記憶用ChromaDBコレクション）だけをリセットする保守スクリプト。

answer_system.txt のプレースホルダー不一致バグ（{question} 等が置換されずに
Geminiへ送られていた問題）により、これまで生成された回答の多くは、実際の質問文や
検索結果を反映していない「壊れた回答」になっている可能性があります。

これらは handle_chat_input() -> register_chat_memory() によって長期記憶
（config.CHAT_COLLECTION_NAME）に保存され続けているため、プロンプトを修正した後も、
この汚染された記憶が検索結果として今後の回答に混入し続けます。

このスクリプトが削除するのは「長期記憶コレクション」だけです。
- PDFのデータ（PDF_CHILD_COLLECTION_NAME / PDF_PARENT_COLLECTION_NAME）は削除されません。
- 画面左のチャットスレッド一覧（chat_sessions.json）も削除されません
  （そちらを消したい場合はアプリのサイドバーの🗑️から個別に削除してください）。

使い方（アプリと同じディレクトリ、同じ仮想環境で実行してください）:
    python reset_chat_memory.py         # 件数を表示し、確認してから削除
    python reset_chat_memory.py --yes   # 確認なしで即削除

念のため、実行前に ./chroma_db フォルダをまるごとコピーしておくことをおすすめします。
"""

import sys

import chromadb

from config import CHAT_COLLECTION_NAME, CHROMA_DB_PATH


def main() -> None:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        collection = client.get_collection(CHAT_COLLECTION_NAME)
    except Exception:
        print(f"コレクション '{CHAT_COLLECTION_NAME}' はまだ存在しません。何もせず終了します。")
        return

    count = collection.count()
    print(f"長期記憶コレクション '{CHAT_COLLECTION_NAME}' には現在 {count} 件のレコードがあります。")

    if count == 0:
        print("削除対象がないため終了します。")
        return

    if "--yes" not in sys.argv:
        answer = input(
            "すべて削除しますか？（プロンプト修正前に生成された、壊れた回答が\n"
            "含まれている可能性があります） [y/N]: "
        )
        if answer.strip().lower() != "y":
            print("キャンセルしました。")
            return

    client.delete_collection(CHAT_COLLECTION_NAME)
    print(f"'{CHAT_COLLECTION_NAME}' を削除しました。次回アプリ起動時に空のコレクションとして再作成されます。")


if __name__ == "__main__":
    main()