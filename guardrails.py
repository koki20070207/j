"""
簡易ガードレール。

本格的なプロンプトインジェクション検知には専用の分類モデル／APIを使うのが望ましいが、
それを毎ターン呼ぶとAPIコストが増える。ここでは「追加のAPI呼び出しを増やさない」ことを
優先し、正規表現による軽量なヒューリスティックだけを実装している。

注意: これは簡易対策であり、巧妙な言い回しはすり抜ける可能性がある。
本番運用するなら専用のガードレールサービス／分類モデルの導入を推奨する。
"""

import re
import unicodedata
from typing import Optional

# 「これまでの指示を無視して」系の典型的な言い回し（日本語・英語）
_INJECTION_PATTERNS = [
    r"これまでの指示(を|は).{0,10}(無視|忘れ)",
    r"システムプロンプト(を|は).{0,10}(見せ|教え|表示|出力)",
    r"あなたの(指示|設定|ルール)(を|は).{0,10}(無視|忘れ|開示)",
    r"ignore (all )?(previous|above) instructions",
    r"reveal (your |the )?system prompt",
    r"disregard (all )?(previous|prior) (instructions|rules)",
    r"you are now (in )?(dan|developer) mode",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def check_input_safety(user_text: str) -> Optional[str]:
    """ユーザー入力に典型的なプロンプトインジェクションの兆候がないか確認する。

    戻り値: 問題があれば警告メッセージ（str）、なければ None。
    ブロックするかどうかは呼び出し側の判断に委ねる（ここでは検知のみ）。
    """
    if not user_text:
        return None

    # 全角英数字やゼロ幅文字を混ぜるだけの単純な正規表現回避を防ぐため、
    # 判定前にNFKC正規化し、ゼロ幅系の文字を除去する。
    # （あくまで簡易対策。巧妙な言い換えまでは防げない点は変わらない）
    normalized = unicodedata.normalize("NFKC", user_text)
    normalized = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", normalized)

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(normalized):
            return (
                "入力内容に、AIへの指示を上書きしようとする表現が含まれている可能性があります。"
                "意図しない動作を避けるため、この質問は送信しませんでした。"
                "言い回しを変えて再度お試しください。"
            )

    return None