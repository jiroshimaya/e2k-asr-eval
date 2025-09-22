"""BEP辞書ファイルの解析処理。"""

from pathlib import Path

import structlog

from e2k_asr_eval.schemas import WordEntry


logger = structlog.get_logger(__name__)

def load_bep_dictionary(dict_path: Path, encoding: str = "utf-8") -> list[WordEntry]:
    """BEP辞書ファイルを読み込み、WordEntryのリストを返す。

    Args:
        dict_path: BEP辞書ファイルのパス
        encoding: ファイルのエンコーディング（デフォルトはUTF-8）

    Returns:
        list[WordEntry]: 解析された単語エントリのリスト

    Raises:
        FileNotFoundError: 辞書ファイルが見つからない場合
        UnicodeDecodeError: ファイルエンコーディングエラー
    """
    with dict_path.open("r", encoding=encoding) as f:
        lines = f.readlines()

    word_entries = []
    for line in lines:
        # 各行を解析してWordEntryを作成
        line = line.split("#", 1)[0]  # コメントを除去
        line = line.strip()
        if not line:
            continue
        columns = line.split()
        if len(columns) != 2:
            logger.warning(f"無効な行をスキップ: {line}")
            continue
        word, kana_gold = columns
        entry = WordEntry(word=word, kana_gold=kana_gold)
        word_entries.append(entry)

    return word_entries
