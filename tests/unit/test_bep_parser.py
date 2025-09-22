from e2k_asr_eval.schemas import WordEntry
from e2k_asr_eval.core.bep_parser import load_bep_dictionary
import pytest
from pathlib import Path

@pytest.fixture
def sample_bep_file(tmp_path: Path) -> Path:
    """一時的なBEP辞書ファイルを作成し、そのパスを返す。"""
    text = (
        "# comment コメント\n"
        "# Another comment\n"
        "ENGLISH イングリッシュ\n"
        "TITEL タイトル\n"       
        )
    bep_file = tmp_path / "sample_bep.dic"
    bep_file.write_text(text, encoding="utf-8")
    return bep_file

class TestLoadBepDictionary:
  def test_正常系(self, sample_bep_file: Path) -> None:
      dictionary = load_bep_dictionary(sample_bep_file)
      assert len(dictionary) == 2
      assert dictionary[0] == WordEntry(word="ENGLISH", kana_gold="イングリッシュ")
      assert dictionary[1] == WordEntry(word="TITEL", kana_gold="タイトル")

