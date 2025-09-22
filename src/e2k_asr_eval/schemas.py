"""英単語カナ変換評価に関するPydanticモデル定義。"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Status types
ProcessingStatus = Literal["pending", "processing", "completed", "failed"]
EvaluationStatus = Literal["not_evaluated", "evaluated", "error"]


class WordEntry(BaseModel):
    """辞書エントリモデル"""
    
    word: str = Field(..., description="英単語")
    kana_gold: str = Field(..., description="正解のカナ表記（ひらがな）")
    
    @field_validator("word")
    def word_not_empty(cls, v):
        if not v.strip():
            raise ValueError("word cannot be empty")
        return v.strip()
    
    @field_validator("kana_gold")
    def kana_gold_not_empty(cls, v):
        if not v.strip():
            raise ValueError("kana_gold cannot be empty")
        return v.strip()


class AudioGenerationResult(BaseModel):
    """音声生成結果モデル"""
    
    word: str = Field(..., description="英単語")
    audio_path: Path = Field(..., description="生成された音声ファイルのパス")
    status: ProcessingStatus = Field(..., description="生成ステータス")
    duration: Optional[float] = Field(None, description="音声の長さ（秒）")
    error_message: Optional[str] = Field(None, description="エラーメッセージ")


class ASRResult(BaseModel):
    """ASR認識結果モデル"""
    
    word: str = Field(..., description="英単語")
    kana_gold: str = Field(..., description="正解のカナ表記")
    kana_pred: str = Field(..., description="予測されたカナ表記")
    confidence: Optional[float] = Field(None, description="信頼度スコア")
    audio_path: Path = Field(..., description="入力音声ファイルのパス")
    processing_time: Optional[float] = Field(None, description="処理時間（秒）")
    status: ProcessingStatus = Field(..., description="認識ステータス")
    error_message: Optional[str] = Field(None, description="エラーメッセージ")


class EvaluationMetrics(BaseModel):
    """評価指標モデル"""
    
    total_count: int = Field(..., description="総テストケース数")
    exact_match_count: int = Field(..., description="完全一致数")
    exact_match_accuracy: float = Field(..., description="完全一致精度")
    character_error_rate: float = Field(..., description="文字誤り率（CER）")
    word_error_rate: float = Field(..., description="単語誤り率（WER）")
    
    # 詳細な誤り分類
    insertion_errors: int = Field(0, description="挿入エラー数")
    deletion_errors: int = Field(0, description="削除エラー数")
    substitution_errors: int = Field(0, description="置換エラー数")


class ErrorAnalysis(BaseModel):
    """誤り分析結果モデル"""
    
    word: str = Field(..., description="英単語")
    kana_gold: str = Field(..., description="正解のカナ表記")
    kana_pred: str = Field(..., description="予測されたカナ表記")
    error_type: Literal["insertion", "deletion", "substitution", "correct"] = Field(..., description="エラータイプ")
    error_detail: str = Field(..., description="具体的なエラー内容")
    character_distance: int = Field(..., description="編集距離")


class ExperimentConfig(BaseModel):
    """実験設定モデル"""
    
    # データ関連
    bep_dict_path: Path = Field(..., description="BEP辞書ファイルのパス")
    output_dir: Path = Field(..., description="出力ディレクトリ")
    max_words: Optional[int] = Field(None, description="処理する最大単語数")
    
    # TTS設定
    openai_api_key: Optional[str] = Field(None, description="OpenAI APIキー")
    tts_voice: str = Field("alloy", description="TTSで使用する音声")
    tts_speed: float = Field(1.0, description="音声速度")
    
    # ASR設定
    asr_model_name: str = Field("japanese-asr-model", description="使用するASRモデル名")
    batch_size: int = Field(1, description="バッチサイズ")
    
    # 評価設定
    enable_detailed_analysis: bool = Field(True, description="詳細分析を有効にするか")
    save_intermediate_results: bool = Field(True, description="中間結果を保存するか")

    @field_validator("output_dir")
    def create_output_dir(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v


class ExperimentResult(BaseModel):
    """実験全体の結果モデル"""
    
    config: ExperimentConfig = Field(..., description="実験設定")
    word_entries: list[WordEntry] = Field(..., description="処理対象の単語エントリ")
    audio_results: list[AudioGenerationResult] = Field(..., description="音声生成結果")
    asr_results: list[ASRResult] = Field(..., description="ASR認識結果")
    metrics: EvaluationMetrics = Field(..., description="評価指標")
    error_analysis: list[ErrorAnalysis] = Field(..., description="誤り分析結果")
    
    start_time: str = Field(..., description="実験開始時刻")
    end_time: Optional[str] = Field(None, description="実験終了時刻")
    total_duration: Optional[float] = Field(None, description="総実行時間（秒）")
