import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers.models.auto.tokenization_auto import AutoTokenizer
import soundfile as sf
from pathlib import Path
from typing import List, Tuple
import logging

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device) # type: ignore
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

logger = logging.getLogger(__name__)

def generate_speech(prompt: str, output_path: str):
    """単一のプロンプトから音声を生成する
    
    Args:
        prompt: 音声化するテキスト
        output_path: 出力ファイルパス
    """
    description = ""
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(output_path, audio_arr, model.config.sampling_rate)

def generate_speech_batch(prompts_and_paths: List[Tuple[str, str]], description: str = ""):
    """複数のプロンプトから音声をバッチ生成する（ループ版）
    
    Args:
        prompts_and_paths: (prompt, output_path) のタプルのリスト
        description: 音声の説明（全プロンプト共通）
    """
    logger.info(f"Starting batch speech generation for {len(prompts_and_paths)} items")
    
    for i, (prompt, output_path) in enumerate(prompts_and_paths):
        try:
            logger.debug(f"Processing item {i+1}/{len(prompts_and_paths)}: '{prompt}' -> '{output_path}'")
            
            # 出力ディレクトリを作成
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 音声生成
            input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            audio_arr = generation.cpu().numpy().squeeze()
            sf.write(output_path, audio_arr, model.config.sampling_rate)
            
            logger.debug(f"Successfully generated: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate speech for '{prompt}': {e}", exc_info=True)
            continue
    
    logger.info("Batch speech generation completed")

def generate_speech_batch_tensor(prompts_and_paths: List[Tuple[str, str]], description: str = "", batch_size: int = 4):
    """真のテンソルバッチ処理による音声生成
    
    Args:
        prompts_and_paths: (prompt, output_path) のタプルのリスト
        description: 音声の説明（全プロンプト共通）
        batch_size: 一度に処理するバッチサイズ
    """
    logger.info(f"Starting tensor batch speech generation for {len(prompts_and_paths)} items (batch_size={batch_size})")
    
    # バッチごとに処理
    for batch_start in range(0, len(prompts_and_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts_and_paths))
        batch_items = prompts_and_paths[batch_start:batch_end]
        
        try:
            logger.debug(f"Processing batch {batch_start//batch_size + 1}: items {batch_start+1}-{batch_end}")
            
            # バッチ内のプロンプトを抽出
            batch_prompts = [item[0] for item in batch_items]
            batch_paths = [item[1] for item in batch_items]
            
            # 出力ディレクトリを作成
            for output_path in batch_paths:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # バッチ用のdescriptionとpromptをトークナイズ
            # 同じdescriptionを全バッチで使用
            descriptions = [description] * len(batch_prompts)
            
            # バッチトークナイズ（パディング付き）
            description_tokens = tokenizer(
                descriptions, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).input_ids.to(device)
            
            prompt_tokens = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).input_ids.to(device)
            
            logger.debug(f"Description tokens shape: {description_tokens.shape}")
            logger.debug(f"Prompt tokens shape: {prompt_tokens.shape}")
            
            # バッチ生成
            with torch.no_grad():
                generations = model.generate(
                    input_ids=description_tokens, 
                    prompt_input_ids=prompt_tokens
                )
            
            # 各生成結果を個別ファイルに保存
            for i, (generation, output_path) in enumerate(zip(generations, batch_paths)):
                try:
                    audio_arr = generation.cpu().numpy().squeeze()
                    # 多次元配列の場合は最初の次元を取る
                    if audio_arr.ndim > 1:
                        audio_arr = audio_arr[0] if audio_arr.shape[0] == 1 else audio_arr
                    
                    sf.write(output_path, audio_arr, model.config.sampling_rate)
                    logger.debug(f"Successfully generated: {output_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to save audio for '{batch_prompts[i]}' to '{output_path}': {e}")
                    continue
            
            logger.debug(f"Completed batch {batch_start//batch_size + 1}")
            
        except Exception as e:
            logger.error(f"Failed to process batch {batch_start//batch_size + 1}: {e}", exc_info=True)
            # バッチ全体が失敗した場合は個別処理にフォールバック
            logger.info("Falling back to individual processing for this batch")
            for prompt, output_path in batch_items:
                try:
                    generate_speech(prompt, output_path)
                except Exception as fallback_e:
                    logger.error(f"Fallback also failed for '{prompt}': {fallback_e}")
    
    logger.info("Tensor batch speech generation completed")
    

if __name__ == "__main__":
    # 単一プロンプトの例
    prompt = "ZYUGANOV".lower()
    generate_speech(prompt, "parler_tts_out.wav")
    
    # バッチ処理の例
    batch_prompts = [
        ("hello", "output/audio/hello.wav"),
        ("world", "output/audio/world.wav"),
        ("zyuganov", "output/audio/zyuganov.wav"),
        ("batch processing", "output/audio/batch_processing.wav")
    ]
    
    # 従来のループ型バッチ処理
    print("=== Loop-based batch processing ===")
    generate_speech_batch(batch_prompts)
    
    # テンソルバッチ処理（より効率的）
    print("=== Tensor-based batch processing ===")
    generate_speech_batch_tensor(batch_prompts, batch_size=2) 
