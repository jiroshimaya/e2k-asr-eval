"""ロギング設定モジュール。"""

import logging
import os
from pathlib import Path
from typing import Any, Literal

import structlog
from rich.logging import RichHandler
from rich.console import Console

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["json", "console", "plain"]


def setup_logging(
    *,
    level: LogLevel | str = "INFO",
    format: LogFormat = "console",
    log_file: str | Path | None = None,
    include_timestamp: bool = True,
    force: bool = False,
) -> None:
    """構造化ログの設定。
    
    Args:
        level: ログレベル (環境変数 LOG_LEVEL でも設定可能)
        format: 出力フォーマット (json/console/plain)
        log_file: ログファイルパス (オプション)
        include_timestamp: タイムスタンプを含めるか
        force: 既存の設定を強制的に再設定するか
    """
    # 環境変数からログレベルを取得
    env_level = os.environ.get("LOG_LEVEL", "").upper()
    if env_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        level = env_level

    # プロセッサの設定
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # タイムスタンプ追加
    if include_timestamp:
        processors.insert(3, structlog.processors.TimeStamper(fmt="iso"))

    # レンダラーの選択
    if format == "json":
        processors.append(structlog.processors.JSONRenderer())
    elif format == "console":
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )
    else:
        processors.append(structlog.processors.KeyValueRenderer())

    # structlogの設定
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 標準ライブラリのloggingも設定
    log_level = getattr(logging, level.upper())
    
    # Richハンドラーを使用してカラフルなコンソール出力
    console = Console(stderr=True)
    rich_handler = RichHandler(
        console=console,
        show_time=include_timestamp,
        show_path=False,
        rich_tracebacks=True
    )
    
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[rich_handler],
        force=force,
    )

    # ファイルハンドラーの追加
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.root.addHandler(file_handler)

    # サードパーティライブラリのログレベル調整
    if level != "DEBUG":
        for logger_name in ["urllib3", "asyncio", "filelock", "transformers", "torch"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str, **context: Any) -> structlog.BoundLogger:
    """構造化ロガーインスタンスを取得。
    
    Args:
        name: ロガー名（通常は __name__）
        **context: 初期コンテキスト
        
    Returns:
        設定済みのstructlogロガー
    """
    logger = structlog.get_logger(name)
    
    if context:
        logger = logger.bind(**context)
    
    return logger