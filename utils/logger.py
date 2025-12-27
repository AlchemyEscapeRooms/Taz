"""Logging configuration for AI Trading Bot."""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_dir: str = "logs",
    level: str = "INFO",
    retention_days: int = 30
) -> None:
    """Configure the logger with daily log files and console output.

    Each day gets its own log file: trading_bot_2025-11-26.log
    """

    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Add console handler with colors
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        colorize=True
    )

    # Add file handler with daily rotation
    # File name includes date: trading_bot_2025-11-26.log
    log_file = str(log_path / "trading_bot_{time:YYYY-MM-DD}.log")
    logger.add(
        log_file,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="00:00",  # Rotate at midnight
        retention=f"{retention_days} days",
        compression="zip"
    )

    logger.info("Logger initialized successfully")


def get_logger(name: Optional[str] = None):
    """Get a logger instance."""
    if name:
        return logger.bind(name=name)
    return logger


# Initialize logger on import
setup_logger()
