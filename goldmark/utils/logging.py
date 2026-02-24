import logging
from logging import Logger
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: str = "INFO", log_file: Optional[Path] = None) -> Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name; typically ``__name__`` from the caller module.
    level:
        Logging level name (INFO, DEBUG, ...).
    log_file:
        Optional path where logs should also be written. The directory is
        created automatically when provided.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Logger already configured
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
