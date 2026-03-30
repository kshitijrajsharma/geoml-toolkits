import logging

from rich.logging import RichHandler
from rich.progress import track as _rich_track


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(RichHandler(show_path=False, markup=True))
        logger.propagate = False
    return logger


def track(sequence, description: str = "Processing...", **kwargs):
    return _rich_track(sequence, description=description, **kwargs)
