import logging
import os
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure the application logging.

    Args:
        level: Logging level (default: logging.INFO).
        log_file: Optional path to a log file. If provided, logs are written
            to both the console and the file. Parent directories are created
            automatically if they do not exist.
    """
    handlers: list = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        handlers=handlers,
    )
