"""
Logging Configuration

Sets up the application's logging system, providing console and file output
with custom filtering for web server noise.
"""

import logging
import re

from backend.utils.paths import project_path


class _SuppressResultPollFilter(logging.Filter):
    """Hide only routine /result polling lines from web server request logs."""

    _pattern = re.compile(r'"?(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) /result(?:\?|\s)')

    def filter(self, record):
        if record.name != "uvicorn.access":
            return True
        return not bool(self._pattern.search(record.getMessage()))


# UVICORN_LOG_CONFIG

UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {},
    "handlers": {},
    "loggers": {
        "uvicorn": {"level": "INFO", "propagate": True},
        "uvicorn.error": {"level": "INFO", "propagate": True},
        "uvicorn.access": {"level": "INFO", "propagate": True},
    },
}


def setup_logging():
    """Configure application logging for console and file output."""
    root_logger = logging.getLogger()

    # Avoid duplicate handlers when setup is called more than once.
    if getattr(root_logger, "_vision_logging_configured", False):
        return

    logs_dir = project_path("data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_SuppressResultPollFilter())

    file_handler = logging.FileHandler(logs_dir / "vision.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


    root_logger._vision_logging_configured = True


def get_logger(name):
    """
    Returns a logger instance with the specified name.
    """
    return logging.getLogger(name)