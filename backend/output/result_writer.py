"""
Result Writer

Saves vision processing results to the SQL database (see backend.core.db)
and, as a lightweight local backup, to a daily JSONL file. The two writes
are independent — a failure in one is logged but does not prevent the
other, so a temporary database hiccup (relevant once this points at a
networked MSSQL server) can't silently lose an inspection result.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from backend.core import db
from backend.core.config_loader import cfg

log = logging.getLogger(__name__)

_write_lock = threading.Lock()

# Pre-resolve output directory at module load time
_result_dir = cfg.get("output", {}).get("result_dir", "data/results")
_output_dir = Path(_result_dir)
if not _output_dir.is_absolute():
    _output_dir = Path(__file__).resolve().parents[2] / _output_dir


def save_result(result: dict):
    """
    Saves a vision processing result to the database, and as a backup to
    a daily JSONL file.
    """
    now = datetime.now()
    result["timestamp"] = now.isoformat()

    try:
        db.insert_result(result)
    except Exception:
        log.exception("Failed to write result to the database; JSONL backup will still be attempted")

    with _write_lock:
        try:
            _output_dir.mkdir(parents=True, exist_ok=True)
            daily_file = _output_dir / f"{now.date().isoformat()}.jsonl"
            with daily_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False))
                handle.write("\n")
        except Exception:
            log.exception("Failed to write result to JSONL backup")