"""
Database Layer

SQL storage for vision results, via SQLAlchemy Core, so the same code works
against SQLite today and MSSQL later - only `database.url` in the config
changes.

    database:
      url: "sqlite:///data/results.db"
      # url: "mssql+pyodbc://user:password@server/db?driver=ODBC+Driver+17+for+SQL+Server"

A fixed set of columns holds what is worth filtering on; everything else on a
result dict (detections, primary, timing, ...) goes into the `details` JSON
blob, so a new field in the result never needs a schema migration.

NOTE for a later MSSQL migration: install pyodbc plus the ODBC driver, and
consider mssql.DATETIME2 if the JSONL's microsecond timestamps matter (plain
DATETIME is ~3.33 ms).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine

from backend.core.config_loader import cfg
from backend.utils.paths import project_path

log = logging.getLogger(__name__)

# Fields that get their own column. Everything else goes into `details`.
_CORE_FIELDS = {
    "timestamp",
    "status",
    "mode",
    "confidence_threshold",
    "processing_time_ms",
    "cycle_time_ms",
    "error",
}

metadata = MetaData()

results_table = Table(
    "results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("status", String(10), nullable=False),
    Column("mode", String(50), nullable=True),
    Column("confidence_threshold", Float, nullable=True),
    Column("processing_time_ms", Float, nullable=True),
    Column("cycle_time_ms", Float, nullable=True),
    Column("error", Text, nullable=True),
    Column("details", Text, nullable=True),  # JSON-encoded, dialect-portable
    Index("ix_results_timestamp", "timestamp"),
)


def _resolve_sqlite_url(url: str) -> str:
    """Resolve relative sqlite:/// paths against the repo root, like every other path in this project."""
    prefix = "sqlite:///"
    if not url.startswith(prefix) or url.startswith("sqlite:////"):
        return url  # not sqlite, or already an absolute path
    relative_part = url[len(prefix):]
    absolute_path = project_path(relative_part).resolve()
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{absolute_path}"


def _build_engine() -> Engine:
    url = cfg.get("database", {}).get("url", "sqlite:///data/results.db")
    url = _resolve_sqlite_url(url)
    log.info("Database engine: %s", url.split("://")[0] + "://...")
    return create_engine(url, future=True)


engine: Engine = _build_engine()


def init_db() -> None:
    """Create the results table if it doesn't exist yet. Safe to call on every startup."""
    metadata.create_all(engine)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            log.warning("Could not parse timestamp %r, using current time", value)
    return datetime.now()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_from_result(result: dict) -> dict:
    """Build the DB row dict for one result."""
    details = {k: v for k, v in result.items() if k not in _CORE_FIELDS}
    return {
        "timestamp": _parse_timestamp(result.get("timestamp")),
        "status": result.get("status") or "NOK",
        "mode": result.get("mode"),
        "confidence_threshold": _safe_float(result.get("confidence_threshold")),
        "processing_time_ms": _safe_float(result.get("processing_time_ms")),
        "cycle_time_ms": _safe_float(result.get("cycle_time_ms")),
        "error": result.get("error"),
        "details": json.dumps(details, ensure_ascii=False) if details else None,
    }


def insert_result(result: dict) -> None:
    """Insert one vision result. Same `result` dict shape as the JSONL writer expects."""
    row = _row_from_result(result)
    with engine.begin() as conn:
        conn.execute(results_table.insert().values(**row))