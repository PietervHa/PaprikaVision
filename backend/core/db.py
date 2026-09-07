"""
Database Layer

SQL storage for vision inspection results, via SQLAlchemy Core. The point
of using SQLAlchemy instead of raw sqlite3/pyodbc calls is portability:
this module works unchanged against SQLite today and against MSSQL later —
only `database.url` in config.yaml needs to change.

    config.yaml:
        database:
          url: "sqlite:///data/results.db"
          # later, once a server is available:
          # url: "mssql+pyodbc://user:password@server/db?driver=ODBC+Driver+17+for+SQL+Server"

Schema design: different detection backends (OCR, template matching,
classifier, YOLO, Roboflow) each log their own extra fields
(searched_word, match_score, best_reference, label, count, ...). Rather
than giving every possible field its own nullable column — which means a
schema migration every time a backend changes — this module keeps a fixed
set of columns for whatever the analytics dashboard and exports need to
filter/aggregate on, and stores everything else as a JSON blob in
`details`. Nothing is lost; nothing needs to change when a backend adds a
new field.

NOTE for the MSSQL migration later: this has only been tested against
SQLite. Two things worth double-checking once a real MSSQL server is
available:
  - `pyodbc` (or `pymssql`) needs to be installed, plus the matching ODBC
    driver on the machine running this.
  - The `DateTime` columns here map to MSSQL's plain DATETIME type
    (~3.33ms precision). The JSONL history has microsecond timestamps; if
    that precision matters, switch to `mssql.DATETIME2` for those columns.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from backend.core.config_loader import cfg

log = logging.getLogger(__name__)

# Fields that get their own column. Everything else on a result dict is
# folded into the `details` JSON blob — see module docstring. Exception:
# for "ocread" results only, "detections" is excluded from that blob and
# gets its own row per item in `result_detections` instead (see below), so
# text/confidence pairs can be queried and sorted directly in SQL instead of
# being buried in JSON. "ocr" and "object_detection" results are unaffected
# and keep their detections inside `details`, same as before this existed.
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

# One row per recognized text item belonging to an "ocread"-mode result in
# `results`. Split out into its own table instead of living inside the
# `details` JSON blob so text and confidence can each be queried, filtered,
# and sorted directly in SQL - e.g. "how accurate are reads of this
# particular text" - rather than having to parse JSON for every row first.
# Only "ocread" results get rows here; "ocr" and "object_detection" results
# still store their detections inside `details`, unchanged.
result_detections_table = Table(
    "result_detections",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("result_id", Integer, ForeignKey("results.id", ondelete="CASCADE"), nullable=False),
    Column("detection_index", Integer, nullable=False),  # original position within the result's detections list
    Column("text", Text, nullable=True),
    Column("confidence", Float, nullable=True),
    Index("ix_result_detections_result_id", "result_id"),
    Index("ix_result_detections_confidence", "confidence"),
    Index("ix_result_detections_text", "text"),
)

# Maintenance-mode accounts. There is no web-facing registration route --
# accounts are created/removed only via tools/manage_users.py, which
# needs shell access to the machine. password_hash is produced by
# backend.core.auth.hash_password(); this module never sees a plaintext
# password.
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(64), nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("created_at", DateTime, nullable=False),
)

# Every maintenance-mode login attempt, successful or not, so "who's been
# logging in" is always auditable. `success` is stored as Integer (0/1)
# rather than Boolean for the same MSSQL-portability reason as the rest of
# this module -- see the module docstring.
login_log_table = Table(
    "login_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("username", String(64), nullable=False),
    Column("success", Integer, nullable=False),
    Column("ip_address", String(64), nullable=True),
    Index("ix_login_log_timestamp", "timestamp"),
)


def _resolve_sqlite_url(url: str) -> str:
    """Resolve relative sqlite:/// paths against the repo root, like every other path in this project."""
    prefix = "sqlite:///"
    if not url.startswith(prefix) or url.startswith("sqlite:////"):
        return url  # not sqlite, or already an absolute path
    relative_part = url[len(prefix):]
    repo_root = Path(__file__).resolve().parents[2]
    absolute_path = (repo_root / relative_part).resolve()
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
    """
    Build the DB row dict for one result. Shared by insert_result() and the
    migration script's bulk insert.

    "detections" is only pulled out of the `details` blob for "ocread"
    results (see `_detection_rows`); "ocr" and "object_detection" results
    keep storing their detections inside `details` exactly as before.
    """
    exclude = set(_CORE_FIELDS)
    if result.get("mode") == "ocread":
        exclude.add("detections")
    details = {k: v for k, v in result.items() if k not in exclude}
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


def _detection_rows(result_id: int, result: dict) -> list[dict]:
    """
    Build one `result_detections` row per item in an "ocread" result's
    detections list (text + confidence, individually queryable/sortable in
    SQL). Only "ocread" results are broken out this way - "ocr" and
    "object_detection" results are unaffected and keep their detections
    inside the `details` JSON blob, same as before this feature existed.
    """
    if result.get("mode") != "ocread":
        return []
    detections = result.get("detections")
    if not isinstance(detections, list):
        return []
    rows = []
    for idx, d in enumerate(detections):
        if not isinstance(d, dict):
            continue
        rows.append({
            "result_id": result_id,
            "detection_index": idx,
            "text": d.get("text"),
            "confidence": _safe_float(d.get("confidence")),
        })
    return rows


def insert_result(result: dict) -> None:
    """
    Insert one inspection result. For "ocread" results, also insert one
    `result_detections` row per detected text (text/confidence split out so
    they can be queried and sorted directly in SQL); "ocr" and
    "object_detection" results are stored exactly as before this feature was
    added, with their detections inside the `details` blob. Same `result`
    dict shape as the JSONL writer expects.
    """
    row = _row_from_result(result)
    with engine.begin() as conn:
        inserted = conn.execute(results_table.insert().values(**row))
        result_id = inserted.inserted_primary_key[0]

        det_rows = _detection_rows(result_id, result)
        if det_rows:
            conn.execute(result_detections_table.insert(), det_rows)


def _date_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    return datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.max.time())


def query_summary(start: date, end: date) -> list[dict]:
    """
    Lightweight query for the analytics dashboard: timestamp/status/
    processing_time_ms only, no `details` blob. Used on every dashboard
    refresh, so keep it cheap.
    """
    start_dt, end_dt = _date_bounds(start, end)
    cols = results_table.c
    stmt = (
        select(cols.timestamp, cols.status, cols.processing_time_ms)
        .where(and_(cols.timestamp >= start_dt, cols.timestamp <= end_dt))
        .order_by(cols.timestamp)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(r) for r in rows]


def query_full(start: date, end: date) -> list[dict]:
    """
    Full query for exports: every core column plus `details` parsed back
    out to individual keys, so an exported row looks like the original
    JSONL record again. For "ocread" results, `detections` is reattached
    from `result_detections` instead of `details` (that's where it's
    stored for that mode - see `insert_result`); "ocr" and
    "object_detection" results already have `detections` in `details` and
    are unaffected. Either way an export row ends up with the same
    "detections": [{"text":..., "confidence":...}] shape as before.
    """
    start_dt, end_dt = _date_bounds(start, end)
    cols = results_table.c
    stmt = (
        select(results_table)
        .where(and_(cols.timestamp >= start_dt, cols.timestamp <= end_dt))
        .order_by(cols.timestamp)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()

        result_ids = [row["id"] for row in rows]
        detections_by_result_id: dict[int, list[dict]] = {}
        if result_ids:
            det_cols = result_detections_table.c
            det_stmt = (
                select(det_cols.result_id, det_cols.text, det_cols.confidence)
                .where(det_cols.result_id.in_(result_ids))
                .order_by(det_cols.result_id, det_cols.detection_index)
            )
            for det_row in conn.execute(det_stmt).mappings().all():
                detections_by_result_id.setdefault(det_row["result_id"], []).append(
                    {"text": det_row["text"], "confidence": det_row["confidence"]}
                )

    results = []
    for row in rows:
        row = dict(row)
        result_id = row["id"]
        item = {k: v for k, v in row.items() if k not in ("id", "details")}
        if isinstance(item.get("timestamp"), datetime):
            item["timestamp"] = item["timestamp"].isoformat()
        details_raw = row.get("details")
        if details_raw:
            try:
                item.update(json.loads(details_raw))
            except (TypeError, json.JSONDecodeError):
                pass
        if result_id in detections_by_result_id:
            item["detections"] = detections_by_result_id[result_id]
        results.append(item)
    return results


# --- Users -----------------------------------------------------------------

def get_user(username: str) -> Optional[dict]:
    """Look up one user by username (case-sensitive). Returns None if not found."""
    cols = users_table.c
    stmt = select(cols.id, cols.username, cols.password_hash, cols.created_at).where(
        cols.username == username
    )
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
    return dict(row) if row else None


def create_user(username: str, password_hash: str) -> None:
    """Insert a new user. Raises ValueError if the username already exists."""
    try:
        with engine.begin() as conn:
            conn.execute(
                users_table.insert().values(
                    username=username,
                    password_hash=password_hash,
                    created_at=datetime.now(),
                )
            )
    except IntegrityError as exc:
        raise ValueError(f"User '{username}' already exists") from exc


def set_password(username: str, password_hash: str) -> bool:
    """Update a user's password hash. Returns True if the user existed."""
    with engine.begin() as conn:
        result = conn.execute(
            users_table.update()
            .where(users_table.c.username == username)
            .values(password_hash=password_hash)
        )
    return result.rowcount > 0


def list_users() -> list[dict]:
    """All users, oldest first. Never includes password_hash."""
    cols = users_table.c
    stmt = select(cols.id, cols.username, cols.created_at).order_by(cols.created_at)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(r) for r in rows]


def delete_user(username: str) -> bool:
    """Delete a user by username. Returns True if a row was deleted."""
    with engine.begin() as conn:
        result = conn.execute(users_table.delete().where(users_table.c.username == username))
    return result.rowcount > 0


# --- Login log ---------------------------------------------------------------

def record_login(username: str, success: bool, ip_address: Optional[str] = None) -> None:
    """Record one login attempt, successful or not."""
    with engine.begin() as conn:
        conn.execute(
            login_log_table.insert().values(
                timestamp=datetime.now(),
                username=username,
                success=1 if success else 0,
                ip_address=ip_address,
            )
        )


def get_recent_failures_since_last_success(username: str, limit: int) -> list[datetime]:
    """
    Failed-login timestamps for `username` since their last successful login
    (or since the beginning of the log if they've never succeeded), newest
    first, capped at `limit` rows. Used to derive lockout state on demand --
    see auth.lockout_until(). Only ever called for usernames that exist, so
    an unknown username can never accumulate a strike count here.
    """
    cols = login_log_table.c
    with engine.connect() as conn:
        last_success = conn.execute(
            select(func.max(cols.timestamp)).where(
                and_(cols.username == username, cols.success == 1)
            )
        ).scalar()

        stmt = select(cols.timestamp).where(
            and_(cols.username == username, cols.success == 0)
        )
        if last_success is not None:
            stmt = stmt.where(cols.timestamp > last_success)
        stmt = stmt.order_by(cols.timestamp.desc()).limit(limit)

        rows = conn.execute(stmt).scalars().all()
    return list(rows)


def clear_login_log() -> int:
    """
    Delete every row from login_log. Returns the number of rows deleted.
    Note: since lockout state is derived from this table (see
    get_recent_failures_since_last_success), clearing it also lifts any
    lockout currently in effect, that's a deliberate, expected side effect,
    not a bug, and gives an operator a manual way to unlock an account.
    """
    with engine.begin() as conn:
        result = conn.execute(login_log_table.delete())
    return result.rowcount


def get_recent_logins(limit: int = 20) -> list[dict]:
    """Most recent login attempts (successful and failed), newest first."""
    cols = login_log_table.c
    stmt = (
        select(cols.timestamp, cols.username, cols.success, cols.ip_address)
        .order_by(cols.timestamp.desc())
        .limit(limit)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [
        {
            "timestamp": row["timestamp"],
            "username": row["username"],
            "success": bool(row["success"]),
            "ip_address": row["ip_address"],
        }
        for row in rows
    ]