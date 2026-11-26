"""
Vision Orchestration

Entry point for a vision cycle. Much thinner than the VisionSoftwareMDE
version, because there is one mode instead of three and no model-loading
dance to manage.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from backend.core.config_loader import cfg
from backend.core.paprika_engine import PaprikaEngine
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_engine: Optional[PaprikaEngine] = None
_app_state: Optional[Any] = None

# Single worker so cycles run sequentially off the caller's thread, matching
# the original project's threading model.
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision_worker")


def _make_engine(app_state):
    """paprika.engine in the config: "belt" (default) - PaprikaEngine - or "rotate" - RotateEngine,
    the rotate unit with MDE's own stem model. Both return the same result contract."""
    block = cfg.get("paprika") if isinstance(cfg.get("paprika"), dict) else {}
    kind = str(block.get("engine", "belt")).strip().lower()
    if kind == "rotate":
        from backend.core.rotate_engine import RotateEngine
        return RotateEngine(cfg, app_state=app_state)
    if kind not in ("belt", "paprika"):
        logger.warning("paprika.engine %r unknown - using the belt engine", kind)
    return PaprikaEngine(cfg, app_state=app_state)


def bind_app_state(app_state) -> None:
    global _engine, _app_state
    _app_state = app_state
    _engine = _make_engine(app_state)
    logger.info("Vision bound: %s", _engine.status())


def get_engine() -> Optional[PaprikaEngine]:
    return _engine


def _empty_result(reason: str) -> dict:
    return {
        "status": "NOK",
        "mode": "paprika",
        "detections": [],
        "primary": None,
        "confidence": 0.0,
        "processing_time_ms": 0.0,
        "failure_reason": reason,
    }


def run_vision(frame, callback=None, profile: bool = False):
    """Run one cycle, synchronously or via callback.

    Signature kept identical to the original project so TCPTriggerServer can be
    reused unchanged.
    """
    if _engine is None:
        result = _empty_result("engine_not_bound")
        if callback:
            callback(result)
            return None
        return result

    if frame is None:
        result = _empty_result("no_frame")
        if callback:
            callback(result)
            return None
        return result

    if _app_state is not None and _app_state.get_vision_mode() == "idle":
        result = _empty_result("mode_idle")
        if callback:
            callback(result)
            return None
        return result

    if not callback:
        return _engine.evaluate(frame)

    # Every callback cycle is a trigger - the TCP server and the 'q' key both
    # come through here, while the overlay calls the engine directly. Marking
    # it here covers both without either caller knowing the overlay exists.
    # Marked before the submit, so the overlay stays off the model while this
    # cycle is still queued on the pool; cleared as soon as the engine is done,
    # so the hold-off counts from the end of inference, not from the end of
    # the database write.
    if _app_state is not None:
        _app_state.trigger_started()

    def worker():
        start = time.perf_counter()
        try:
            try:
                result = _engine.evaluate(frame)
            finally:
                if _app_state is not None:
                    _app_state.trigger_finished()
        except Exception as exc:
            logger.error("Vision worker exception: %s", exc)
            result = _empty_result("vision_exception")
            result["error"] = str(exc)
            result["processing_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
        callback(result)

    try:
        _pool.submit(worker)
    except Exception:
        # Never reached the pool, so the worker that would clear the mark will
        # never run. Left set, it would hold the overlay off for good.
        if _app_state is not None:
            _app_state.trigger_finished()
        raise
    return None
