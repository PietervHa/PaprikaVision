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


def bind_app_state(app_state) -> None:
    global _engine, _app_state
    _app_state = app_state
    _engine = PaprikaEngine(cfg, app_state=app_state)
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

    def worker():
        start = time.perf_counter()
        try:
            result = _engine.evaluate(frame)
        except Exception as exc:
            logger.error("Vision worker exception: %s", exc)
            result = _empty_result("vision_exception")
            result["error"] = str(exc)
            result["processing_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
        callback(result)

    _pool.submit(worker)
    return None
