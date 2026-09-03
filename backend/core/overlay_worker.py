"""
Overlay Worker

Continuously evaluates the camera feed so the HMI can draw a live orientation
overlay, independent of PLC-triggered inspections.

Why this is a separate thread and not part of the frame generator
-----------------------------------------------------------------
The MJPEG generator is pinned to ~30 fps. A pose inference is 20-80 ms on CPU
and the shape backend is a few ms, but neither is free, and running detection
inline would couple the video frame rate to inference latency - so the live
view would visibly stutter whenever the model slowed down. Instead this thread
scans on its own clock and the stream redraws the most recent completed result
at full speed.

The overlay is display-only. It never increments a counter, never writes a
result, and never reaches the PLC. That path stays exclusively driven by the
trigger, so what the operator sees can never be mistaken for what the machine
acted on.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from backend.core.config_loader import cfg
from backend.utils.logger import get_logger

log = get_logger(__name__)


class OverlayWorker:
    def __init__(self, camera, app_state, engine) -> None:
        self._camera = camera
        self._app_state = app_state
        self._engine = engine

        hmi_cfg = cfg.get("hmi") if isinstance(cfg.get("hmi"), dict) else {}
        self._interval_s = max(0.0, float(hmi_cfg.get("overlay_interval_s", 0.15)))

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_error = ""

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.is_running():
            return True
        if self._engine is None or not self._engine.is_ready():
            log.warning("Overlay not started: engine is not ready")
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="overlay_worker", daemon=True)
        self._thread.start()
        log.info("Overlay worker started (interval %.2fs)", self._interval_s)
        return True

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # Bounded join - the worker may be mid-inference, and blocking a web
            # handler until that finishes would hang the HMI.
            thread.join(timeout=2.0)
        self._thread = None
        self._app_state.clear_overlay()
        log.info("Overlay worker stopped")

    def toggle(self) -> bool:
        if self.is_running():
            self.stop()
            return False
        return self.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            cycle_start = time.perf_counter()
            try:
                frame = self._camera.get_frame()
                if frame is None:
                    self._stop_event.wait(0.2)
                    continue

                result = self._engine.evaluate(frame)
                self._app_state.set_overlay(
                    detections=result.get("detections") or [],
                    primary=result.get("primary"),
                    scan_ms=float(result.get("processing_time_ms", 0.0)),
                    ts=time.time(),
                )
                self._last_error = ""
            except Exception as exc:
                self._last_error = str(exc)
                log.error("Overlay cycle failed: %s", exc)
                self._stop_event.wait(0.5)
                continue

            remaining = self._interval_s - (time.perf_counter() - cycle_start)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def status(self) -> dict:
        return {
            "running": self.is_running(),
            "interval_s": self._interval_s,
            "error": self._last_error,
        }
