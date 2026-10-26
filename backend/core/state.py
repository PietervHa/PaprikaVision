"""
Application State

Thread-safe container shared by the camera, vision, web and PLC threads.
Carried over from VisionSoftwareMDE with the OCR/classifier fields removed and
the live overlay state added.
"""

import threading
import time

from backend.core.config_loader import cfg

VALID_VISION_MODES = ("paprika", "idle")


class AppState:
    def __init__(self):
        self.lock = threading.Lock()

        self.latest_result = {
            "status": "NOK",
            "mode": "paprika",
            "detections": [],
            "primary": None,
            "processing_time_ms": 0,
        }

        self.vision_mode = str(cfg.get("vision_mode", "paprika"))
        self.camera_rotation = 0

        # Live overlay state. Kept separate from latest_result on purpose:
        # latest_result is the record of a *triggered* inspection - counted,
        # written to the database, answered to the PLC. The overlay is a
        # continuously refreshed preview that must never touch any of that.
        self.overlay_detections = []
        self.overlay_primary = None
        self.overlay_ts = 0.0
        self.overlay_scan_ms = 0.0

        # Triggered cycles in flight, and when the last one finished. The
        # overlay preview and a trigger share one model behind one lock, so the
        # overlay reads this to stay off the model while the line is asking.
        # A count rather than a flag: TCP triggers can queue behind each other
        # on the vision pool, and the first one finishing does not mean the
        # line has stopped asking.
        self.triggers_in_flight = 0
        self.last_trigger_end = 0.0

    # ------------------------------------------------------------- inspection

    def update_result(self, result: dict):
        with self.lock:
            self.latest_result = result

    def get_snapshot(self) -> dict:
        with self.lock:
            return {
                "result": dict(self.latest_result),
                "vision_mode": self.vision_mode,
            }

    # ---------------------------------------------------------------- overlay

    def set_overlay(self, detections: list, primary, scan_ms: float, ts: float):
        with self.lock:
            self.overlay_detections = detections
            self.overlay_primary = primary
            self.overlay_scan_ms = scan_ms
            self.overlay_ts = ts

    def get_overlay(self, max_age_s: float = 2.0, now: float = 0.0) -> dict:
        with self.lock:
            age = now - self.overlay_ts if self.overlay_ts else None
            stale = age is None or (max_age_s > 0 and age > max_age_s)
            return {
                "detections": [] if stale else list(self.overlay_detections),
                "primary": None if stale else self.overlay_primary,
                "age_s": age,
                "stale": stale,
                "scan_ms": self.overlay_scan_ms,
            }

    def clear_overlay(self):
        with self.lock:
            self.overlay_detections = []
            self.overlay_primary = None
            self.overlay_ts = 0.0

    # ------------------------------------------------------------ trigger gate

    def trigger_started(self):
        with self.lock:
            self.triggers_in_flight += 1

    def trigger_finished(self, now: float = 0.0):
        with self.lock:
            self.triggers_in_flight = max(0, self.triggers_in_flight - 1)
            self.last_trigger_end = now or time.monotonic()

    def overlay_held(self, pause_s: float, now: float = 0.0) -> bool:
        """True while the overlay must not start a new inference.

        Held while any trigger is in flight, and for pause_s after the last one
        finished. pause_s <= 0 never holds - the behaviour from before this
        existed, where the overlay and the trigger simply took turns on the
        model lock.

        Monotonic time, not wall time: a clock step from NTP must not be able
        to pause the overlay for an hour or cut the hold-off short.
        """
        if pause_s <= 0:
            return False
        now = now or time.monotonic()
        with self.lock:
            if self.triggers_in_flight > 0:
                return True
            return bool(self.last_trigger_end) and now - self.last_trigger_end < pause_s

    # ------------------------------------------------------------------ modes

    def get_vision_mode(self) -> str:
        with self.lock:
            return self.vision_mode

    def set_vision_mode(self, value: str):
        with self.lock:
            if value in VALID_VISION_MODES:
                self.vision_mode = value

    def get_camera_rotation(self) -> int:
        with self.lock:
            return self.camera_rotation

    def rotate_camera(self):
        with self.lock:
            self.camera_rotation = (self.camera_rotation + 1) % 4