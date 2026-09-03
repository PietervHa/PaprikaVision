"""
Application State

Thread-safe container shared by the camera, vision, web and PLC threads.
Carried over from VisionSoftwareMDE with the OCR/classifier fields removed and
the live overlay state added.
"""

import threading

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
        self.counters = {"ok": 0, "nok": 0, "total": 0, "reorient": 0, "reject": 0}

        self.vision_mode = str(cfg.get("vision_mode", "paprika"))
        self.maintenance_mode = False
        self.maintenance_session_token = ""
        self.maintenance_session_user = ""
        self.camera_rotation = 0

        # Live overlay state. Kept separate from latest_result on purpose:
        # latest_result is the record of a *triggered* inspection - counted,
        # written to the database, answered to the PLC. The overlay is a
        # continuously refreshed preview that must never touch any of that.
        self.overlay_detections = []
        self.overlay_primary = None
        self.overlay_ts = 0.0
        self.overlay_scan_ms = 0.0

    # ------------------------------------------------------------- inspection

    def update_result(self, result: dict):
        with self.lock:
            self.latest_result = result

    def increment_counter(self, status: str, placement: str = ""):
        with self.lock:
            self.counters["total"] += 1
            if status == "OK":
                self.counters["ok"] += 1
            else:
                self.counters["nok"] += 1
            if placement in ("reorient", "reject"):
                self.counters[placement] += 1

    def reset_counters(self):
        with self.lock:
            self.counters = {"ok": 0, "nok": 0, "total": 0, "reorient": 0, "reject": 0}

    def get_snapshot(self) -> dict:
        with self.lock:
            return {
                "result": dict(self.latest_result),
                "counters": dict(self.counters),
                "maintenance_mode": self.maintenance_mode,
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

    # ------------------------------------------------------------------ modes

    def get_vision_mode(self) -> str:
        with self.lock:
            return self.vision_mode

    def set_vision_mode(self, value: str):
        with self.lock:
            if value in VALID_VISION_MODES:
                self.vision_mode = value

    def get_maintenance_mode(self) -> bool:
        with self.lock:
            return self.maintenance_mode

    def set_maintenance_mode(self, value: bool):
        with self.lock:
            self.maintenance_mode = bool(value)

    def get_maintenance_session_token(self) -> str:
        with self.lock:
            return self.maintenance_session_token

    def set_maintenance_session_token(self, value: str):
        with self.lock:
            self.maintenance_session_token = str(value or "")

    def get_maintenance_session_user(self) -> str:
        with self.lock:
            return self.maintenance_session_user

    def set_maintenance_session_user(self, value: str):
        with self.lock:
            self.maintenance_session_user = str(value or "")

    def get_camera_rotation(self) -> int:
        with self.lock:
            return self.camera_rotation

    def rotate_camera(self):
        with self.lock:
            self.camera_rotation = (self.camera_rotation + 1) % 4
