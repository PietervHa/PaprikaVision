"""
Rotate Engine

The rotate unit's answer from a frame: which way the stem points. Same interface and the same
result contract as PaprikaEngine (see its module docstring), so the HMI, the overlay, the PLC
trigger, the database and the file source all work unchanged - chosen in the config with

    paprika:
      engine: rotate                     # "belt" (default) | "rotate"
      rotate:
        model_path: models/rotate/rotate_kp.onnx    # + rotate_kp.json next to it
        threads: 1
        point_threshold: null            # null = the one in model.json (tuned on validation)

Why not the belt engine with other settings: the belt's rules (standing fruit, end-on, edge cut,
groove and shape fallbacks, span ratio) are about fruit lying on a belt. On a rotate-unit frame
they reject good answers - which is what made the first rotate test look bad. Here there is one
model and one rule: an angle when both points are confident and apart, else "no stem".

The model is MDE's own keypoint model (backend/detection/paprika/rotate_keypoints.py). Two angle
conventions meet here, both measured from the stem base to the end of the straight part:

    orientation.angle_deg   this app's screen convention (orientation.py): 0 = right, 90 = UP the
                            image - so the overlay's arrow and everything downstream read it right
    angle_rotate            the rotate unit's convention: -180..180, 0 = +X, y DOWN (-90 = up) -
                            the labels', MDE Annotation's and Aurora's (SendResult) convention
    angle_plc               angle_rotate with paprika.frame applied (offset, then invert), kept in
                            -180..180: with the defaults exactly what Aurora sends the PLC
The label on the HMI shows angle_plc, so it reads the same as Aurora.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from backend.detection.paprika import orientation as orient
from backend.detection.paprika.pose_detector import PoseModelConfigError
from backend.detection.paprika.rotate_keypoints import POINT_NAMES, RotateKeypointModel
from backend.utils.logger import get_logger
from backend.utils.paths import project_path

log = get_logger(__name__)

PLACEMENT_PLACE = "place"
PLACEMENT_UNKNOWN = "unknown"
MIN_STEM_PX = 2.0


def _empty(reason: str, total_ms: float = 0.0, timing: Optional[dict] = None, error: Optional[str] = None) -> dict:
    result = {"status": "NOK", "mode": "paprika", "detections": [], "primary": None, "confidence": 0.0,
              "processing_time_ms": total_ms, "failure_reason": reason, "timing": timing or {}}
    if error:
        result["error"] = error
    return result


class RotateEngine:
    def __init__(self, cfg: dict, app_state=None) -> None:
        block = cfg.get("paprika", {}) or {}
        rot = block.get("rotate", {}) or {}
        frame_cfg = block.get("frame", {}) or {}
        self._angle_offset_deg = float(frame_cfg.get("angle_offset_deg", 0.0))
        self._angle_invert = bool(frame_cfg.get("angle_invert", False))
        self._model = RotateKeypointModel(
            project_path(rot.get("model_path"), "models/rotate/rotate_kp.onnx"),
            threads=int(rot.get("threads", 1)),
            point_threshold=rot.get("point_threshold"),
        )
        self._app_state = app_state
        try:
            self._model.load_model()
        except Exception as exc:                       # reported by status() and every evaluate()
            self._model.error = str(exc)
            log.error("Rotate model not loaded: %s", exc)

    # ------------------------------------------------------------- PaprikaEngine's interface
    def is_ready(self) -> bool:
        return self._model.is_ready()

    def load_model(self) -> None:
        """Load the model if that has not happened yet. A model that cannot run stops the app at the
        start with one clear message (main.py turns PoseModelConfigError into "Refusing to start")."""
        if self._model.is_ready():
            return
        try:
            self._model.load_model()
        except Exception as exc:
            raise PoseModelConfigError(f"rotate model {self._model.path}: {exc}") from exc

    def warm_up(self, frame_shape) -> None:
        self._model.warm_up(frame_shape)

    def status(self) -> dict:
        status = self._model.status()
        status.update({"engine": "rotate", "angle_offset_deg": self._angle_offset_deg, "angle_invert": self._angle_invert})
        return status

    def _timing(self, total_ms: float, t: dict) -> dict:
        detect_ms = t.get("detect_ms")
        return {"backend": "mde-keypoints", "lock_wait_ms": t.get("lock_wait_ms"), "inference_ms": t.get("inference_ms"),
                "detect_ms": detect_ms,
                "post_ms": None if detect_ms is None else round(max(0.0, total_ms - float(detect_ms)), 2),
                "total_ms": total_ms}

    # ------------------------------------------------------------- evaluation
    def evaluate(self, frame: np.ndarray) -> dict:
        start = time.perf_counter()
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return _empty("no_frame")
        if not self.is_ready():
            return _empty("detector_unavailable", round((time.perf_counter() - start) * 1000, 2),
                          error=self._model.error)
        t: dict = {}
        try:
            points = self._model.predict(frame, timing=t)
        except Exception as exc:
            log.error("Rotate model failed: %s", exc)
            total = round((time.perf_counter() - start) * 1000, 2)
            return _empty("detector_error", total, self._timing(total, t), error=str(exc))
        detection = self._detection(points, frame.shape[1], frame.shape[0])
        total = round((time.perf_counter() - start) * 1000, 2)
        found = detection["placement"] == PLACEMENT_PLACE
        result = {
            "status": "OK" if found else "NOK",
            "mode": "paprika",
            "detections": [detection],
            "primary": detection,
            "confidence": float(detection["orientation"].get("confidence", 0.0)),
            "processing_time_ms": total,
            "timing": self._timing(total, t),
        }
        if not found:
            result["failure_reason"] = "no_stem_found"
        return result

    def _detection(self, points: list, width: int, height: int) -> dict:
        thr = self._model.threshold
        a, b = self._model.angle_from, self._model.angle_to
        base, end = points[a], points[b]
        length = math.hypot(end[0] - base[0], end[1] - base[1])
        found = base[2] >= thr and end[2] >= thr and length >= MIN_STEM_PX
        conf = min(base[2], end[2])
        angle_rotate = angle_plc = None
        if found:
            angle_rotate = math.degrees(math.atan2(end[1] - base[1], end[0] - base[0]))     # y down
            angle_screen = math.degrees(math.atan2(-(end[1] - base[1]), end[0] - base[0]))  # y up, as orientation.py
            angle_plc = angle_rotate + self._angle_offset_deg
            angle_plc = -angle_plc if self._angle_invert else angle_plc
            angle_plc = (angle_plc + 180.0) % 360.0 - 180.0
            result = orient.Orientation(angle_deg=angle_screen, axis_deg=angle_screen % 180.0, source="keypoints",
                                        confidence=conf, flip_confidence=conf, pose=orient.POSE_LYING,
                                        stem_present=True, stem_span_px=length)
            label = f"stem {angle_plc:.0f} deg"    # "deg": the overlay's Hershey fonts are ASCII-only
        else:
            result = orient.Orientation(source="keypoints", confidence=conf, pose=orient.POSE_STEM_NOT_FOUND,
                                        stem_present=False, notes=[f"point confidence {base[2]:.2f} / {end[2]:.2f}, "
                                                                   f"threshold {thr:.2f}"])
            label = "no stem"
        # A box around the stem, only as the anchor for the overlay's label - not drawn (draw_box False):
        # the model finds the stem, not the fruit's outline, so a box would suggest something it is not.
        margin = max(40.0, 0.6 * length)
        xs, ys = (base[0], end[0]), (base[1], end[1])
        x1, y1 = max(0.0, min(xs) - margin), max(0.0, min(ys) - margin)
        x2, y2 = min(width - 1.0, max(xs) + margin), min(height - 1.0, max(ys) + margin)
        names = list(POINT_NAMES) + [f"point_{i}" for i in range(len(POINT_NAMES), len(points))]
        return {
            "label": label,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "draw_box": False,
            "center": [int((base[0] + end[0]) / 2), int((base[1] + end[1]) / 2)],
            "confidence": round(float(max(base[2], end[2])), 3),
            "keypoints": {
                names[i]: {"x": round(float(x), 1), "y": round(float(y), 1), "confidence": round(float(c), 3),
                           "visible": bool(c >= thr)}
                for i, (x, y, c) in enumerate(points)
            },
            "orientation": result.to_dict(),
            "angle_rotate": None if angle_rotate is None else round(angle_rotate, 2),
            "angle_plc": None if angle_plc is None else round(angle_plc, 2),
            "placement": PLACEMENT_PLACE if found else PLACEMENT_UNKNOWN,
            "simulated": False,
            "colour": "",
            "stem_method": "keypoints",
            "stem_quality": round(float(conf), 3),
            "stem_spread_deg": 0.0,
        }
