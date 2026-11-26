"""
Rotate-unit stem model: MDE's own keypoint model (no YOLO, no Ultralytics)

Runs the model MDE Annotation trains for the rotate unit (mde_annotation/keypoints, format
"mde-keypoints-1"): runs_keypoints/<name>/model.onnx with its model.json, which this app expects
as models/rotate/rotate_kp.onnx + models/rotate/rotate_kp.json.

The model looks at the whole frame - there is one paprika per rotate-unit frame - and returns
two points with a confidence each:

    point 0   stem_end    the end of the straight part of the stem
    point 1   stem_base   where the stem leaves the fruit

The frame is prepared exactly as the training prepared it: BGR -> RGB, letterboxed to the model's
fixed input with cv2.warpAffine (centred, bilinear, padding 114), divided by 255; the
normalisation is inside the model. The points come back in input pixels and are taken back to
frame pixels here.

model.json holds what the training decided: the confidence threshold tuned on the validation
frames, and which point the angle runs from and to. A threshold in the config overrides it.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.utils.logger import get_logger

log = get_logger(__name__)

PAD = 114
POINT_NAMES = ("stem_end", "stem_base")


def letterbox_matrix(width: int, height: int, in_w: int, in_h: int) -> np.ndarray:
    """The training's letterbox: scale to fit, centred (MDE Annotation keypoints.data)."""
    s = min(in_w / width, in_h / height)
    return np.array([[s, 0.0, (in_w - width * s) / 2.0], [0.0, s, (in_h - height * s) / 2.0]], np.float64)


class RotateKeypointModel:
    def __init__(self, model_path: Path, threads: int = 1, point_threshold: Optional[float] = None) -> None:
        self.path = Path(model_path)
        self.threads = max(1, int(threads))
        self._threshold_override = None if point_threshold is None else float(point_threshold)
        self._session = None
        self._lock = threading.Lock()
        self.input_w = self.input_h = 0
        self.threshold = 0.5
        self.angle_from, self.angle_to = 1, 0
        self.settings_source = "defaults"
        self.error: Optional[str] = None

    # ------------------------------------------------------------- loading
    def load_model(self) -> None:
        import onnxruntime as ort

        if not self.path.exists():
            self.error = f"model file not found: {self.path}"
            raise FileNotFoundError(self.error)
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(self.path), options, providers=["CPUExecutionProvider"])
        shape = session.get_inputs()[0].shape
        out = session.get_outputs()[0].shape
        if len(shape) != 4 or not all(isinstance(v, int) and v > 0 for v in shape[2:]) or len(out) != 3 or out[2] != 3:
            raise ValueError(f"{self.path.name} is not an MDE keypoint model (input {shape}, output {out}); "
                             "expected input [1, 3, H, W] and output [1, K, 3]")
        self.input_h, self.input_w = int(shape[2]), int(shape[3])
        self._read_settings()
        self._session = session
        self.error = None
        log.info("Rotate model %s loaded: input %dx%d, threshold %.2f (%s), angle from point %d to %d",
                 self.path.name, self.input_w, self.input_h, self.threshold, self.settings_source,
                 self.angle_from, self.angle_to)

    def _read_settings(self) -> None:
        meta_path = self.path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("format") not in (None, "mde-keypoints-1"):
                log.warning("%s: format %s, expected mde-keypoints-1", meta_path.name, meta.get("format"))
            self.threshold = float(meta.get("threshold", self.threshold))
            angle = meta.get("angle") or {}
            self.angle_from = int(angle.get("from", self.angle_from))
            self.angle_to = int(angle.get("to", self.angle_to))
            self.settings_source = meta_path.name
        else:
            log.warning("%s not found next to the model: threshold %.2f by default - copy the training's model.json "
                        "there as %s", meta_path.name, self.threshold, meta_path.name)
        if self._threshold_override is not None:
            self.threshold = self._threshold_override
            self.settings_source = "config"

    def is_ready(self) -> bool:
        return self._session is not None

    def warm_up(self, frame_shape) -> None:
        if self._session is None:
            return
        h, w = int(frame_shape[0]), int(frame_shape[1])
        self.predict(np.zeros((h, w, 3), np.uint8))

    def status(self) -> dict:
        return {
            "backend": "mde-keypoints",
            "model": str(self.path),
            "ready": self.is_ready(),
            "input": f"{self.input_w}x{self.input_h}" if self.input_w else None,
            "point_threshold": self.threshold,
            "settings_from": self.settings_source,
            "threads": self.threads,
            "error": self.error,
        }

    # ------------------------------------------------------------- running
    def predict(self, frame_bgr: np.ndarray, timing: Optional[dict] = None) -> list:
        """The points in frame pixels: [(x, y, confidence)] in the model's order."""
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        A = letterbox_matrix(w, h, self.input_w, self.input_h)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if frame_bgr.ndim == 3 else cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
        boxed = cv2.warpAffine(rgb, A, (self.input_w, self.input_h), flags=cv2.INTER_LINEAR,
                               borderValue=(PAD, PAD, PAD))
        tensor = np.ascontiguousarray(boxed.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        t_lock = time.perf_counter()
        with self._lock:
            t_run = time.perf_counter()
            out = self._session.run(None, {self._session.get_inputs()[0].name: tensor})[0][0]
            t_done = time.perf_counter()
        s, tx, ty = A[0, 0], A[0, 2], A[1, 2]
        points = [((float(x) - tx) / s, (float(y) - ty) / s, float(c)) for x, y, c in out]
        if timing is not None:
            timing["lock_wait_ms"] = round((t_run - t_lock) * 1000, 2)
            timing["inference_ms"] = round((t_done - t_run) * 1000, 2)
            timing["detect_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return points
