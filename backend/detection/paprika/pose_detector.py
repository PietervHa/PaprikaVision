"""
Paprika Detector

Finds paprikas in a frame and hands each one to the orientation engine.

Three backends, selected by `paprika.backend` in config:

  "shape"      Classical segmentation. No model, no training, no GPU. Finds
               saturated blobs against the belt and orients them from
               silhouette alone. Works TODAY, before a single image has been
               annotated - which is the point: the HMI, the PLC handshake, the
               overlay and the angle convention can all be commissioned and
               signed off while the dataset is still being collected. It is
               also a genuine fallback if the model ever fails to load.

  "pose"       YOLOv8/YOLO11 pose model predicting one box plus the two
               landmarks per fruit. The production backend once trained.

  "simulator"  Synthetic fruit sweeping through 360 degrees, ignoring the
               camera entirely. For exercising the PLC integration on a bench
               with no camera and no product.

All three return the identical contract, so switching backend is a config edit
and nothing downstream can tell the difference.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import cv2
import numpy as np

from backend.detection.paprika import classical
from backend.detection.paprika import orientation as orient
from backend.detection.paprika.orientation import Keypoint
from backend.utils.logger import get_logger

log = get_logger(__name__)

VALID_BACKENDS = ("shape", "pose", "simulator")


class PaprikaDetector:
    """Detects paprikas and returns box + landmarks for each.

    Args:
        cfg_block: the `paprika` section of the config.
    """

    def __init__(self, cfg_block: dict) -> None:
        cfg_block = cfg_block if isinstance(cfg_block, dict) else {}
        self._cfg = cfg_block

        backend = str(cfg_block.get("backend", "shape")).strip().lower()
        if backend not in VALID_BACKENDS:
            log.warning("Unknown paprika backend '%s'; falling back to 'shape'", backend)
            backend = "shape"
        self.backend = backend

        self.confidence = float(cfg_block.get("confidence", 0.45))
        self.max_detections = int(cfg_block.get("max_detections", 8))

        shape_cfg = cfg_block.get("shape") if isinstance(cfg_block.get("shape"), dict) else {}
        self._saturation_floor = int(shape_cfg.get("saturation_floor", 80))
        belt = shape_cfg.get("belt_hue") or [96, 145]
        self._belt_hue = (int(belt[0]), int(belt[1]))
        self._value_floor = int(shape_cfg.get("value_floor", 45))
        stem = shape_cfg.get("stem_hue") or [33, 92]
        self._stem_hue = (int(stem[0]), int(stem[1]))
        self._use_roi = bool(shape_cfg.get("use_roi", True))
        self._selfcheck = bool(shape_cfg.get("stem_selfcheck", True))
        self._restrict_to_belt = bool(shape_cfg.get("restrict_to_belt", True))
        self._belt_warned = False
        self._min_area_px = int(shape_cfg.get("min_area_px", 4000))
        self._max_area_ratio = float(
            shape_cfg.get("max_area_ratio", classical.DEFAULT_MAX_AREA_RATIO)
        )

        pose_cfg = cfg_block.get("pose") if isinstance(cfg_block.get("pose"), dict) else {}
        self._model_path = str(pose_cfg.get("model_path", "models/paprika_pose.pt"))
        self._imgsz = int(pose_cfg.get("imgsz", 640))
        self._kp_confidence = float(pose_cfg.get("keypoint_confidence", 0.30))

        self._model = None
        self._model_lock = threading.Lock()
        self._model_failed = False

        self._sim_start = time.time()

        log.info("Paprika detector backend: %s", self.backend)

    # ------------------------------------------------------------- lifecycle

    def is_ready(self) -> bool:
        if self.backend in ("shape", "simulator"):
            return True
        return self._load_model() is not None

    def _load_model(self):
        """Lazily load the pose model, mirroring the old project's YOLO pattern."""
        if self._model is not None:
            return self._model
        if self._model_failed:
            return None

        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
            log.info("Paprika pose model loaded: %s", self._model_path)
            return self._model
        except Exception as exc:
            log.error("Failed to load paprika pose model '%s': %s", self._model_path, exc)
            self._model_failed = True
            return None

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "ready": self.is_ready(),
            "model_path": self._model_path if self.backend == "pose" else "",
            "confidence": self.confidence,
        }

    # -------------------------------------------------------------- dispatch

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Return raw detections: [{bbox, confidence, keypoints}].

        `keypoints` maps landmark name -> Keypoint, and may be empty (the shape
        backend never produces landmarks). Orientation is NOT computed here -
        that belongs to the engine, so all three backends stay comparable.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        if self.backend == "pose":
            return self._detect_pose(frame)
        if self.backend == "simulator":
            return self._detect_simulator(frame)
        return self._detect_shape(frame)

    # ----------------------------------------------------------- shape route

    def _detect_shape(self, frame: np.ndarray) -> list[dict]:
        """Classical detection: fruit by hue, stem by colour or morphology.

        Emits real keypoints, not just a box. That is the difference between a
        backend that finds something and a backend the machine can run on:
        without stem_end and blossom_end the engine has no source for the angle
        at all, and rightly reports "unknown" on every fruit.

        Colour-independent in the fruit detection - nothing selects on paprika
        colour, it only filters the belt out.
        """
        belt = None
        if self._restrict_to_belt:
            belt = classical.belt_mask_raw(frame, self._belt_hue)
            if belt is None and not self._belt_warned:
                log.warning(
                    "No belt recognised in the frame - searching the whole image. "
                    "Check the camera framing and paprika.shape.belt_hue."
                )
                self._belt_warned = True

        fruits = classical.find_fruit(
            frame,
            belt_hue=self._belt_hue,
            saturation_floor=self._saturation_floor,
            value_floor=self._value_floor,
            stem_hue=self._stem_hue,
            min_area=self._min_area_px,
            max_area_ratio=self._max_area_ratio,
            max_fruit=self.max_detections,
            use_roi=self._use_roi,
            selfcheck=self._selfcheck,
            belt_mask=belt,
        )

        detections: list[dict] = []
        for fruit in fruits:
            landmarks: dict[str, Keypoint] = {}
            if fruit.stem_end is not None and fruit.blossom_end is not None:
                # The keypoint confidence IS the stem localisation quality.
                # Feeding a fixed 0.85 here was what let a loosely pinned stem
                # on a green fruit reach the PLC looking as certain as a stem
                # separated cleanly by colour.
                quality = float(fruit.stem_quality or 0.5)
                landmarks["stem_end"] = Keypoint(
                    fruit.stem_end[0], fruit.stem_end[1], quality, True
                )
                landmarks["blossom_end"] = Keypoint(
                    fruit.blossom_end[0],
                    fruit.blossom_end[1],
                    quality,
                    # On a standing fruit the blossom end is underneath.
                    # Marking it occluded is exactly what the engine needs to
                    # tell stem-up from stem-down.
                    not fruit.standing,
                )

            x1, y1, x2, y2 = fruit.bbox
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    # No model score available, so these are honestly fixed:
                    # found is found. An invented variable score would imply a
                    # confidence that nothing supports.
                    "confidence": round(0.55 + 0.35 * float(fruit.stem_quality or 0.0), 3),
                    "keypoints": landmarks,
                    "area_px": fruit.area,
                    "stem_method": fruit.stem_method,
                    "stem_quality": round(float(fruit.stem_quality or 0.0), 3),
                    "stem_spread_deg": float(fruit.stem_spread_deg or 0.0),
                    "colour": fruit.colour,
                    "unpickable_reason": fruit.unpickable_reason,
                    "edge_clipped": fruit.edge_clipped,
                }
            )
        return detections

    # ------------------------------------------------------------ pose route

    def _detect_pose(self, frame: np.ndarray) -> list[dict]:
        with self._model_lock:
            model = self._load_model()
            if model is None:
                return []
            results = model.predict(
                frame, imgsz=self._imgsz, conf=self.confidence, verbose=False
            )

        detections: list[dict] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)
            if boxes is None:
                continue

            box_xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.empty((0, 4))
            box_conf = boxes.conf.cpu().numpy() if len(boxes) else np.empty((0,))

            kp_xy = None
            kp_conf = None
            if keypoints is not None and getattr(keypoints, "xy", None) is not None:
                kp_xy = keypoints.xy.cpu().numpy()
                if getattr(keypoints, "conf", None) is not None:
                    kp_conf = keypoints.conf.cpu().numpy()

            for i in range(len(box_xyxy)):
                x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy[i])
                landmarks: dict[str, Keypoint] = {}

                if kp_xy is not None and i < len(kp_xy):
                    for k, name in enumerate(orient.KEYPOINT_NAMES):
                        if k >= len(kp_xy[i]):
                            break
                        kx, ky = float(kp_xy[i][k][0]), float(kp_xy[i][k][1])
                        kc = float(kp_conf[i][k]) if kp_conf is not None and i < len(kp_conf) else 1.0

                        # Ultralytics emits (0, 0) for a landmark it declined to
                        # place. Treated as absent rather than as a real point
                        # at the image origin, which would otherwise produce a
                        # confident angle pointing at the top-left corner.
                        if kx == 0.0 and ky == 0.0:
                            continue

                        landmarks[name] = Keypoint(
                            x=kx,
                            y=ky,
                            confidence=kc,
                            # Below the landmark threshold means "I think it is
                            # here but I cannot see it" - the occluded case the
                            # annotation spec trains for. Still positioned, just
                            # not visible, which is what tells the engine the
                            # fruit may be standing stem-down.
                            visible=kc >= self._kp_confidence,
                        )

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(float(box_conf[i]), 3) if i < len(box_conf) else 0.0,
                        "keypoints": landmarks,
                        "area_px": max(0, (x2 - x1) * (y2 - y1)),
                    }
                )

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[: self.max_detections]

    # ------------------------------------------------------- simulator route

    def _detect_simulator(self, frame: np.ndarray) -> list[dict]:
        """One synthetic fruit rotating at 30 deg/s, ignoring the camera.

        Exists so the PLC handshake and the angle convention can be verified on
        a bench: the angle is known exactly, so any mismatch at the PLC end is
        unambiguously a convention error rather than a detection error.
        """
        h, w = frame.shape[:2]
        angle = (time.time() - self._sim_start) * 30.0 % 360.0

        cx, cy = w / 2.0, h / 2.0
        half_len = min(w, h) * 0.18
        radians = math.radians(angle)
        dx, dy = math.cos(radians) * half_len, -math.sin(radians) * half_len

        stem = (cx + dx, cy + dy)
        blossom = (cx - dx, cy - dy)

        pad = min(w, h) * 0.09
        x1 = int(min(stem[0], blossom[0]) - pad)
        y1 = int(min(stem[1], blossom[1]) - pad)
        x2 = int(max(stem[0], blossom[0]) + pad)
        y2 = int(max(stem[1], blossom[1]) + pad)

        return [
            {
                "bbox": [max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)],
                "confidence": 0.99,
                "keypoints": {
                    "stem_end": Keypoint(stem[0], stem[1], 0.99, True),
                    "blossom_end": Keypoint(blossom[0], blossom[1], 0.99, True),
                },
                "area_px": max(0, (x2 - x1) * (y2 - y1)),
                "simulated": True,
            }
        ]
