"""
Paprika Engine

The single place that turns a camera frame into the answer the machine acts
on: where the fruit is, which way its stem points, and whether that answer is
trustworthy enough to place on.

Result contract
---------------
    {
      "status": "OK" | "NOK",
      "mode": "paprika",
      "detections": [ ...every fruit found... ],
      "primary": { ...the one fruit the PLC should act on, or None... },
      "confidence": float,
      "processing_time_ms": float,
      "failure_reason": str            # only when status is NOK
    }

Each detection carries:

    bbox            [x1, y1, x2, y2] pixels
    center          [cx, cy] pixels
    confidence      detector confidence for the fruit itself
    keypoints       {name: {x, y, confidence, visible}}
    orientation     see orientation.Orientation.to_dict()
    angle_plc       orientation angle mapped into the machine's frame
    placement       "place" | "reject" | "reorient" | "unknown"
    label           short human string for the HMI overlay

Why `primary` exists
--------------------
The belt can show several fruit at once, but the actuator acts on one. Picking
it here rather than in the PLC keeps the choice in the place that can see the
whole frame, and keeps the TCP response a single fixed-shape line.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from backend.detection.paprika import orientation as orient
from backend.detection.paprika.orientation import Keypoint, Orientation
from backend.detection.paprika.pose_detector import PaprikaDetector
from backend.utils.logger import get_logger

log = get_logger(__name__)

PLACEMENT_PLACE = "place"
PLACEMENT_REORIENT = "reorient"
PLACEMENT_REJECT = "reject"
PLACEMENT_UNKNOWN = "unknown"


def _round_or_none(value, digits: int = 2):
    """Round for transport, preserving None as a distinct 'no angle' state."""
    return None if value is None else round(float(value), digits)


class PaprikaEngine:
    """Owns the detector and applies the placement policy."""

    def __init__(self, cfg: dict, app_state=None) -> None:
        self._app_state = app_state
        block = cfg.get("paprika") if isinstance(cfg.get("paprika"), dict) else {}
        self._cfg = block

        self._detector = PaprikaDetector(block)

        self._use_shape_crosscheck = bool(block.get("shape_crosscheck", False))

        # Bug die dit voorkomt: shape_crosscheck stond voor "gebruik het
        # silhouet als TWEEDE mening naast de keypoints". Maar de klassieke
        # backend leverde geen keypoints, dus met de cross-check uit had die
        # helemaal geen hoekbron meer en rapporteerde hij "onbekend" op elke
        # vrucht - detectie zonder antwoord.
        #
        # Beide kanten zijn nu gerepareerd: de klassieke backend levert echte
        # keypoints uit steeldetectie, en de silhouetschatter draait alleen nog
        # waar hij daadwerkelijk iets toevoegt. Twee instellingen die elkaar
        # stilzwijgend uitschakelden, is precies het soort koppeling dat je
        # alleen merkt als je het draait.
        self._backend = self._detector.backend
        shape_cfg = block.get("shape") if isinstance(block.get("shape"), dict) else {}
        self._saturation_floor = int(shape_cfg.get("saturation_floor", 80))
        belt = shape_cfg.get("belt_hue") or [96, 145]
        self._belt_hue = (int(belt[0]), int(belt[1]))
        self._min_span_ratio = float(block.get("min_span_ratio", 0.18))

        # Placement policy thresholds.
        policy = block.get("policy") if isinstance(block.get("policy"), dict) else {}
        self._min_angle_confidence = float(policy.get("min_angle_confidence", 0.45))
        self._min_flip_confidence = float(policy.get("min_flip_confidence", 0.40))
        self._reject_standing = bool(policy.get("reject_standing", True))

        # Machine frame mapping. Changing how the vision zero lines up with the
        # actuator zero must never require a code change - it is a commissioning
        # adjustment, done once per machine, by whoever is standing at it.
        frame_cfg = block.get("frame") if isinstance(block.get("frame"), dict) else {}
        self._angle_offset_deg = float(frame_cfg.get("angle_offset_deg", 0.0))
        self._angle_invert = bool(frame_cfg.get("angle_invert", False))

        # Which fruit the actuator acts on when several are visible.
        self._primary_rule = str(block.get("primary_rule", "largest")).strip().lower()

    # ------------------------------------------------------------------ state

    def is_ready(self) -> bool:
        return self._detector.is_ready()

    def status(self) -> dict:
        status = self._detector.status()
        status.update(
            {
                "shape_crosscheck": self._use_shape_crosscheck,
                "angle_offset_deg": self._angle_offset_deg,
                "angle_invert": self._angle_invert,
                "primary_rule": self._primary_rule,
            }
        )
        return status

    # ------------------------------------------------------------- evaluation

    def _placement_for(self, result: Orientation) -> str:
        """Decide what the machine should do with this fruit.

        Deliberately conservative. A wrong angle puts a paprika down backwards;
        an honest "I don't know" just sends it round again. The costs are not
        symmetric, so the thresholds are not either.
        """
        if result.pose in (orient.POSE_STANDING_STEM_UP, orient.POSE_STANDING_STEM_DOWN):
            # A fruit stood on its end has no meaningful in-plane rotation. The
            # machine has to topple it and look again; there is nothing to
            # place from this view.
            return PLACEMENT_REJECT if self._reject_standing else PLACEMENT_REORIENT

        if result.angle_deg is None:
            return PLACEMENT_UNKNOWN

        if result.confidence < self._min_angle_confidence:
            return PLACEMENT_UNKNOWN

        if result.flip_confidence < self._min_flip_confidence:
            # The axis is solid but which end carries the stem is not. Placing
            # now is a coin flip, so hand it back for another look.
            return PLACEMENT_REORIENT

        return PLACEMENT_PLACE

    @staticmethod
    def _label_for(result: Orientation, placement: str) -> str:
        if result.pose == orient.POSE_STANDING_STEM_UP:
            return "standing, stem up"
        if result.pose == orient.POSE_STANDING_STEM_DOWN:
            return "standing, stem down"
        if result.angle_deg is None:
            return "orientation unknown"
        suffix = "" if placement == PLACEMENT_PLACE else f" ({placement})"
        # "deg", not the degree sign: this string is drawn with OpenCV's Hershey
        # fonts, which are ASCII-only and render anything else as "??".
        return f"stem {result.angle_deg:.0f} deg{suffix}"

    def _evaluate_one(self, frame: np.ndarray, detection: dict) -> dict:
        bbox = detection["bbox"]
        landmarks: dict = detection.get("keypoints") or {}

        stem: Optional[Keypoint] = landmarks.get("stem_end")
        blossom: Optional[Keypoint] = landmarks.get("blossom_end")

        # Het silhouet draait alleen als tweede mening naast bestaande
        # keypoints. Zonder keypoints zou het de enige bron zijn, en op
        # blokpaprika haalt die maar 66% op de vraag welk uiteinde de steel is -
        # dan is "geen hoek" een eerlijker antwoord dan een muntworp.
        use_shape = self._use_shape_crosscheck and stem is not None and blossom is not None

        result = orient.estimate(
            frame=frame if use_shape else None,
            bbox=bbox,
            stem=stem,
            blossom=blossom,
            use_shape=use_shape,
            saturation_floor=self._saturation_floor,
            belt_hue=self._belt_hue,
            min_span_ratio=self._min_span_ratio,
        )

        placement = self._placement_for(result)
        x1, y1, x2, y2 = bbox

        return {
            "label": self._label_for(result, placement),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "confidence": float(detection.get("confidence", 0.0)),
            "keypoints": {
                name: {
                    "x": round(float(kp.x), 1),
                    "y": round(float(kp.y), 1),
                    "confidence": round(float(kp.confidence), 3),
                    "visible": bool(kp.visible),
                }
                for name, kp in landmarks.items()
            },
            "orientation": result.to_dict(),
            "angle_plc": _round_or_none(
                orient.apply_frame_convention(
                    result.angle_deg, self._angle_offset_deg, self._angle_invert
                )
            ),
            "placement": placement,
            "simulated": bool(detection.get("simulated", False)),
        }

    def _pick_primary(self, detections: list[dict]) -> Optional[dict]:
        """Choose the fruit the actuator acts on.

        "largest" is the default because on a belt the biggest silhouette is
        normally the one fully in view, rather than one half-entering frame
        whose angle is being measured from a partial fruit.
        """
        if not detections:
            return None

        placeable = [d for d in detections if d["placement"] == PLACEMENT_PLACE]
        pool = placeable or detections

        if self._primary_rule == "confidence":
            return max(pool, key=lambda d: d["orientation"].get("confidence", 0.0))
        if self._primary_rule == "centered":
            return min(pool, key=lambda d: abs(d["center"][0]) + abs(d["center"][1]))
        return max(
            pool,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        )

    def evaluate(self, frame: np.ndarray) -> dict:
        """Full cycle: detect, orient, decide."""
        start = time.perf_counter()

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {
                "status": "NOK",
                "mode": "paprika",
                "detections": [],
                "primary": None,
                "confidence": 0.0,
                "processing_time_ms": 0.0,
                "failure_reason": "no_frame",
            }

        if not self.is_ready():
            return {
                "status": "NOK",
                "mode": "paprika",
                "detections": [],
                "primary": None,
                "confidence": 0.0,
                "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
                "failure_reason": "detector_unavailable",
            }

        try:
            raw = self._detector.detect(frame)
        except Exception as exc:
            log.error("Paprika detection failed: %s", exc)
            return {
                "status": "NOK",
                "mode": "paprika",
                "detections": [],
                "primary": None,
                "confidence": 0.0,
                "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
                "failure_reason": "detector_error",
                "error": str(exc),
            }

        detections = [self._evaluate_one(frame, d) for d in raw]
        primary = self._pick_primary(detections)

        processing_time_ms = round((time.perf_counter() - start) * 1000, 2)

        if primary is None:
            return {
                "status": "NOK",
                "mode": "paprika",
                "detections": [],
                "primary": None,
                "confidence": 0.0,
                "processing_time_ms": processing_time_ms,
                "failure_reason": "no_paprika_detected",
            }

        placeable = primary["placement"] == PLACEMENT_PLACE

        result = {
            # OK means "the machine may act on this angle", not "a fruit was
            # seen". A detected fruit whose orientation cannot be trusted is a
            # NOK with a reason, because acting on it is the failure mode this
            # whole engine exists to prevent.
            "status": "OK" if placeable else "NOK",
            "mode": "paprika",
            "detections": detections,
            "primary": primary,
            "confidence": float(primary["orientation"].get("confidence", 0.0)),
            "processing_time_ms": processing_time_ms,
        }
        if not placeable:
            result["failure_reason"] = f"not_placeable_{primary['placement']}"
        return result
