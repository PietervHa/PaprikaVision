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

import math
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

# Why the detector said a fruit is unusable, mapped onto the pose it is
# recorded as. The keys are the REASON_* values in
# backend/detection/paprika/classical.py; they are wire tokens that also end up
# in stored results, so they are matched here as literals exactly as the rest
# of this file does.
#
# Anything unrecognised falls through to POSE_UPSIDE_DOWN, which is the
# conservative default: it rejects, and a new reason showing up as "upside
# down" in the log is at least visible rather than silently placeable.
_UNPICKABLE_POSES = {
    "edge_clipped": orient.POSE_INCOMPLETE,
    "no_stem": orient.POSE_STEM_NOT_FOUND,
}

# Kept beside the mapping above so a new reason cannot end up with a pose but
# no label, which is how an HMI ends up showing a raw enum to an operator.
_UNPICKABLE_LABELS = {
    orient.POSE_INCOMPLETE: "incomplete in frame",
    orient.POSE_STEM_NOT_FOUND: "stem not found",
    orient.POSE_UPSIDE_DOWN: "upside down",
}


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

        # The bug this prevents: shape_crosscheck meant "use the silhouette as a
        # SECOND opinion alongside the keypoints". But the classical backend
        # produced no keypoints, so with the cross-check off it had no source of
        # an angle at all and reported "unknown" on every fruit - detection
        # without an answer.
        #
        # Both sides are fixed now: the classical backend emits real keypoints
        # from stem detection, and the silhouette estimator only runs where it
        # actually adds something. Two settings that silently disabled each
        # other is exactly the kind of coupling you only notice by running it.
        self._backend = self._detector.backend
        shape_cfg = block.get("shape") if isinstance(block.get("shape"), dict) else {}
        self._saturation_floor = int(shape_cfg.get("saturation_floor", 80))
        belt = shape_cfg.get("belt_hue") or [96, 145]
        self._belt_hue = (int(belt[0]), int(belt[1]))
        self._min_span_ratio = float(block.get("min_span_ratio", 0.18))

        # When the classical backend cannot find a stem at all on a fully-
        # visible fruit, ask the silhouette instead of rejecting outright: a
        # paprika is measurably wider at the stem end (the shoulder, where
        # the calyx sits) than at the blossom end, and that is readable from
        # the outline alone - no stem required. See _shape_only_estimate().
        #
        # This is attempted ONLY on a fruit shape_orientation() itself calls
        # "lying" (elongated). A round silhouette - standing on either end -
        # is left exactly as before: detected and rejected, with no angle
        # invented for it, because there is no wider end to read when the
        # camera is looking straight down the long axis rather than along
        # its side.
        self._stemless_shape_fallback = bool(block.get("stemless_shape_fallback", True))

        # Placement policy thresholds.
        policy = block.get("policy") if isinstance(block.get("policy"), dict) else {}
        self._min_angle_confidence = float(policy.get("min_angle_confidence", 0.45))
        self._min_flip_confidence = float(policy.get("min_flip_confidence", 0.40))
        self._reject_standing = bool(policy.get("reject_standing", True))
        # Maximum measured movement of the stem direction under a lighting
        # change before the fruit is sent round again instead of placed.
        self._max_stem_spread_deg = float(policy.get("max_stem_spread_deg", 6.0))

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
                "stemless_shape_fallback": self._stemless_shape_fallback,
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
        if result.pose in (
            orient.POSE_UPSIDE_DOWN,
            orient.POSE_STEM_NOT_FOUND,
            orient.POSE_INCOMPLETE,
        ):
            return PLACEMENT_REJECT

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

    def _shape_only_estimate(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> Optional[Orientation]:
        """Silhouette-only orientation for a fruit whose stem could not be found.

        Only ever called for REASON_NO_STEM, never for an edge-clipped fruit -
        a partial silhouette has no trustworthy width profile either, and
        that case is handled by the caller before this is reached.

        Runs shape_orientation() directly rather than through orient.estimate(),
        which fuses it against keypoints that simply do not exist on this
        path. Returns None - never a half-finished Orientation - whenever the
        mask cannot be produced, or the fruit turns out to be round: a round
        silhouette means the fruit is standing on one end or the other, and
        there is no wider end to read when the camera is looking straight
        down the long axis rather than along its side. "Detect it, do not
        guess an angle for it" is the correct answer there, not a gap to
        work around.

        The angle this returns, when it returns one, still goes through the
        normal _placement_for() gate below like any other estimate - a weak
        width signal (this backend's median on real fruit was 0.074, per the
        measurement behind paprika.shape_crosscheck) is sent for
        reorientation rather than placed on a guess.
        """
        if frame is None:
            return None
        mask = orient.segment_fruit(
            frame, bbox, saturation_floor=self._saturation_floor, belt_hue=self._belt_hue
        )
        if mask is None:
            return None

        result = orient.shape_orientation(mask)
        if result.pose != orient.POSE_LYING or result.angle_deg is None:
            return None

        result.source = "shape_only"
        result.notes.append("no_stem")
        result.notes.append("shape_fallback")
        return result

    def _evaluate_one(self, frame: np.ndarray, detection: dict) -> dict:
        bbox = detection["bbox"]
        landmarks: dict = detection.get("keypoints") or {}

        stem: Optional[Keypoint] = landmarks.get("stem_end")
        blossom: Optional[Keypoint] = landmarks.get("blossom_end")

        # The silhouette runs only as a second opinion alongside existing
        # keypoints. Without keypoints it would be the only source, and on
        # blocky paprika it manages just 66% on the question of which end holds
        # the stem - "no angle" is a more honest answer than a coin toss.
        use_shape = self._use_shape_crosscheck and stem is not None and blossom is not None

        # Unpickable is an outcome, not a failure. No angle is computed here on
        # purpose: even a good estimate would not help the robot, because it
        # cannot pick up an upside-down fruit and turn it over. Returning a
        # number would imply an action that does not exist.
        reason = str(detection.get("unpickable_reason") or "")
        result: Optional[Orientation] = None

        if reason == "no_stem" and self._stemless_shape_fallback:
            # No stem found, but the fruit is fully in frame. Before writing
            # it off, ask the one estimator that never looks at the stem.
            # Returns None (and falls through to the plain reject below)
            # whenever the silhouette turns out to be round - that is a
            # standing or upside-down fruit, and it should be detected, not
            # guessed at.
            result = self._shape_only_estimate(frame, bbox)

        if reason and result is None:
            pose = _UNPICKABLE_POSES.get(reason, orient.POSE_UPSIDE_DOWN)
            unusable = Orientation(
                source="classical",
                pose=pose,
                stem_present=False,
                notes=[reason],
            )
            x1, y1, x2, y2 = bbox
            return {
                "label": _UNPICKABLE_LABELS.get(pose, "unusable"),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                "confidence": float(detection.get("confidence", 0.0)),
                "keypoints": {},
                "orientation": unusable.to_dict(),
                "angle_plc": None,
                # Unchanged by the stem_not_found split, and deliberately so:
                # this is a reporting distinction, not a policy one. A fruit
                # whose stem cannot be found is exactly as unplaceable as it
                # was before it had its own name.
                "placement": PLACEMENT_REJECT,
                "simulated": False,
                "colour": detection.get("colour", ""),
                "stem_method": detection.get("stem_method", "none"),
            }

        if result is None:
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

        # A stem direction that moves with the light is not a direction. This
        # is measured per fruit by re-running the detection at two other gains,
        # so it reflects this fruit under this light rather than an average
        # taken over the dataset.
        spread = float(detection.get("stem_spread_deg", 0.0) or 0.0)
        if placement == PLACEMENT_PLACE and spread > self._max_stem_spread_deg:
            placement = PLACEMENT_REORIENT
            result.notes.append(f"stem_unstable={spread:.0f}deg")
            result.flip_confidence = min(result.flip_confidence, 0.3)

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
            # Diagnostics, not a decision. Without these the logs cannot show
            # whether an erratic result came from the fruit colour or from the
            # stem method, which is exactly what you want to know when
            # something is behaving inconsistently.
            "colour": detection.get("colour", ""),
            "stem_method": detection.get("stem_method", "none"),
            "stem_quality": detection.get("stem_quality", 0.0),
            "stem_spread_deg": round(float(detection.get("stem_spread_deg", 0.0) or 0.0), 1),
        }

    def _pick_primary(
        self,
        detections: list[dict],
        frame_width: int = 0,
        frame_height: int = 0,
    ) -> Optional[dict]:
        """Choose the fruit the actuator acts on.

        "largest" is the default because on a belt the biggest silhouette is
        normally the one fully in view, rather than one half-entering frame
        whose angle is being measured from a partial fruit.

        Args:
            frame_width, frame_height: size of the frame the detections came
                from. Only the "centered" rule needs them, but it needs them
                absolutely: centre-ness is meaningless without knowing where
                the centre is.
        """
        if not detections:
            return None

        placeable = [d for d in detections if d["placement"] == PLACEMENT_PLACE]
        # An unusable fruit must never displace a usable one, not even if it is
        # larger: what matters is what the robot can actually act on.
        usable = [d for d in detections if d["placement"] != PLACEMENT_REJECT]
        pool = placeable or usable or detections

        if self._primary_rule == "confidence":
            return max(pool, key=lambda d: d["orientation"].get("confidence", 0.0))

        if self._primary_rule == "centered":
            # Distance from the middle of the frame. This used to measure from
            # the image ORIGIN, which made "centered" quietly mean "nearest the
            # top-left corner" - i.e. it preferred the fruit just entering the
            # frame, the exact opposite of the intent, and on a belt running
            # left to right it picked a different fruit every time. A rule that
            # names the centre has to be told where the centre is, which is why
            # the frame size is now a parameter rather than an assumption.
            if frame_width > 0 and frame_height > 0:
                cx, cy = frame_width / 2.0, frame_height / 2.0
                return min(
                    pool,
                    key=lambda d: math.hypot(d["center"][0] - cx, d["center"][1] - cy),
                )
            # No frame size means the centre is unknowable. Fall through to
            # "largest" rather than silently reinstating the origin bug.
            log.warning(
                "primary_rule='centered' needs the frame size; falling back to 'largest'"
            )

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
        frame_height, frame_width = frame.shape[:2]
        primary = self._pick_primary(detections, frame_width, frame_height)

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