"""
Belt crop

Works out which part of the frame is actually the belt, so the HMI can show
that instead of the whole sensor.

Why crop at all
---------------
The camera sees more than the product: pale structural strips down both sides,
machine frame, whatever is behind the line. None of it is inspectable and all
of it is transmitted thirty times a second and then scaled down in the browser,
so the part the operator actually needs ends up small. Cropping to the belt
makes the fruit fill the view and cuts the JPEG down at the same time.

The box is cached rather than recomputed per frame. A belt does not move, so
finding it thirty times a second would be pure waste - but it is refreshed
periodically anyway, because a camera can be nudged and a stale crop would
quietly hide product rather than fail visibly.

The crop is display-only. Detection keeps running on the full frame, and
coordinates stay in full-frame terms right up to the point of drawing. Cropping
before detection would have been faster still, but it would mean the boxes and
the angles were measured in one coordinate system and the PLC told about
another - and that is the kind of mismatch that is invisible until a robot
places a paprika in the wrong spot.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from backend.detection.paprika import classical
from backend.utils.logger import get_logger

log = get_logger(__name__)


class BeltCrop:
    """Finds and caches the belt's bounding box.

    Args:
        belt_hue:   OpenCV hue range of the belt.
        margin:     fraction of the belt size kept around it, so fruit hanging
                    over the edge stay visible rather than being cut in half by
                    the very view meant to show them.
        refresh_s:  how often the box is recomputed.
    """

    def __init__(
        self,
        belt_hue: tuple[int, int] = classical.DEFAULT_BELT_HUE,
        margin: float = 0.02,
        refresh_s: float = 10.0,
    ) -> None:
        self._belt_hue = belt_hue
        self._margin = float(margin)
        self._refresh_s = float(refresh_s)

        self._lock = threading.Lock()
        self._box: Optional[tuple[int, int, int, int]] = None
        self._computed_at = 0.0
        self._shape: Optional[tuple] = None
        self._failed_once = False

    def _compute(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        belt = classical.belt_mask_raw(frame, self._belt_hue)
        if belt is None:
            return None

        ys, xs = np.nonzero(belt)
        if len(xs) < 100:
            return None

        # The mask is held at BELT_SCALE, so its coordinates are scaled back up
        # to frame coordinates here.
        scale = 1.0 / classical.BELT_SCALE
        height, width = frame.shape[:2]
        x1, x2 = int(xs.min() * scale), int(xs.max() * scale)
        y1, y2 = int(ys.min() * scale), int(ys.max() * scale)

        pad_x = int((x2 - x1) * self._margin)
        pad_y = int((y2 - y1) * self._margin)

        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )

    def box_for(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Current crop box, recomputing when the cache has expired."""
        if frame is None or frame.size == 0:
            return None

        now = time.time()
        shape = frame.shape[:2]
        with self._lock:
            # A cached box belongs to the frame size it was measured on. Reusing
            # it across a size change hands back a box that runs off the image.
            same_frame = self._shape == shape
            fresh = (
                same_frame
                and self._box is not None
                and (now - self._computed_at) < self._refresh_s
            )
            if fresh:
                return self._box

        box = self._compute(frame)

        with self._lock:
            self._computed_at = now
            self._shape = shape
            if box is None:
                # Show the whole frame rather than guessing. An operator seeing
                # the full sensor knows something is off; an operator seeing a
                # confidently wrong crop does not.
                self._box = None
                if not self._failed_once:
                    log.warning(
                        "No belt found for the HMI crop - showing the full frame. "
                        "Check the camera framing and paprika.shape.belt_hue."
                    )
                    self._failed_once = True
            else:
                self._box = box
                self._failed_once = False
            return self._box

    def invalidate(self) -> None:
        """Force a recompute, e.g. after the camera has been rotated."""
        with self._lock:
            self._box = None
            self._computed_at = 0.0
            self._shape = None


def crop_frame_and_detections(
    frame: np.ndarray,
    detections: list[dict],
    box: Optional[tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[dict]]:
    """Crop a frame and shift the detections to match.

    Returns copies: the caller's detections are the same objects the overlay
    thread and the PLC path read, and shifting them in place would move the
    coordinates the machine acts on to suit a display decision.
    """
    if box is None:
        return frame, detections

    x1, y1, x2, y2 = box
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return frame, detections

    shifted = []
    for detection in detections:
        moved = dict(detection)

        bbox = detection.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            moved["bbox"] = [bbox[0] - x1, bbox[1] - y1, bbox[2] - x1, bbox[3] - y1]

        center = detection.get("center")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            moved["center"] = [center[0] - x1, center[1] - y1]

        keypoints = detection.get("keypoints")
        if isinstance(keypoints, dict):
            moved["keypoints"] = {
                name: {**kp, "x": kp["x"] - x1, "y": kp["y"] - y1}
                for name, kp in keypoints.items()
                if isinstance(kp, dict) and "x" in kp and "y" in kp
            }

        shifted.append(moved)

    return cropped, shifted
