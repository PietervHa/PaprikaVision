"""backend/utils/annotate.py

Draws the orientation result onto the live frame.

The arrow is the important part, not the box. A bounding box says a paprika is
present; the operator already knew that. The arrow says which way the machine
thinks the stem points, which is the one thing they cannot verify by eye
against the HMI unless it is drawn. If the arrow is wrong, it is wrong in a way
anybody standing at the machine can see instantly - which is exactly the
property you want while commissioning.

Colour encodes the placement decision, not the fruit. Green means the machine
will act on this angle; amber means it will send the fruit round again; red
means it will not act at all. Colouring by fruit colour would be prettier and
would tell the operator nothing they need.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

# BGR.
_COLOR_PLACE = (90, 190, 70)
_COLOR_REORIENT = (60, 180, 235)
_COLOR_REJECT = (60, 60, 220)
_COLOR_UNKNOWN = (170, 170, 170)
_COLOR_STEM_KP = (80, 220, 80)
_COLOR_BLOSSOM_KP = (220, 140, 60)
_COLOR_OCCLUDED = (120, 120, 120)

_PLACEMENT_COLORS = {
    "place": _COLOR_PLACE,
    "reorient": _COLOR_REORIENT,
    "reject": _COLOR_REJECT,
    "unknown": _COLOR_UNKNOWN,
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for_placement(placement: str) -> tuple[int, int, int]:
    return _PLACEMENT_COLORS.get(str(placement or "").lower(), _COLOR_UNKNOWN)


def _text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    return (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)


def _chip(frame, x, y, text, bg, font_scale=0.5):
    """Filled label chip with its top-left at (x, y), clamped into frame."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, 1)
    h, w = frame.shape[:2]
    x = max(0, min(w - tw - 12, x))
    y = max(0, min(h - th - baseline - 8, y))
    cv2.rectangle(frame, (x, y), (x + tw + 10, y + th + baseline + 6), bg, -1)
    cv2.putText(
        frame, text, (x + 5, y + th + 3), _FONT, font_scale, _text_color_for(bg), 1, cv2.LINE_AA
    )
    return y + th + baseline + 6


def draw_orientation_arrow(
    frame: np.ndarray,
    center: tuple[int, int],
    angle_deg: float,
    length_px: float,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """Draw an arrow from the fruit centre toward the stem.

    `angle_deg` follows the engine's screen convention: 0 points right, 90
    points up the image. The sign flip on dy converts back to image space.
    """
    radians = math.radians(float(angle_deg))
    dx = math.cos(radians) * length_px
    dy = -math.sin(radians) * length_px

    cx, cy = int(center[0]), int(center[1])
    tip = (int(round(cx + dx)), int(round(cy + dy)))
    tail = (int(round(cx - dx * 0.35)), int(round(cy - dy * 0.35)))

    cv2.arrowedLine(frame, tail, tip, color, thickness, cv2.LINE_AA, tipLength=0.28)


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    primary_center: Optional[list] = None,
    show_keypoints: bool = True,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw every detection. Modifies and returns `frame`."""
    if frame is None or not isinstance(frame, np.ndarray) or not detections:
        return frame

    for detection in detections:
        if not isinstance(detection, dict):
            continue

        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = (int(v) for v in bbox)
        placement = str(detection.get("placement", "unknown"))
        color = color_for_placement(placement)

        is_primary = (
            primary_center is not None and detection.get("center") == list(primary_center)
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_primary else 1)

        orientation = detection.get("orientation") or {}
        angle = orientation.get("angle_deg")
        center = detection.get("center") or [(x1 + x2) // 2, (y1 + y2) // 2]

        if angle is not None:
            arrow_len = max(24.0, min(x2 - x1, y2 - y1) * 0.55)
            draw_orientation_arrow(frame, center, angle, arrow_len, color, 3 if is_primary else 2)
        else:
            # No angle is a real state, not a rendering gap - mark it, so an
            # empty box never reads as "the arrow just failed to draw".
            radius = max(10, int(min(x2 - x1, y2 - y1) * 0.18))
            cv2.circle(frame, (int(center[0]), int(center[1])), radius, color, 2)

        if show_keypoints:
            for name, kp in (detection.get("keypoints") or {}).items():
                if not isinstance(kp, dict):
                    continue
                px, py = int(kp.get("x", 0)), int(kp.get("y", 0))
                visible = bool(kp.get("visible", True))
                if not visible:
                    kp_color = _COLOR_OCCLUDED
                elif name == "stem_end":
                    kp_color = _COLOR_STEM_KP
                else:
                    kp_color = _COLOR_BLOSSOM_KP
                # Hollow marker for an occluded landmark, so "inferred position"
                # is visually distinct from "actually seen".
                cv2.circle(frame, (px, py), 5, kp_color, -1 if visible else 2)
                cv2.circle(frame, (px, py), 6, (20, 20, 20), 1)

        label = str(detection.get("label", placement))
        next_y = _chip(frame, x1, y1 - 22, label, color, font_scale)

        flip_conf = orientation.get("flip_confidence", 0.0)
        if angle is not None and flip_conf < 0.4:
            # The single most useful warning on this screen: the axis is fine
            # but the machine is not sure which end the stem is on.
            _chip(frame, x1, next_y + 2, "stem end uncertain", _COLOR_REORIENT, font_scale - 0.06)

    return frame


def draw_hud(
    frame: np.ndarray,
    lines: list[str],
    color: tuple[int, int, int] = (35, 35, 35),
    font_scale: float = 0.5,
) -> np.ndarray:
    """Stacked status lines in the top-left corner."""
    if frame is None or not lines:
        return frame
    y = 4
    for line in lines:
        y = _chip(frame, 4, y, line, color, font_scale) + 2
    return frame
