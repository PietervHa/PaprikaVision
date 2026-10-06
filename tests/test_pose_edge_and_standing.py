"""Two defects seen on the belt with backend: pose.

1. A fruit running off the frame was placed with a confident angle. _detect_pose
   set an "edge_clipped" key that nothing in the codebase reads - the engine
   keys off unpickable_reason - so the flag was inert.

2. A fruit lying down with both landmarks correctly placed was reported
   "standing, stem up". The silhouette reconsideration opened on any fruit whose
   landmarks sat closer than 0.45 of the box diagonal; a real lying fruit
   measured about 0.43 and fell through. A genuinely standing fruit is under
   0.18 - that is what min_span_ratio means.
"""

import numpy as np
import pytest

from backend.core.paprika_engine import PaprikaEngine, _UNPICKABLE_POSES
from backend.detection.paprika import end_on, orientation as orient
from backend.detection.paprika.classical import REASON_EDGE_CLIPPED
from backend.detection.paprika.orientation import Keypoint, Orientation
from backend.detection.paprika.pose_detector import (
    PaprikaDetector, BBOX_EDGE_CUT_THRESHOLD,
)


def _engine(**block):
    base = {
        "backend": "pose",
        "shape": {"saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
                  "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {"min_angle_confidence": 0.45, "min_flip_confidence": 0.40},
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
    }
    base.update(block)
    return PaprikaEngine({"paprika": base})


# ------------------------------------------------------------ edge clipping

def test_the_reason_the_engine_reads_is_the_one_that_maps():
    assert REASON_EDGE_CLIPPED in _UNPICKABLE_POSES
    assert _UNPICKABLE_POSES[REASON_EDGE_CLIPPED] == orient.POSE_INCOMPLETE


@pytest.mark.parametrize("bbox,clipped", [
    ((0, 300, 200, 600), True),        # runs off the left
    ((458, 0, 915, 280), True),        # cut at the top, as in the belt capture
    ((300, 300, 500, 600), False),     # well inside
])
def test_bbox_edge_measure(bbox, clipped):
    ratio = PaprikaDetector._bbox_edge_cut(bbox, (1092, 885, 3))
    assert (ratio > BBOX_EDGE_CUT_THRESHOLD) is clipped


def test_a_clipped_fruit_is_not_placed_and_carries_no_angle():
    detection = {
        "bbox": (0, 0, 200, 300), "center": (100.0, 150.0), "confidence": 0.95,
        "colour": "red", "stem_method": "pose",
        "unpickable_reason": REASON_EDGE_CLIPPED,
        "keypoints": {
            "stem_end": Keypoint(x=180.0, y=40.0, confidence=0.9, visible=True),
            "blossom_end": Keypoint(x=20.0, y=260.0, confidence=0.9, visible=True),
        },
    }
    out = _engine()._evaluate_one(np.zeros((400, 400, 3), np.uint8), detection)
    assert out["orientation"]["pose"] == orient.POSE_INCOMPLETE
    assert out["placement"] != "place"
    assert out["orientation"]["angle_deg"] is None


# ---------------------------------------------------------------- standing

def _lying(span):
    return Orientation(source="keypoints", pose=orient.POSE_LYING, angle_deg=40.0,
                       axis_deg=40.0, stem_span_px=span,
                       confidence=0.86, flip_confidence=1.0)


def test_a_lying_fruit_at_043_is_left_alone(monkeypatch):
    """The exact case from the belt: correct landmarks, span 0.43 of the
    diagonal, reported "standing, stem up"."""
    monkeypatch.setattr(end_on, "end_on_probability", lambda *a, **k: 0.99)
    monkeypatch.setattr(orient, "segment_fruit",
                        lambda *a, **k: np.ones((5, 5), np.uint8) * 255)
    bbox = (0, 0, 438, 500)                       # diagonal ~665
    out = _engine()._reconsider_standing(
        np.zeros((600, 600, 3), np.uint8), bbox, _lying(287.0), None
    )
    assert out.pose == orient.POSE_LYING, "0.43 of the diagonal is not standing"
    assert out.angle_deg == 40.0


def test_a_genuinely_standing_fruit_is_still_reconsidered(monkeypatch):
    """Landmarks nearly on top of each other - what standing looks like."""
    monkeypatch.setattr(end_on, "end_on_probability", lambda *a, **k: 0.99)
    monkeypatch.setattr(orient, "segment_fruit",
                        lambda *a, **k: np.ones((5, 5), np.uint8) * 255)
    engine = _engine(end_on_overrides_lying=True, end_on_span_ratio_max=0.22)
    out = engine._reconsider_standing(
        np.zeros((600, 600, 3), np.uint8), (0, 0, 438, 500), _lying(90.0), None
    )
    assert out.pose in (orient.POSE_STANDING_STEM_UP, orient.POSE_STANDING_STEM_DOWN)
    assert out.angle_deg is None


def test_the_uncertain_shape_crosscheck_does_not_fire_on_pose():
    """A dormant path that woke up because an unrelated field got populated.

    uncertain_shape_crosscheck is entirely about the CLASSICAL stem search -
    hue finding the stem on red, morphology plus a brightness self-check
    carrying green. The pose model has neither.

    It was gated on the fruit's colour, and pose detections used to report
    colour="" so it never fired. Giving them a real colour - a fix for a
    different problem - switched it on, and it answered 6 of 181 placed fruit
    with the whole tail: 93, 68, 54 and 37 degrees out, against a keypoint p90
    of 10.8.
    """
    import copy
    import cv2
    from backend.core.paprika_engine import PaprikaEngine
    from backend.detection.paprika import orientation as orient
    from backend.detection.paprika.orientation import Keypoint, Orientation

    block = {
        "shape": {"saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
                  "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {"min_angle_confidence": 0.45, "min_flip_confidence": 0.40,
                   "uncertain_shape_crosscheck": True,
                   "uncertain_shape_colours": ["green"]},
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
    }
    detection = {
        "bbox": (40, 40, 260, 240), "center": (150.0, 140.0), "confidence": 0.9,
        "colour": "green", "stem_method": "pose",
        "keypoints": {
            "stem_end": Keypoint(x=200.0, y=100.0, confidence=0.30, visible=True),
            "blossom_end": Keypoint(x=100.0, y=180.0, confidence=0.30, visible=True),
        },
    }
    frame = np.full((320, 320, 3), (190, 110, 45), np.uint8)
    cv2.ellipse(frame, (150, 140), (105, 95), 20, 0, 360, (55, 165, 70), -1)

    original = orient.shape_orientation
    orient.shape_orientation = lambda *a, **k: Orientation(
        source="shape", pose=orient.POSE_LYING, angle_deg=95.0, axis_deg=95.0,
        confidence=0.9, flip_confidence=0.9)
    try:
        results = {}
        for backend in ("pose", "shape"):
            engine = PaprikaEngine({"paprika": {**block, "backend": backend}})
            out = engine._evaluate_one(frame, copy.deepcopy(detection))
            results[backend] = out["orientation"].get("source")
    finally:
        orient.shape_orientation = original

    assert results["pose"] == "keypoints", "the silhouette must not answer for pose"
    assert results["shape"] == "fused", "but it must still help the shape backend"
