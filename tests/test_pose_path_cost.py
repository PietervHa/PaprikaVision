"""The marginal-stem gate must not fire on a backend that has no stem blob.

stem_area_ratio measures a morphology blob against the fruit it sits on. The
pose backend has no such blob and emits no such field, so reading the default
of 0.0 made every pose detection look "marginal": the end-on classifier ran on
all of them - about 5ms a fruit - and any scoring above the threshold had its
landmarks discarded and was reported stemless.

A missing measurement is not a small measurement.
"""

import numpy as np
import pytest

from backend.core.paprika_engine import PaprikaEngine
from backend.detection.paprika import end_on
from backend.detection.paprika.orientation import Keypoint


def _engine():
    return PaprikaEngine({"paprika": {
        "backend": "pose",
        "shape": {"saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
                  "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {"min_angle_confidence": 0.45, "min_flip_confidence": 0.40,
                   "detect_end_on": True},
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
        "end_on_overrides_lying": False,
    }})


def _detection(stem_method):
    return {
        "bbox": (40, 40, 260, 240), "center": (150.0, 140.0), "confidence": 0.9,
        "colour": "green", "stem_method": stem_method,
        "keypoints": {
            "stem_end": Keypoint(x=240.0, y=60.0, confidence=0.85, visible=True),
            "blossom_end": Keypoint(x=60.0, y=220.0, confidence=0.85, visible=True),
        },
    }


@pytest.fixture
def frame():
    import cv2
    image = np.full((320, 320, 3), (190, 110, 45), np.uint8)
    cv2.ellipse(image, (150, 140), (105, 95), 20, 0, 360, (55, 165, 70), -1)
    return image


def test_the_classifier_is_not_run_for_a_pose_detection(frame, monkeypatch):
    calls = {"n": 0}

    def counted(*args, **kwargs):
        calls["n"] += 1
        return 0.99

    monkeypatch.setattr(end_on, "end_on_probability", counted)
    _engine()._evaluate_one(frame, _detection("pose"))
    assert calls["n"] == 0, "pose detections have no stem blob to call marginal"


def test_it_still_runs_where_the_ratio_means_something(frame, monkeypatch):
    """The gate exists for a real failure - a morphology speck being read as a
    calyx - and must keep working on the backend that has one."""
    calls = {"n": 0}

    def counted(*args, **kwargs):
        calls["n"] += 1
        return 0.01

    monkeypatch.setattr(end_on, "end_on_probability", counted)
    detection = _detection("morphology")
    detection["stem_area_ratio"] = 0.002          # a speck
    _engine()._evaluate_one(frame, detection)
    assert calls["n"] == 1


def test_a_solid_morphology_stem_is_not_second_guessed(frame, monkeypatch):
    calls = {"n": 0}

    def counted(*args, **kwargs):
        calls["n"] += 1
        return 0.99

    monkeypatch.setattr(end_on, "end_on_probability", counted)
    detection = _detection("morphology")
    detection["stem_area_ratio"] = 0.06           # a real calyx
    _engine()._evaluate_one(frame, detection)
    assert calls["n"] == 0


def test_a_half_visible_fruit_is_flagged_incomplete():
    """_UNPICKABLE_POSES keys off "edge_clipped", and the pose path never set
    it - so a fruit running off the frame was measured as though it were whole,
    landmarks partly outside the image and all, and came out "standing, stem
    up" instead of "incomplete in frame".

    Threshold calibrated against classical.edge_cut_ratio on 170 fruit from 99
    belt frames: clipped fruit score 0.231-0.406, clean fruit 0.000-0.269.
    """
    from backend.detection.paprika.pose_detector import (
        PaprikaDetector, BBOX_EDGE_CUT_THRESHOLD,
    )
    shape = (1092, 885, 3)
    assert PaprikaDetector._bbox_edge_cut((0, 300, 200, 600), shape) > BBOX_EDGE_CUT_THRESHOLD
    assert PaprikaDetector._bbox_edge_cut((300, 300, 500, 600), shape) < BBOX_EDGE_CUT_THRESHOLD


def test_confidence_no_longer_opens_the_standing_gate():
    """A fruit lying down with both landmarks correctly placed far apart is not
    standing, whatever the model's confidence in them. Pose confidences sit
    around 0.5 on ordinary fruit, so "or confidence < 0.55" opened this gate on
    most of the crop - and the end-on classifier, which scores ordinary stemmed
    fruit 0.84-0.98, then called them standing with the stem up.
    """
    import numpy as np
    from backend.core.paprika_engine import PaprikaEngine
    from backend.detection.paprika import end_on, orientation as orient
    from backend.detection.paprika.orientation import Orientation

    engine = PaprikaEngine({"paprika": {
        "backend": "pose",
        "shape": {"saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
                  "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {"min_angle_confidence": 0.45, "min_flip_confidence": 0.40},
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
    }})
    calls = {"n": 0}
    original = end_on.end_on_probability
    end_on.end_on_probability = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), 0.99)[1]
    try:
        lying = Orientation(source="keypoints", pose=orient.POSE_LYING,
                            angle_deg=40.0, stem_span_px=260.0,
                            confidence=0.50, flip_confidence=0.50)
        out = engine._reconsider_standing(np.zeros((400, 400, 3), np.uint8),
                                          (0, 0, 300, 300), lying, None)
    finally:
        end_on.end_on_probability = original

    assert calls["n"] == 0, "well-separated landmarks must not be reconsidered"
    assert out.pose == orient.POSE_LYING
