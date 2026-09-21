"""The silhouette overruling a "lying" verdict on a standing fruit.

Keypoints only report standing when the two landmarks nearly coincide, and the
pose model has almost never seen that - 5.5% standing fruit and 0% occluded
landmarks in training. It spreads them apart instead, so a standing fruit comes
out "lying" with a guessed angle, or "human check needed" when the landmarks
are uncertain. The silhouette does not share that blind spot.
"""

import numpy as np
import pytest

from backend.core.paprika_engine import PaprikaEngine
from backend.detection.paprika import end_on, orientation as orient
from backend.detection.paprika.orientation import Keypoint, Orientation


def _engine(**policy):
    block = {
        "backend": "shape",
        "shape": {"saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
                  "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {"min_angle_confidence": 0.45, "min_flip_confidence": 0.40},
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
    }
    block.update(policy)
    return PaprikaEngine({"paprika": block})


def _lying(span=60.0, confidence=0.3):
    return Orientation(source="keypoints", pose=orient.POSE_LYING, angle_deg=42.0,
                       axis_deg=42.0, stem_span_px=span, confidence=confidence,
                       flip_confidence=confidence)


FRAME = np.zeros((200, 200, 3), np.uint8)
BBOX = (0, 0, 200, 200)


def test_a_confident_well_separated_pair_is_left_alone(monkeypatch):
    """The gate that makes this safe. On fruit with strong stem evidence the
    end-on classifier scores 0.84-0.98 - higher than the fruit it should catch
    - so it must never overrule a convincing reading."""
    monkeypatch.setattr(end_on, "end_on_probability", lambda *a, **k: 0.99)
    engine = _engine()
    result = _lying(span=180.0, confidence=0.95)     # long baseline, confident
    out = engine._reconsider_standing(FRAME, BBOX, result, None)
    assert out.pose == orient.POSE_LYING
    assert out.angle_deg == 42.0


def test_an_unconvincing_lying_verdict_is_reconsidered(monkeypatch):
    monkeypatch.setattr(end_on, "end_on_probability", lambda *a, **k: 0.99)
    monkeypatch.setattr(orient, "segment_fruit",
                        lambda *a, **k: np.ones((5, 5), np.uint8) * 255)
    engine = _engine()
    out = engine._reconsider_standing(FRAME, BBOX, _lying(span=60.0), None)
    assert out.pose in (orient.POSE_STANDING_STEM_UP, orient.POSE_STANDING_STEM_DOWN)


def test_the_angle_is_withdrawn_not_just_relabelled(monkeypatch):
    """A rotation angle for a fruit standing on its end is not an unknown
    quantity - it is not a quantity. Leaving a number attached to a standing
    verdict is how it gets read as one."""
    monkeypatch.setattr(end_on, "end_on_probability", lambda *a, **k: 0.99)
    monkeypatch.setattr(orient, "segment_fruit",
                        lambda *a, **k: np.ones((5, 5), np.uint8) * 255)
    out = _engine()._reconsider_standing(FRAME, BBOX, _lying(), None)
    assert out.angle_deg is None
    assert out.axis_deg is None
    assert any("standing_from_silhouette" in n for n in out.notes)


@pytest.mark.parametrize("visible,expected", [
    (True, orient.POSE_STANDING_STEM_UP),
    (False, orient.POSE_STANDING_STEM_DOWN),
])
def test_stem_visibility_decides_which_end_is_up(monkeypatch, visible, expected):
    """Straight up versus upside down. The model reports landmark visibility
    usefully even when it misplaces the landmark - that is the occluded case
    ANNOTATION_SPEC section 3 trains for."""
    monkeypatch.setattr(end_on, "end_on_probability", lambda *a, **k: 0.99)
    monkeypatch.setattr(orient, "segment_fruit",
                        lambda *a, **k: np.ones((5, 5), np.uint8) * 255)
    stem = Keypoint(x=10.0, y=10.0, confidence=0.9, visible=visible)
    out = _engine()._reconsider_standing(FRAME, BBOX, _lying(), stem)
    assert out.pose == expected


def test_a_low_end_on_score_leaves_the_verdict_alone(monkeypatch):
    monkeypatch.setattr(end_on, "end_on_probability", lambda *a, **k: 0.02)
    monkeypatch.setattr(orient, "segment_fruit",
                        lambda *a, **k: np.ones((5, 5), np.uint8) * 255)
    out = _engine()._reconsider_standing(FRAME, BBOX, _lying(), None)
    assert out.pose == orient.POSE_LYING
    assert out.angle_deg == 42.0
    assert any(n.startswith("end_on_p=") for n in out.notes), "must record what it saw"


def test_an_unreadable_silhouette_changes_nothing(monkeypatch):
    monkeypatch.setattr(end_on, "end_on_probability", lambda *a, **k: None)
    monkeypatch.setattr(orient, "segment_fruit",
                        lambda *a, **k: np.ones((5, 5), np.uint8) * 255)
    out = _engine()._reconsider_standing(FRAME, BBOX, _lying(), None)
    assert out.pose == orient.POSE_LYING


def test_it_can_be_switched_off():
    engine = _engine(end_on_overrides_lying=False)
    assert engine._end_on_overrides_lying is False
    out = engine._reconsider_standing(FRAME, BBOX, _lying(), None)
    assert out.pose == orient.POSE_LYING


def test_a_pose_that_is_already_standing_is_untouched(monkeypatch):
    monkeypatch.setattr(end_on, "end_on_probability", lambda *a, **k: 0.99)
    result = Orientation(source="keypoints", pose=orient.POSE_STANDING_STEM_UP)
    out = _engine()._reconsider_standing(FRAME, BBOX, result, None)
    assert out.pose == orient.POSE_STANDING_STEM_UP
