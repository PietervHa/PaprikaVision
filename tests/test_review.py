"""The "human check needed" verdict.

Between min_usable_confidence and min_angle_confidence the machine has a
measurement it does not trust enough to act on - that is "unknown", and the
angle is still shown so an operator can see what it was leaning towards. Below
the floor there is no measurement: the numbers are whatever the noise happened
to be, and showing one invites somebody to read meaning into it.
"""

import numpy as np
import pytest

from backend.core.paprika_engine import (
    PaprikaEngine, PLACEMENT_PLACE, PLACEMENT_REVIEW, PLACEMENT_UNKNOWN,
)
from backend.detection.paprika import orientation as orient
from backend.detection.paprika.orientation import Orientation
from backend.core.state import AppState


def _engine(**policy):
    block = {
        "backend": "shape",
        "shape": {"saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
                  "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {"min_angle_confidence": 0.45, "min_flip_confidence": 0.40,
                   "min_usable_confidence": 0.15, **policy},
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
    }
    return PaprikaEngine({"paprika": block})


def _lying(confidence, flip=0.9, angle=10.0):
    return Orientation(pose=orient.POSE_LYING, angle_deg=angle,
                       confidence=confidence, flip_confidence=flip)


@pytest.mark.parametrize("confidence", [0.0, 0.05, 0.12, 0.1499])
def test_below_the_floor_is_a_review(confidence):
    assert _engine()._placement_for(_lying(confidence)) == PLACEMENT_REVIEW


@pytest.mark.parametrize("confidence", [0.15, 0.30, 0.44])
def test_between_the_thresholds_is_still_unknown(confidence):
    """The middle band is unchanged - this must not quietly swallow it."""
    assert _engine()._placement_for(_lying(confidence)) == PLACEMENT_UNKNOWN


def test_above_both_still_places():
    assert _engine()._placement_for(_lying(0.9)) == PLACEMENT_PLACE


def test_the_floor_is_configurable():
    assert _engine(min_usable_confidence=0.40)._placement_for(_lying(0.30)) \
        == PLACEMENT_REVIEW
    assert _engine(min_usable_confidence=0.0)._placement_for(_lying(0.01)) \
        == PLACEMENT_UNKNOWN


def test_the_label_carries_no_number():
    """The whole point. An angle next to "uncertain" still gets read as an
    angle, so the label must not offer one."""
    label = PaprikaEngine._label_for(_lying(0.05), PLACEMENT_REVIEW)
    assert label == "human check needed"
    assert not any(ch.isdigit() for ch in label)


def test_review_still_labels_a_standing_fruit_as_review():
    """Pose wording must not leak past the verdict: "standing, stem up" next to
    a withheld angle reads as a measurement that was never made."""
    result = Orientation(pose=orient.POSE_STANDING_STEM_UP, angle_deg=10.0,
                         confidence=0.05)
    assert PaprikaEngine._label_for(result, PLACEMENT_REVIEW) == "human check needed"


def test_counters_keep_reviews_apart_from_rejects():
    """A rising reject count is about the fruit; a rising review count is about
    the detector. Summing them hides which one you are looking at."""
    state = AppState()
    assert state.counters["review"] == 0
    state.increment_counter("NOK", placement="review", pose="lying")
    state.increment_counter("NOK", placement="reject", pose="incomplete")
    assert state.counters["review"] == 1
    assert state.counters["reject"] == 1
    state.reset_counters()
    assert state.counters["review"] == 0


def test_review_has_its_own_overlay_colour():
    from backend.utils.annotate import color_for_placement
    review = color_for_placement("review")
    assert review not in (
        color_for_placement("place"),
        color_for_placement("reorient"),
        color_for_placement("reject"),
        color_for_placement("unknown"),
    )
