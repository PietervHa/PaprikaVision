"""Refusing to place when the two estimators point different ways.

Measured over 802 hand-labelled fruit: the keypoint error against the clicked
truth climbs monotonically with how far the silhouette disagrees, from a 3
degree median under 10 degrees of disagreement to 135 degrees above 120. So
disagreement is not a tie to be broken - neither estimator is reliably better -
it is a warning that one of them is wrong and nothing here can say which.
"""

import pytest

from backend.core.paprika_engine import (
    PaprikaEngine, PLACEMENT_PLACE, PLACEMENT_REORIENT,
)
from backend.detection.paprika import orientation as orient
from backend.detection.paprika.orientation import Orientation


def _engine(**policy):
    block = {
        "backend": "shape",
        "shape": {"saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
                  "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {"min_angle_confidence": 0.45, "min_flip_confidence": 0.40, **policy},
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
    }
    return PaprikaEngine({"paprika": block})


def _good(agreement=None):
    return Orientation(source="fused", pose=orient.POSE_LYING, angle_deg=42.0,
                       confidence=0.9, flip_confidence=0.9, agreement_deg=agreement)


def test_agreeing_estimators_still_place():
    assert _engine()._placement_for(_good(agreement=4.0)) == PLACEMENT_PLACE


def test_a_large_disagreement_is_not_placed():
    """The band where 77% of fruit were more than 20 degrees out."""
    assert _engine()._placement_for(_good(agreement=50.0)) == PLACEMENT_REORIENT


def test_it_reorients_rather_than_rejects():
    """The fruit is fine - two readings simply cannot both be right, and
    another pass may settle it. Binning it would be throwing away produce for
    a disagreement between two pieces of software."""
    assert _engine()._placement_for(_good(agreement=90.0)) != "reject"


def test_confidence_alone_would_not_have_caught_it():
    """fuse() cuts confidence by 0.6 on a disagreement, but 0.9 x 0.6 = 0.54,
    which still clears min_angle_confidence of 0.45. That is why this gate has
    to be a hard stop rather than another confidence nudge."""
    engine = _engine()
    softened = _good(agreement=None)
    softened.confidence = 0.9 * 0.6
    assert engine._placement_for(softened) == PLACEMENT_PLACE, (
        "the confidence cut on its own leaves the fruit placeable"
    )


def test_no_second_opinion_means_no_gate():
    """agreement_deg is None whenever the silhouette could not be measured.
    Absence of a second opinion is not disagreement."""
    assert _engine()._placement_for(_good(agreement=None)) == PLACEMENT_PLACE


@pytest.mark.parametrize("threshold,agreement,expected", [
    (20.0, 25.0, PLACEMENT_REORIENT),
    (45.0, 25.0, PLACEMENT_PLACE),
])
def test_the_threshold_is_configurable(threshold, agreement, expected):
    engine = _engine(max_estimator_disagreement_deg=threshold)
    assert engine._placement_for(_good(agreement=agreement)) == expected


def test_a_fruit_with_no_angle_is_unaffected():
    """The gate is about two directions disagreeing. With no angle there is no
    direction, and the existing verdicts own that case."""
    result = Orientation(source="fused", pose=orient.POSE_LYING, angle_deg=None,
                         confidence=0.9, agreement_deg=90.0)
    assert _engine()._placement_for(result) != PLACEMENT_PLACE
