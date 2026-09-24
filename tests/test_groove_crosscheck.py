"""The groove cross-check: a third signal that only ever flags or nudges.

Keypoints and shape both start from the same evidence in the end - a stem the
pose model has to see, or a mask shape_orientation() has to segment out of the
same frame - so on a fruit where that evidence is misleading they can be
confidently wrong together. `orient.groove_crosscheck()` compares whatever
axis the pipeline settled on against the fruit's own surface grooves, read
from texture rather than either mask. See the "third signal" section of
orientation.py's module docstring.

Two things this is NOT allowed to do, tested directly below: decide the
angle, or touch the flip. A groove has no front or back.
"""

import numpy as np
import pytest

from backend.core.paprika_engine import (
    PaprikaEngine, PLACEMENT_PLACE, PLACEMENT_REORIENT,
)
from backend.detection.paprika import orientation as orient
from backend.detection.paprika.orientation import Orientation

# --------------------------------------------------------------- orient.py


def _placed(angle=42.0, confidence=0.9):
    return Orientation(source="fused", pose=orient.POSE_LYING, angle_deg=angle,
                       confidence=confidence, flip_confidence=0.9)


def test_agreeing_groove_axis_nudges_confidence_up():
    result = orient.groove_crosscheck(_placed(angle=42.0), 40.0, 0.6)
    assert result.groove_agreement_deg == pytest.approx(2.0)
    assert result.confidence > 0.9
    assert "groove_agrees" in result.notes


def test_disagreeing_groove_axis_flags_and_cuts_confidence():
    result = orient.groove_crosscheck(_placed(angle=42.0, confidence=0.9), 120.0, 0.6)
    assert result.groove_agreement_deg > 30.0
    assert result.confidence == pytest.approx(0.9 * 0.6)
    assert any("groove_axis_disagreement" in note for note in result.notes)


def test_low_coherence_reading_is_ignored():
    """Below the coherence floor the "axis" is just noise - agreeing or
    disagreeing with it means nothing, so neither should be recorded."""
    result = orient.groove_crosscheck(_placed(), 120.0, 0.10, min_coherence=0.35)
    assert result.groove_agreement_deg is None
    assert result.confidence == pytest.approx(0.9)


def test_no_groove_reading_is_a_noop():
    result = orient.groove_crosscheck(_placed(), None, None)
    assert result.groove_agreement_deg is None
    assert result.confidence == pytest.approx(0.9)


def test_no_settled_angle_is_a_noop():
    """Nothing to cross-check a fruit that never got an angle in the first
    place - the existing verdicts (unknown, review, ...) already own that."""
    result = orient.groove_crosscheck(_placed(angle=None), 40.0, 0.6)
    assert result.angle_deg is None
    assert result.groove_agreement_deg is None


def test_never_sets_angle_or_touches_flip():
    """A groove reading is an axis, not a direction - it must never be
    allowed to decide which end holds the stem."""
    before_angle, before_flip = 42.0, 0.9
    result = orient.groove_crosscheck(
        _placed(angle=before_angle, confidence=0.9), 120.0, 0.9
    )
    assert result.angle_deg == before_angle
    assert result.flip_confidence == before_flip


def test_axis_has_no_front_or_back():
    """A groove axis of 175 and a settled direction of 10 describe the same
    line through the fruit - near-zero disagreement, not near 180."""
    result = orient.groove_crosscheck(_placed(angle=10.0), 175.0, 0.6)
    assert result.groove_agreement_deg < 20.0


def test_axis_difference_folds_onto_0_90():
    assert orient.axis_difference(10.0, 175.0) == pytest.approx(15.0)
    assert orient.axis_difference(10.0, 100.0) == pytest.approx(90.0)
    assert orient.axis_difference(10.0, 10.0) == pytest.approx(0.0)


# -------------------------------------------------------------- engine wiring


def _engine(**policy):
    block = {
        "backend": "shape",
        "shape": {"saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
                  "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {"min_angle_confidence": 0.45, "min_flip_confidence": 0.40, **policy},
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
    }
    return PaprikaEngine({"paprika": block})


def _good():
    return Orientation(source="fused", pose=orient.POSE_LYING, angle_deg=42.0,
                       confidence=0.9, flip_confidence=0.9)


def test_groove_disagreement_reorients_a_placeable_fruit(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(engine, "_raw_groove_axis", lambda frame, bbox: (120.0, 0.6))
    result = _good()
    assert engine._placement_for(result) == PLACEMENT_PLACE  # before the crosscheck

    result = engine._apply_groove_crosscheck(
        np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5), "green", result
    )
    assert engine._placement_for(result) == PLACEMENT_REORIENT


def test_groove_agreement_still_places(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(engine, "_raw_groove_axis", lambda frame, bbox: (44.0, 0.6))
    result = engine._apply_groove_crosscheck(
        np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5), "green", _good()
    )
    assert engine._placement_for(result) == PLACEMENT_PLACE


def test_crosscheck_is_colour_gated_to_green_by_default(monkeypatch):
    """Colour finds the stem directly on red (87% hit rate) - nothing
    measured yet says the extra groove_axis call earns its keep there, so
    the default colour list leaves it out. Same reasoning as
    uncertain_shape_colours."""
    engine = _engine()
    monkeypatch.setattr(engine, "_raw_groove_axis", lambda frame, bbox: (120.0, 0.9))
    result = engine._apply_groove_crosscheck(
        np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5), "red", _good()
    )
    assert result.groove_agreement_deg is None
    assert engine._placement_for(result) == PLACEMENT_PLACE


def test_crosscheck_colours_are_configurable(monkeypatch):
    engine = _engine(groove_crosscheck_colours=["red"])
    monkeypatch.setattr(engine, "_raw_groove_axis", lambda frame, bbox: (120.0, 0.9))
    result = engine._apply_groove_crosscheck(
        np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5), "red", _good()
    )
    assert engine._placement_for(result) == PLACEMENT_REORIENT


def test_crosscheck_can_be_switched_off(monkeypatch):
    engine = _engine(groove_crosscheck=False)
    monkeypatch.setattr(engine, "_raw_groove_axis", lambda frame, bbox: (120.0, 0.9))
    result = engine._apply_groove_crosscheck(
        np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5), "green", _good()
    )
    assert result.groove_agreement_deg is None


def test_a_groove_only_result_is_not_compared_to_itself(monkeypatch):
    """A fruit already answered from grooves (the no-stem or low-confidence
    fallback) has no independent second groove reading to check it against -
    running the same measurement twice and calling it a cross-check would be
    circular."""
    engine = _engine()
    monkeypatch.setattr(
        engine, "_raw_groove_axis",
        lambda frame, bbox: (999.0, 0.9),  # would disagree wildly if compared
    )
    groove_result = Orientation(source="grooves", pose=orient.POSE_LYING,
                                angle_deg=30.0, confidence=0.6, flip_confidence=0.0)
    result = engine._apply_groove_crosscheck(
        np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5), "green", groove_result
    )
    assert result.groove_agreement_deg is None
    assert result.angle_deg == 30.0


def test_no_groove_signal_available_is_a_noop(monkeypatch):
    """No usable mask, no contrast, or too smooth a surface - groove_axis()
    itself returns None for all of these. Absence of a reading is not
    disagreement, same principle as the kp-vs-shape gate."""
    engine = _engine()
    monkeypatch.setattr(engine, "_raw_groove_axis", lambda frame, bbox: None)
    result = engine._apply_groove_crosscheck(
        np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5), "green", _good()
    )
    assert result.groove_agreement_deg is None
    assert engine._placement_for(result) == PLACEMENT_PLACE


@pytest.mark.parametrize("threshold,axis,expected", [
    (20.0, 65.0, PLACEMENT_REORIENT),   # 23deg off axis, over a tight threshold
    (45.0, 65.0, PLACEMENT_PLACE),      # same disagreement, a looser threshold
])
def test_the_threshold_is_configurable(monkeypatch, threshold, axis, expected):
    engine = _engine(max_groove_disagreement_deg=threshold)
    monkeypatch.setattr(engine, "_raw_groove_axis", lambda frame, bbox: (axis, 0.6))
    result = engine._apply_groove_crosscheck(
        np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5), "green", _good()
    )
    assert engine._placement_for(result) == expected
