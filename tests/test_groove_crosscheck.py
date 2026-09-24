"""
groove_crosscheck: a settled axis checked against the fruit's own grooves.

Not the same mechanism as groove_axis_fallback / _groove_axis_estimate,
which only ever runs on a fruit with no better source and SUPPLIES an axis.
This is the opposite case: a fruit that already has an axis from keypoints
and/or shape - fuse() is internally satisfied with it - checked against a
third, independent signal before it is trusted enough to place.

Why that is worth doing separately from test_disagreement_gate.py: that gate
compares keypoints against shape, and both are read off the same mask - their
agreement is real evidence, but it cannot catch the two of them being
confidently wrong TOGETHER. Grooves are read from surface texture instead, so
they are the one signal here that does not share that blind spot.

These tests pin, in order: the primitives (orient.axis_difference,
orient.groove_crosscheck), the hard-stop in _placement_for, the engine
wiring (_apply_groove_crosscheck) in isolation, and finally a full
evaluate()-through-_evaluate_one() pass on a synthetic fruit - so the whole
pipeline, not just the pieces, is on record as doing what it is meant to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.paprika_engine import (  # noqa: E402
    PaprikaEngine, PLACEMENT_PLACE, PLACEMENT_REORIENT,
)
from backend.detection.paprika import end_on as end_on_model  # noqa: E402
from backend.detection.paprika import orientation as orient  # noqa: E402
from backend.detection.paprika.orientation import Keypoint, Orientation  # noqa: E402

from tests.test_orientation import COLORS, render_paprika  # noqa: E402


# ------------------------------------------------------------- axis_difference


def test_axis_difference_folds_the_180_wrap():
    """0 and 180 are the SAME axis - unlike angular_difference, which scores
    directions and would call them maximally different."""
    assert orient.axis_difference(0.0, 180.0) == pytest.approx(0.0)
    assert orient.axis_difference(0.0, 0.0) == pytest.approx(0.0)


def test_axis_difference_maxes_out_at_90():
    """Perpendicular axes are as different as two axes can be."""
    assert orient.axis_difference(0.0, 90.0) == pytest.approx(90.0)


def test_axis_difference_wraps_past_90():
    """10 and 170 are 160 degrees apart as directions, but as AXES the true
    separation is 20 - the short way round through the 180 wrap."""
    assert orient.axis_difference(10.0, 170.0) == pytest.approx(20.0)


# ------------------------------------------------------------ groove_crosscheck


def _settled(angle=40.0, confidence=0.9, flip_confidence=0.9):
    return Orientation(source="fused", pose=orient.POSE_LYING, angle_deg=angle,
                       axis_deg=None if angle is None else angle % 180.0,
                       confidence=confidence, flip_confidence=flip_confidence)


def test_agreeing_grooves_are_recorded_without_a_confidence_cut():
    result = orient.groove_crosscheck(_settled(angle=40.0), 42.0, coherence=0.9)
    assert result.groove_agreement_deg == pytest.approx(2.0)
    assert result.confidence == pytest.approx(0.9)
    assert "groove_agrees" in result.notes


def test_disagreeing_grooves_cut_confidence_but_never_the_angle():
    result = orient.groove_crosscheck(_settled(angle=40.0, confidence=0.9), 100.0,
                                       coherence=0.9)
    assert result.groove_agreement_deg == pytest.approx(60.0)
    assert result.confidence < 0.9
    assert result.angle_deg == pytest.approx(40.0), (
        "grooves cannot say which end carries the stem and must never move "
        "the settled angle"
    )
    assert result.flip_confidence == pytest.approx(0.9), (
        "grooves cannot speak to the flip at all"
    )
    assert any("groove_disagreement" in n for n in result.notes)


def test_a_faint_groove_reading_is_skipped_not_trusted():
    """Below min_coherence the axis is the direction of whatever noise was
    strongest - the same floor _groove_axis_estimate uses elsewhere, so one
    fruit is not held to two different ideas of "faint"."""
    result = orient.groove_crosscheck(_settled(), 100.0, coherence=0.10,
                                       min_coherence=0.35)
    assert result.groove_agreement_deg is None
    assert any("groove_crosscheck_skipped" in n for n in result.notes)


def test_no_settled_angle_means_nothing_to_check():
    result = orient.groove_crosscheck(_settled(angle=None), 100.0, coherence=0.9)
    assert result.groove_agreement_deg is None


def test_no_groove_reading_means_nothing_to_check():
    result = orient.groove_crosscheck(_settled(), None, coherence=0.9)
    assert result.groove_agreement_deg is None


def test_the_disagreement_threshold_is_configurable():
    agrees = orient.groove_crosscheck(_settled(angle=0.0), 25.0, coherence=0.9,
                                       max_disagreement_deg=30.0)
    assert agrees.confidence == pytest.approx(0.9)

    disagrees = orient.groove_crosscheck(_settled(angle=0.0), 25.0, coherence=0.9,
                                          max_disagreement_deg=10.0)
    assert disagrees.confidence < 0.9


# --------------------------------------------------------------- _placement_for


def _bare_engine(**policy):
    block = {
        "backend": "shape",
        "shape": {"saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
                  "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {"min_angle_confidence": 0.45, "min_flip_confidence": 0.40, **policy},
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
    }
    return PaprikaEngine({"paprika": block})


def _good(groove_agreement=None):
    return Orientation(source="fused", pose=orient.POSE_LYING, angle_deg=42.0,
                       confidence=0.9, flip_confidence=0.9,
                       groove_agreement_deg=groove_agreement)


def test_grooves_agreeing_still_place():
    assert _bare_engine()._placement_for(_good(groove_agreement=4.0)) == PLACEMENT_PLACE


def test_a_large_groove_disagreement_is_a_hard_stop():
    """Mirrors max_estimator_disagreement_deg's own hard-stop test almost
    exactly, on purpose - it is the same policy applied to a third
    estimator."""
    assert _bare_engine()._placement_for(_good(groove_agreement=50.0)) == PLACEMENT_REORIENT


def test_it_reorients_rather_than_rejects():
    """The fruit is fine - grooves and keypoints simply cannot both be
    right, and another pass may settle it."""
    assert _bare_engine()._placement_for(_good(groove_agreement=80.0)) != "reject"


def test_confidence_alone_would_not_have_caught_it():
    """groove_crosscheck() cuts confidence by 0.6 on a disagreement, but
    0.9 x 0.6 = 0.54 still clears min_angle_confidence of 0.45 - why this
    has to be a hard stop in _placement_for rather than another confidence
    nudge, the same reasoning as the keypoint/shape gate."""
    engine = _bare_engine()
    softened = _good(groove_agreement=None)
    softened.confidence = 0.9 * 0.6
    assert engine._placement_for(softened) == PLACEMENT_PLACE, (
        "the confidence cut on its own leaves the fruit placeable"
    )


def test_no_groove_reading_means_no_gate():
    """groove_agreement_deg is None whenever no groove reading was ever
    compared. Absence of a second opinion is not disagreement."""
    assert _bare_engine()._placement_for(_good(groove_agreement=None)) == PLACEMENT_PLACE


@pytest.mark.parametrize("threshold,disagreement,expected", [
    (20.0, 25.0, PLACEMENT_REORIENT),
    (45.0, 25.0, PLACEMENT_PLACE),
])
def test_the_threshold_is_configurable(threshold, disagreement, expected):
    engine = _bare_engine(max_groove_disagreement_deg=threshold)
    assert engine._placement_for(_good(groove_agreement=disagreement)) == expected


def test_a_fruit_with_no_angle_is_unaffected():
    """The gate is about two axes disagreeing. With no angle there is no
    axis, and the existing verdicts own that case."""
    result = Orientation(source="fused", pose=orient.POSE_LYING, angle_deg=None,
                         confidence=0.9, groove_agreement_deg=90.0)
    assert _bare_engine()._placement_for(result) != PLACEMENT_PLACE


# ---------------------------------------------------- _apply_groove_crosscheck


def test_apply_groove_crosscheck_skips_a_groove_sourced_result(monkeypatch):
    """Comparing the groove axis to itself would always agree and say
    nothing about the fruit - this is what keeps the two mechanisms
    (fallback and crosscheck) from stacking into a no-op self-endorsement."""
    engine = _bare_engine(groove_crosscheck=True)
    monkeypatch.setattr(end_on_model, "groove_axis", lambda *a, **k: (999.0, 0.9))
    monkeypatch.setattr(orient, "segment_fruit",
                        lambda *a, **k: np.ones((5, 5), np.uint8) * 255)
    result = Orientation(source="grooves", pose=orient.POSE_LYING, angle_deg=40.0,
                         confidence=0.9)
    out = engine._apply_groove_crosscheck(
        np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5), result, "green"
    )
    assert out.groove_agreement_deg is None


def test_apply_groove_crosscheck_handles_a_missing_frame():
    engine = _bare_engine(groove_crosscheck=True)
    result = Orientation(source="fused", pose=orient.POSE_LYING, angle_deg=40.0,
                         confidence=0.9)
    out = engine._apply_groove_crosscheck(None, (0, 0, 5, 5), result, "green")
    assert out is result
    assert out.groove_agreement_deg is None


def test_apply_groove_crosscheck_handles_an_unreadable_mask(monkeypatch):
    engine = _bare_engine(groove_crosscheck=True)
    monkeypatch.setattr(orient, "segment_fruit", lambda *a, **k: None)
    result = Orientation(source="fused", pose=orient.POSE_LYING, angle_deg=40.0,
                         confidence=0.9)
    out = engine._apply_groove_crosscheck(
        np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5), result, "green"
    )
    assert out.groove_agreement_deg is None


# --------------------------------------------------- full pipeline, end to end


def _full_engine(**overrides):
    """Mirrors test_orientation.py's own _engine(): a shape-backend
    PaprikaEngine that can run evaluate() on a rendered frame, not just take
    hand-built Orientation objects."""
    block = {
        "backend": "shape",
        "shape_crosscheck": False,
        "stemless_shape_fallback": True,
        "shape": {
            "saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
            "min_area_px": 3000, "max_area_ratio": 0.7,
        },
        "policy": {
            "min_angle_confidence": 0.45, "min_flip_confidence": 0.40,
            "reject_standing": True,
        },
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
        "primary_rule": "largest",
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(block.get(key), dict):
            block[key] = {**block[key], **value}
        else:
            block[key] = value
    return PaprikaEngine({"paprika": block})


def _redetect(baseline):
    """The primary detection evaluate() already found, repackaged as the
    plain dict _evaluate_one() takes - same pattern test_orientation.py uses
    to re-drive a single fruit through the engine a second time."""
    return {
        "bbox": baseline["bbox"],
        "confidence": 0.9,
        "colour": baseline["colour"],
        "stem_method": baseline["stem_method"],
        "keypoints": {
            name: Keypoint(v["x"], v["y"], v["confidence"], v["visible"])
            for name, v in baseline["keypoints"].items()
        },
    }


def test_off_by_default_leaves_a_disagreeing_fruit_untouched(monkeypatch):
    """Unmeasured against labelled data, so it ships off - see
    config/default.yaml. Disabled must mean disabled, not disabled-ish."""
    image, _, _ = render_paprika(40, with_stem=True, color=COLORS["green"])
    engine = _full_engine()  # groove_crosscheck not set -> defaults to off
    baseline = engine.evaluate(image)["primary"]
    assert baseline["placement"] == "place"

    monkeypatch.setattr(end_on_model, "groove_axis", lambda *a, **k: (130.0, 0.9))
    evaluated = engine._evaluate_one(image, _redetect(baseline))

    assert evaluated["placement"] == "place"
    assert evaluated["orientation"]["groove_agreement_deg"] is None


def test_switched_on_downgrades_a_disagreeing_green_fruit(monkeypatch):
    """The end-to-end case the whole mechanism exists for: a fruit that
    would otherwise place, sent for another look because its own grooves
    disagree with the axis keypoints and shape were satisfied with."""
    image, _, _ = render_paprika(40, with_stem=True, color=COLORS["green"])
    engine = _full_engine(policy={"groove_crosscheck": True})
    baseline = engine.evaluate(image)["primary"]
    assert baseline["placement"] == "place", \
        "sanity check: would place with no groove opinion at all"

    # ~90 degrees off the fruit's real axis - well past the 30deg default.
    monkeypatch.setattr(end_on_model, "groove_axis", lambda *a, **k: (130.0, 0.9))
    evaluated = engine._evaluate_one(image, _redetect(baseline))

    assert evaluated["placement"] == "reorient"
    assert evaluated["orientation"]["groove_agreement_deg"] > 80.0
    # Reorient keeps the angle on the record (same as the keypoint/shape
    # disagreement gate and the stem-spread gate) - it is withheld only for
    # "review" and "unknown", where there is no usable measurement at all.
    assert evaluated["orientation"]["angle_deg"] is not None
    assert any("groove_disagreement" in n for n in evaluated["orientation"]["notes"])


def test_switched_on_still_places_when_grooves_agree(monkeypatch):
    image, _, _ = render_paprika(40, with_stem=True, color=COLORS["green"])
    engine = _full_engine(policy={"groove_crosscheck": True})
    baseline = engine.evaluate(image)["primary"]

    monkeypatch.setattr(end_on_model, "groove_axis", lambda *a, **k: (41.0, 0.9))
    evaluated = engine._evaluate_one(image, _redetect(baseline))

    assert evaluated["placement"] == "place"
    assert evaluated["orientation"]["groove_agreement_deg"] < 5.0


def test_colour_gate_leaves_red_fruit_unaffected(monkeypatch):
    """Colour finds the stem directly on red - see uncertain_shape_colours
    for the identical reasoning - so a groove disagreement there is not the
    failure mode this exists for, and the default scope (green only) must
    not touch it."""
    image, _, _ = render_paprika(40, with_stem=True, color=COLORS["red"])
    engine = _full_engine(policy={"groove_crosscheck": True})
    baseline = engine.evaluate(image)["primary"]
    assert baseline["colour"] == "red"
    assert baseline["placement"] == "place"

    monkeypatch.setattr(end_on_model, "groove_axis", lambda *a, **k: (130.0, 0.9))
    evaluated = engine._evaluate_one(image, _redetect(baseline))

    assert evaluated["placement"] == "place"
    assert evaluated["orientation"]["groove_agreement_deg"] is None


def test_a_low_coherence_reading_does_not_downgrade_placement(monkeypatch):
    """The runtime gate, exercised through the real pipeline rather than the
    primitive directly: a groove axis that disagrees strongly but was too
    faint to trust must not cost the fruit its placement."""
    image, _, _ = render_paprika(40, with_stem=True, color=COLORS["green"])
    engine = _full_engine(policy={"groove_crosscheck": True})
    baseline = engine.evaluate(image)["primary"]

    monkeypatch.setattr(end_on_model, "groove_axis", lambda *a, **k: (130.0, 0.10))
    evaluated = engine._evaluate_one(image, _redetect(baseline))

    assert evaluated["placement"] == "place"
    assert evaluated["orientation"]["groove_agreement_deg"] is None
