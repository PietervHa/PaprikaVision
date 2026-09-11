"""The end-on classifier, and the guarantee that it currently decides nothing.

Fitted on 254 hand-labelled crops (57 end-on, 197 side-on) across six sessions
and three fruit colours. AUC 0.927; leave-one-session-out 0.872, which is the
honest figure because a fold sharing a session shares lighting, belt and often
the same physical fruit.
"""

import cv2
import numpy as np
import pytest

from backend.detection.paprika import end_on


def _disc(radius=70, size=200, texture=None):
    """A plain fruit-ish blob, optionally with something at its centre."""
    crop = np.full((size, size, 3), (60, 40, 40), np.uint8)
    cv2.circle(crop, (size // 2, size // 2), radius, (40, 40, 200), -1)
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), radius, 255, -1)
    if texture:
        texture(crop, size)
    return crop, mask


def test_probability_is_between_zero_and_one():
    crop, mask = _disc()
    p = end_on.end_on_probability(crop, mask)
    assert p is None or 0.0 <= p <= 1.0


def test_unreadable_fruit_returns_none_not_a_guess():
    """None means "could not measure" and must never be read as side-on."""
    crop, mask = _disc(radius=8, size=40)
    assert end_on.end_on_probability(crop, mask) is None


def test_mismatched_mask_is_refused():
    crop, _ = _disc()
    assert end_on.end_on_probability(crop, np.zeros((10, 10), np.uint8)) is None


def test_none_inputs_are_refused():
    crop, mask = _disc()
    assert end_on.end_on_probability(None, mask) is None
    assert end_on.end_on_probability(crop, None) is None


BASELINE = {"centre_texture": 1.12, "solidity": 0.963, "par": 0.30}


def test_each_coefficient_points_the_way_it_was_fitted():
    """Guards against the constants being pasted back in the wrong order - the
    feature order is positional, so a transposition would still run and would
    silently invert two of the three signals.

    Directions come from the labelled medians: end-on fruit have MORE texture
    at the centre (1.39 vs 1.01, the blossom scar), LESS convex outlines (0.964
    vs 0.978, lobes bumping out all round) and LESS groove alignment (0.13 vs
    0.40, grooves converging rather than running as parallel bands).
    """
    base = end_on.score(BASELINE)
    assert end_on.score({**BASELINE, "centre_texture": 1.60}) > base
    assert end_on.score({**BASELINE, "solidity": 0.99}) < base
    assert end_on.score({**BASELINE, "par": 0.60}) < base


def test_a_typical_end_on_fruit_scores_above_the_threshold():
    """Median measurements of the 57 labelled end-on crops."""
    assert end_on.score({"centre_texture": 1.39, "solidity": 0.964, "par": 0.128}) \
        > end_on.DEFAULT_END_ON_THRESHOLD


def test_a_typical_side_on_fruit_scores_below_it():
    """Median measurements of the 197 labelled side-on crops."""
    assert end_on.score({"centre_texture": 1.01, "solidity": 0.978, "par": 0.396}) \
        < end_on.DEFAULT_END_ON_THRESHOLD


def test_features_returns_the_three_it_was_fitted_on():
    """The coefficient order in end_on.py is positional, so the feature set is
    part of the contract, not an implementation detail."""
    measured = end_on.features(*_disc())
    if measured is not None:
        assert set(measured) == {"centre_texture", "solidity", "par"}


def test_classifier_reports_but_does_not_decide_by_default(monkeypatch):
    """The guarantee that makes shipping this safe: with end_on_decides off,
    a confident end-on verdict still leaves the placement exactly where
    stem_not_found put it."""
    from backend.core.paprika_engine import PaprikaEngine
    from backend.detection.paprika import orientation as orient

    block = {
        "backend": "shape", "stemless_shape_fallback": False,
        "shape": {"saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
                  "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {"min_angle_confidence": 0.45, "min_flip_confidence": 0.40},
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
    }
    monkeypatch.setattr(end_on, "end_on_probability", lambda *a, **k: 0.99)
    monkeypatch.setattr(orient, "segment_fruit",
                        lambda *a, **k: np.ones((5, 5), np.uint8) * 255)

    engine = PaprikaEngine({"paprika": block})
    pose, notes = engine._end_on_verdict(np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5),
                                         orient.POSE_STEM_NOT_FOUND)
    assert pose == orient.POSE_STEM_NOT_FOUND, "reporting mode must not change the pose"
    assert "end_on_detected" in notes


def test_it_does_decide_when_switched_on(monkeypatch):
    from backend.core.paprika_engine import PaprikaEngine
    from backend.detection.paprika import orientation as orient

    block = {
        "backend": "shape", "stemless_shape_fallback": False,
        "shape": {"saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
                  "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {"min_angle_confidence": 0.45, "min_flip_confidence": 0.40,
                   "end_on_decides": True},
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
    }
    monkeypatch.setattr(end_on, "end_on_probability", lambda *a, **k: 0.99)
    monkeypatch.setattr(orient, "segment_fruit",
                        lambda *a, **k: np.ones((5, 5), np.uint8) * 255)

    engine = PaprikaEngine({"paprika": block})
    pose, _ = engine._end_on_verdict(np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5),
                                     orient.POSE_STEM_NOT_FOUND)
    assert pose == orient.POSE_UPSIDE_DOWN


def test_groove_axis_refuses_a_textureless_fruit():
    """The trap that a relative measure walks into.

    Coherence is a ratio, so a surface with no grooves at all still scores
    highly: on a flat drawn disc the percentile cut selects antialiasing noise
    and the doubled-angle resultant came back at 0.54, indistinguishable from a
    deeply grooved pepper. An absolute contrast floor is what separates "the
    grooves all run this way" from "there are no grooves".
    """
    crop, mask = _disc()
    assert end_on.groove_axis(crop, mask) is None


def _ripple(crop, size, horizontal=True):
    """Soft parallel bands, the way a lobed surface actually shades.

    Hard-drawn lines are no good here: they saturate the Sobel response, so a
    quarter of the pixels share the maximum value and the percentile cut
    selects nothing at all. Real fruit shade continuously.
    """
    axis = np.arange(size, dtype=np.float32)
    wave = (np.sin(axis / 6.0) * 40).astype(np.int16)
    band = wave[:, None] if horizontal else wave[None, :]
    shaded = np.clip(crop.astype(np.int16) + band[..., None], 0, 255)
    crop[:] = shaded.astype(np.uint8)


def test_groove_axis_reads_parallel_bands():
    crop, mask = _disc(texture=lambda c, s: _ripple(c, s, horizontal=True))
    measured = end_on.groove_axis(crop, mask)
    assert measured is not None
    axis, coherence = measured
    # Horizontal bands, so the axis they define is horizontal: 0 or 180.
    assert min(abs(axis - 0.0), abs(axis - 180.0)) < 15.0
    assert coherence > 0.5


def test_groove_axis_is_an_orientation_not_a_direction():
    """0-180 only. Which end carries the stem is not knowable from grooves,
    and a function that returned 0-360 would invite a caller to believe it."""
    axis, _ = end_on.groove_axis(
        *_disc(texture=lambda c, s: _ripple(c, s, horizontal=True))
    )
    assert 0.0 <= axis < 180.0


def test_groove_estimate_claims_no_flip_confidence():
    """The guarantee that stops a groove axis being placed on a coin toss."""
    from backend.core.paprika_engine import PaprikaEngine
    block = {
        "backend": "shape", "stemless_shape_fallback": False,
        "shape": {"saturation_floor": 80, "belt_hue": [96, 145], "value_floor": 45,
                  "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {"min_angle_confidence": 0.45, "min_flip_confidence": 0.40},
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
    }
    engine = PaprikaEngine({"paprika": block})
    import numpy as np
    from backend.detection.paprika import end_on as module, orientation as orient
    import pytest as _pytest
    original = module.groove_axis
    module.groove_axis = lambda *a, **k: (42.0, 0.9)
    orig_seg = orient.segment_fruit
    orient.segment_fruit = lambda *a, **k: np.ones((5, 5), np.uint8) * 255
    try:
        result = engine._groove_axis_estimate(np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5))
    finally:
        module.groove_axis = original
        orient.segment_fruit = orig_seg
    assert result is not None
    assert result.angle_deg == 42.0
    assert result.flip_confidence == 0.0, "a groove axis must never claim a stem end"
