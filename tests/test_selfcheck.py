"""The stem self-check must not score its own failure as the fruit's fault.

This check only ever runs on fruit where colour cannot find the stem - in
practice, green. Measured over the labelled green crops, 39 of 40 gain-variant
attempts could not re-find the stem at all, so every green fruit reaching this
code was scored as maximally unstable and went out as "human check needed" with
the stem plainly visible in the picture.
"""

import numpy as np
import pytest

import backend.detection.paprika.classical as classical


def _region(shape=(120, 120, 3)):
    return np.zeros(shape, np.uint8)


def test_returns_none_when_the_control_cannot_see_the_stem(monkeypatch):
    """The whole point. If the check's own re-detection fails at UNCHANGED
    brightness, nothing the gain variants do afterwards is evidence."""
    monkeypatch.setattr(classical, "_direction_at_gain", lambda *a, **k: None)
    assert classical._stem_selfcheck(_region(), (96, 145), 80, 45, 30.0) is None


def test_returns_180_when_the_control_sees_it_but_a_variant_loses_it(monkeypatch):
    """Now it IS evidence: the stem was visible at this exact brightness and an
    8% change lost it."""
    calls = {"n": 0}

    def fake(region, gain, *a, **k):
        calls["n"] += 1
        return 30.0 if gain == classical.SELFCHECK_CONTROL_GAIN else None

    monkeypatch.setattr(classical, "_direction_at_gain", fake)
    assert classical._stem_selfcheck(_region(), (96, 145), 80, 45, 30.0) == 180.0


def test_measures_the_spread_when_every_variant_works(monkeypatch):
    angles = {1.0: 30.0, 0.92: 34.0, 1.08: 41.0}
    monkeypatch.setattr(classical, "_direction_at_gain",
                        lambda region, gain, *a, **k: angles[gain])
    spread = classical._stem_selfcheck(_region(), (96, 145), 80, 45, 30.0)
    assert spread == pytest.approx(11.0, abs=0.5)


def test_the_control_runs_at_unchanged_brightness():
    """A control at any other gain would not be a control."""
    assert classical.SELFCHECK_CONTROL_GAIN == 1.0
    assert classical.SELFCHECK_CONTROL_GAIN not in classical.STEM_SELFCHECK_GAINS


def test_an_unavailable_check_leaves_the_fruit_no_worse_off(monkeypatch):
    """It must fall back to what morphology itself reported - the same value
    used when the check is switched off. An unavailable check that zeroed the
    quality was strictly worse than never having asked."""
    monkeypatch.setattr(classical, "_direction_at_gain", lambda *a, **k: None)

    import cv2
    frame = np.full((260, 260, 3), (200, 120, 40), np.uint8)
    cv2.circle(frame, (130, 130), 78, (60, 170, 70), -1)
    cv2.ellipse(frame, (130, 52), (11, 26), 0, 0, 360, (70, 190, 95), -1)

    found = classical.find_fruit(frame, belt_hue=(96, 145),
                                 saturation_floor=80, value_floor=45)
    for fruit in found:
        if fruit.stem_method == "morphology":
            assert fruit.stem_selfcheck == "unavailable"
            assert fruit.stem_spread_deg == 0.0
            assert fruit.stem_quality > 0.0, (
                "an unrunnable check must not zero the quality"
            )


def test_selfcheck_state_reaches_the_detection_dict():
    """The engine notes it, so it has to survive the trip."""
    from backend.detection.paprika.classical import ClassicalFruit
    assert hasattr(ClassicalFruit(bbox=(0, 0, 1, 1), mask=np.zeros((1, 1), np.uint8),
                                  area=1, centroid=(0.0, 0.0), hue=50.0, colour="green"),
                   "stem_selfcheck")
