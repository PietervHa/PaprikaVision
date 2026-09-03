"""
Regression tests for the orientation engine.

These matter more than usual here. `orientation.py` contains several tuned
constants - the 4.0 flip-confidence scaling, the 1.12 elongation floor, the
0.18 standing span ratio - and every one of them is the kind of number that
gets nudged during a debugging session and never put back. Each test below
pins a behaviour that a nudge would break.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.paprika_engine import PaprikaEngine  # noqa: E402
from backend.detection.paprika import orientation as orient  # noqa: E402
from backend.detection.paprika.orientation import Keypoint  # noqa: E402

BELT = (198, 198, 196)
COLORS = {
    "red": (40, 40, 215),
    "yellow": (45, 215, 235),
    "green": (70, 175, 75),
    "orange": (30, 145, 240),
}


def render_paprika(
    angle_deg: float,
    size: int = 400,
    length: int = 150,
    shoulder_w: int = 94,
    tip_w: int = 50,
    color=COLORS["red"],
):
    """Synthetic paprika: wide shoulder tapering to a narrower tip.

    Built pointing +x (stem to the right, i.e. 0 deg) then rotated
    counter-clockwise on screen by angle_deg.

    Returns (image, stem_xy, blossom_xy).
    """
    image = np.zeros((size, size, 3), np.uint8)
    image[:] = BELT
    cx = cy = size // 2

    profile = []
    steps = 140
    for i in range(steps + 1):
        u = i / steps
        t = -length / 2 + u * length
        w = tip_w + (shoulder_w - tip_w) * (u ** 1.35)
        w *= math.sqrt(max(0.0, 1 - (2 * u - 1) ** 8))
        profile.append((t, w / 2))

    polygon = profile + [(t, -h) for t, h in reversed(profile)]

    radians = math.radians(angle_deg)
    ca, sa = math.cos(radians), math.sin(radians)
    points = [
        [int(round(cx + t * ca - h * sa)), int(round(cy - (t * sa + h * ca)))]
        for t, h in polygon
    ]
    cv2.fillPoly(image, [np.array(points, np.int32)], color)

    stem = (cx + ca * length / 2, cy - sa * length / 2)
    blossom = (cx - ca * length / 2, cy + sa * length / 2)
    return image, stem, blossom


def bbox_of(image):
    mask = orient.segment_fruit(image)
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


# ------------------------------------------------------------------- angles


def test_vector_to_angle_is_screen_intuitive():
    """0 right, 90 up, 180 left, 270 down. Image y grows downward."""
    assert orient.vector_to_angle(1, 0) == pytest.approx(0.0)
    assert orient.vector_to_angle(0, -1) == pytest.approx(90.0)
    assert orient.vector_to_angle(-1, 0) == pytest.approx(180.0)
    assert orient.vector_to_angle(0, 1) == pytest.approx(270.0)


def test_angular_difference_wraps():
    assert orient.angular_difference(350, 10) == pytest.approx(20.0)
    assert orient.angular_difference(10, 350) == pytest.approx(20.0)
    assert orient.angular_difference(0, 180) == pytest.approx(180.0)


@pytest.mark.parametrize("offset,invert,expected", [
    (0, False, 30.0),
    (90, False, 120.0),
    (0, True, 330.0),
    (180, True, 150.0),
])
def test_plc_frame_mapping(offset, invert, expected):
    assert orient.apply_frame_convention(30.0, offset, invert) == pytest.approx(expected)


def test_plc_frame_mapping_preserves_none():
    """No angle must stay no angle, never become 0 - which is a valid angle."""
    assert orient.apply_frame_convention(None, 90.0, True) is None


# -------------------------------------------------------------- shape route


@pytest.mark.parametrize("color_name", list(COLORS))
@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 225, 270, 315])
def test_shape_recovers_angle_for_every_colour(color_name, angle):
    """The stemless path: orientation from silhouette alone, colour-agnostic."""
    image, _, _ = render_paprika(angle, color=COLORS[color_name])
    result = orient.shape_orientation(orient.segment_fruit(image))
    assert result.angle_deg is not None
    assert orient.angular_difference(result.angle_deg, angle) < 5.0


def test_taper_breaks_the_180_degree_tie():
    """PCA alone gives an axis; the taper is what picks the end."""
    image, _, _ = render_paprika(30)
    result = orient.shape_orientation(orient.segment_fruit(image))
    assert orient.angular_difference(result.angle_deg, 30) < 5.0
    assert result.flip_confidence > 0.55


def test_symmetric_fruit_reports_low_flip_confidence():
    """A symmetric silhouette must admit it cannot tell the ends apart,
    rather than guessing and sounding confident about it."""
    image, _, _ = render_paprika(30, shoulder_w=74, tip_w=74)
    result = orient.shape_orientation(orient.segment_fruit(image))
    assert result.flip_confidence < 0.2


def test_round_silhouette_reports_no_axis():
    """A fruit stood on its end has no meaningful in-plane rotation."""
    image = np.full((400, 400, 3), BELT, np.uint8)
    cv2.circle(image, (200, 200), 70, COLORS["red"], -1)
    result = orient.shape_orientation(orient.segment_fruit(image))
    assert result.angle_deg is None
    assert result.elongation < 1.12


# ----------------------------------------------------------- keypoint route


def test_keypoints_give_exact_angle():
    image, stem, blossom = render_paprika(115)
    x1, y1, x2, y2 = bbox_of(image)
    diagonal = math.hypot(x2 - x1, y2 - y1)

    result = orient.keypoint_orientation(
        Keypoint(*stem, 0.9, True), Keypoint(*blossom, 0.9, True), diagonal
    )
    assert result.pose == orient.POSE_LYING
    assert orient.angular_difference(result.angle_deg, 115) < 2.0


def test_overlapping_keypoints_mean_standing_not_a_random_angle():
    """Both landmarks on the same spot => axis points at the camera. Returning
    a plausible angle here would be worse than returning none."""
    result = orient.keypoint_orientation(
        Keypoint(202, 201, 0.9, True), Keypoint(200, 200, 0.9, True), 180
    )
    assert result.angle_deg is None
    assert result.pose == orient.POSE_STANDING_STEM_UP


def test_hidden_stem_means_standing_stem_down():
    """Visibility, not position, is what distinguishes the two standing poses -
    which is exactly why ANNOTATION_SPEC insists on labelling occluded
    landmarks rather than dropping them."""
    result = orient.keypoint_orientation(
        Keypoint(202, 201, 0.7, False), Keypoint(200, 200, 0.7, True), 180
    )
    assert result.pose == orient.POSE_STANDING_STEM_DOWN


def test_missing_keypoint_yields_no_angle():
    result = orient.keypoint_orientation(None, Keypoint(10, 10, 0.9), 100)
    assert result.angle_deg is None
    assert "missing_keypoint" in result.notes


# ------------------------------------------------------------------ fusion


def test_shape_rescues_a_stemless_fruit_the_model_got_backwards():
    """THE case this whole design exists for.

    Stem broke off, so the pose model sees no stem, hedges, and puts the
    landmarks the wrong way round with low confidence. The silhouette is
    unambiguous, so it takes the flip and the final answer is correct.
    """
    image, stem, blossom = render_paprika(200)
    x1, y1, x2, y2 = bbox_of(image)
    diagonal = math.hypot(x2 - x1, y2 - y1)

    # Landmarks deliberately swapped, with the low confidence a hedging model
    # would report.
    wrong = orient.keypoint_orientation(
        Keypoint(*blossom, 0.30, True), Keypoint(*stem, 0.30, True), diagonal
    )
    assert orient.angular_difference(wrong.angle_deg, 200) > 150  # confirms it is backwards

    shape = orient.shape_orientation(orient.segment_fruit(image))
    fused = orient.fuse(wrong, shape)

    assert orient.angular_difference(fused.angle_deg, 200) < 5.0
    assert "flip_taken_from_shape" in fused.notes


def test_confident_keypoints_keep_the_call():
    image, stem, blossom = render_paprika(200)
    x1, y1, x2, y2 = bbox_of(image)
    diagonal = math.hypot(x2 - x1, y2 - y1)

    good = orient.keypoint_orientation(
        Keypoint(*stem, 0.95, True), Keypoint(*blossom, 0.95, True), diagonal
    )
    fused = orient.fuse(good, orient.shape_orientation(orient.segment_fruit(image)))

    assert orient.angular_difference(fused.angle_deg, 200) < 3.0
    assert "estimators_agree" in fused.notes


def test_axis_disagreement_lowers_confidence_rather_than_averaging():
    """Averaging two directions 90 deg apart yields a number confidently
    perpendicular to both. Disagreement must degrade confidence instead."""
    keypoints = orient.Orientation(
        angle_deg=10.0, axis_deg=10.0, source="keypoints",
        confidence=0.9, flip_confidence=0.9, pose=orient.POSE_LYING,
    )
    shape = orient.Orientation(
        angle_deg=100.0, axis_deg=100.0, source="shape",
        confidence=0.8, flip_confidence=0.8, pose=orient.POSE_LYING,
    )
    fused = orient.fuse(keypoints, shape)

    assert fused.angle_deg == pytest.approx(10.0)   # keypoints keep the axis
    assert fused.confidence < 0.9                    # but pay for the dispute
    assert any("axis_disagreement" in note for note in fused.notes)


# ------------------------------------------------------------------ engine


def _engine(**overrides):
    block = {
        "backend": "shape",
        "shape_crosscheck": True,
        "shape": {"saturation_floor": 60, "min_area_px": 3000, "max_area_ratio": 0.7},
        "policy": {
            "min_angle_confidence": 0.45,
            "min_flip_confidence": 0.40,
            "reject_standing": True,
        },
        "frame": {"angle_offset_deg": 0.0, "angle_invert": False},
        "primary_rule": "largest",
    }
    block.update(overrides)
    return PaprikaEngine({"paprika": block})


def test_engine_places_a_clear_fruit():
    image, _, _ = render_paprika(25)
    result = _engine().evaluate(image)

    assert result["status"] == "OK"
    primary = result["primary"]
    assert primary["placement"] == "place"
    assert orient.angular_difference(primary["angle_plc"], 25) < 5.0


def test_engine_sends_an_ambiguous_fruit_for_reorientation():
    """Symmetric fruit: the axis is fine, the stem end is a coin flip. The
    machine must not act on a coin flip."""
    image, _, _ = render_paprika(25, shoulder_w=76, tip_w=74)
    result = _engine().evaluate(image)

    assert result["status"] == "NOK"
    assert result["primary"]["placement"] == "reorient"
    assert result["failure_reason"] == "not_placeable_reorient"


def test_engine_reports_nok_on_empty_belt():
    image = np.full((400, 400, 3), BELT, np.uint8)
    result = _engine().evaluate(image)

    assert result["status"] == "NOK"
    assert result["failure_reason"] == "no_paprika_detected"
    assert result["primary"] is None


def test_engine_applies_the_plc_frame_mapping():
    image, _, _ = render_paprika(0)
    result = _engine(frame={"angle_offset_deg": 90.0, "angle_invert": True}).evaluate(image)

    # vision 0 -> invert -> 0 -> +90 -> 90
    assert orient.angular_difference(result["primary"]["angle_plc"], 90.0) < 5.0


def test_labels_stay_ascii_for_opencv():
    """Overlay text is drawn with Hershey fonts, which render anything
    non-ASCII as '??'."""
    image, _, _ = render_paprika(25)
    label = _engine().evaluate(image)["primary"]["label"]
    assert label.isascii(), f"non-ASCII label would render as ?? : {label!r}"
