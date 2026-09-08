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

# Blue belt, like the real line. Strongly saturated, so this immediately tests
# that segmentation works on hue and not on saturation - a grey test belt would
# hide exactly that bug.
BELT = (200, 90, 30)
STEM_COLOUR = (60, 150, 70)
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
    with_stem: bool = False,
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

    if with_stem:
        # A short green stem on the calyx side, so the classical backend takes
        # the same path here as it does on the real images.
        tip = (int(round(stem[0] + ca * 42)), int(round(stem[1] - sa * 42)))
        cv2.line(image, (int(stem[0]), int(stem[1])), tip, STEM_COLOUR, 13, cv2.LINE_AA)

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
        "shape_crosscheck": False,
        "stemless_shape_fallback": True,
        "shape": {
            "saturation_floor": 80,
            "belt_hue": [96, 145],
            "value_floor": 45,
            "min_area_px": 3000,
            "max_area_ratio": 0.7,
        },
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
    image, _, _ = render_paprika(25, with_stem=True)
    result = _engine().evaluate(image)

    assert result["status"] == "OK"
    primary = result["primary"]
    assert primary["placement"] == "place"
    assert orient.angular_difference(primary["angle_plc"], 25) < 5.0


def test_standing_stemless_fruit_is_reported_stem_not_found():
    """A round, stemless silhouette is standing on one end or the other.

    It used to be reported as `upside_down`, which asserts something the
    backend never established: that the fruit is lying blossom-up. All it
    really knows is that it could not find a stem. Conflating the two makes the
    upside_down counter unusable as a diagnosis, because a climbing number
    could equally mean the infeed is tipping fruit or that the stem detector
    has stopped coping with this cultivar.

    This must stay a plain, angle-free reject regardless of the shape
    fallback below: looking straight down the long axis, a round silhouette
    has no wider end to read, so shape_orientation() itself declines to
    guess (elongation below min_elongation) and the fallback has nothing to
    offer here. Standing fruit is detected, not oriented - on purpose.
    """
    image = np.full((400, 400, 3), BELT, np.uint8)
    cv2.circle(image, (200, 200), 70, COLORS["red"], -1)
    result = _engine().evaluate(image)
    primary = result["primary"]

    assert result["status"] == "NOK"
    assert primary["placement"] == "reject"
    assert primary["orientation"]["pose"] == orient.POSE_STEM_NOT_FOUND
    assert primary["orientation"]["pose"] != orient.POSE_UPSIDE_DOWN
    assert primary["angle_plc"] is None
    assert primary["orientation"]["angle_deg"] is None
    # The detector's own reason is kept on the record, so a result read back
    # from the database can still be traced to the branch that produced it.
    assert "no_stem" in primary["orientation"]["notes"]


def test_lying_stemless_fruit_gets_a_shape_based_angle():
    """The case paprika.stemless_shape_fallback exists for.

    Stem broke off in the crate, but the fruit is clearly lying on its side -
    elongated, shoulder wider than tip. Outright rejecting this throws away
    a perfectly readable silhouette, so it is measured directly instead of
    being folded into the round-and-standing case above.
    """
    image, _, _ = render_paprika(25, with_stem=False)
    result = _engine().evaluate(image)
    primary = result["primary"]

    assert primary["orientation"]["pose"] == orient.POSE_LYING
    assert primary["orientation"]["source"] == "shape_only"
    assert orient.angular_difference(primary["angle_plc"], 25) < 5.0
    # Strong, well-tapered synthetic shoulder: confident enough to place, not
    # just to measure. A weaker real-world taper is expected to land on
    # "reorient" instead - see the flip-confidence gate this still goes
    # through, unchanged, in _placement_for().
    assert primary["placement"] == "place"
    assert result["status"] == "OK"
    # Provenance survives onto the record, same as the standing case.
    assert "no_stem" in primary["orientation"]["notes"]
    assert "shape_fallback" in primary["orientation"]["notes"]


def test_stemless_shape_fallback_can_be_switched_off():
    """paprika.stemless_shape_fallback: false restores the old behaviour.

    A commissioning engineer who finds the fallback guessing wrong on their
    cultivar needs a config change, not a code change, to turn it off.
    """
    image, _, _ = render_paprika(25, with_stem=False)
    result = _engine(stemless_shape_fallback=False).evaluate(image)
    primary = result["primary"]

    assert primary["placement"] == "reject"
    assert primary["orientation"]["pose"] == orient.POSE_STEM_NOT_FOUND
    assert primary["orientation"]["angle_deg"] is None


def test_stem_not_found_and_upside_down_are_counted_apart():
    """The counters are the reason the split exists, so pin them.

    A fruit the backend could not find a stem on must not increment the
    upside_down counter, and the new key must exist from the start rather than
    appearing only once the first such fruit goes past.
    """
    from backend.core.state import AppState

    state = AppState()
    assert state.counters["stem_not_found"] == 0
    assert state.counters["upside_down"] == 0

    state.increment_counter("NOK", placement="reject", pose=orient.POSE_STEM_NOT_FOUND)

    counters = state.get_snapshot()["counters"]
    assert counters["stem_not_found"] == 1
    assert counters["upside_down"] == 0
    assert counters["reject"] == 1

    state.reset_counters()
    assert state.counters["stem_not_found"] == 0


def test_fruit_running_off_the_frame_is_incomplete_not_stem_not_found():
    """Half out of frame says something about the frame, not about the fruit -
    it may have a perfectly good stem you simply cannot see. Incomplete takes
    precedence over stem_not_found for exactly that reason: counting a clipped
    fruit as one the stem detector failed on would send somebody looking at the
    lighting when the answer is the camera framing or the trigger timing."""
    image, _, _ = render_paprika(90, with_stem=True)
    # Remove the bottom half: the fruit now continues outside the frame. The
    # cut then spans 43% of the diameter, well above EDGE_CUT_THRESHOLD - a
    # fruit merely grazing the border should specifically NOT fall under it, and
    # that is what the threshold separates.
    cropped = image[: image.shape[0] // 2, :]

    result = _engine().evaluate(cropped)
    primary = result["primary"]

    assert primary["placement"] == "reject"
    assert primary["orientation"]["pose"] == orient.POSE_INCOMPLETE
    assert primary["angle_plc"] is None


def test_centered_primary_rule_measures_from_the_frame_centre():
    """"centered" must mean the middle of the frame, not the origin.

    It used to measure |cx| + |cy| from (0, 0), so it actually preferred the
    fruit nearest the TOP-LEFT CORNER - the one just entering the frame, whose
    angle is measured from a partial silhouette. That is the precise opposite
    of what the rule is for, and on a belt running left to right it silently
    picked a different fruit on every frame.
    """
    def fake(cx, cy):
        return {
            "bbox": [cx - 40, cy - 30, cx + 40, cy + 30],
            "center": [cx, cy],
            "placement": "place",
            "orientation": {"confidence": 0.9},
        }

    width, height = 900, 420
    entering = fake(90, 45)          # near the origin
    centred = fake(450, 210)         # actually in the middle

    engine = _engine(primary_rule="centered")
    assert engine._pick_primary([entering, centred], width, height) is centred
    # Order must not decide it.
    assert engine._pick_primary([centred, entering], width, height) is centred


def test_centered_rule_without_a_frame_size_falls_back_to_largest():
    """No frame size means the centre is unknowable.

    Falling back to "largest" is the safe answer; quietly measuring from the
    origin again would reinstate the bug in the one code path nobody looks at.
    """
    def fake(cx, cy, half_w):
        return {
            "bbox": [cx - half_w, cy - 30, cx + half_w, cy + 30],
            "center": [cx, cy],
            "placement": "place",
            "orientation": {"confidence": 0.9},
        }

    small_near_origin = fake(60, 40, 20)
    large_far_away = fake(700, 380, 90)

    engine = _engine(primary_rule="centered")
    picked = engine._pick_primary([small_near_origin, large_far_away], 0, 0)
    assert picked is large_far_away


def test_unstable_stem_is_sent_for_reorientation_not_placed():
    """A stem direction that moves when the light changes is not a direction.

    This is the check that fixed erratic scans on green fruit, where the stem
    has to be found by shape rather than by colour. It is gated on a measured
    quantity, so the test drives it through the measurement: a fruit reported
    with a large stem spread must never come back as placeable.
    """
    image, _, _ = render_paprika(40, with_stem=True)
    engine = _engine()

    result = engine.evaluate(image)
    assert result["primary"]["placement"] == "place"

    # Same detection, but reported as having an unstable stem direction.
    detection = {
        "bbox": result["primary"]["bbox"],
        "confidence": 0.9,
        "keypoints": {},
        "stem_spread_deg": 30.0,
        "stem_method": "morphology",
        "colour": "green",
    }
    from backend.detection.paprika.orientation import Keypoint

    kp = result["primary"]["keypoints"]
    if kp:
        detection["keypoints"] = {
            name: Keypoint(v["x"], v["y"], v["confidence"], v["visible"])
            for name, v in kp.items()
        }
    evaluated = engine._evaluate_one(image, detection)

    assert evaluated["placement"] == "reorient"
    assert any("stem_unstable" in n for n in evaluated["orientation"]["notes"])


def test_stem_found_by_colour_is_treated_as_stable():
    """The colour route measured stable to well under a degree, so it must not
    be dragged down by the shape route's gate."""
    image, _, _ = render_paprika(40, with_stem=True)
    result = _engine().evaluate(image)
    primary = result["primary"]

    assert primary["stem_method"] == "hue"
    assert primary["stem_spread_deg"] == 0.0
    assert primary["placement"] == "place"


def test_a_usable_fruit_is_never_displaced_by_a_rejected_one():
    """An unusable fruit must never become primary while a usable one is
    present, not even if it is larger - what matters is what the robot can
    actually act on."""
    import numpy as np

    canvas = np.full((420, 900, 3), BELT, np.uint8)
    good, _, _ = render_paprika(30, size=400, with_stem=True)
    bad, _, _ = render_paprika(30, size=400, length=190, shoulder_w=130,
                               tip_w=110, with_stem=False)
    canvas[10:410, 470:870] = good
    canvas[10:410, 10:410] = bad

    result = _engine().evaluate(canvas)
    assert result["primary"]["placement"] == "place"


def test_classical_backend_finds_the_stem_on_every_colour():
    """Stem detection must not depend on fruit colour - on green the colour
    method fails by definition and morphology has to take over."""
    from backend.detection.paprika import classical

    for name, colour in COLORS.items():
        image, _, _ = render_paprika(40, color=colour, with_stem=True)
        fruits = classical.find_fruit(image)
        assert fruits, f"no fruit found on {name}"
        assert fruits[0].stem_end is not None, f"no stem found on {name}"


def test_engine_reports_nok_on_empty_belt():
    image = np.full((400, 400, 3), BELT, np.uint8)
    result = _engine().evaluate(image)

    assert result["status"] == "NOK"
    assert result["failure_reason"] == "no_paprika_detected"
    assert result["primary"] is None


def test_engine_applies_the_plc_frame_mapping():
    image, _, _ = render_paprika(0, with_stem=True)
    result = _engine(frame={"angle_offset_deg": 90.0, "angle_invert": True}).evaluate(image)

    # vision 0 -> invert -> 0 -> +90 -> 90
    assert orient.angular_difference(result["primary"]["angle_plc"], 90.0) < 5.0


def test_labels_stay_ascii_for_opencv():
    """Overlay text is drawn with Hershey fonts, which render anything
    non-ASCII as '??'."""
    image, _, _ = render_paprika(25)
    label = _engine().evaluate(image)["primary"]["label"]
    assert label.isascii(), f"non-ASCII label would render as ?? : {label!r}"