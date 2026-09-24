"""
Orientation Geometry

Turns a detected paprika into the thing the PLC actually wants: an angle, a
position, and a statement about which way up the fruit is lying.

Two independent estimators live here, because relying on the stem alone fails
on exactly the fruit you care about most.

1. Keypoints. The pose model predicts `stem_end` and `blossom_end`. The vector
   between them is the orientation, full 360 degrees, no ambiguity. This is the
   primary estimator.

2. Shape. A paprika is not symmetric along its long axis: the stem end carries
   the shoulder and is measurably wider, and the blossom end tapers to the
   lobed tip. PCA on the fruit mask gives the axis; the width profile along
   that axis says which end is the shoulder. This works on a fruit with no
   stem at all, because it never looks at the stem.

   It also breaks on a fruit that DOES still have a stem, if that stem is
   left in the mask: a visible stub sticking past the shoulder pulls that
   end's slice of the width profile down towards the stem's own width
   instead of the shoulder's, and on a real stemmed fruit that swing is
   large enough to flip the sign with a confident-looking margin - not an
   occasional miss, the measured case below flips every single time. So
   before any of that geometry runs, `shape_orientation()` first asks
   classical.stem_by_morphology() whether a stem-shaped protrusion is
   sitting on the silhouette at all. Found: it is stripped out of the mask
   the width profile is measured on, and its own position relative to the
   body becomes the flip signal directly - a visible stem, however short,
   is stronger evidence than a taper ever was, and it settles fruit the
   width profile alone cannot (a near-round blokpaprika with no taper to
   read). Not found: shape_orientation() falls back to the width profile
   exactly as before.

Neither is trusted blindly. `fuse()` combines them and reports which one won,
so a disagreement is visible in the log and on the HMI instead of silently
picking one.

A third signal that does not share their blind spot
-----------------------------------------------------
Keypoints and shape both end up looking at the same evidence: a stem the
pose model has to see, or a mask shape_orientation() has to segment out of
the same frame. On a fruit where that evidence is misleading - a shadow
that reads as a stem stub, a segmentation gap that flattens the taper -
both can be confidently wrong together, and agreeing with itself is not
independent confirmation.

`end_on.groove_axis()` reads the fruit's own surface ridges instead of its
outline: a pepper's grooves run stem to blossom, and their shared direction
is the axis, measured from texture rather than from either mask. It cannot
say which end holds the stem - a groove looks the same from both ends, see
`end_on.groove_axis`'s own docstring - so `groove_crosscheck()` below never
touches `angle_deg` or `flip_confidence`. What it does is compare that axis
to whatever `fuse()` already settled on and, where a fruit's own grooves
disagree by more than `policy.max_groove_disagreement_deg`, say so loudly
enough for `PaprikaEngine._placement_for()` to reorient rather than place -
the same "disagreement is a warning, not a tie to break" policy this module
already applies to keypoints vs. shape, extended to a signal that does not
run through either one's mask. Gated on `policy.min_groove_coherence`
throughout, same as the runtime's groove fallback: a smooth fruit has no
grooves to disagree WITH.

Angle convention
----------------
`angle_deg` is the direction pointing FROM the blossom end TOWARD the stem end
- that is, "where the stem is". Measured in degrees, 0-360:

    0   = stem points right   (+x on screen)
    90  = stem points up      (toward the top of the image)
    180 = stem points left
    270 = stem points down

Screen-intuitive, so an operator watching the HMI can sanity-check it without
converting anything. It is deliberately NOT the robot's frame - see
`apply_frame_convention()` and the `angle_offset_deg` / `angle_invert` config,
which map this onto whatever zero and handedness the PLC expects without
touching any code.

The 180-degree problem
----------------------
PCA gives an axis, not a direction: it cannot tell a paprika from the same
paprika rotated 180 degrees. Everything in `shape_orientation()` that looks
asymmetric - width profile, mass skew - exists purely to break that tie. When
the fruit is too symmetric to call, `flip_confidence` comes back low rather
than the function guessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np

from backend.detection.paprika.classical import stem_by_morphology

# Canonical keypoint order. Everything - the annotation spec, the dataset
# validator, the trained model's output, and this module - agrees on this
# ordering. Changing it means re-exporting the dataset, so it lives in exactly
# one place.
KEYPOINT_NAMES: tuple[str, ...] = ("stem_end", "blossom_end")
KP_STEM = 0
KP_BLOSSOM = 1

# Pose classes.
POSE_LYING = "lying"
POSE_STANDING_STEM_UP = "standing_stem_up"
POSE_STANDING_STEM_DOWN = "standing_stem_down"
POSE_UNKNOWN = "unknown"
# Stem pointing down or tucked under the fruit: you are looking at the blossom
# end. There is no in-plane angle to measure, and the robot cannot pick it up
# and turn it over anyway, so this is an end state and not a failed measurement.
POSE_UPSIDE_DOWN = "upside_down"
# No stem could be found on a fruit that is fully in frame. Deliberately NOT
# POSE_UPSIDE_DOWN: that pose asserts something about the fruit (it is lying
# blossom-up), while this one only reports what the backend managed to see.
# Both are rejected, and identically - the distinction buys nothing at the
# actuator and everything in the log. A climbing upside_down count points at
# the infeed, a climbing stem_not_found count points at the stem detector, the
# lighting or the cultivar, and those are fixed by different people on
# different days. Conflated, the number tells you only that something is wrong.
POSE_STEM_NOT_FOUND = "stem_not_found"
# Fruit continues outside the frame. Says nothing about the fruit and
# everything about the image - the centroid and angle would come from half a
# paprika.
POSE_INCOMPLETE = "incomplete"


# --------------------------------------------------------------------- types


@dataclass
class Keypoint:
    """One predicted landmark.

    Attributes:
        x, y:       pixel position in the full frame.
        confidence: model confidence for this landmark.
        visible:    False when the model believes the landmark exists but is
                    hidden (fruit standing on its blossom end, stem pointing
                    away from the camera). A hidden landmark still has a
                    usable position - that is the whole reason the annotation
                    spec insists on labelling occluded landmarks rather than
                    dropping them.
    """

    x: float
    y: float
    confidence: float = 0.0
    visible: bool = True

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Orientation:
    """The orientation verdict for one fruit."""

    angle_deg: Optional[float] = None
    axis_deg: Optional[float] = None
    source: str = "none"           # keypoints | shape | fused | none
    confidence: float = 0.0        # certainty about the axis
    flip_confidence: float = 0.0   # certainty about which end is the stem
    pose: str = POSE_UNKNOWN
    elongation: float = 0.0        # major/minor axis ratio of the mask
    agreement_deg: Optional[float] = None  # keypoint vs shape disagreement
    groove_agreement_deg: Optional[float] = None  # settled axis vs groove axis
    stem_present: bool = False
    stem_span_px: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("angle_deg", "axis_deg", "agreement_deg", "groove_agreement_deg"):
            if data[key] is not None:
                data[key] = round(float(data[key]), 2)
        for key in ("confidence", "flip_confidence", "elongation", "stem_span_px"):
            data[key] = round(float(data[key]), 3)
        return data


# ----------------------------------------------------------------- primitives


def _norm360(angle: float) -> float:
    return angle % 360.0


def _norm180(angle: float) -> float:
    """Fold an angle onto 0-180, i.e. treat it as an axis rather than a ray."""
    return angle % 180.0


def angular_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two directions, 0-180."""
    diff = abs(_norm360(a) - _norm360(b)) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def axis_difference(a: float, b: float) -> float:
    """Smallest difference between two AXES (0-180 orientations), 0-90.

    An axis has no front or back - a groove reading of 175 and a keypoint
    direction of 10 describe the same line through the fruit, not a near
    180-degree disagreement. Folding both onto 0-180 first and then onto
    0-90 is what `tools/eval_estimators.py`'s `axis_error_deg` already does
    for scoring against ground truth; this is the same formula, in one
    place, so the runtime cross-check and the offline scorer cannot drift
    apart the way keypoint and shape angle math briefly did before
    `vector_to_angle` was pulled out for the same reason.
    """
    return abs((_norm180(a) - _norm180(b) + 90.0) % 180.0 - 90.0)


def vector_to_angle(dx: float, dy: float) -> float:
    """Convert an image-space vector to the screen-intuitive convention.

    Image y grows downward, so dy is negated to make 90 degrees mean "up on
    screen". Getting this backwards is the single easiest bug to introduce
    here, which is why it is one function used everywhere rather than an
    inline atan2 in three places.
    """
    return _norm360(math.degrees(math.atan2(-dy, dx)))


def apply_frame_convention(
    angle_deg: Optional[float],
    offset_deg: float = 0.0,
    invert: bool = False,
) -> Optional[float]:
    """Map the screen convention onto the robot/PLC convention.

    Args:
        angle_deg:  angle in this module's convention, or None.
        offset_deg: added after any inversion. Set this to whatever makes the
                    machine's zero line up with the vision zero.
        invert:     flip handedness (counter-clockwise to clockwise). Needed
                    when the actuator counts the opposite way round.
    """
    if angle_deg is None:
        return None
    value = -float(angle_deg) if invert else float(angle_deg)
    return _norm360(value + float(offset_deg))


# ------------------------------------------------------------ keypoint route


def keypoint_orientation(
    stem: Optional[Keypoint],
    blossom: Optional[Keypoint],
    reference_size_px: float,
    min_span_ratio: float = 0.18,
) -> Orientation:
    """Derive orientation from the two predicted landmarks.

    Args:
        stem:               the `stem_end` landmark, or None if not predicted.
        blossom:            the `blossom_end` landmark, or None.
        reference_size_px:  a size to judge the landmark separation against -
                            normally the bounding box diagonal. Used to decide
                            whether the fruit is lying down or standing on end.
        min_span_ratio:     separation below this fraction of reference_size_px
                            means the long axis is pointing at the camera, so
                            the on-screen angle is meaningless.

    Returns:
        An Orientation. When the fruit is standing, `angle_deg` is None and
        `pose` says which end is up - because a rotation angle for a fruit
        standing on its end is not just unknown, it is not a real quantity,
        and returning a plausible-looking number for it would be worse than
        returning nothing.
    """
    result = Orientation(source="keypoints")

    if stem is None or blossom is None:
        result.notes.append("missing_keypoint")
        return result

    dx = stem.x - blossom.x
    dy = stem.y - blossom.y
    span = math.hypot(dx, dy)

    result.stem_span_px = span
    result.stem_present = bool(stem.visible)

    reference = max(1.0, float(reference_size_px))
    span_ratio = span / reference

    # Both landmarks projecting onto nearly the same point means the fruit's
    # long axis runs into the camera - it is standing on one end.
    if span_ratio < min_span_ratio:
        result.pose = POSE_STANDING_STEM_UP if stem.visible else POSE_STANDING_STEM_DOWN
        result.confidence = min(stem.confidence, blossom.confidence)
        result.flip_confidence = result.confidence
        result.notes.append(f"standing (span_ratio={span_ratio:.2f})")
        return result

    result.pose = POSE_LYING
    result.angle_deg = vector_to_angle(dx, dy)
    result.axis_deg = _norm180(result.angle_deg)

    # Axis confidence tracks the weaker of the two landmarks; a long baseline
    # between them makes the direction less sensitive to per-landmark jitter,
    # so short spans are penalised.
    pair_conf = min(stem.confidence, blossom.confidence)
    result.confidence = float(np.clip(pair_conf * min(1.0, span_ratio / 0.5), 0.0, 1.0))

    # Knowing which end is which is exactly what the keypoints encode, so the
    # flip is as certain as the landmarks themselves.
    result.flip_confidence = pair_conf
    return result


# --------------------------------------------------------------- shape route


def segment_fruit(
    frame: np.ndarray,
    bbox: Optional[tuple[int, int, int, int]] = None,
    saturation_floor: int = 80,
    belt_hue: tuple[int, int] = (96, 145),
    value_floor: int = 45,
) -> Optional[np.ndarray]:
    """Isolate the fruit from the belt inside `bbox`.

    Two rules, and both are needed:

    1. Exclude the belt's own hue. A blue belt is *strongly* saturated, so
       saturation alone cannot separate it from fruit - which is why the
       earlier saturation-only version of this function failed on the real
       line. Hue can, and comfortably: measured across the sample set, the
       belt sits in a tight band at hue 100-125 while every paprika colour
       falls outside it. Blue is a deliberate choice in food handling exactly
       because no produce is blue.

    2. Require saturation and brightness. This is what removes the pale
       structural strips at the edges of the belt, which are unsaturated and
       would otherwise survive rule 1, along with deep shadow.

    Args:
        frame:            full BGR frame.
        bbox:             (x1, y1, x2, y2) to crop to. Whole frame when None.
        saturation_floor: minimum saturation for a fruit pixel.
        belt_hue:         (low, high) OpenCV hue range of the belt, excluded.
                          Set to (0, 0) on a neutral belt to fall back to
                          saturation alone. Verify on your own line by
                          sampling belt pixels in HSV (OpenCV hue is 0-179,
                          not 0-359); tools/tune_shape.py writes
                          fruit | mask | overlay montages that show directly
                          whether the belt is being excluded.
        value_floor:      minimum brightness, to drop shadow.

    Returns:
        A uint8 mask (0/255) the size of the crop, containing only the largest
        blob, or None when nothing survives. Largest-blob-only matters because
        a neighbouring fruit clipping the corner of the box would otherwise
        drag the PCA axis off the fruit being measured.
    """
    if frame is None or frame.size == 0:
        return None

    if bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
    else:
        crop = frame

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    keep = (saturation >= saturation_floor) & (value >= value_floor)
    low, high = belt_hue
    if high > low:
        keep &= ~((hue >= low) & (hue <= high))
    mask = keep.astype(np.uint8) * 255

    # Close specular highlights (which read as unsaturated and punch holes in
    # the middle of a glossy fruit) before measuring anything.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def _principal_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """PCA over mask pixel coordinates.

    Returns:
        (centroid, major_unit_vector, major_sd, minor_sd)
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    cov = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    major = eigenvectors[:, 0]
    major_sd = float(math.sqrt(max(eigenvalues[0], 1e-9)))
    minor_sd = float(math.sqrt(max(eigenvalues[1], 1e-9)))
    return centroid, major, major_sd, minor_sd


# A protrusion may not remove more than this fraction of the mask's own area
# before it stops being treated as a stem. A genuine stem stub is nowhere
# close: across every protrusion stem_by_morphology() found on the 254
# labelled crops in data/debug/labelling, the largest was 32% of the fruit's
# area, p95 was 7% and the median 2%. Above 50% what was "removed" is most of
# the silhouette - a shadow crease or a segmentation gap read as an opening -
# and treating that as a stem would hollow the fruit out rather than clean it
# up.
PROTRUSION_MAX_AREA_RATIO = 0.5


def shape_orientation(
    mask: np.ndarray,
    bins: int = 12,
    end_fraction: float = 0.3,
    min_elongation: float = 1.12,
    detect_protrusion: bool = True,
) -> Orientation:
    """Estimate orientation from fruit silhouette alone.

    The stem end of a paprika is its widest part - the shoulder where the
    calyx sits. The blossom end tapers. Measuring the width profile along the
    long axis therefore says which end is which, without ever looking for a
    stem. That is the entire point of this estimator: it is the one that still
    works on a fruit whose stem broke off in the crate.

    But most fruit reaching this path have NOT lost their stem - they simply
    were not trusted by the colour/morphology route, or shape is being run
    as a crosscheck alongside it. A visible stem stub, left in the mask, is
    not neutral: it drags that end's slice of the width profile toward the
    stem's own (narrow) width instead of the shoulder's, and on a real
    stemmed fruit the swing is large enough to flip the sign with a
    confident-looking margin every time - see the module docstring. So
    before any width-profile geometry runs, this asks
    classical.stem_by_morphology() - the same colour-independent "open with a
    kernel wider than the stem" search classical.py uses - whether a
    stem-shaped protrusion sits on the silhouette at all:

    - Found: it is subtracted out of the mask the width profile is measured
      on, so the profile describes the body alone, and the protrusion's own
      position relative to the body becomes the flip signal directly. A
      visible stem, however short, is stronger evidence than a taper - it is
      also the one signal that still works on a near-round blokpaprika, which
      has essentially no taper to read (see `shape_crosscheck` in
      config/default.yaml for how badly the taper-only version of this
      function did on real, mostly-round fruit).
    - Not found: falls back to the width profile exactly as before. Nothing
      here changes a stemless fruit's answer.

    Args:
        mask:              uint8 mask of one fruit (0/255).
        bins:              slices along the long axis for the width profile.
        end_fraction:      fraction of the length at each end that counts as
                           "the end" when comparing widths.
        min_elongation:    below this major/minor ratio the BODY (after any
                           protrusion is removed) is too round for a long axis
                           to mean anything from width alone - but a
                           protrusion found on a round body still yields an
                           angle, since its direction does not depend on the
                           body having a taper.
        detect_protrusion: set False to measure the raw mask exactly as the
                           original width-profile-only version did - useful
                           for comparing the two directly in tools/tune_shape.py.

    Returns:
        An Orientation carrying angles only - no positions - so nothing here
        needs the crop offset added back on: a direction measured inside the
        crop is the same direction in the full frame. `flip_confidence` is the
        number to watch: it is how strongly the ends differed (width route) or
        how well the protrusion's own kernel-scale agreement held up
        (protrusion route), and it collapses toward zero on a fruit shape
        cannot settle either way.
    """
    result = Orientation(source="shape")

    if mask is None or mask.size == 0:
        result.notes.append("empty_mask")
        return result

    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        result.notes.append("mask_too_small")
        return result

    # Strip a stem-shaped protrusion, if any, before any axis is measured -
    # see the docstring above for why leaving it in corrupts the geometry
    # rather than merely adding noise to it.
    protrusion: Optional[np.ndarray] = None
    protrusion_quality = 0.0
    body = mask
    if detect_protrusion:
        found, quality = stem_by_morphology(mask)
        if found is not None:
            candidate_body = cv2.subtract(mask, found)
            if int((candidate_body > 0).sum()) >= (1.0 - PROTRUSION_MAX_AREA_RATIO) * int(
                (mask > 0).sum()
            ):
                body, protrusion, protrusion_quality = candidate_body, found, quality
            else:
                result.notes.append("protrusion_rejected_too_large")

    body_ys, body_xs = np.nonzero(body)
    if len(body_xs) < 30:
        # A genuine stem stub can never take the body below this - getting
        # here means the "protrusion" WAS most of the fruit. Measure the
        # untouched mask rather than reporting nothing.
        body = mask
        protrusion = None
        body_ys, body_xs = ys, xs

    points = np.column_stack([body_xs, body_ys]).astype(np.float64)
    centroid, major, major_sd, minor_sd = _principal_axis(points)

    elongation = major_sd / max(minor_sd, 1e-6)
    result.elongation = elongation

    # Where the protrusion sits relative to the body, in body-radii. Computed
    # up front because it is needed both to decide whether a round body still
    # gets an answer and, later, as the confidence in that answer.
    protrusion_centroid: Optional[np.ndarray] = None
    protrusion_offset_ratio = 0.0
    if protrusion is not None:
        p_ys, p_xs = np.nonzero(protrusion)
        if len(p_xs) > 0:
            protrusion_centroid = np.array([p_xs.mean(), p_ys.mean()])
            body_radius = math.sqrt(max(int((body > 0).sum()), 1) / math.pi)
            protrusion_offset_ratio = float(
                np.linalg.norm(protrusion_centroid - centroid) / max(body_radius, 1e-6)
            )

    if elongation < min_elongation and protrusion_centroid is None:
        # Round silhouette, no visible stem either: the fruit is standing on
        # an end, pointing at the camera. Which end is up is not answerable
        # from the outline alone.
        result.pose = POSE_UNKNOWN
        result.notes.append(f"too_round (elongation={elongation:.2f})")
        return result

    result.pose = POSE_LYING

    minor = np.array([-major[1], major[0]])
    centred = points - centroid
    t = centred @ major           # position along the long axis
    s = centred @ minor           # offset across it

    t_min, t_max = float(t.min()), float(t.max())
    length = max(t_max - t_min, 1e-6)

    # Width profile: how wide the BODY is at each slice along its length. Runs
    # unconditionally, even when a protrusion will decide the answer below -
    # it is cheap, and comparing the two signals is exactly what tells a
    # frame worth a second look apart from one where they simply agree.
    edges = np.linspace(t_min, t_max, bins + 1)
    widths = np.zeros(bins)
    for i in range(bins):
        in_bin = (t >= edges[i]) & (t < edges[i + 1] if i < bins - 1 else t <= edges[i + 1])
        if np.any(in_bin):
            slice_s = s[in_bin]
            widths[i] = float(slice_s.max() - slice_s.min())

    end_bins = max(1, int(round(bins * end_fraction)))
    width_low = float(np.mean(widths[:end_bins]))     # the -major end
    width_high = float(np.mean(widths[-end_bins:]))   # the +major end
    width_scale = max(width_low + width_high, 1e-6)
    # Positive => the +major end is the wider (stem) end.
    width_signal = (width_high - width_low) / width_scale

    # Second, independent tie-breaker: where the mass sits. A shape that is
    # fatter at one end has its centroid pulled toward that end relative to
    # the midpoint of its own length.
    midpoint = (t_min + t_max) / 2.0
    mass_signal = float(np.clip((0.0 - midpoint) / (length / 2.0), -1.0, 1.0))
    combined = 0.75 * width_signal + 0.25 * mass_signal

    if protrusion_centroid is not None:
        direction = protrusion_centroid - centroid
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-6:
            stem_direction = direction / direction_norm
            width_profile_direction = major if combined >= 0 else -major
            if float(stem_direction @ width_profile_direction) < 0:
                # The taper and the visible stem point different ways. The
                # protrusion still wins - it is direct evidence, the taper is
                # an inference - but a frame where they disagree is worth
                # finding again later, not silently overwritten.
                result.notes.append("protrusion_overrides_width_profile")

            result.axis_deg = _norm180(vector_to_angle(stem_direction[0], stem_direction[1]))
            result.angle_deg = vector_to_angle(stem_direction[0], stem_direction[1])

            # First-pass estimate, not yet checked against labelled angles:
            # a protrusion that barely clears the centroid is weaker evidence
            # than one standing a full body-radius clear of it. Floor of 0.5
            # rather than 0.0 because reaching this point already required
            # stem_by_morphology's own two-of-three kernel-scale agreement -
            # this is additional evidence on top of that, not the only
            # evidence. Re-check with tools/tune_shape.py once labelled
            # angles exist for real crops, the same way the 4.0 below was.
            result.confidence = float(
                np.clip(0.5 + 0.5 * min(1.0, protrusion_offset_ratio), 0.0, 1.0)
            )
            # stem_by_morphology's own quality is kernel-scale agreement,
            # which is exactly "how much do I trust this IS the stem" - the
            # same question flip_confidence answers here.
            result.flip_confidence = float(protrusion_quality)

            result.notes.append(
                f"protrusion_offset_ratio={protrusion_offset_ratio:.2f} "
                f"protrusion_quality={protrusion_quality:.2f}"
            )
            result.notes.append(
                f"width_signal={width_signal:+.3f} mass_signal={mass_signal:+.3f}"
            )
            return result

        # A protrusion sitting exactly on the centroid carries no direction of
        # its own - fall through to the width profile as if none were found.
        result.notes.append("protrusion_at_centroid_ignored")

    if elongation < min_elongation:
        # Reachable only via the fallthrough above: a protrusion was found
        # but gave no usable direction, and the body itself is too round for
        # the width profile to answer either.
        result.pose = POSE_UNKNOWN
        result.notes.append(f"too_round (elongation={elongation:.2f})")
        return result

    stem_direction = major if combined >= 0 else -major

    result.axis_deg = _norm180(vector_to_angle(major[0], major[1]))
    result.angle_deg = vector_to_angle(stem_direction[0], stem_direction[1])

    # Axis confidence rises with elongation: a long thin fruit has an
    # unmistakable axis, a nearly round one does not.
    result.confidence = float(np.clip((elongation - 1.0) / 1.0, 0.0, 1.0))
    # Flip confidence is how asymmetric the fruit actually was. A perfectly
    # symmetric silhouette gives 0, and the caller should then lean on the
    # keypoints instead of this estimate.
    #
    # The 4.0 is calibrated, not arbitrary. Measured on a synthetic fruit with
    # a pronounced 95px shoulder tapering to a 52px tip - a stronger taper than
    # most real paprikas - `combined` only reaches about 0.17, because
    # averaging over the end 30% of the length deliberately blunts the
    # extremes. Scaling by 4 puts that clear-cut case near 0.68, comfortably
    # above shape_flip_override, while a near-symmetric fruit still lands
    # around 0.1. Re-check this against real masks once the dataset lands:
    # tools/tune_shape.py prints `combined` per image for exactly this.
    result.flip_confidence = float(np.clip(abs(combined) * 4.0, 0.0, 1.0))

    if protrusion is not None:
        # Found, but gave no usable direction (handled above) - noted here so
        # a fruit with a real but centred stem is distinguishable in the log
        # from one that never had a protrusion candidate at all.
        result.notes.append("protrusion_found_but_ambiguous")
    result.notes.append(f"width_signal={width_signal:+.3f} mass_signal={mass_signal:+.3f}")
    return result


# ---------------------------------------------------------------- fusion


def fuse(
    keypoint_result: Orientation,
    shape_result: Optional[Orientation],
    disagreement_threshold_deg: float = 35.0,
    shape_flip_override: float = 0.55,
) -> Orientation:
    """Combine the two estimators into the answer the machine acts on.

    Policy, in order:

    - Keypoints win the axis whenever they produced one. They are trained on
      this exact fruit under this exact lighting; the shape estimator is a
      geometric prior that knows nothing about either.
    - Shape can override the *flip* - and only the flip - when the keypoints
      are unsure which end is the stem but the silhouette clearly is. This is
      the stemless case: the model sees no stem, hedges on `stem_end`, and the
      shoulder-versus-tip asymmetry settles it.
    - Shape supplies the whole answer when the keypoints produced nothing.
    - Disagreement is recorded, never averaged away. Averaging two directions
      that differ by 170 degrees produces a number that is confidently
      perpendicular to both, which is the worst possible failure for a
      placement machine. A flagged disagreement can be rejected; a plausible
      wrong angle cannot.
    """
    if shape_result is None:
        return keypoint_result

    have_kp = keypoint_result.angle_deg is not None
    have_shape = shape_result.angle_deg is not None

    if not have_kp and not have_shape:
        merged = keypoint_result
        # A round silhouette corroborates "standing" even without landmarks.
        if shape_result.elongation and shape_result.elongation < 1.15:
            merged.notes.append("shape_agrees_standing")
        merged.elongation = shape_result.elongation
        return merged

    if not have_kp:
        shape_result.notes.append("keypoints_unavailable")
        return shape_result

    if not have_shape:
        keypoint_result.notes.append("shape_unavailable")
        return keypoint_result

    merged = Orientation(
        angle_deg=keypoint_result.angle_deg,
        axis_deg=keypoint_result.axis_deg,
        source="fused",
        confidence=keypoint_result.confidence,
        flip_confidence=keypoint_result.flip_confidence,
        pose=keypoint_result.pose,
        elongation=shape_result.elongation,
        stem_present=keypoint_result.stem_present,
        stem_span_px=keypoint_result.stem_span_px,
        notes=list(keypoint_result.notes) + list(shape_result.notes),
    )

    disagreement = angular_difference(keypoint_result.angle_deg, shape_result.angle_deg)
    merged.agreement_deg = disagreement

    # A ~180 degree split means the two agree on the axis and disagree only on
    # which end the stem is - a flip dispute, not an axis dispute.
    flipped_disagreement = abs(180.0 - disagreement)

    if flipped_disagreement < disagreement_threshold_deg:
        if shape_result.flip_confidence > shape_flip_override > keypoint_result.flip_confidence:
            merged.angle_deg = shape_result.angle_deg
            merged.flip_confidence = shape_result.flip_confidence
            merged.notes.append("flip_taken_from_shape")
        else:
            # Keypoints keep the call, but the dispute is on the record and the
            # confidence is cut so a downstream threshold can catch it.
            merged.flip_confidence *= 0.5
            merged.notes.append("flip_disputed")
    elif disagreement > disagreement_threshold_deg:
        merged.confidence *= 0.6
        merged.notes.append(f"axis_disagreement={disagreement:.0f}deg")
    else:
        # Independent agreement is real evidence, so allow a modest boost.
        merged.confidence = float(min(1.0, merged.confidence * 1.1))
        merged.notes.append("estimators_agree")

    return merged


def groove_crosscheck(
    result: Orientation,
    groove_axis_deg: Optional[float],
    coherence: Optional[float],
    min_coherence: float = 0.35,
    disagreement_threshold_deg: float = 30.0,
) -> Orientation:
    """Check a settled axis against the fruit's own surface grooves.

    See the module docstring for why this exists as a third signal rather
    than a third vote: it reads texture, not either mask, so it can catch
    the case where keypoints and shape are wrong in the same direction
    because they were looking at the same misleading evidence.

    Deliberately narrow. This never sets `angle_deg` or touches
    `flip_confidence` - a groove has no front or back, so it is not
    evidence about which end the stem is on, only about the line through
    the fruit. It only ever:

    - records the disagreement in `groove_agreement_deg`, so
      `PaprikaEngine._placement_for()` can reorient on it exactly as it
      already does for `agreement_deg` (keypoints vs. shape), and
    - nudges `confidence` up on agreement or down on disagreement, the same
      modest way `fuse()` does for its own two estimators - not a
      substitute for the hard stop, since a confidence multiplier alone was
      already shown not to reliably trigger it (see
      `tests/test_disagreement_gate.py::test_confidence_alone_would_not_have_caught_it`).

    A no-op whenever there is nothing to compare: no settled angle, no
    groove reading, or a groove reading too faint to trust
    (`coherence < min_coherence` - the same floor the runtime's own groove
    fallback uses, for the same reason: below it the "axis" is just the
    direction of whatever noise happened to be strongest).
    """
    if result.angle_deg is None or groove_axis_deg is None or coherence is None:
        return result
    if coherence < min_coherence:
        return result

    disagreement = axis_difference(result.angle_deg, groove_axis_deg)
    result.groove_agreement_deg = disagreement

    if disagreement > disagreement_threshold_deg:
        result.confidence *= 0.6
        result.notes.append(f"groove_axis_disagreement={disagreement:.0f}deg")
    else:
        result.confidence = float(min(1.0, result.confidence * 1.05))
        result.notes.append("groove_agrees")

    return result


# ------------------------------------------------------------------ helpers


def estimate(
    frame: Optional[np.ndarray],
    bbox: tuple[int, int, int, int],
    stem: Optional[Keypoint],
    blossom: Optional[Keypoint],
    use_shape: bool = True,
    saturation_floor: int = 80,
    belt_hue: tuple[int, int] = (96, 145),
    min_span_ratio: float = 0.18,
) -> Orientation:
    """Run both estimators for one detection and fuse them.

    Args:
        frame:            full BGR frame. Pass None to skip shape analysis.
        bbox:             (x1, y1, x2, y2) of this fruit.
        stem, blossom:    landmarks in FULL-FRAME pixel coordinates.
        use_shape:        disable to run keypoints only (faster, and the right
                          choice if the belt turns out to be as saturated as
                          the fruit).
        saturation_floor: passed to segment_fruit().
        min_span_ratio:   passed to keypoint_orientation().
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    diagonal = math.hypot(max(1, x2 - x1), max(1, y2 - y1))

    kp_result = keypoint_orientation(stem, blossom, diagonal, min_span_ratio=min_span_ratio)

    shape_result: Optional[Orientation] = None
    if use_shape and frame is not None:
        mask = segment_fruit(
            frame, bbox, saturation_floor=saturation_floor, belt_hue=belt_hue
        )
        if mask is not None:
            shape_result = shape_orientation(mask)

    return fuse(kp_result, shape_result)
