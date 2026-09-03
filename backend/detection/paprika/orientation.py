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

Neither is trusted blindly. `fuse()` combines them and reports which one won,
so a disagreement is visible in the log and on the HMI instead of silently
picking one.

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
    stem_present: bool = False
    stem_span_px: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("angle_deg", "axis_deg", "agreement_deg"):
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
                          saturation alone. Verify with tools/measure_belt.py.
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


def shape_orientation(
    mask: np.ndarray,
    bins: int = 12,
    end_fraction: float = 0.3,
    min_elongation: float = 1.12,
) -> Orientation:
    """Estimate orientation from fruit silhouette alone.

    The stem end of a paprika is its widest part - the shoulder where the
    calyx sits. The blossom end tapers. Measuring the width profile along the
    long axis therefore says which end is which, without ever looking for a
    stem. That is the entire point of this estimator: it is the one that still
    works on a fruit whose stem broke off in the crate.

    Args:
        mask:            uint8 mask of one fruit (0/255).
        bins:            slices along the long axis for the width profile.
        end_fraction:    fraction of the length at each end that counts as
                         "the end" when comparing widths.
        min_elongation:  below this major/minor ratio the fruit is too round
                         for a long axis to mean anything - report standing
                         rather than inventing an axis out of noise.

    Returns:
        An Orientation whose coordinates are relative to the mask, so the
        caller must add the crop offset back on. `flip_confidence` is the
        number to watch: it is how strongly the two ends actually differed,
        and it collapses toward zero on a symmetric fruit.
    """
    result = Orientation(source="shape")

    if mask is None or mask.size == 0:
        result.notes.append("empty_mask")
        return result

    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        result.notes.append("mask_too_small")
        return result

    points = np.column_stack([xs, ys]).astype(np.float64)
    centroid, major, major_sd, minor_sd = _principal_axis(points)

    elongation = major_sd / max(minor_sd, 1e-6)
    result.elongation = elongation

    if elongation < min_elongation:
        # Round silhouette: the fruit is standing on an end, pointing at the
        # camera. Which end is up is not answerable from the outline alone.
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

    # Width profile: how wide the fruit is at each slice along its length.
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
