"""
Classical paprika detection: fruit, stem and calyx without a model

Shared by the `shape` backend in pose_detector.py and by
tools/pre_annotate.py, so the runtime and the pre-annotation are guaranteed to
see the same thing. If the two drifted apart you would be annotating a dataset
against a different definition from the one the machine applies.

Segmentation: hue, not saturation
---------------------------------
The blue belt is strongly saturated, so saturation cannot separate it from
fruit - that is what the first version broke on. Hue can, comfortably:
measured across 160 images the belt sits in a tight band at hue 100-125 and
every paprika colour falls outside it. Blue is chosen in food handling for
exactly this reason: no produce is blue. Saturation and brightness floors do
the rest, removing the pale edge strips and deep shadow.

Stem: two methods, in this order
--------------------------------
1. Colour. On a red, orange or yellow fruit the green stem is trivially
   separable. Measured hit rate 87%, and this is the more accurate of the two.

2. Morphology. An opening with a kernel wider than the stem lifts it off the
   fruit; the difference is the stem. Colour-independent, so it also works on a
   green paprika where method 1 fails by definition. Measured hit rate 86% on
   green fruit, and on non-green fruit it agrees with method 1 to within 13 px
   median - that is validation against a known reference, not an assumption.

Together they cover well over 90%. What remains is genuinely stemless or has
the stem underneath the fruit, and that should become an explicit "I don't
know".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Calibrated on the real line. Re-check with tools/tune_shape.py.
DEFAULT_BELT_HUE = (96, 145)
DEFAULT_SATURATION_FLOOR = 80
DEFAULT_VALUE_FLOOR = 45
DEFAULT_STEM_HUE = (33, 92)

# Largest a fruit may be as a fraction of the frame. Measured on 243 real
# paprikas: the 99th percentile is 10.1% of the frame. The old 0.7 let through
# blobs seven times larger than any paprika, which is how a crate at the edge
# of the frame became a "fruit" - and, because the morphology kernel scales
# with the blob, a 630 ms one.
DEFAULT_MAX_AREA_RATIO = 0.18

# Radius, in pixels, that a fruit is scaled to before the stem morphology runs.
#
# Capping the kernel instead was tried and is not enough: morphological cost
# grows with the SQUARE of the kernel, so a 121x121 opening costs 110 ms on its
# own and nine of them per fruit put a frame at 1.4 seconds. Normalising the
# fruit first makes the kernel small and constant - about 2 ms per opening -
# so the cost of a frame no longer depends on how big the object in it is.
#
# It also removes a source of inconsistency: at a fixed kernel, a large and a
# small paprika were being eroded by different relative amounts.
STEM_NORMALISED_RADIUS = 60

# Belt detection: the belt is the one large blue region. Below this fraction of
# the frame it is not recognisable as a belt and the restriction is skipped
# rather than guessed at.
BELT_MIN_FRAME_FRACTION = 0.15
BELT_SCALE = 0.25
BELT_RING_DILATION = 21
# How much of the ring around a fruit may be foreign - neither belt nor other
# produce - before it is treated as something standing beside the belt rather
# than product on it.
MAX_FOREIGN_RING = 0.45

GREEN_FRUIT_HUE = (33, 95)
MIN_STEM_AREA_HUE = 150

# A stem can never be this large a share of its own fruit. Measured over 129
# fruit the ratio is 0.052 median and 0.094 at p95, so 0.20 rejects only the
# cases where the "stem" is in fact a neighbouring green fruit that the colour
# route swallowed - which is how a touching green pepper used to disappear into
# the red one beside it.
MAX_STEM_AREA_RATIO = 0.20

# How far a pixel's hue must sit from the FRUIT'S OWN hue to count as stem.
# Measured against the fruit rather than against a fixed green band, because
# stems are not all the same green: a dried olive stub sits near hue 33, right
# on the edge of any band wide enough to be useful, and a band widened to catch
# it starts swallowing orange fruit instead.
STEM_HUE_DELTA = 18.0
STEM_MIN_SATURATION = 55
STEM_MIN_VALUE = 30

# Size limits expressed against the BELT WIDTH rather than the frame.
#
# Frame-relative limits were a mistake, and an expensive one: they encode how
# the camera happens to be framed, so a closer lens silently rejected every
# fruit as "too large" and the detector returned nothing at all. Measured on the
# original set a paprika spans 0.33 of the belt width (p99 0.456), and that
# ratio is a property of the product and the machine - it does not change when
# the lens does.
MIN_FRUIT_AREA_PER_BELT2 = 0.008
MAX_FRUIT_AREA_PER_BELT2 = 0.22
# A blob up to this is a candidate for SPLITTING, not a rejection. Two touching
# peppers measured 0.26 on the images where detection was failing entirely.
MAX_BLOB_AREA_PER_BELT2 = 0.60
MIN_PART_AREA_PER_BELT2 = 0.015
MIN_STEM_AREA_PER_BELT2 = 0.00015
# When the belt runs off both sides of the frame its measured width is only a
# lower bound - the real belt is wider - so every size expressed against it is
# understated. Rather than reject fruit for being "too large" on a tightly
# framed camera, the upper limits are relaxed by this factor.
BELT_CUTOFF_SLACK = 3.0

# Splitting touching fruit.
# Both halves must be plausible fruit in their own right: the 1st percentile of
# real fruit area is 7161 px, so anything under that is a stem or a sliver, not
# a second paprika. Requiring it of BOTH halves is what stops a stem being
# split off as if it were fruit.
SPLIT_MIN_PART_AREA = 7500
# A split part is inherently less solid than a whole fruit: it has a concave cut
# edge where its neighbour was. Requiring whole-fruit solidity of it meant a
# single ragged part vetoed the entire split, leaving two peppers merged as one.
#
# Size is what actually keeps stems from being split off as fruit: a stem is
# about 5% of its fruit, which is 0.004 of belt width squared, far below the
# 0.015 a part must reach. Solidity only has to exclude genuinely stringy
# fragments.
SPLIT_MIN_SOLIDITY = 0.55
# Distance-transform threshold as a fraction of the peak. Higher separates more
# eagerly and risks cutting one fruit in two.
SPLIT_DISTANCE_RATIO = 0.45
# Hue separation, in OpenCV units, before two regions count as different fruit
# rather than one fruit and its stem.
SPLIT_MIN_HUE_SEPARATION = 20
# Fixed seed for the hue clustering below. cv2.kmeans draws its initial centres
# from OpenCV's global RNG, and nothing was seeding it, so the SAME blob
# clustered twice produced different centres, a different measured hue
# separation, and therefore a different answer to "is this one fruit or two".
# Measured on a real frame: 25 identical evaluations gave 4 different verdicts,
# the most common being one paprika placed TWICE at angles 180 degrees apart.
# Any constant will do; what matters is that detection is a function of the
# frame and nothing else.
SPLIT_HUE_RNG_SEED = 0
# A split part must still look like a paprika. Area and solidity alone let a
# shadowed lobe through, so elongation is checked too.
#
# Calibrated against every split observed over a 1968-frame run, measured on
# the part masks themselves rather than on a bbox crop:
#
#     two touching fruit          1.21 / 1.27      legitimate
#     two touching fruit          1.26 / 1.35      legitimate
#     two touching fruit          1.27 / 2.11      legitimate
#     fruit + clipped neighbour   1.47 / 2.95      legitimate
#     fruit + its own lobe        1.98 / 3.58      PHANTOM
#
# 3.2 sits in the gap. An earlier value of 2.0 looked safe on a seven-frame
# sample and cost two real placements on the full run - the gap I claimed was
# empty had two legitimate splits in it. This one is narrow (2.95 to 3.58) and
# rests on a single phantom, so re-check it when more data arrives rather than
# trusting it the way the first number was trusted.
SPLIT_MAX_PART_ELONGATION = 3.2

# Above this, a detection that sits mostly inside a larger one is a piece of
# that larger one - a shadowed band, a lobe cut off by the saturation floor -
# rather than a second fruit. Only applied together with a shape test: a whole
# healthy fruit can legitimately have its box swallowed by a merged blob's box
# beside it, and dropping that would lose real fruit. See _drop_contained().
CONTAINED_MIN_OVERLAP = 0.70
CONTAINED_MAX_AREA_RATIO = 0.50
CONTAINED_MAX_ELONGATION = 2.5
CONTAINED_MIN_SOLIDITY = 0.75

# Only blobs that could plausibly hold two fruit are examined at all. A single
# compact paprika needs no splitting, and attempting it on every blob doubled
# the cost of a frame for nothing.
SPLIT_TRY_MIN_AREA = SPLIT_MIN_PART_AREA * 2
SPLIT_TRY_MAX_SOLIDITY = 0.93
# Pixels sampled for the hue clustering. Clustering every pixel of a 50,000 px
# blob is wasted work: the hue distribution is the same either way.
SPLIT_HUE_SAMPLE = 3000
# Circular resultant length below which a blob's hue is considered spread out
# enough to be worth clustering. One fruit of one colour gives a value very
# close to 1; two colours pressed together pull it down. Costs one pass over a
# sample of pixels, against a k-means over all of them.
SPLIT_HUE_UNIMODAL_R = 0.985
SPLIT_NORMALISED_RADIUS = 90
MIN_STEM_AREA_MORPH = 200

# Kernel scales for the morphological stem search, and how many must agree.
# Three scales spread around the original single 0.55 value: if a protrusion
# only survives at one of them it was a kernel artefact, not a stem.
STEM_KERNEL_SCALES = (0.45, 0.55, 0.68)
STEM_MIN_AGREEING_SCALES = 2
STEM_AGREEMENT_RADIUS_RATIO = 0.35

# Fraction of stem pixels averaged to place the calyx. Small enough to stay at
# the base, large enough that no single pixel can move it.
CALYX_NEAREST_FRACTION = 0.10

# On a green fruit the stem cannot be separated by hue, but it is consistently
# LESS SATURATED than the flesh: measured across 82 green fruit the median drop
# is 19 saturation units, with the same sign in 88% of cases. Morphology says
# roughly where the stem is; this sharpens its outline using actual pixel
# values instead of kernel geometry, which is what makes the calyx - and
# therefore the angle - hold still between frames.
STEM_SATURATION_DROP = 12
STEM_REFINE_DILATION = 11
# The refinement is only accepted when it still covers this much of the
# morphological candidate. In the 12% of fruit where the stem is not the less
# saturated part, it would otherwise wander off onto a shadow.
STEM_REFINE_MIN_OVERLAP = 0.35

STEM_QUALITY_HUE = 0.92

# Angular disagreement between kernel scales at which the stem direction is
# considered worthless. Chosen against the policy: a fruit whose scales differ
# by this much is exactly the fruit whose reported angle jumps between frames.
STEM_ANGLE_SPREAD_LIMIT_DEG = 25.0

# Self-check for morphological stems: re-run the detection on the same fruit at
# these gains and see whether the stem direction moves with the light.
#
# This measures the disturbance itself instead of a proxy for it, which is why
# it works where three earlier attempts did not. Scale agreement, orientation
# confidence and mask jitter all failed to separate stable from unstable fruit
# - each caught 1 in 5. Measured across 32 morphological fruit, this separates
# them cleanly: 1.1 degrees median spread on the stable ones against 29.1 on
# the unstable ones.
STEM_SELFCHECK_GAINS = (0.92, 1.08)
# Above this spread the stem direction is treated as not established. At 6
# degrees it catches 6 of 7 genuinely unstable fruit and sends 7 of 25 sound
# ones round again. That trade is deliberate and follows the same asymmetry as
# the rest of the policy: a misplaced paprika leaves the cell, an extra loop
# does not.
STEM_SELFCHECK_LIMIT_DEG = 6.0
SELFCHECK_MAX_DIMENSION = 220

# If the calyx sits closer to the centroid than this fraction of the fruit
# radius, the long axis points into the camera and the fruit is standing on end.
STANDING_RADIUS_RATIO = 0.30

# If the fruit reaches the frame border within this margin, it is only
# partially in view. Measured on the set: 22 of the 30 fruit where no stem was
# found were simply running off the edge of the frame. Naming that separately
# does more than clean up the statistics - it is a materially different case, as
# an incomplete fruit may have a perfectly good stem you just cannot see.
EDGE_MARGIN_PX = 6

# How much of the diameter may be cut away before the fruit counts as
# incomplete. Measured on the set: fruit merely grazing the border sit around
# 0.16, fruit genuinely running off it around 0.69, so 0.30 falls cleanly
# between them. Simply checking whether the box touches the border rejected 34%
# of all fruit, most of which were almost entirely in view.
EDGE_CUT_THRESHOLD = 0.30

REASON_EDGE_CLIPPED = "edge_clipped"
REASON_NO_STEM = "no_stem"

# Downscale factor for the candidate search. The expensive steps - above all
# the morphological opening with a large kernel - then run only on the cropped
# fruit instead of on the whole frame.
CANDIDATE_SCALE = 0.25
# Generous, and deliberately so. The crop must never cut through a fruit: the
# component inside the crop would shrink, the centroid would be wrong and the
# angle would shift with it. Overlap costs almost nothing because duplicates are
# filtered out further down - a sliced fruit costs a wrong answer.
ROI_PADDING_RATIO = 0.25


@dataclass
class ClassicalFruit:
    """One fruit as found classically.

    `mask` is cropped to `bbox`, so mask[y, x] corresponds to frame pixel
    (bbox[0] + x, bbox[1] + y). This is deliberate: keeping a mask the size of
    the search region while the box is in frame coordinates gives you two
    coordinate systems that look identical and quietly produce wrong answers -
    and it keeps a megabyte-sized array alive per fruit that nobody needs.
    """

    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area: int
    centroid: tuple[float, float]
    hue: float
    colour: str
    stem_end: Optional[tuple[float, float]] = None
    blossom_end: Optional[tuple[float, float]] = None
    stem_method: str = "none"          # hue | morphology | none
    # How well the stem was localised, 0-1. Not the same as "was a stem found":
    # morphology often finds one but pins it loosely, and a loosely pinned stem
    # is exactly what produces an angle that jumps between frames. Carrying it
    # forward lets the policy decline to place instead of placing on a guess.
    stem_quality: float = 0.0
    # Measured angular movement of the stem direction under a lighting change.
    stem_spread_deg: float = 0.0
    standing: bool = False
    edge_clipped: bool = False
    # Why no coordinates can be derived here. Empty means the fruit is usable.
    unpickable_reason: str = ""


def belt_is_cut_off(belt: Optional[np.ndarray], frame_shape: tuple, margin: int = 4) -> bool:
    """True when the belt reaches both side edges of the frame.

    Then its measured width is a lower bound, not the width, and any limit
    derived from it is understated by an unknown amount.
    """
    if belt is None:
        return False
    columns = np.nonzero(belt.any(axis=0))[0]
    if len(columns) == 0:
        return False
    edge = frame_shape[1] * BELT_SCALE - 1 - margin
    return bool(columns.min() <= margin and columns.max() >= edge)


def belt_width(belt: Optional[np.ndarray]) -> Optional[float]:
    """Width of the belt in pixels, used as the scale reference.

    How large a paprika should be is expressed against this instead of against
    the frame, so one configuration works whatever the camera resolution or how
    close the lens sits.
    """
    if belt is None:
        return None
    ys, xs = np.nonzero(belt)
    if len(xs) < 100:
        return None
    # Divided back out because the mask is held at BELT_SCALE.
    return float(xs.max() - xs.min() + 1) / BELT_SCALE


def belt_mask_raw(
    bgr: np.ndarray,
    belt_hue: tuple[int, int] = DEFAULT_BELT_HUE,
) -> Optional[np.ndarray]:
    """The belt surface itself: the one large blue region, holes NOT filled.

    Deliberately unfilled. Filling was tried and drops fruit at the frame edge,
    because such a fruit is not an enclosed hole in the belt and no
    hole-filling method will treat it as one - which would silently stop
    reporting the 18% of product that runs off the frame, the very case the
    incomplete counter exists for.

    Returns None when no plausible belt is visible, so the caller can fall back
    to searching the whole frame and say so. A camera knocked askew should
    degrade loudly, not quietly stop finding product.
    """
    # Computed at quarter scale. The belt is by far the largest thing in the
    # frame, so a quarter of the pixels locate it just as well, and doing it at
    # full resolution cost 17 ms per frame - more than all the rest of the
    # detection put together.
    small = cv2.resize(bgr, None, fx=BELT_SCALE, fy=BELT_SCALE,
                       interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[:, :, 0], hsv[:, :, 1]

    low, high = belt_hue
    blue = ((hue >= low) & (hue <= high) & (saturation >= 60)).astype(np.uint8) * 255

    count, labels, stats, _ = cv2.connectedComponentsWithStats(blue, 8)
    if count <= 1:
        return None

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    small_area = small.shape[0] * small.shape[1]
    if stats[largest, cv2.CC_STAT_AREA] < small_area * BELT_MIN_FRAME_FRACTION:
        return None

    # Returned at BELT_SCALE, not resized up. Callers scale their coordinates
    # instead, which is one multiply against a full-frame array allocation.
    return (labels == largest).astype(np.uint8) * 255


def foreign_ring_fraction(
    blob: np.ndarray,
    belt: np.ndarray,
    produce: np.ndarray,
    offset: tuple[int, int],
    frame_shape: tuple,
) -> float:
    """How much of the ring around a blob is neither belt nor produce.

    Asks what surrounds the object rather than what it overlaps, because a
    paprika on the belt is ringed by belt while a crate or a guard beside the
    belt is not, whatever colour it happens to be.

    Other fruit count as acceptable neighbours. Testing against belt alone was
    tried and dropped eight real peppers: clustered green fruit are ringed by
    each other rather than by belt, which is normal on a full belt and must not
    read as "off the belt".

    Ring pixels outside the frame are excluded rather than counted against the
    fruit - running off the edge is a separate condition, already handled by
    edge_cut_ratio.
    """
    kernel = np.ones((BELT_RING_DILATION,) * 2, np.uint8)
    ring = cv2.subtract(cv2.dilate(blob, kernel), blob)

    ys, xs = np.nonzero(ring)
    if len(xs) == 0:
        return 1.0

    # The belt and produce masks are kept at BELT_SCALE rather than resized up
    # to the full frame. Scaling two full-frame masks per call cost more than
    # every lookup they served, and the ring test only needs to know which side
    # of a boundary a pixel is on - a quarter-resolution answer to that is the
    # same answer.
    ox, oy = offset
    gx = ((xs + ox) * BELT_SCALE).astype(np.int32)
    gy = ((ys + oy) * BELT_SCALE).astype(np.int32)
    inside = (gx >= 0) & (gx < belt.shape[1]) & (gy >= 0) & (gy < belt.shape[0])
    if inside.sum() < 20:
        # Almost the whole ring is off-frame, so there is nothing to judge by.
        # Accepted here and left to edge_cut_ratio, which is the check that
        # actually speaks to this situation.
        return 0.0

    gx, gy = gx[inside], gy[inside]
    known = (belt[gy, gx] > 0) | (produce[gy, gx] > 0)
    return float(1.0 - known.mean())


def fruit_mask(
    bgr: np.ndarray,
    belt_hue: tuple[int, int] = DEFAULT_BELT_HUE,
    saturation_floor: int = DEFAULT_SATURATION_FLOOR,
    value_floor: int = DEFAULT_VALUE_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate fruit from the belt. Returns (mask, hsv)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    keep = (saturation >= saturation_floor) & (value >= value_floor)
    low, high = belt_hue
    if high > low:
        keep &= ~((hue >= low) & (hue <= high))

    mask = keep.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, hsv


def circular_hue(hue_channel: np.ndarray, mask: np.ndarray) -> float:
    """Mean hue, computed circularly - red sits at both 0 and 179, so a plain
    arithmetic mean would give exactly the wrong answer there."""
    values = hue_channel[mask > 0].astype(np.float64) * 2 * np.pi / 180.0
    mean = math.atan2(np.sin(values).mean(), np.cos(values).mean())
    return (math.degrees(mean) / 2.0) % 180.0


def colour_name(hue: float) -> str:
    if hue < 12 or hue > 168:
        return "red"
    if hue < 24:
        return "orange"
    if hue < 33:
        return "yellow"
    if hue < 95:
        return "green"
    return "unknown"


def _solidity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    return float(cv2.contourArea(contour) / max(1.0, cv2.contourArea(hull)))


def _elongation(mask: np.ndarray) -> float:
    """Long-axis over short-axis of the pixel distribution.

    Second central moments rather than minAreaRect: a rectangle is fitted to
    the extremes and so is dominated by the one stray pixel furthest out, while
    the moments describe the whole blob. This is the same quantity
    orientation.shape_orientation reports, so a number here can be compared
    directly against one from there.

    Returns 1.0 for anything too small or too degenerate to measure, which
    reads as "perfectly round" and therefore never vetoes a split on its own.
    """
    moments = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    if moments["m00"] <= 0:
        return 1.0
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]
    common = math.sqrt(max(0.0, 4.0 * mu11 ** 2 + (mu20 - mu02) ** 2))
    major = (mu20 + mu02 + common) / 2.0
    minor = (mu20 + mu02 - common) / 2.0
    if minor <= 1e-6:
        return 1.0
    return math.sqrt(major / minor)


def _plausible_parts(
    parts: list[np.ndarray], min_part_area: float = SPLIT_MIN_PART_AREA
) -> Optional[list[np.ndarray]]:
    """Accept a split only if every part could be a paprika on its own.

    Three tests, and they fail differently. Area rejects stems and specks.
    Solidity rejects stringy fragments. Elongation rejects the case the other
    two miss: a shadowed lobe of ONE fruit, which is large enough and solid
    enough to pass as a paprika but far too long and thin to be one. That was
    reaching the actuator as a second fruit with its own angle.

    Rejecting the split is the conservative outcome. It leaves the blob whole,
    so the fruit is measured once - possibly badly - rather than twice with
    contradictory answers.
    """
    if len(parts) < 2:
        return None
    for part in parts:
        if int((part > 0).sum()) < min_part_area:
            return None
        if _solidity(part) < SPLIT_MIN_SOLIDITY:
            return None
        if _elongation(part) > SPLIT_MAX_PART_ELONGATION:
            return None
    return parts


def split_by_distance(
    blob: np.ndarray, min_part_area: float = SPLIT_MIN_PART_AREA
) -> Optional[list[np.ndarray]]:
    """Split touching fruit of the SAME colour, via distance transform.

    Two paprikas pressed together form one connected region, and the waist
    between them is the thinnest part of it. The distance transform makes that
    waist explicit and watershed cuts there.

    Runs on a downscaled copy for the same reason the stem morphology does: the
    cut only has to land in the right place to within a few pixels, and both
    the distance transform and the watershed cost scale with area.
    """
    area = int((blob > 0).sum())
    if area <= 0:
        return None

    radius = math.sqrt(area / math.pi)
    scale = min(1.0, SPLIT_NORMALISED_RADIUS / max(radius, 1.0))
    if scale < 1.0:
        work = cv2.resize(blob, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    else:
        work = blob

    distance = cv2.distanceTransform(work, cv2.DIST_L2, 5)
    peak = float(distance.max())
    if peak <= 0:
        return None

    _, cores = cv2.threshold(distance, SPLIT_DISTANCE_RATIO * peak, 255, 0)
    cores = cores.astype(np.uint8)

    count, markers = cv2.connectedComponents(cores)
    if count <= 2:
        return None                                   # one core, one fruit

    unknown = cv2.subtract(work, cores)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(work, cv2.COLOR_GRAY2BGR), markers)

    parts = []
    for label in range(2, count + 1):
        part = ((markers == label) & (work > 0)).astype(np.uint8) * 255
        if int((part > 0).sum()) == 0:
            continue
        if scale < 1.0:
            part = cv2.resize(part, (blob.shape[1], blob.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            part = cv2.bitwise_and(part, blob)
        parts.append(part)
    return _plausible_parts(parts, min_part_area)


def split_by_hue(
    blob: np.ndarray, hue_channel: np.ndarray,
    min_part_area: float = SPLIT_MIN_PART_AREA,
) -> Optional[list[np.ndarray]]:
    """Split touching fruit of DIFFERENT colours.

    Colour is a far stronger cue than shape when a green and a red pepper are
    pressed together: the waist between them may be barely visible while the
    hue boundary is unmistakable.

    The guards matter more than the clustering. Without them this splits every
    fruit from its own green stem, which is a 45-unit hue step and clusters
    beautifully - and is completely wrong.
    """
    selected = blob > 0
    ys, xs = np.nonzero(selected)
    values = hue_channel[ys, xs].astype(np.float32)
    if len(values) < 1000:
        return None

    # Clustered on the unit circle so red, which sits at both 0 and 179, is not
    # torn into two clusters by the wrap-around.
    def to_circle(v):
        radians = v * 2 * np.pi / 180.0
        return np.stack([np.cos(radians), np.sin(radians)], axis=1).astype(np.float32)

    # Fitted on a sample, then applied to every pixel. The hue distribution of
    # a 50,000 px blob is fully described by a few thousand of them.
    if len(values) > SPLIT_HUE_SAMPLE:
        idx = np.linspace(0, len(values) - 1, SPLIT_HUE_SAMPLE).astype(np.int64)
        sample = to_circle(values[idx])
    else:
        sample = to_circle(values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    # Seeded immediately before the draw, not once at import: OpenCV's RNG is
    # global and advances on every use, so seeding at startup would only make
    # the FIRST call repeatable. Note this does reset the global stream, which
    # is harmless here because this is the only OpenCV RNG consumer in the
    # detector - if another is ever added, both need thinking about together.
    cv2.setRNGSeed(SPLIT_HUE_RNG_SEED)
    _, _, centers = cv2.kmeans(sample, 2, None, criteria, 4, cv2.KMEANS_PP_CENTERS)

    angles = [math.atan2(c[1], c[0]) for c in centers]
    separation = abs(math.degrees(angles[0] - angles[1])) % 360.0
    separation = min(separation, 360.0 - separation) / 2.0    # back to OpenCV hue
    if separation < SPLIT_MIN_HUE_SEPARATION:
        return None

    points = to_circle(values)
    distances = np.stack([
        ((points - centers[c]) ** 2).sum(axis=1) for c in (0, 1)
    ], axis=1)
    flat = distances.argmin(axis=1)

    parts = []
    for cluster in (0, 1):
        component = np.zeros_like(blob)
        component[ys[flat == cluster], xs[flat == cluster]] = 255
        component = cv2.morphologyEx(component, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

        count, lab, stats, _ = cv2.connectedComponentsWithStats(component, 8)
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] >= min_part_area:
                parts.append((lab == i).astype(np.uint8) * 255)

    return _plausible_parts(parts, min_part_area)


def split_touching(
    blob: np.ndarray, hue_channel: np.ndarray,
    min_part_area: float = SPLIT_MIN_PART_AREA,
) -> list[np.ndarray]:
    """Separate a blob into individual fruit, or return it unchanged.

    Colour first, then shape. A colour boundary is direct evidence that two
    different fruit are present; a thin waist is only a hint, and a single
    lumpy paprika has waists too.
    """
    # Cheap gate first. A blob too small to hold two fruit, or compact enough
    # to be one, is left alone - which is most of them.
    area = int((blob > 0).sum())
    if area < min_part_area * 2:
        return [blob]
    if area < min_part_area * 4 and _solidity(blob) > SPLIT_TRY_MAX_SOLIDITY:
        return [blob]

    # Is there more than one colour here at all? A cheap test before an
    # expensive one: clustering a single-coloured blob can only ever return the
    # answer we already have.
    ys, xs = np.nonzero(blob)
    step = max(1, len(xs) // SPLIT_HUE_SAMPLE)
    sampled = hue_channel[ys[::step], xs[::step]].astype(np.float32) * 2 * np.pi / 180.0
    resultant = math.hypot(float(np.cos(sampled).mean()), float(np.sin(sampled).mean()))

    parts = None
    if resultant < SPLIT_HUE_UNIMODAL_R:
        parts = split_by_hue(blob, hue_channel, min_part_area)
    if parts is None:
        parts = split_by_distance(blob, min_part_area)
    return parts if parts else [blob]


def _stem_by_hue(
    hsv: np.ndarray, fruit: np.ndarray, min_area: float = MIN_STEM_AREA_HUE
) -> Optional[np.ndarray]:
    """Find the stem as the part of the fruit whose colour is not the fruit's.

    Measured against the fruit's own hue, not against a fixed green band. On a
    red pepper with a short dried stub, a fixed band returned 25 pixels and
    found nothing; measured against the fruit's own colour it returns 408 and
    finds the stub. Where both approaches work they agree to 1 px median, 4 px
    at p90.

    On a green fruit the stem shares the flesh's hue, so nothing is returned
    and the caller falls back to morphology. That is the correct outcome here,
    not a failure.

    Works inside the fruit's bounding box rather than over the whole crop. The
    difference is not cosmetic: computing the hue delta across every pixel of
    the region for every fruit in it doubled the time per frame.
    """
    ys, xs = np.nonzero(fruit)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    window = fruit[y1:y2, x1:x2]
    hue = hsv[y1:y2, x1:x2, 0]
    saturation = hsv[y1:y2, x1:x2, 1]
    value = hsv[y1:y2, x1:x2, 2]

    fruit_hue = circular_hue(hue, window)
    delta = np.abs(hue.astype(np.float32) - fruit_hue)
    delta = np.minimum(delta, 180.0 - delta)          # hue is circular

    candidate = (
        (delta >= STEM_HUE_DELTA)
        & (window > 0)
        & (saturation >= STEM_MIN_SATURATION)
        & (value >= STEM_MIN_VALUE)
    ).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    blobs = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > min_area]
    if not blobs:
        return None

    largest = max(blobs, key=lambda i: stats[i, cv2.CC_STAT_AREA])

    # A stem that is a fifth of its own fruit is not a stem. Without this the
    # colour route absorbs a touching green pepper into the red one beside it,
    # and the green one is never reported at all.
    if stats[largest, cv2.CC_STAT_AREA] > int((fruit > 0).sum()) * MAX_STEM_AREA_RATIO:
        return None

    stem = np.zeros_like(fruit)
    stem[y1:y2, x1:x2] = (labels == largest).astype(np.uint8) * 255
    return stem


def _stem_by_morphology(fruit: np.ndarray) -> tuple[Optional[np.ndarray], float]:
    """Stem as a thin protrusion, independent of colour.

    The fruit is first scaled to a fixed reference radius, so the morphology
    always runs with a small kernel on a small mask. Cost is then constant
    rather than growing with the square of the fruit size, and the amount of
    erosion a fruit receives no longer depends on how big it happens to be.

    Runs at three kernel scales and requires at least two to agree. A single
    scale is what made green fruit unstable: the kernel derives from the fruit,
    so a few pixels of segmentation difference between frames changed what
    survived the opening, and the reported angle jumped. Requiring agreement
    turns that brittleness into an honest "no stem".
    """
    area = int((fruit > 0).sum())
    if area <= 0:
        return None, 0.0

    radius = math.sqrt(area / math.pi)
    if radius < 4:
        return None, 0.0

    # Only ever downscale. Blowing a small fruit up would invent detail that
    # the segmentation never resolved.
    scale = min(1.0, STEM_NORMALISED_RADIUS / radius)
    if scale < 1.0:
        work = cv2.resize(fruit, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    else:
        work = fruit

    work_radius = radius * scale

    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for kernel_scale in STEM_KERNEL_SCALES:
        size = max(5, int(work_radius * kernel_scale)) | 1     # odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

        body = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
        stem = cv2.subtract(work, body)
        stem = cv2.morphologyEx(stem, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        min_area = max(20, int(MIN_STEM_AREA_MORPH * scale * scale))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(stem, 8)
        blobs = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > min_area]
        if not blobs:
            continue

        largest = max(blobs, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        candidates.append(
            ((labels == largest).astype(np.uint8) * 255, np.array(centroids[largest]))
        )

    if len(candidates) < STEM_MIN_AGREEING_SCALES:
        return None, 0.0

    tolerance = max(8.0, work_radius * STEM_AGREEMENT_RADIUS_RATIO)
    points = [c for _, c in candidates]
    reference = np.median(np.stack(points), axis=0)
    agreeing = [
        (m, c) for m, c in candidates
        if float(np.linalg.norm(c - reference)) <= tolerance
    ]
    if len(agreeing) < STEM_MIN_AGREEING_SCALES:
        return None, 0.0

    merged = agreeing[0][0].copy()
    for mask, _ in agreeing[1:]:
        merged = cv2.bitwise_or(merged, mask)

    centre = _mask_centroid(work)
    angles = []
    for mask, _ in agreeing:
        calyx = _calyx_from_stem(mask, centre)
        angles.append(math.degrees(math.atan2(-(calyx[1] - centre[1]),
                                              calyx[0] - centre[0])) % 360.0)

    spread_deg = 0.0
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            diff = abs(angles[i] - angles[j]) % 360.0
            spread_deg = max(spread_deg, min(diff, 360.0 - diff))

    agreement = 1.0 - min(1.0, spread_deg / STEM_ANGLE_SPREAD_LIMIT_DEG)
    coverage = len(agreeing) / len(STEM_KERNEL_SCALES)
    quality = float(np.clip(0.20 + 0.65 * agreement + 0.15 * coverage, 0.0, 1.0))

    if scale < 1.0:
        merged = cv2.resize(
            merged, (fruit.shape[1], fruit.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return merged, quality


def _stem_direction(fruit: np.ndarray) -> Optional[float]:
    """Direction from the fruit centroid to the calyx, or None."""
    stem, _ = _stem_by_morphology(fruit)
    if stem is None:
        return None
    centre = _mask_centroid(fruit)
    calyx = _calyx_from_stem(stem, centre, fruit)
    return math.degrees(math.atan2(-(calyx[1] - centre[1]), calyx[0] - centre[0])) % 360.0


def _stem_selfcheck(
    region: np.ndarray,
    belt_hue: tuple[int, int],
    saturation_floor: int,
    value_floor: int,
    baseline_deg: float,
) -> float:
    """How far the stem direction moves when the light changes.

    Returns the largest angular disagreement in degrees, or a large value when
    a gain variant loses the stem entirely - losing the stem under an 8%
    lighting change is itself the strongest possible evidence that this stem
    was never solidly established.
    """
    angles = [baseline_deg]

    # Downscaled first. This check only has to answer whether the direction
    # moves by more than a few degrees, and at full resolution it was the
    # single most expensive thing in a busy frame.
    longest = max(region.shape[:2])
    if longest > SELFCHECK_MAX_DIMENSION:
        factor = SELFCHECK_MAX_DIMENSION / longest
        region = cv2.resize(region, None, fx=factor, fy=factor,
                            interpolation=cv2.INTER_AREA)

    for gain in STEM_SELFCHECK_GAINS:
        scaled = np.clip(region.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        mask, _ = fruit_mask(scaled, belt_hue, saturation_floor, value_floor)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return 180.0
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

        direction = _stem_direction((labels == largest).astype(np.uint8) * 255)
        if direction is None:
            return 180.0
        angles.append(direction)

    spread = 0.0
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            diff = abs(angles[i] - angles[j]) % 360.0
            spread = max(spread, min(diff, 360.0 - diff))
    return spread


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


def _refine_stem_by_saturation(
    stem_mask: np.ndarray, fruit: np.ndarray, saturation: np.ndarray
) -> np.ndarray:
    """Sharpen a morphological stem using the saturation drop at the stem.

    Returns the refined mask, or the original when the refinement does not
    look like the same object - a refinement that no longer overlaps what
    morphology found is not a better stem, it is a different thing.
    """
    search = cv2.dilate(stem_mask, np.ones((STEM_REFINE_DILATION,) * 2, np.uint8))
    body = cv2.subtract(fruit, search)
    if int((body > 0).sum()) < 300:
        return stem_mask

    body_saturation = float(np.median(saturation[body > 0]))
    threshold = body_saturation - STEM_SATURATION_DROP

    refined = ((search > 0) & (saturation < threshold)).astype(np.uint8) * 255
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(refined, 8)
    blobs = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > MIN_STEM_AREA_MORPH // 2]
    if not blobs:
        return stem_mask

    largest = max(blobs, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    candidate = (labels == largest).astype(np.uint8) * 255

    overlap = float(((candidate > 0) & (stem_mask > 0)).sum()) / max(1, int((stem_mask > 0).sum()))
    if overlap < STEM_REFINE_MIN_OVERLAP:
        return stem_mask
    return candidate


def _calyx_from_stem(
    stem_mask: np.ndarray,
    centroid: np.ndarray,
    fruit: Optional[np.ndarray] = None,
) -> np.ndarray:
    """The point where the stem meets the fruit shoulder.

    Deliberately the base and not the stem's own centroid: the annotation spec
    asks for the calyx. Stem length varies enormously and stems snap off, so
    the tip is not a fixed anatomical point whereas the base is.

    Averaged over the nearest decile of stem pixels rather than taking the
    single closest one. A single pixel is the most noise-sensitive estimator
    available - one stray pixel from a slightly different threshold moves the
    calyx, and with it the reported angle.

    Defining this instead as the stem/body attachment overlap was tried and
    measured worse: red went from 0.3 to 1.4 degrees p90 and the worst green
    case from 37 to 45. Anatomically that definition is the more correct one,
    but the attachment region is itself sensitive to the dilation used to find
    it, and that sensitivity outweighed the theoretical gain. Kept here as a
    note so it is not re-attempted blind.
    """
    pixels = np.column_stack(np.nonzero(stem_mask)[::-1]).astype(np.float64)
    distances = np.linalg.norm(pixels - centroid, axis=1)
    take = max(1, int(len(pixels) * CALYX_NEAREST_FRACTION))
    nearest = np.argpartition(distances, take - 1)[:take]
    return pixels[nearest].mean(axis=0)


def _opposite_end(fruit: np.ndarray, calyx: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Blossom_end: walk from the centroid away from the calyx until the mask
    ends. Not a mirror of the calyx - a paprika is not symmetric about its
    centroid, so mirroring places the point systematically too far or too short.
    """
    direction = centroid - calyx
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return centroid
    direction = direction / norm

    height, width = fruit.shape
    last_inside = centroid
    for step in range(int(max(height, width))):
        point = centroid + direction * step
        x, y = int(round(point[0])), int(round(point[1]))
        if not (0 <= x < width and 0 <= y < height):
            break
        if fruit[y, x] == 0:
            break
        last_inside = point

    # Pull in slightly: the edge point lies on the silhouette boundary, and the
    # blossom scar sits just inside it.
    return centroid + (last_inside - centroid) * 0.88


def _candidate_boxes(
    bgr: np.ndarray,
    belt_hue: tuple[int, int],
    saturation_floor: int,
    value_floor: int,
    area_min: float,
    area_max: float,
) -> list[tuple[int, int, int, int]]:
    """Find candidate regions on a downscaled image.

    Only to decide WHERE to look, never to measure. A paprika is hundreds of
    pixels across, so at quarter scale it stays comfortably detectable while
    sixteen times fewer pixels pass through the morphology - which is exactly
    where the time was going.
    """
    small = cv2.resize(bgr, None, fx=CANDIDATE_SCALE, fy=CANDIDATE_SCALE,
                       interpolation=cv2.INTER_AREA)
    mask, _ = fruit_mask(small, belt_hue, saturation_floor, value_floor)

    # Half as strict as the real threshold. This step only decides WHERE to
    # look; blobs of the wrong size are still rejected during the
    # full-resolution measurement. A missed candidate is final, a superfluous
    # candidate costs a few milliseconds.
    factor = CANDIDATE_SCALE * CANDIDATE_SCALE
    scaled_min = max(20, int(area_min * factor * 0.5))
    scaled_max = area_max * factor * 1.5

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    height, width = bgr.shape[:2]

    boxes = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < scaled_min or area > scaled_max:
            continue

        x = stats[i, cv2.CC_STAT_LEFT] / CANDIDATE_SCALE
        y = stats[i, cv2.CC_STAT_TOP] / CANDIDATE_SCALE
        w = stats[i, cv2.CC_STAT_WIDTH] / CANDIDATE_SCALE
        h = stats[i, cv2.CC_STAT_HEIGHT] / CANDIDATE_SCALE

        # Crop generously: at quarter scale the boundary is imprecise, and a
        # stem falling just outside the tight box would otherwise be lost -
        # precisely the detail this whole module turns on.
        pad = max(w, h) * ROI_PADDING_RATIO
        boxes.append(
            (
                max(0, int(x - pad)),
                max(0, int(y - pad)),
                min(width, int(x + w + pad)),
                min(height, int(y + h + pad)),
            )
        )

    return _merge_boxes(boxes)


def _merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge overlapping candidates.

    Two touching fruit would otherwise each get their own crop running straight
    through the other, and you would be measuring two half fruit instead of two
    whole ones.
    """
    merged: list[list[int]] = []
    for box in sorted(boxes):
        placed = False
        for existing in merged:
            if (
                box[0] < existing[2] and existing[0] < box[2]
                and box[1] < existing[3] and existing[1] < box[3]
            ):
                existing[0] = min(existing[0], box[0])
                existing[1] = min(existing[1], box[1])
                existing[2] = max(existing[2], box[2])
                existing[3] = max(existing[3], box[3])
                placed = True
                break
        if not placed:
            merged.append(list(box))
    return [tuple(b) for b in merged]


def find_fruit(
    bgr: np.ndarray,
    belt_hue: tuple[int, int] = DEFAULT_BELT_HUE,
    saturation_floor: int = DEFAULT_SATURATION_FLOOR,
    value_floor: int = DEFAULT_VALUE_FLOOR,
    stem_hue: tuple[int, int] = DEFAULT_STEM_HUE,
    min_area: int = 4000,
    max_area_ratio: float = DEFAULT_MAX_AREA_RATIO,
    max_fruit: int = 8,
    use_roi: bool = True,
    selfcheck: bool = True,
    belt_mask: Optional[np.ndarray] = None,
    split: bool = True,
) -> list[ClassicalFruit]:
    """Find every fruit with, where possible, its stem and calyx.

    Three stages. A coarse search on a downscaled image decides where to look;
    each region is then segmented at full resolution; and any region holding
    more than one fruit is separated before anything is measured on it.

    Splitting before measuring is not a detail. A centroid, an axis and a stem
    taken from two merged peppers belong to neither of them, so measuring first
    and separating afterwards cannot be repaired later.

    Set `use_roi=False` to run on the whole frame, or `split=False` to leave
    touching fruit merged - both useful for checking what each stage
    contributes.
    """
    if bgr is None or bgr.size == 0:
        return []

    height, width = bgr.shape[:2]
    frame_area = height * width

    if belt_mask is None:
        belt_mask = belt_mask_raw(bgr, belt_hue)

    # Size limits come from the belt when one is visible, and only fall back to
    # frame fractions when it is not. Belt-relative limits describe the product
    # and the machine; frame-relative ones describe the lens.
    width_belt = belt_width(belt_mask)
    if width_belt is not None:
        # Upper limits only. The lower limits stay put: a belt that is cut off
        # is wider than measured, so the smallest plausible fruit can only grow,
        # never shrink.
        slack = BELT_CUTOFF_SLACK if belt_is_cut_off(belt_mask, bgr.shape) else 1.0
        area_min = max(600.0, MIN_FRUIT_AREA_PER_BELT2 * width_belt ** 2)
        area_max_fruit = MAX_FRUIT_AREA_PER_BELT2 * width_belt ** 2 * slack
        area_max_blob = MAX_BLOB_AREA_PER_BELT2 * width_belt ** 2 * slack
        part_min = MIN_PART_AREA_PER_BELT2 * width_belt ** 2
        stem_area_min = max(60.0, MIN_STEM_AREA_PER_BELT2 * width_belt ** 2)
    else:
        area_min = float(min_area)
        area_max_fruit = frame_area * max_area_ratio
        area_max_blob = frame_area * max_area_ratio
        part_min = float(SPLIT_MIN_PART_AREA)
        stem_area_min = float(MIN_STEM_AREA_HUE)

    produce_mask = None
    if belt_mask is not None:
        small = cv2.resize(bgr, None, fx=BELT_SCALE, fy=BELT_SCALE,
                           interpolation=cv2.INTER_AREA)
        produce_mask, _ = fruit_mask(small, belt_hue, saturation_floor, value_floor)

    if use_roi:
        regions = _candidate_boxes(
            bgr, belt_hue, saturation_floor, value_floor, area_min, area_max_blob
        )
    else:
        regions = [(0, 0, width, height)]

    fruits: list[ClassicalFruit] = []
    seen: list[tuple[int, int]] = []

    def measure(blob, crop, hsv, hue_channel, origin) -> None:
        """Measure one separated fruit and append it to the results."""
        rx1, ry1 = origin
        area = int((blob > 0).sum())
        if area < area_min or area > area_max_fruit:
            return

        ys, xs = np.nonzero(blob)
        centroid = np.array([xs.mean(), ys.mean()])

        # On the belt, or not product. Checked before any stem work: the
        # expensive part of this function must never run on something that was
        # never a candidate in the first place.
        if belt_mask is not None:
            foreign = foreign_ring_fraction(
                blob, belt_mask, produce_mask, (rx1, ry1), bgr.shape
            )
            if foreign > MAX_FOREIGN_RING:
                return

        bx, by = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - bx + 1), int(ys.max() - by + 1)
        bbox = (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh)

        # Overlapping ROIs can offer up the same fruit twice.
        if any(abs(bbox[0] - a) < 20 and abs(bbox[1] - b) < 20 for a, b in seen):
            return
        seen.append((bbox[0], bbox[1]))

        hue_value = circular_hue(hue_channel, blob)

        # Crop the mask to its own box so mask and bbox share one coordinate
        # system. The measurements below still work in crop coordinates; only
        # what leaves this function is translated to frame coordinates.
        fruit = ClassicalFruit(
            bbox=bbox,
            mask=blob[by:by + bh, bx:bx + bw].copy(),
            area=area,
            centroid=(float(rx1 + centroid[0]), float(ry1 + centroid[1])),
            hue=round(hue_value, 1),
            colour=colour_name(hue_value),
        )
        fruit.edge_clipped = edge_cut_ratio(fruit, bgr.shape) > EDGE_CUT_THRESHOLD

        # Colour first - more accurate. Morphology as the fallback, and the
        # only thing left on a green fruit.
        stem_mask = None
        if not (GREEN_FRUIT_HUE[0] < hue_value < GREEN_FRUIT_HUE[1]):
            stem_mask = _stem_by_hue(hsv, blob, stem_area_min)
            if stem_mask is not None:
                fruit.stem_method = "hue"
                fruit.stem_quality = STEM_QUALITY_HUE
        if stem_mask is None:
            stem_mask, quality = _stem_by_morphology(blob)
            if stem_mask is not None:
                fruit.stem_method = "morphology"
                stem_mask = _refine_stem_by_saturation(stem_mask, blob, hsv[:, :, 1])

                # Only for stems found by shape, and only on fruit that could
                # actually be placed. Running it on a fruit already heading for
                # reject spends the cost on an answer nobody uses.
                if selfcheck and not fruit.edge_clipped:
                    centre_local = _mask_centroid(blob)
                    calyx_local = _calyx_from_stem(stem_mask, centre_local, blob)
                    baseline = math.degrees(
                        math.atan2(-(calyx_local[1] - centre_local[1]),
                                   calyx_local[0] - centre_local[0])
                    ) % 360.0
                    pad = 12
                    region = crop[max(0, by - pad):by + bh + pad,
                                  max(0, bx - pad):bx + bw + pad]
                    spread = _stem_selfcheck(
                        region, belt_hue, saturation_floor, value_floor, baseline
                    )
                    fruit.stem_spread_deg = round(float(spread), 1)
                    fruit.stem_quality = float(
                        np.clip(1.0 - spread / (STEM_SELFCHECK_LIMIT_DEG * 2.0), 0.0, 1.0)
                    )
                else:
                    fruit.stem_quality = quality

        if stem_mask is None:
            # Incomplete-in-frame takes precedence: in that case "no stem found"
            # says nothing about the fruit and everything about the image.
            fruit.unpickable_reason = (
                REASON_EDGE_CLIPPED if fruit.edge_clipped else REASON_NO_STEM
            )
            fruits.append(fruit)
            return

        if fruit.edge_clipped:
            # A stem was found, but the fruit continues outside the frame, so
            # the centroid - and therefore the angle - is wrong.
            fruit.unpickable_reason = REASON_EDGE_CLIPPED

        calyx = _calyx_from_stem(stem_mask, centroid, blob)
        fruit.stem_end = (float(rx1 + calyx[0]), float(ry1 + calyx[1]))

        radius = math.sqrt(area / math.pi)
        if float(np.linalg.norm(calyx - centroid)) < radius * STANDING_RADIUS_RATIO:
            fruit.standing = True
            fruit.blossom_end = fruit.centroid
        else:
            other = _opposite_end(blob, calyx, centroid)
            fruit.blossom_end = (float(rx1 + other[0]), float(ry1 + other[1]))

        fruits.append(fruit)

    for rx1, ry1, rx2, ry2 in regions:
        crop = bgr[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue

        mask, hsv = fruit_mask(crop, belt_hue, saturation_floor, value_floor)
        hue_channel = hsv[:, :, 0]

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            # A blob too large for one fruit is a candidate for SPLITTING, not
            # a rejection. Rejecting outright is what made two touching peppers
            # vanish entirely instead of becoming two detections.
            if area < area_min or area > area_max_blob:
                continue

            # Cut the component down to its own box before anything else
            # touches it. Every step below scans whole arrays, and scanning the
            # whole region for a fruit occupying a tenth of it was the single
            # largest cost in a busy frame.
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])

            component = ((labels[by:by + bh, bx:bx + bw] == i).astype(np.uint8) * 255)
            sub_crop = crop[by:by + bh, bx:bx + bw]
            sub_hsv = hsv[by:by + bh, bx:bx + bw]
            sub_hue = sub_hsv[:, :, 0]

            parts = (
                split_touching(component, sub_hue, part_min)
                if split else [component]
            )
            for part in parts:
                measure(part, sub_crop, sub_hsv, sub_hue, (rx1 + bx, ry1 + by))

    fruits.sort(key=lambda f: f.area, reverse=True)
    fruits = _drop_contained(fruits)
    return fruits[:max_fruit]


def _bbox_containment(small: tuple, big: tuple) -> float:
    """How much of `small`'s box lies inside `big`'s, as a fraction of `small`."""
    overlap_w = max(0, min(small[2], big[2]) - max(small[0], big[0]))
    overlap_h = max(0, min(small[3], big[3]) - max(small[1], big[1]))
    small_area = max(1, (small[2] - small[0]) * (small[3] - small[1]))
    return (overlap_w * overlap_h) / small_area


def _drop_contained(fruits: list[ClassicalFruit]) -> list[ClassicalFruit]:
    """Remove detections that are a piece of a larger detection.

    The splitter guards blobs it splits, but nothing guarded blobs that arrive
    already separate. A shadowed band along a fruit's edge, dark enough to fall
    below the saturation floor, becomes its own connected component and then
    its own fruit - with its own angle, sent to the actuator as if a second
    paprika were lying there.

    Containment alone is not enough to act on, and this is the trap. Over a
    1968-frame run four detections were mostly inside a larger one:

        211x40  elongation 4.68              a shadow sliver, and it was PLACED
        214x79  elongation 2.34 solidity 0.64  a shadow band, also PLACED
        219x104 elongation 1.83               an edge strip, already rejected
        281x240 elongation 1.21 solidity 0.98  A WHOLE HEALTHY FRUIT

    The last one is why the shape test is not optional: two fruit side by side,
    one of them merged with a third into a wide blob, leaves a perfectly good
    fruit's box sitting inside its neighbour's. Dropping on containment alone
    would have thrown it away. Requiring the small one to ALSO be misshapen -
    too long and thin, or too ragged - separates all four correctly.
    """
    if len(fruits) < 2:
        return fruits

    kept: list[ClassicalFruit] = []
    for fruit in fruits:
        swallowed = False
        for other in fruits:
            if other is fruit or other.area <= fruit.area:
                continue
            if _bbox_containment(fruit.bbox, other.bbox) < CONTAINED_MIN_OVERLAP:
                continue
            if fruit.area / max(1, other.area) > CONTAINED_MAX_AREA_RATIO:
                continue
            misshapen = (
                _elongation(fruit.mask) > CONTAINED_MAX_ELONGATION
                or _solidity(fruit.mask) < CONTAINED_MIN_SOLIDITY
            )
            if misshapen:
                swallowed = True
                break
        if not swallowed:
            kept.append(fruit)
    return kept


def edge_cut_ratio(fruit: ClassicalFruit, frame_shape: tuple, margin: int = EDGE_MARGIN_PX) -> float:
    """What fraction of the fruit outline lies on the frame border.

    A fruit merely grazing the border has a few pixels there; a fruit running
    off the frame has a long straight cut. The ratio between that cut and the
    fruit diameter distinguishes the two, and it beats simply checking whether
    the box touches the border - that also rejected fruit which were almost
    entirely in view.
    """
    height, width = frame_shape[:2]
    x1, y1, _, _ = fruit.bbox
    ys, xs = np.nonzero(fruit.mask)
    if len(xs) == 0:
        return 0.0

    gx, gy = xs + x1, ys + y1
    on_border = int(
        ((gx <= margin) | (gx >= width - 1 - margin)
         | (gy <= margin) | (gy >= height - 1 - margin)).sum()
    )

    # on_border counts AREA in a band (2*margin+1) wide, not the length of the
    # cut. Dividing by the band width turns it back into a length, so the result
    # lands neatly between 0 and 1 and reads as "what fraction of the diameter
    # has been cut away".
    chord = on_border / float(2 * margin + 1)
    diameter = 2.0 * math.sqrt(max(1, fruit.area) / math.pi)
    return float(min(1.5, chord / max(1.0, diameter)))