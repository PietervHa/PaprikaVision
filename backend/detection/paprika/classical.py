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

GREEN_FRUIT_HUE = (33, 95)
MIN_STEM_AREA_HUE = 150
MIN_STEM_AREA_MORPH = 200

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
    standing: bool = False
    edge_clipped: bool = False
    # Why no coordinates can be derived here. Empty means the fruit is usable.
    unpickable_reason: str = ""


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


def _stem_by_hue(
    hue_channel: np.ndarray, fruit: np.ndarray, stem_hue: tuple[int, int]
) -> Optional[np.ndarray]:
    """Green stem on a non-green fruit."""
    low, high = stem_hue
    green = (
        (hue_channel >= low) & (hue_channel <= high) & (fruit > 0)
    ).astype(np.uint8) * 255
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)
    blobs = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > MIN_STEM_AREA_HUE]
    if not blobs:
        return None
    largest = max(blobs, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


def _stem_by_morphology(fruit: np.ndarray) -> Optional[np.ndarray]:
    """Stem as a thin protrusion, independent of colour.

    The kernel scales with the fruit rather than being fixed: a large and a
    small paprika have a similar stem-to-fruit ratio but very different absolute
    sizes, so a fixed kernel would leave the stem in place on one and eat half
    the fruit on the other.
    """
    area = int((fruit > 0).sum())
    if area <= 0:
        return None

    radius = int(math.sqrt(area / math.pi))
    size = max(9, int(radius * 0.55)) | 1     # oneven maken
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    body = cv2.morphologyEx(fruit, cv2.MORPH_OPEN, kernel)
    stem = cv2.subtract(fruit, body)
    stem = cv2.morphologyEx(stem, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(stem, 8)
    blobs = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > MIN_STEM_AREA_MORPH]
    if not blobs:
        return None
    largest = max(blobs, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


def _calyx_from_stem(stem_mask: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """The point on the stem nearest the fruit's centroid.

    Deliberately the base and not the stem's own centroid: the annotation spec
    asks for the calyx, where the stem attaches to the shoulder. Stem length
    varies enormously and stems snap off, so the tip is not a fixed anatomical
    point whereas the base is.
    """
    pixels = np.column_stack(np.nonzero(stem_mask)[::-1]).astype(np.float64)
    distances = np.linalg.norm(pixels - centroid, axis=1)
    return pixels[int(np.argmin(distances))]


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
    min_area: int,
    max_area_ratio: float,
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

    frame_area = small.shape[0] * small.shape[1]
    # Half as strict as the real threshold. This step only decides WHERE to
    # look; fruit found to be too small are still rejected during the
    # full-resolution measurement. A missed candidate is final, a superfluous
    # candidate costs a few milliseconds.
    scaled_min = max(20, int(min_area * CANDIDATE_SCALE * CANDIDATE_SCALE * 0.5))

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    height, width = bgr.shape[:2]

    boxes = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < scaled_min or area > frame_area * max_area_ratio:
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
    max_area_ratio: float = 0.7,
    max_fruit: int = 8,
    use_roi: bool = True,
) -> list[ClassicalFruit]:
    """Find every fruit with, where possible, its stem and calyx.

    Two stages: first a coarse search on a downscaled image for where something
    lies, then a precise measurement at full resolution within each region
    found. Every measurement therefore comes from the original pixels; the
    downscale only decides where to look.

    Set `use_roi=False` to run everything on the whole frame. Useful for
    checking that the two paths produce the same result.
    """
    if bgr is None or bgr.size == 0:
        return []

    height, width = bgr.shape[:2]
    frame_area = height * width

    if use_roi:
        regions = _candidate_boxes(
            bgr, belt_hue, saturation_floor, value_floor, min_area, max_area_ratio
        )
    else:
        regions = [(0, 0, width, height)]

    fruits: list[ClassicalFruit] = []
    seen: list[tuple[int, int, int, int]] = []

    for rx1, ry1, rx2, ry2 in regions:
        crop = bgr[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue

        mask, hsv = fruit_mask(crop, belt_hue, saturation_floor, value_floor)
        hue_channel = hsv[:, :, 0]

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area > frame_area * max_area_ratio:
                continue

            blob = (labels == i).astype(np.uint8) * 255
            ys, xs = np.nonzero(blob)
            centroid = np.array([xs.mean(), ys.mean()])

            bbox = (
                rx1 + int(stats[i, cv2.CC_STAT_LEFT]),
                ry1 + int(stats[i, cv2.CC_STAT_TOP]),
                rx1 + int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
                ry1 + int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]),
            )

            # Overlapping ROIs can offer up the same fruit twice.
            if any(
                abs(bbox[0] - s[0]) < 20 and abs(bbox[1] - s[1]) < 20 for s in seen
            ):
                continue
            seen.append(bbox)

            hue_value = circular_hue(hue_channel, blob)

            # Crop the mask to its own box so mask and bbox share one coordinate
            # system. The measurements below still work in crop coordinates;
            # only what leaves this function is translated to frame coordinates.
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])

            fruit = ClassicalFruit(
                bbox=bbox,
                mask=blob[by:by + bh, bx:bx + bw].copy(),
                area=area,
                centroid=(float(rx1 + centroid[0]), float(ry1 + centroid[1])),
                hue=round(hue_value, 1),
                colour=colour_name(hue_value),
            )

            fruit.edge_clipped = (
                edge_cut_ratio(fruit, bgr.shape) > EDGE_CUT_THRESHOLD
            )

            # Colour first - more accurate. Morphology as the fallback, and the
            # only thing left on a green fruit.
            stem_mask = None
            if not (GREEN_FRUIT_HUE[0] < hue_value < GREEN_FRUIT_HUE[1]):
                stem_mask = _stem_by_hue(hue_channel, blob, stem_hue)
                if stem_mask is not None:
                    fruit.stem_method = "hue"
            if stem_mask is None:
                stem_mask = _stem_by_morphology(blob)
                if stem_mask is not None:
                    fruit.stem_method = "morphology"

            if stem_mask is None:
                # Incomplete-in-frame takes precedence: in that case "no stem
                # found" says nothing about the fruit and everything about the
                # image.
                fruit.unpickable_reason = (
                    REASON_EDGE_CLIPPED if fruit.edge_clipped else REASON_NO_STEM
                )
                fruits.append(fruit)
                continue

            if fruit.edge_clipped:
                # A stem was found, but the fruit continues outside the frame,
                # so the centroid - and therefore the angle - is wrong.
                fruit.unpickable_reason = REASON_EDGE_CLIPPED

            calyx = _calyx_from_stem(stem_mask, centroid)
            fruit.stem_end = (float(rx1 + calyx[0]), float(ry1 + calyx[1]))

            radius = math.sqrt(area / math.pi)
            if float(np.linalg.norm(calyx - centroid)) < radius * STANDING_RADIUS_RATIO:
                fruit.standing = True
                fruit.blossom_end = fruit.centroid
            else:
                other = _opposite_end(blob, calyx, centroid)
                fruit.blossom_end = (float(rx1 + other[0]), float(ry1 + other[1]))

            fruits.append(fruit)

    fruits.sort(key=lambda f: f.area, reverse=True)
    return fruits[:max_fruit]


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
