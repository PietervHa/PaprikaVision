"""
End-on detection from the silhouette

Answers one question about a fruit whose stem could not be found: is the camera
looking down the fruit's axis (blossom scar or calyx facing it - standing or
upside down), or at its side with the stem merely hidden?

Why this is not obvious
-----------------------
A blokpaprika is close to a rounded cube, so it is round in outline from every
direction. Elongation, which an earlier revision used for exactly this
decision, scores AUC 0.685 - barely better than a coin. Three other silhouette
measures were tried first and all three separated a hand-picked set of eight
frames perfectly, then misclassified more than half of a 239-fruit dataset. The
signal they were reading was how pronounced a given pepper's lobes happen to
be, which is fruit-to-fruit variation rather than viewing geometry.

What works, measured against 254 hand-labelled crops from six sessions across
red, green and orange fruit:

    centre_texture   AUC 0.797   the blossom scar or calyx is a TEXTURE anomaly
                                 in the middle of the silhouette; end-on 1.39
                                 against side-on 1.01
    par              AUC 0.790   groove alignment: grooves run stem-to-blossom,
                                 so side-on they cross the fruit as parallel
                                 bands, end-on they converge and cancel
    solidity         AUC 0.663   end-on the lobes bump out around the WHOLE
                                 perimeter, so the outline is less convex

Together: AUC 0.927. Leave-one-session-out, which is the honest figure because
a fold sharing a session shares lighting, belt and often the same physical
fruit, it falls to 0.872.

Deliberately NOT a decision
---------------------------
0.872 is a good deal better than chance and nowhere near good enough to bin
fruit on. At the shipped threshold this catches under half the end-on fruit,
so treat a negative as "no opinion", never as "definitely side-on".

The coefficients below come from a logistic regression fitted offline. They are
inlined rather than loaded so the runtime keeps no model file and no sklearn
dependency; retraining means re-running the fit and pasting three numbers.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

# Fitted on 254 labelled crops (57 end-on, 197 side-on). Order matters:
# centre_texture, solidity, par.
_FEATURE_MEAN = (1.123765, 0.963388, 0.301354)
_FEATURE_SCALE = (0.442896, 0.032140, 0.198535)
_FEATURE_COEF = (1.373774, -0.768903, -1.701308)
_INTERCEPT = -2.125812

# Chosen from the leave-one-session-out sweep. At 0.45 the model finds 26 of 57
# end-on fruit and mislabels 5 of 197 side-on ones - roughly 46% recall at 84%
# precision. Lower catches more and costs more; 0.35 reaches 32 of 57 but
# mislabels 16. Precision is the one worth protecting: a side-on fruit called
# end-on is a fruit wrongly accused of being unpickable.
DEFAULT_END_ON_THRESHOLD = 0.45

# Below this the mask is too small for the interior statistics to mean
# anything - the erosion leaves nothing to measure.
_MIN_MASK_PX = 2000


def _interior(mask: np.ndarray) -> np.ndarray:
    """Mask eroded away from its own edge.

    The silhouette boundary is the strongest gradient in the picture and has
    nothing to do with surface grooves, so it has to be excluded or it swamps
    every orientation measurement. The kernel scales with the fruit so the same
    proportion is removed whatever the working distance.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    size = max(5, int(math.sqrt(max(1, int((mask > 0).sum()))) * 0.12)) | 1
    eroded = cv2.erode(binary, np.ones((size, size), np.uint8)) > 0
    return eroded if eroded.sum() >= 300 else (mask > 0)


def _solidity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 1.0
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 1.0
    return float(cv2.contourArea(contour) / hull_area)


def features(crop_bgr: np.ndarray, mask: np.ndarray) -> Optional[dict]:
    """The three measurements, or None when the fruit is too small to read.

    Args:
        crop_bgr: the fruit's own crop, in BGR.
        mask:     its segmentation mask, same height and width as the crop.
    """
    if crop_bgr is None or mask is None:
        return None
    if mask.shape[:2] != crop_bgr.shape[:2]:
        return None
    if int((mask > 0).sum()) < _MIN_MASK_PX:
        return None

    grey = cv2.GaussianBlur(
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (7, 7), 0
    )
    interior = _interior(mask)
    ys, xs = np.nonzero(interior)
    if len(ys) < 300:
        return None

    # --- par: do the grooves run one way, or do they converge? --------------
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=5)
    magnitude = np.hypot(gx, gy)[interior]
    angle = np.arctan2(gy[interior], gx[interior])
    strong = magnitude > np.percentile(magnitude, 75)
    if strong.sum() < 80:
        return None
    weights, angles = magnitude[strong], angle[strong]
    # Doubled because a groove has an orientation, not a direction: a band at
    # 10 degrees and one at 190 are the same band.
    par = float(
        np.hypot(
            (weights * np.cos(2 * angles)).sum(),
            (weights * np.sin(2 * angles)).sum(),
        ) / weights.sum()
    )

    # --- centre_texture: is there something in the middle? ------------------
    whole = mask > 0
    yy, xx = np.nonzero(whole)
    cy, cx = yy.mean(), xx.mean()
    radius = np.hypot(yy - cy, xx - cx)
    span = radius.max()
    if span <= 0:
        return None
    core = radius < 0.30 * span
    ring = (radius > 0.45 * span) & (radius < 0.80 * span)
    if core.sum() < 50 or ring.sum() < 50:
        return None
    fine = np.hypot(
        cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3),
    )[whole]
    ring_energy = fine[ring].mean()
    if ring_energy <= 1e-6:
        return None
    centre_texture = float(fine[core].mean() / ring_energy)

    return {
        "centre_texture": centre_texture,
        "solidity": _solidity(mask),
        "par": par,
    }


def score(measured: dict) -> float:
    """Turn the three measurements into P(end-on).

    Split out from the measuring so the fitted direction of each coefficient
    can be checked without having to synthesise an image that produces a given
    feature value - which is harder than it sounds, since a plain drawn disc
    has too little interior gradient to measure at all.
    """
    values = (measured["centre_texture"], measured["solidity"], measured["par"])
    z = _INTERCEPT + sum(
        ((v - m) / s) * c
        for v, m, s, c in zip(values, _FEATURE_MEAN, _FEATURE_SCALE, _FEATURE_COEF)
    )
    return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z)))))


def end_on_probability(crop_bgr: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """P(the camera is looking down this fruit's axis), or None if unreadable.

    None means "could not measure" and must not be read as "side-on".
    """
    measured = features(crop_bgr, mask)
    return None if measured is None else score(measured)
