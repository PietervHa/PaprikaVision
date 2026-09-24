"""
Paprika Engine

The single place that turns a camera frame into the answer the machine acts
on: where the fruit is, which way its stem points, and whether that answer is
trustworthy enough to place on.

Result contract
---------------
    {
      "status": "OK" | "NOK",
      "mode": "paprika",
      "detections": [ ...every fruit found... ],
      "primary": { ...the one fruit the PLC should act on, or None... },
      "confidence": float,
      "processing_time_ms": float,
      "failure_reason": str            # only when status is NOK
    }

Each detection carries:

    bbox            [x1, y1, x2, y2] pixels
    center          [cx, cy] pixels
    confidence      detector confidence for the fruit itself
    keypoints       {name: {x, y, confidence, visible}}
    orientation     see orientation.Orientation.to_dict()
    angle_plc       orientation angle mapped into the machine's frame
    placement       "place" | "reject" | "reorient" | "unknown"
    label           short human string for the HMI overlay

Why `primary` exists
--------------------
The belt can show several fruit at once, but the actuator acts on one. Picking
it here rather than in the PLC keeps the choice in the place that can see the
whole frame, and keeps the TCP response a single fixed-shape line.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from backend.detection.paprika import end_on as end_on_model
from backend.detection.paprika import orientation as orient
from backend.detection.paprika.orientation import Keypoint, Orientation
from backend.detection.paprika.pose_detector import PaprikaDetector
from backend.utils.logger import get_logger

log = get_logger(__name__)

PLACEMENT_PLACE = "place"
PLACEMENT_REORIENT = "reorient"
PLACEMENT_REJECT = "reject"
PLACEMENT_UNKNOWN = "unknown"
# Below min_usable_confidence the measurement is not weak, it is absent - the
# numbers that come out are whatever the noise happened to be. "unknown" already
# means "measured, and not good enough to act on"; this means "there is nothing
# here worth calling a measurement, send a person". Kept apart from unknown so
# the counters separate a detector working badly from a detector not working at
# all, which are different things to go and fix.
PLACEMENT_REVIEW = "review"

# Why the detector said a fruit is unusable, mapped onto the pose it is
# recorded as. The keys are the REASON_* values in
# backend/detection/paprika/classical.py; they are wire tokens that also end up
# in stored results, so they are matched here as literals exactly as the rest
# of this file does.
#
# Anything unrecognised falls through to POSE_UPSIDE_DOWN, which is the
# conservative default: it rejects, and a new reason showing up as "upside
# down" in the log is at least visible rather than silently placeable.
_UNPICKABLE_POSES = {
    "edge_clipped": orient.POSE_INCOMPLETE,
    "no_stem": orient.POSE_STEM_NOT_FOUND,
}

# Kept beside the mapping above so a new reason cannot end up with a pose but
# no label, which is how an HMI ends up showing a raw enum to an operator.
_UNPICKABLE_LABELS = {
    orient.POSE_INCOMPLETE: "incomplete in frame",
    orient.POSE_STEM_NOT_FOUND: "stem not found",
    orient.POSE_UPSIDE_DOWN: "upside down",
}


def _round_or_none(value, digits: int = 2):
    """Round for transport, preserving None as a distinct 'no angle' state."""
    return None if value is None else round(float(value), digits)


class PaprikaEngine:
    """Owns the detector and applies the placement policy."""

    def __init__(self, cfg: dict, app_state=None) -> None:
        self._app_state = app_state
        block = cfg.get("paprika") if isinstance(cfg.get("paprika"), dict) else {}
        self._cfg = block

        self._detector = PaprikaDetector(block)

        self._use_shape_crosscheck = bool(block.get("shape_crosscheck", False))

        # The bug this prevents: shape_crosscheck meant "use the silhouette as a
        # SECOND opinion alongside the keypoints". But the classical backend
        # produced no keypoints, so with the cross-check off it had no source of
        # an angle at all and reported "unknown" on every fruit - detection
        # without an answer.
        #
        # Both sides are fixed now: the classical backend emits real keypoints
        # from stem detection, and the silhouette estimator only runs where it
        # actually adds something. Two settings that silently disabled each
        # other is exactly the kind of coupling you only notice by running it.
        self._backend = self._detector.backend
        shape_cfg = block.get("shape") if isinstance(block.get("shape"), dict) else {}
        self._saturation_floor = int(shape_cfg.get("saturation_floor", 80))
        belt = shape_cfg.get("belt_hue") or [96, 145]
        self._belt_hue = (int(belt[0]), int(belt[1]))
        self._min_span_ratio = float(block.get("min_span_ratio", 0.18))
        # A standing fruit is only recognised by keypoints when the model puts
        # the two landmarks nearly on top of each other. The pose model has
        # almost never seen that: its training set was 5.5% standing fruit and
        # 0% occluded landmarks, so it has essentially no example of the two
        # ends coinciding and it spreads them apart instead. The fruit then
        # comes out "lying" with a guessed angle, or - when the landmarks are
        # uncertain - as "human check needed".
        #
        # The silhouette does not have that blind spot. end_on.py reads the
        # fruit's own outline and surface, is backend-independent, and was
        # measured at AUC 0.872 leave-one-session-out. Consulting it when the
        # keypoint evidence for "lying" is weak recovers exactly the case the
        # training data is missing, without waiting for more labels.
        self._end_on_overrides_lying = bool(
            block.get("end_on_overrides_lying", True)
        )
        # Only consulted when the keypoints are UNCONVINCING about lying.
        # Measured previously: on fruit with strong stem evidence the end-on
        # classifier scores 0.84-0.98, higher than the fruit it is meant to
        # catch, so it must never be allowed to overrule a confident reading.
        # These two gates are what keep it on the population it works on.
        self._end_on_span_ratio_max = float(
            block.get("end_on_span_ratio_max", 0.45)
        )
        # Kept so an existing config does not fail to load; no longer consulted.
        # See _reconsider_standing for why confidence was the wrong signal.
        self._end_on_conf_max = float(block.get("end_on_conf_max", 0.55))

        # When the classical backend cannot find a stem at all on a fully-
        # visible fruit, ask the silhouette instead of rejecting outright: a
        # paprika is measurably wider at the stem end (the shoulder, where
        # the calyx sits) than at the blossom end, and that is readable from
        # the outline alone - no stem required. See _shape_only_estimate().
        #
        # Two shapes get an answer here, not one:
        #
        #   - "lying" (elongated): the width-profile fallback below measures
        #     a real angle from the taper.
        #   - round (standing on one end): there is no wider end to read
        #     looking straight down the long axis, so no angle is invented -
        #     but a round, fully-segmented silhouette on which colour AND
        #     morphology both failed to find a stem is reported as
        #     STANDING_STEM_DOWN rather than a plain reject. A stem pointing
        #     AT the camera on a stand-up fruit shows as an isolated
        #     protrusion near the centre of a round blob, which is exactly
        #     the shape both stem routes are built to catch; one finding
        #     nothing at all is itself the evidence that the stem is
        #     underneath, not that the detector missed something visible.
        #     This used to be folded into STEM_NOT_FOUND. It was pulled back
        #     out for the same reason STEM_NOT_FOUND was split from
        #     UPSIDE_DOWN in the first place: the two failure populations -
        #     "expected, the fruit is genuinely blossom-up" versus "something
        #     the stem detector should have caught did not get caught" -
        #     point at different people, and conflating them makes both
        #     counters unusable as a diagnosis.
        self._stemless_shape_fallback = bool(block.get("stemless_shape_fallback", True))

        # Placement policy thresholds.
        policy = block.get("policy") if isinstance(block.get("policy"), dict) else {}
        self._min_angle_confidence = float(policy.get("min_angle_confidence", 0.45))
        # Floor below which no angle is reported at all, not even as a guess
        # the operator could overrule.
        self._min_usable_confidence = float(policy.get("min_usable_confidence", 0.15))
        # Refuse to place when the two estimators point in different
        # directions. Measured against 802 hand-labelled fruit, the keypoint
        # error rises monotonically with how far the silhouette disagrees:
        #
        #   disagreement    n    median err   over 20 deg
        #     0-10 deg    439        3.0          10%
        #    10-20 deg    102        4.9          15%
        #    20-35 deg     56        8.0          30%
        #    35-60 deg     22       53.5          77%
        #   60-120 deg     17       91.2          82%
        #  120-181 deg      5      135.7         100%
        #
        # Neither estimator is reliably better - on green the keypoints win on
        # median and the silhouette on p90 - so there is nothing to gain by
        # reweighting them. What they give, cheaply, is a second opinion: where
        # they diverge, one of them is wrong and nothing here can say which.
        #
        # fuse() already measures this and cuts the confidence by 0.6, but a
        # fruit at 0.9 lands on 0.54 and still clears min_angle_confidence, so
        # it still gets placed. This is the hard stop.
        self._max_estimator_disagreement_deg = float(
            policy.get("max_estimator_disagreement_deg", 30.0)
        )
        # Withhold the angle from "unknown" as well, not only from "review".
        # "unknown" already means the ANGLE is not trusted - as distinct from
        # "reorient", which means the angle is trusted and the stem END is not.
        # Printing a number beside a verdict that says the number cannot be
        # relied on invites somebody to rely on it, which is the whole reason
        # these verdicts exist.
        self._hide_unknown_angle = bool(policy.get("hide_unknown_angle", True))
        # Refuse to PLACE on a stem that was never verified, or never seen.
        self._require_verified_stem = bool(
            policy.get("require_verified_stem", True)
        )
        self._min_flip_confidence = float(policy.get("min_flip_confidence", 0.40))
        self._reject_standing = bool(policy.get("reject_standing", True))
        # A fruit whose stem could not be found is not the same as a fruit that
        # cannot be picked. Very often the stem is simply facing away or tucked
        # underneath, and one more pass down the line shows it. Rejecting
        # throws away good produce for a limitation of the view; reorienting
        # costs a cycle and keeps the fruit. It matters most on green, where
        # _stem_by_hue returns nothing by design and this branch will carry
        # most of the crop.
        self._reorient_stem_not_found = bool(
            policy.get("reorient_stem_not_found", True)
        )
        # Label-only for now, on purpose. The classifier is measured at AUC
        # 0.872 leave-one-session-out, which is far better than chance and
        # nowhere near good enough to bin fruit on. Running it in report mode
        # lets it accumulate field evidence about its own accuracy without
        # having cost a single fruit if it turns out to be optimistic - and it
        # will be somewhat optimistic, because it was fitted on six sessions of
        # one line. Set end_on_decides once the logs say it has earned it.
        self._detect_end_on = bool(policy.get("detect_end_on", True))
        self._end_on_decides = bool(policy.get("end_on_decides", False))
        self._end_on_threshold = float(
            policy.get("end_on_threshold", end_on_model.DEFAULT_END_ON_THRESHOLD)
        )
        # A stem smaller than this fraction of the fruit is a speck, not a
        # calyx, and the end-on classifier is consulted instead of trusting it.
        # The classifier CANNOT be applied to fruit with a solid stem: fitted
        # on stemless crops, it scores normal stemmed fruit 0.84-0.98, higher
        # than the fruit it is meant to catch. Restricting it to fruit whose
        # stem evidence is thin keeps it on the population it was measured on.
        self._min_stem_area_ratio = float(policy.get("min_stem_area_ratio", 0.015))
        # Give a stemless fruit that is clearly NOT end-on an axis read from
        # its grooves, instead of refusing to answer at all.
        self._groove_axis_fallback = bool(policy.get("groove_axis_fallback", True))
        self._min_groove_coherence = float(policy.get("min_groove_coherence", 0.35))
        # Maximum measured movement of the stem direction under a lighting
        # change before the fruit is sent round again instead of placed.
        self._max_stem_spread_deg = float(policy.get("max_stem_spread_deg", 6.0))
        # A stem WAS found (this is not the no_stem/groove path above) but the
        # verdict it produced is not placeable. Rather than trust it anyway or
        # give up and send the fruit to review, ask shape_orientation() to
        # settle just the flip - the same fuse() arbitration shape_crosscheck
        # already does, run here only when the un-helped verdict needed help.
        # Scoped to green by default: colour finds the stem directly on
        # red/orange (measured hit rate 87%), so there is nothing uncertain
        # there worth a second opinion; on green, colour cannot see the stem
        # at all and quality comes entirely from morphology plus a brightness
        # self-check that is measurably more fragile - see stem_by_morphology
        # and _stem_selfcheck in classical.py for why.
        self._uncertain_shape_crosscheck = bool(policy.get("uncertain_shape_crosscheck", True))
        self._uncertain_shape_colours = {
            str(colour).strip().lower()
            for colour in (policy.get("uncertain_shape_colours") or ["green"])
        }

        # Machine frame mapping. Changing how the vision zero lines up with the
        # actuator zero must never require a code change - it is a commissioning
        # adjustment, done once per machine, by whoever is standing at it.
        frame_cfg = block.get("frame") if isinstance(block.get("frame"), dict) else {}
        self._angle_offset_deg = float(frame_cfg.get("angle_offset_deg", 0.0))
        self._angle_invert = bool(frame_cfg.get("angle_invert", False))

        # Which fruit the actuator acts on when several are visible.
        self._primary_rule = str(block.get("primary_rule", "largest")).strip().lower()

    # ------------------------------------------------------------------ state

    def is_ready(self) -> bool:
        return self._detector.is_ready()

    def status(self) -> dict:
        status = self._detector.status()
        status.update(
            {
                "shape_crosscheck": self._use_shape_crosscheck,
                "stemless_shape_fallback": self._stemless_shape_fallback,
                "uncertain_shape_crosscheck": self._uncertain_shape_crosscheck,
                "uncertain_shape_colours": sorted(self._uncertain_shape_colours),
                "angle_offset_deg": self._angle_offset_deg,
                "angle_invert": self._angle_invert,
                "primary_rule": self._primary_rule,
            }
        )
        return status

    # ------------------------------------------------------------- evaluation

    def _placement_for(self, result: Orientation) -> str:
        """Decide what the machine should do with this fruit.

        Deliberately conservative. A wrong angle puts a paprika down backwards;
        an honest "I don't know" just sends it round again. The costs are not
        symmetric, so the thresholds are not either.
        """
        if result.pose == orient.POSE_STEM_NOT_FOUND:
            return (
                PLACEMENT_REORIENT if self._reorient_stem_not_found
                else PLACEMENT_REJECT
            )

        if result.pose in (orient.POSE_UPSIDE_DOWN, orient.POSE_INCOMPLETE):
            return PLACEMENT_REJECT

        if result.pose in (orient.POSE_STANDING_STEM_UP, orient.POSE_STANDING_STEM_DOWN):
            # A fruit stood on its end has no meaningful in-plane rotation. The
            # machine has to topple it and look again; there is nothing to
            # place from this view.
            return PLACEMENT_REJECT if self._reject_standing else PLACEMENT_REORIENT

        # Checked before the confidence gates: a fruit the two estimators
        # disagree about is not a low-confidence measurement, it is two
        # measurements that cannot both be right. Reorient rather than reject -
        # the fruit is fine and another pass may settle it.
        if (
            result.agreement_deg is not None
            and result.agreement_deg > self._max_estimator_disagreement_deg
            and result.angle_deg is not None
        ):
            return PLACEMENT_REORIENT

        if result.angle_deg is None:
            return PLACEMENT_UNKNOWN

        # Checked before the ordinary confidence gate, because it is a
        # different statement. Between the two thresholds the machine has a
        # real measurement it does not trust enough to act on; below the floor
        # it has no measurement, and printing a number next to it invites
        # somebody to read meaning into noise.
        if result.confidence < self._min_usable_confidence:
            return PLACEMENT_REVIEW

        if result.confidence < self._min_angle_confidence:
            return PLACEMENT_UNKNOWN

        if result.flip_confidence < self._min_flip_confidence:
            # The axis is solid but which end carries the stem is not. Placing
            # now is a coin flip, so hand it back for another look.
            return PLACEMENT_REORIENT

        return PLACEMENT_PLACE

    @staticmethod
    def _label_for(result: Orientation, placement: str) -> str:
        if placement == PLACEMENT_REVIEW:
            # Deliberately carries no number. The angle behind this verdict is
            # kept in the notes for anyone reading the record back, but it is
            # not put in front of an operator as though it were an answer.
            return "human check needed"
        if result.pose == orient.POSE_STANDING_STEM_UP:
            return "standing, stem up"
        if result.pose == orient.POSE_STANDING_STEM_DOWN:
            return "standing, stem down"
        if result.angle_deg is None:
            return "orientation unknown"
        suffix = "" if placement == PLACEMENT_PLACE else f" ({placement})"
        # "deg", not the degree sign: this string is drawn with OpenCV's Hershey
        # fonts, which are ASCII-only and render anything else as "??".
        return f"stem {result.angle_deg:.0f} deg{suffix}"

    def _shape_only_estimate(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> tuple[Optional[Orientation], list[str]]:
        """Silhouette-only orientation for a fruit whose stem could not be found.

        Returns (orientation, notes). The orientation is None whenever the
        silhouette cannot support one; the notes still describe what was seen,
        so declining to answer is recorded as a measurement rather than as
        silence.

        Only ever called for REASON_NO_STEM, never for an edge-clipped fruit -
        a partial silhouette has no trustworthy width profile either, and
        that case is handled by the caller before this is reached.

        Runs shape_orientation() directly rather than through orient.estimate(),
        which fuses it against keypoints that simply do not exist on this
        path. Returns None - never a half-finished Orientation - only when
        the mask cannot be produced at all, or comes back too small to say
        anything ("mask_too_small": genuinely no evidence, still a plain
        reject). A silhouette that was actually measured and turned out
        round is not that case - see below.

        The angle this returns, when it returns one, still goes through the
        normal _placement_for() gate below like any other estimate - a weak
        width signal (this backend's median on real fruit was 0.074, per the
        measurement behind paprika.shape_crosscheck) is sent for
        reorientation rather than placed on a guess.
        """
        if frame is None:
            return None, []
        mask = orient.segment_fruit(
            frame, bbox, saturation_floor=self._saturation_floor, belt_hue=self._belt_hue
        )
        if mask is None:
            return None, []

        result = orient.shape_orientation(mask)

        if result.pose == orient.POSE_LYING and result.angle_deg is not None:
            result.source = "shape_only"
            result.notes.append("no_stem")
            result.notes.append("shape_fallback")
            return result, []

        # A round silhouette used to be reported as POSE_STANDING_STEM_DOWN,
        # on the reasoning that a fruit with no visible stem and no long axis
        # must be standing on end. Measurement does not support it: a
        # blokpaprika is close to a rounded cube, so it is round in outline
        # from every direction, and over 225 red-fruit frames 43% of correctly
        # PLACED fruit measured below the same 1.12 roundness line. Two frames
        # make the point on their own - one fruit genuinely blossom-up
        # measured 1.084, another lying flat on its side measured 1.073.
        #
        # So roundness cannot tell "standing" from "lying with the stem hidden
        # or facing away", and claiming otherwise put a confident pose on a
        # coin toss. What this branch actually knows is that no stem was
        # found, which is what it now says. The reject is unchanged; only the
        # claim is. The measurement is passed back as a note so the reason is
        # still on the record.
        if result.pose == orient.POSE_UNKNOWN and result.elongation > 0.0:
            return None, [f"round_silhouette (elongation={result.elongation:.2f})"]

        return None, []

    def _reconsider_standing(self, frame, bbox, result, stem):
        """Ask the silhouette whether a "lying" fruit is really standing.

        Runs only when the keypoints are unconvincing: either the two landmarks
        sit close together (already near the standing threshold) or the pair
        confidence is low. A confident, well-separated pair is left alone,
        because the end-on classifier is measurably unreliable on fruit whose
        stem evidence is strong and would overrule good readings.

        Which end is up comes from the stem landmark's own visibility, exactly
        as keypoint_orientation decides it - a stem the model could see means
        stem-up, one it placed but could not see means stem-down. That is the
        occluded case ANNOTATION_SPEC section 3 describes, and it is the one
        piece of this the model does report usefully even when it misplaces
        the landmark.

        The angle is withdrawn, not merely relabelled. A rotation angle for a
        fruit standing on its end is not an unknown quantity, it is not a
        quantity at all, and leaving a plausible number attached to a standing
        verdict is how it ends up being read as one.
        """
        if not self._end_on_overrides_lying or result.pose != orient.POSE_LYING:
            return result
        if frame is None:
            return result

        x1, y1, x2, y2 = bbox
        diagonal = math.hypot(max(1, x2 - x1), max(1, y2 - y1))
        span_ratio = (result.stem_span_px or 0.0) / max(1.0, diagonal)
        # Landmark separation ONLY. The confidence clause that used to sit here
        # was wrong in kind: a fruit lying down with both landmarks correctly
        # placed far apart is not standing, whatever the model's confidence in
        # them. Pose keypoint confidences sit around 0.5 on ordinary fruit, so
        # "or confidence < 0.55" opened the gate on most of the crop - and the
        # end-on classifier, which scores ordinary stemmed fruit 0.84 to 0.98,
        # then called them standing with the stem up.
        #
        # Standing is a geometric claim: seen down its own axis, a fruit's two
        # ends project close together. That is what span_ratio measures and it
        # is the only thing that should open this gate.
        unconvincing = span_ratio < self._end_on_span_ratio_max
        if not unconvincing:
            return result

        probability = self._end_on_probability(frame, bbox)
        if probability is None:
            return result
        result.notes.append(f"end_on_p={probability:.2f}")
        if probability <= self._end_on_threshold:
            return result

        stem_visible = bool(stem is not None and getattr(stem, "visible", False))
        result.pose = (
            orient.POSE_STANDING_STEM_UP if stem_visible
            else orient.POSE_STANDING_STEM_DOWN
        )
        result.notes.append(
            f"standing_from_silhouette (span_ratio={span_ratio:.2f}, "
            f"was angle={result.angle_deg:.0f}deg)"
            if result.angle_deg is not None else "standing_from_silhouette"
        )
        result.angle_deg = None
        result.axis_deg = None
        return result

    def _end_on_probability(self, frame, bbox) -> Optional[float]:
        """P(looking down this fruit's axis), or None when it cannot be read."""
        if frame is None:
            return None
        mask = orient.segment_fruit(
            frame, bbox, saturation_floor=self._saturation_floor, belt_hue=self._belt_hue
        )
        if mask is None:
            return None
        return end_on_model.end_on_probability(
            frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], mask
        )

    def _groove_axis_estimate(self, frame, bbox) -> Optional[Orientation]:
        """An axis for a stemless fruit that is clearly lying on its side.

        The outline cannot give it - a blokpaprika on its side measures about
        1.07 elongation and the principal axis of a shape that round is noise.
        The grooves can: they run stem to blossom, so side-on they cross the
        fruit as parallel bands whose shared direction is the axis.

        flip_confidence is left at zero on purpose. Grooves give an
        ORIENTATION, not a direction - both ends look alike, and nothing here
        establishes which one carries the stem. The angle is reported so the
        operator and the log can see it, and the existing flip policy sends the
        fruit for another look rather than placing it on a coin toss.
        """
        if not self._groove_axis_fallback or frame is None:
            return None
        mask = orient.segment_fruit(
            frame, bbox, saturation_floor=self._saturation_floor, belt_hue=self._belt_hue
        )
        if mask is None:
            return None
        measured = end_on_model.groove_axis(
            frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], mask
        )
        if measured is None:
            return None
        axis, coherence = measured
        if coherence < self._min_groove_coherence:
            # A smooth fruit with no readable grooves. The axis would be the
            # direction of whatever noise happened to be strongest.
            return None
        return Orientation(
            source="grooves",
            pose=orient.POSE_LYING,
            angle_deg=axis,
            axis_deg=axis,
            stem_present=False,
            confidence=float(coherence),
            flip_confidence=0.0,
            notes=[f"groove_axis coherence={coherence:.2f}", "stem_end_unknown"],
        )

    def _end_on_verdict(
        self, frame: np.ndarray, bbox, pose: str
    ) -> tuple[str, list[str]]:
        """Is this stemless fruit end-on? Reported always, acted on only if asked.

        Returns the pose to record and any notes. With end_on_decides off - the
        default - the pose is returned unchanged and only a note is added, so
        the classifier's opinion lands in the results and the logs while every
        placement stays exactly where it was.
        """
        if not self._detect_end_on or frame is None:
            return pose, []
        mask = orient.segment_fruit(
            frame, bbox, saturation_floor=self._saturation_floor, belt_hue=self._belt_hue
        )
        if mask is None:
            return pose, []
        probability = end_on_model.end_on_probability(
            frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], mask
        )
        if probability is None:
            # Unreadable is not "side-on". Say nothing rather than imply an
            # answer that was never computed.
            return pose, ["end_on=unreadable"]

        notes = [f"end_on_p={probability:.2f}"]
        if probability <= self._end_on_threshold:
            return pose, notes

        notes.append("end_on_detected")
        if not self._end_on_decides:
            # Reporting only: the fruit still goes wherever stem_not_found
            # sends it. Below half the end-on fruit are caught at this
            # threshold, so a fruit NOT flagged means nothing either way.
            return pose, notes
        return orient.POSE_UPSIDE_DOWN, notes

    def _evaluate_one(self, frame: np.ndarray, detection: dict) -> dict:
        bbox = detection["bbox"]
        landmarks: dict = detection.get("keypoints") or {}

        stem: Optional[Keypoint] = landmarks.get("stem_end")
        blossom: Optional[Keypoint] = landmarks.get("blossom_end")

        # The silhouette runs only as a second opinion alongside existing
        # keypoints. Without keypoints it would be the only source, and on
        # blocky paprika it manages just 66% on the question of which end holds
        # the stem - "no angle" is a more honest answer than a coin toss.
        use_shape = self._use_shape_crosscheck and stem is not None and blossom is not None

        # Unpickable is an outcome, not a failure. No angle is computed here on
        # purpose: even a good estimate would not help the robot, because it
        # cannot pick up an upside-down fruit and turn it over. Returning a
        # number would imply an action that does not exist.
        # A "stem" too small to be a calyx is not evidence. Where the stem is
        # marginal AND the silhouette says the camera is looking down the
        # fruit's axis, the speck is discarded and the fruit is handled as
        # what it actually is: stemless, and pointing at the lens.
        #
        # Both halves are load-bearing. Size alone would need a 1.3% threshold
        # to catch all four known cases and would cost 24 of 239 placements.
        # The classifier alone cannot be used here at all - fitted on stemless
        # crops, it scores ordinary stemmed fruit 0.84-0.98, ABOVE the fruit it
        # is meant to catch. Together they fire on 5 of 239 placed fruit, four
        # of which are the four being placed at a guessed angle while lying
        # blossom-up.
        reason = str(detection.get("unpickable_reason") or "")
        endon_note: list[str] = []
        # stem_area_ratio measures a morphology blob against the fruit it sits
        # on, so it only means anything where a blob was found. The pose
        # backend has no such blob and emits no such field, and reading the
        # default of 0.0 made "marginal" always true: every pose detection ran
        # the end-on classifier, roughly 5ms a fruit, and any that scored above
        # the threshold had its landmarks thrown away and was reported
        # stemless. A missing measurement is not a small measurement.
        stem_route = str(detection.get("stem_method", "none"))
        ratio_is_meaningful = stem_route in ("hue", "morphology")
        if (
            self._detect_end_on
            and not reason
            and detection.get("keypoints")
            and ratio_is_meaningful
            and float(detection.get("stem_area_ratio", 0.0) or 0.0)
                < self._min_stem_area_ratio
        ):
            probability = self._end_on_probability(frame, bbox)
            if probability is not None:
                endon_note = [f"end_on_p={probability:.2f}"]
                if probability > self._end_on_threshold:
                    endon_note.append("stem_speck_rejected")
                    detection = {
                        **detection,
                        "keypoints": {},
                        "stem_method": "none",
                        "unpickable_reason": "no_stem",
                    }
                    reason = "no_stem"

        result: Optional[Orientation] = None
        # Diagnostics from the shape fallback survive even when it declines to
        # produce an orientation - "I looked and the silhouette was round" is
        # a different record from "I never looked".
        fallback_notes: list[str] = list(endon_note)

        if reason == "no_stem" and self._stemless_shape_fallback:
            # No stem found, but the fruit is fully in frame. Before writing
            # it off, ask the one estimator that never looks at the stem.
            # Returns None (and falls through to the plain reject below)
            # whenever the silhouette turns out to be round - that is a
            # standing or upside-down fruit, and it should be detected, not
            # guessed at.
            result, fallback_notes = self._shape_only_estimate(frame, bbox)

        # Decided BEFORE the unpickable branch below, because that branch
        # commits to a verdict and returns. An earlier revision set `result`
        # from inside it and the assignment simply had no effect - the fruit
        # was already on its way out as unmeasurable.
        if reason == "no_stem" and result is None:
            looks_end_on, extra = self._end_on_verdict(
                frame, bbox, orient.POSE_STEM_NOT_FOUND
            )
            fallback_notes = [*fallback_notes, *extra]
            if looks_end_on == orient.POSE_STEM_NOT_FOUND:
                # Not end-on and no stem: the fruit IS lying there with a
                # measurable axis, so measure it rather than reporting nothing.
                # An orientation with no flip still beats no orientation - the
                # operator sees where it lies, and the flip policy decides what
                # to do about the end nobody can identify.
                from_grooves = self._groove_axis_estimate(frame, bbox)
                if from_grooves is not None:
                    from_grooves.notes.extend(fallback_notes)
                    result = from_grooves
                    reason = ""
            else:
                reason = reason or "no_stem"
                fallback_notes = [*fallback_notes]

        if reason and result is None:
            pose = _UNPICKABLE_POSES.get(reason, orient.POSE_UPSIDE_DOWN)
            if pose == orient.POSE_STEM_NOT_FOUND and self._end_on_decides:
                pose, _ = self._end_on_verdict(frame, bbox, pose)
            unusable = Orientation(
                source="classical",
                pose=pose,
                stem_present=False,
                notes=[reason, *fallback_notes],
            )
            x1, y1, x2, y2 = bbox
            return {
                "label": _UNPICKABLE_LABELS.get(pose, "unusable"),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                "confidence": float(detection.get("confidence", 0.0)),
                "keypoints": {},
                "orientation": unusable.to_dict(),
                "angle_plc": None,
                # upside_down and incomplete are end states - nothing the line
                # can do changes them. stem_not_found is not: it describes what
                # this view failed to show, not what the fruit is, so by
                # default it goes round again instead of into the bin. See
                # paprika.policy.reorient_stem_not_found.
                "placement": self._placement_for(unusable),
                "simulated": False,
                "colour": detection.get("colour", ""),
                "stem_method": detection.get("stem_method", "none"),
            }

        if result is None:
            result = orient.estimate(
                frame=frame if use_shape else None,
                bbox=bbox,
                stem=stem,
                blossom=blossom,
                use_shape=use_shape,
                saturation_floor=self._saturation_floor,
                belt_hue=self._belt_hue,
                min_span_ratio=self._min_span_ratio,
            )

        result = self._reconsider_standing(frame, bbox, result, stem)

        placement = self._placement_for(result)

        # A stem WAS found here (this branch is only reached when reason was
        # never "no_stem" above), so the fruit is not stemless - it simply was
        # not trusted enough to place outright. Tightening stem detection to
        # push more green fruit into the stemless fallback above does not fix
        # that; it only trades one failure mode for the other, since that
        # fallback runs the same width-profile geometry a real visible stem
        # can corrupt (see the shape_orientation docstring in orientation.py
        # for the measured version of that failure). Consulting the shape
        # estimator here instead asks it to settle exactly what fuse() already
        # knows how to settle - the flip - using the same arbitration
        # shape_crosscheck performs, without another dial on the stem search.
        if (
            placement != PLACEMENT_PLACE
            and self._uncertain_shape_crosscheck
            and not use_shape
            and frame is not None
            and detection.get("colour", "").strip().lower() in self._uncertain_shape_colours
        ):
            uncertain_mask = orient.segment_fruit(
                frame, bbox, saturation_floor=self._saturation_floor, belt_hue=self._belt_hue
            )
            shape_result = (
                orient.shape_orientation(uncertain_mask) if uncertain_mask is not None else None
            )
            if shape_result is not None and shape_result.angle_deg is not None:
                if result.confidence < self._min_usable_confidence:
                    # fuse() always keeps the keypoint axis on the reasoning
                    # that it was "produced" by a detector trained on this
                    # exact fruit - but a self-check that collapsed quality
                    # to (near) zero is the detector itself saying it does not
                    # trust that axis either. Below the same floor that sends
                    # a fruit to review anyway, keeping it would only be
                    # honouring a technicality, not real evidence. Let shape
                    # stand in fully here, the same as the genuinely stemless
                    # case above, rather than asking fuse()'s modest
                    # agreement bonus to climb out of a near-zero start.
                    shape_result.notes.extend(result.notes)
                    shape_result.notes.append(
                        f"uncertain_stem_shape_crosscheck (stem confidence {result.confidence:.2f})"
                    )
                    result = shape_result
                else:
                    # Confidence is usable, only the flip (or the margin) was
                    # in question - fuse()'s normal arbitration already
                    # handles exactly this.
                    fused = orient.fuse(result, shape_result)
                    fused.notes.append("uncertain_stem_shape_crosscheck")
                    result = fused
                placement = self._placement_for(result)

        # A fruit whose confidence collapsed still has a body lying on a belt,
        # and its grooves still run along its axis. Falling back to them turns
        # "human check needed" into an axis the operator and the log can use.
        # Deliberately reached from HERE rather than only from the stem-not-
        # found path: these fruit DO have a stem, it is simply not trusted, and
        # an earlier revision wired the fallback somewhere they never pass.
        #
        # No flip is claimed, so the fruit still goes round again rather than
        # being placed on an axis with an unknown end.
        if placement == PLACEMENT_REVIEW and self._groove_axis_fallback:
            from_grooves = self._groove_axis_estimate(frame, bbox)
            if from_grooves is not None:
                from_grooves.notes.extend(result.notes)
                from_grooves.notes.append(
                    f"low_confidence_fallback (was {result.confidence:.2f})"
                )
                result = from_grooves
                placement = self._placement_for(result)

        # Two guards on PLACING, both from the first ground-truth measurement
        # this project has had: 470 hand-clicked stem positions scored against
        # the detector. Overall the machine is good - placed fruit sit at a
        # median of 2.4 degrees and 90% inside 10 - but the tail reaching the
        # actuator was concentrated in two identifiable groups.
        #
        # shape_only: the stemless fallback. No stem was found, so the end is
        # inferred from the silhouette alone. It was 1.2% of placements and 3
        # of the 15 worst, landing 30 to 105 degrees out. An estimator that
        # never saw a stem should not be trusted to say which end it is on.
        #
        # selfcheck_unavailable: the stability check could not run, so nothing
        # verified this stem. 6.2% of placements and 5 of the 15 worst,
        # including the single worst at 173 degrees off. Unverified is not the
        # same as verified-good, and placing on it was reading it as the latter.
        #
        # Both fall back to reorient rather than reject: the fruit is fine, the
        # measurement is simply not good enough to act on, and another pass may
        # produce one that is. Costs about 7% of placements on the test set.
        if placement == PLACEMENT_PLACE:
            if self._require_verified_stem and result.source == "shape_only":
                placement = PLACEMENT_REORIENT
                result.notes.append("not_placed: shape_only")
            elif (
                self._require_verified_stem
                and str(detection.get("stem_selfcheck", "skipped")) == "unavailable"
            ):
                placement = PLACEMENT_REORIENT
                result.notes.append("not_placed: stem unverified")

        # Below the usable floor the angle is withdrawn, not merely flagged.
        # Leaving it in the result means the HMI dial still swings to it and
        # the operator still reads a number - which is the thing this verdict
        # exists to prevent. It is kept in the notes instead, where anyone
        # reading the record back can see what was discarded and why.
        withhold = placement == PLACEMENT_REVIEW or (
            placement == PLACEMENT_UNKNOWN and self._hide_unknown_angle
        )
        if withhold and result.angle_deg is not None:
            result.notes.append(
                f"angle_withheld={result.angle_deg:.0f}deg "
                f"confidence={result.confidence:.2f}"
            )
            result.angle_deg = None
            result.axis_deg = None

        # A stem direction that moves with the light is not a direction. This
        # is measured per fruit by re-running the detection at two other gains,
        # so it reflects this fruit under this light rather than an average
        # taken over the dataset.
        #
        # An unavailable check reports a spread of 0.0, so it cannot trip the
        # threshold below - no extra guard is needed here, only a note, so that
        # a record showing spread 0 is not later mistaken for a rock-steady
        # stem when in truth nobody managed to measure it.
        if str(detection.get("stem_selfcheck", "skipped")) == "unavailable":
            result.notes.append("selfcheck_unavailable")
        spread = float(detection.get("stem_spread_deg", 0.0) or 0.0)
        if placement == PLACEMENT_PLACE and spread > self._max_stem_spread_deg:
            placement = PLACEMENT_REORIENT
            result.notes.append(f"stem_unstable={spread:.0f}deg")
            result.flip_confidence = min(result.flip_confidence, 0.3)

        x1, y1, x2, y2 = bbox

        return {
            "label": self._label_for(result, placement),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "confidence": float(detection.get("confidence", 0.0)),
            "keypoints": {
                name: {
                    "x": round(float(kp.x), 1),
                    "y": round(float(kp.y), 1),
                    "confidence": round(float(kp.confidence), 3),
                    "visible": bool(kp.visible),
                }
                for name, kp in landmarks.items()
            },
            "orientation": result.to_dict(),
            "angle_plc": _round_or_none(
                orient.apply_frame_convention(
                    result.angle_deg, self._angle_offset_deg, self._angle_invert
                )
            ),
            "placement": placement,
            "simulated": bool(detection.get("simulated", False)),
            # Diagnostics, not a decision. Without these the logs cannot show
            # whether an erratic result came from the fruit colour or from the
            # stem method, which is exactly what you want to know when
            # something is behaving inconsistently.
            "colour": detection.get("colour", ""),
            "stem_method": detection.get("stem_method", "none"),
            "stem_quality": detection.get("stem_quality", 0.0),
            "stem_spread_deg": round(float(detection.get("stem_spread_deg", 0.0) or 0.0), 1),
        }

    def _pick_primary(
        self,
        detections: list[dict],
        frame_width: int = 0,
        frame_height: int = 0,
    ) -> Optional[dict]:
        """Choose the fruit the actuator acts on.

        "largest" is the default because on a belt the biggest silhouette is
        normally the one fully in view, rather than one half-entering frame
        whose angle is being measured from a partial fruit.

        Args:
            frame_width, frame_height: size of the frame the detections came
                from. Only the "centered" rule needs them, but it needs them
                absolutely: centre-ness is meaningless without knowing where
                the centre is.
        """
        if not detections:
            return None

        placeable = [d for d in detections if d["placement"] == PLACEMENT_PLACE]
        # An unusable fruit must never displace a usable one, not even if it is
        # larger: what matters is what the robot can actually act on.
        usable = [d for d in detections if d["placement"] != PLACEMENT_REJECT]
        pool = placeable or usable or detections

        if self._primary_rule == "confidence":
            return max(pool, key=lambda d: d["orientation"].get("confidence", 0.0))

        if self._primary_rule == "centered":
            # Distance from the middle of the frame. This used to measure from
            # the image ORIGIN, which made "centered" quietly mean "nearest the
            # top-left corner" - i.e. it preferred the fruit just entering the
            # frame, the exact opposite of the intent, and on a belt running
            # left to right it picked a different fruit every time. A rule that
            # names the centre has to be told where the centre is, which is why
            # the frame size is now a parameter rather than an assumption.
            if frame_width > 0 and frame_height > 0:
                cx, cy = frame_width / 2.0, frame_height / 2.0
                return min(
                    pool,
                    key=lambda d: math.hypot(d["center"][0] - cx, d["center"][1] - cy),
                )
            # No frame size means the centre is unknowable. Fall through to
            # "largest" rather than silently reinstating the origin bug.
            log.warning(
                "primary_rule='centered' needs the frame size; falling back to 'largest'"
            )

        return max(
            pool,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        )

    def evaluate(self, frame: np.ndarray) -> dict:
        """Full cycle: detect, orient, decide."""
        start = time.perf_counter()

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {
                "status": "NOK",
                "mode": "paprika",
                "detections": [],
                "primary": None,
                "confidence": 0.0,
                "processing_time_ms": 0.0,
                "failure_reason": "no_frame",
            }

        if not self.is_ready():
            return {
                "status": "NOK",
                "mode": "paprika",
                "detections": [],
                "primary": None,
                "confidence": 0.0,
                "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
                "failure_reason": "detector_unavailable",
            }

        try:
            raw = self._detector.detect(frame)
        except Exception as exc:
            log.error("Paprika detection failed: %s", exc)
            return {
                "status": "NOK",
                "mode": "paprika",
                "detections": [],
                "primary": None,
                "confidence": 0.0,
                "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
                "failure_reason": "detector_error",
                "error": str(exc),
            }

        detections = [self._evaluate_one(frame, d) for d in raw]
        frame_height, frame_width = frame.shape[:2]
        primary = self._pick_primary(detections, frame_width, frame_height)

        processing_time_ms = round((time.perf_counter() - start) * 1000, 2)

        if primary is None:
            return {
                "status": "NOK",
                "mode": "paprika",
                "detections": [],
                "primary": None,
                "confidence": 0.0,
                "processing_time_ms": processing_time_ms,
                "failure_reason": "no_paprika_detected",
            }

        placeable = primary["placement"] == PLACEMENT_PLACE

        result = {
            # OK means "the machine may act on this angle", not "a fruit was
            # seen". A detected fruit whose orientation cannot be trusted is a
            # NOK with a reason, because acting on it is the failure mode this
            # whole engine exists to prevent.
            "status": "OK" if placeable else "NOK",
            "mode": "paprika",
            "detections": detections,
            "primary": primary,
            "confidence": float(primary["orientation"].get("confidence", 0.0)),
            "processing_time_ms": processing_time_ms,
        }
        if not placeable:
            result["failure_reason"] = f"not_placeable_{primary['placement']}"
        return result