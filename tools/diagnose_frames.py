"""
Frame diagnosis

    python -m tools.diagnose_frames data/debug/failures
    python -m tools.diagnose_frames data/debug/failures --masks
    python -m tools.diagnose_frames one_frame.jpg --out data/debug/friday

Runs the real detector and the real engine over raw frames and prints every
number that decided the outcome, one line per fruit.

Why this exists: a screenshot of the HMI shows you the verdict and nothing
else. When a fruit comes back with the wrong angle, the question is which of
about a dozen quantities went wrong first - was the stem found by colour or by
morphology, did the self-check see it move, was the silhouette consulted at
all, did the split gate fire, was the box inflated by a neighbour - and none of
that is on screen. Guessing which one it was from a picture is how you end up
tuning a threshold that was never involved.

It calls PaprikaDetector and PaprikaEngine directly rather than reimplementing
the pipeline, so what it reports is what the machine did, not an approximation
of it. If this tool and the running system ever disagree, that is a bug in one
of them and worth knowing about.

The silhouette estimator is run on every fruit regardless of the
shape_crosscheck setting, because its opinion is diagnostic even when the
policy ignores it. Where it is reported as disagreeing, the reference is the
KEYPOINTS, not ground truth - the tool has no idea where the stem really is.
Two estimators disagreeing tells you to go and look at that frame; it does not
tell you which one was wrong.

Outputs land in data/debug/ (gitignored):
    <out>/ann_<name>.jpg    the frame with the overlay the HMI would have drawn
    <out>/mask_<name>_f<i>.png   segmentation mask per fruit, with --masks
    <out>/records.jsonl     one JSON record per fruit, for later analysis
    <out>/report.txt        a copy of everything printed below
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detection.paprika import classical  # noqa: E402
from backend.detection.paprika import orientation as orient  # noqa: E402
from backend.detection.paprika.pose_detector import PaprikaDetector  # noqa: E402
from backend.core.paprika_engine import PaprikaEngine  # noqa: E402
from backend.utils.annotate import draw_detections  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# Disagreement beyond this counts as "the two estimators picked opposite ends"
# rather than "they differ a bit". Well clear of the 30-40 degrees that ordinary
# axis noise produces, well below 180.
OPPOSITE_ENDS_DEG = 120.0


def _round(value, places: int = 3):
    """Printed numbers are for reading. Full float precision in the console
    hides the digits that matter behind fifteen that do not; records.jsonl
    keeps the unrounded values for anything that needs them."""
    return None if value is None else round(float(value), places)


class Tee:
    """Print to the console and to report.txt at once.

    The console output is the point of the tool, and it is also the thing you
    want to paste into a bug report an hour later. Writing it twice by hand is
    how the two versions drift apart.
    """

    def __init__(self, path: Path) -> None:
        self._handle = path.open("w", encoding="utf-8")

    def __call__(self, line: str = "") -> None:
        print(line)
        self._handle.write(line + "\n")

    def close(self) -> None:
        self._handle.close()


def load_config(explicit: str | None) -> dict:
    """Load the same config the application uses.

    Falls back to reading the YAML directly if config_loader is unhappy: this
    is a diagnostic tool and should still run on a machine that has the frames
    but not a fully wired application config.
    """
    import yaml

    if explicit:
        return yaml.safe_load(Path(explicit).read_text(encoding="utf-8")) or {}
    try:
        from backend.core.config_loader import cfg

        return cfg
    except (Exception, SystemExit):
        # SystemExit is deliberate: config_loader calls sys.exit() when the
        # file is missing, and that must not kill this tool.
        default = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
        if not default.exists():
            raise SystemExit(f"No config found - looked for {default}")
        return yaml.safe_load(default.read_text(encoding="utf-8")) or {}


def gather_frames(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    frames = sorted(
        p for p in target.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    return frames


def belt_limits(frame: np.ndarray, belt_hue: tuple[int, int]) -> dict:
    """Reproduce the size limits find_fruit() derives from the belt.

    These are the numbers every area test in the detector is measured against,
    and they are invisible at runtime. A fruit that is "too big" or a blob that
    "should have been split" only makes sense next to them.
    """
    belt = classical.belt_mask_raw(frame, belt_hue)
    width = classical.belt_width(belt)
    if width is None:
        return {
            "belt_width_px": None,
            "belt_cut_off": None,
            "part_min": float(classical.SPLIT_MIN_PART_AREA),
            "split_solidity_gate_below": float(classical.SPLIT_MIN_PART_AREA * 4),
            "area_min": None,
            "area_max_fruit": None,
        }
    cut = classical.belt_is_cut_off(belt, frame.shape)
    slack = classical.BELT_CUTOFF_SLACK if cut else 1.0
    part_min = classical.MIN_PART_AREA_PER_BELT2 * width ** 2
    return {
        "belt_width_px": float(width),
        "belt_cut_off": bool(cut),
        "part_min": float(part_min),
        # Above this, split_touching() stops consulting solidity and always
        # attempts a split. Worth seeing per fruit: a compact single paprika
        # sitting above the gate is one distance-transform artefact away from
        # being reported as two.
        "split_solidity_gate_below": float(part_min * 4),
        "area_min": float(max(600.0, classical.MIN_FRUIT_AREA_PER_BELT2 * width ** 2)),
        "area_max_fruit": float(classical.MAX_FRUIT_AREA_PER_BELT2 * width ** 2 * slack),
    }


def silhouette_opinion(
    frame: np.ndarray, bbox, saturation_floor: int, belt_hue: tuple[int, int]
) -> tuple[dict, np.ndarray | None]:
    """What shape_orientation() thinks, whether or not the policy asked it."""
    mask = orient.segment_fruit(
        frame, bbox, saturation_floor=saturation_floor, belt_hue=belt_hue
    )
    if mask is None:
        return {"shape_pose": None}, None
    result = orient.shape_orientation(mask)
    area = int((mask > 0).sum())
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = None
    if contours:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        solidity = float(cv2.contourArea(contour) / max(1.0, cv2.contourArea(hull)))
    return (
        {
            "shape_pose": result.pose,
            "shape_angle_deg": result.angle_deg,
            "shape_flip_confidence": result.flip_confidence,
            "shape_elongation": result.elongation,
            "shape_notes": list(result.notes),
            "mask_area_px": area,
            "mask_solidity": solidity,
        },
        mask,
    )


def analyse(args) -> int:
    cfg = load_config(args.config)
    block = cfg.get("paprika") or {}
    shape_cfg = block.get("shape") if isinstance(block.get("shape"), dict) else {}
    policy = block.get("policy") if isinstance(block.get("policy"), dict) else {}

    belt = shape_cfg.get("belt_hue") or [96, 145]
    belt_hue = (int(belt[0]), int(belt[1]))
    saturation_floor = int(shape_cfg.get("saturation_floor", 80))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = Tee(out_dir / "report.txt")
    records_path = out_dir / "records.jsonl"
    records_handle = records_path.open("w", encoding="utf-8")

    frames = gather_frames(Path(args.frames))
    if not frames:
        say(f"No images found under {args.frames}")
        records_handle.close()
        say.close()
        return 1

    detector = PaprikaDetector(block)
    engine = PaprikaEngine(cfg)

    say(f"backend={detector.backend}  belt_hue={belt_hue}  "
        f"saturation_floor={saturation_floor}  "
        f"value_floor={shape_cfg.get('value_floor', 45)}")
    say(f"policy: min_angle_confidence={policy.get('min_angle_confidence', 0.45)}  "
        f"min_flip_confidence={policy.get('min_flip_confidence', 0.40)}  "
        f"max_stem_spread_deg={policy.get('max_stem_spread_deg', 6.0)}  "
        f"reject_standing={policy.get('reject_standing', True)}")
    say(f"shape_crosscheck={block.get('shape_crosscheck')}  "
        f"stemless_shape_fallback={block.get('stemless_shape_fallback')}  "
        f"min_span_ratio={block.get('min_span_ratio')}  "
        f"primary_rule={block.get('primary_rule')!r}")
    say(f"{len(frames)} frame(s) from {args.frames}")
    say()

    rows: list[dict] = []

    for path in frames:
        frame = cv2.imread(str(path))
        if frame is None:
            say(f"!! could not read {path}")
            continue

        height, width = frame.shape[:2]
        limits = belt_limits(frame, belt_hue)

        say("=" * 100)
        say(f"{path.name}   {width}x{height}")
        if limits["belt_width_px"]:
            say(f"  belt_width={limits['belt_width_px']:.0f}px "
                f"cut_off={limits['belt_cut_off']}  "
                f"area_min={limits['area_min']:,.0f}  "
                f"area_max_fruit={limits['area_max_fruit']:,.0f}  "
                f"part_min={limits['part_min']:,.0f}  "
                f"split_solidity_gate_below={limits['split_solidity_gate_below']:,.0f}")
        else:
            say("  belt NOT detected - absolute pixel fallbacks in use, and every "
                "area limit below is a fixed number rather than a belt fraction")

        raw = detector.detect(frame)
        result = engine.evaluate(frame)
        primary_center = (result.get("primary") or {}).get("center")

        say(f"  detector -> {len(raw)} raw   engine -> status={result['status']}, "
            f"{len(result['detections'])} detection(s)")

        # Detections come back in the same order the detector produced them, so
        # they pair up by index. Guard anyway: an engine that ever reorders or
        # drops one would otherwise silently mislabel every line below.
        if len(raw) != len(result["detections"]):
            say("  !! raw and evaluated counts differ - pairing by index is unsafe, "
                "fields from the two sources may not belong to the same fruit")

        for index, detection in enumerate(result["detections"]):
            source = raw[index] if index < len(raw) else {}
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox
            orientation = detection["orientation"]

            shape_view, mask = silhouette_opinion(
                frame, bbox, saturation_floor, belt_hue
            )

            landmarks = source.get("keypoints") or {}
            stem_kp = landmarks.get("stem_end")
            blossom_kp = landmarks.get("blossom_end")
            span = span_ratio = keypoint_angle = None
            diagonal = math.hypot(max(1, x2 - x1), max(1, y2 - y1))
            if stem_kp is not None and blossom_kp is not None:
                dx = stem_kp.x - blossom_kp.x
                dy = stem_kp.y - blossom_kp.y
                span = math.hypot(dx, dy)
                span_ratio = span / diagonal
                keypoint_angle = orient.vector_to_angle(dx, dy)

            disagreement = None
            if keypoint_angle is not None and shape_view.get("shape_angle_deg") is not None:
                disagreement = orient.angular_difference(
                    keypoint_angle, shape_view["shape_angle_deg"]
                )

            mask_area = shape_view.get("mask_area_px")
            gate = limits["split_solidity_gate_below"]
            # Not an error on its own - it only means the cheap "this is one
            # compact fruit, leave it alone" shortcut did not apply, so the
            # blob went to the splitter on its merits.
            gate_bypassed = bool(mask_area and mask_area > gate)

            row = {
                "frame": path.name,
                "index": index,
                "bbox": [int(v) for v in bbox],
                "bbox_w": int(x2 - x1),
                "bbox_h": int(y2 - y1),
                "bbox_diagonal": round(diagonal, 1),
                "colour": source.get("colour", ""),
                "stem_method": source.get("stem_method", "none"),
                "stem_spread_deg": source.get("stem_spread_deg", 0.0),
                "unpickable_reason": source.get("unpickable_reason", ""),
                "keypoint_confidence": (
                    round(float(stem_kp.confidence), 3) if stem_kp is not None else None
                ),
                "span_px": None if span is None else round(span, 1),
                "span_ratio": None if span_ratio is None else round(span_ratio, 3),
                "keypoint_angle_deg": (
                    None if keypoint_angle is None else round(keypoint_angle, 1)
                ),
                "label": detection["label"],
                "placement": detection["placement"],
                "pose": orientation.get("pose"),
                "orientation_source": orientation.get("source"),
                "angle_deg": orientation.get("angle_deg"),
                "angle_plc": detection.get("angle_plc"),
                "confidence": orientation.get("confidence"),
                "flip_confidence": orientation.get("flip_confidence"),
                "notes": orientation.get("notes", []),
                "split_gate_bypassed": gate_bypassed,
                "shape_vs_keypoint_deg": (
                    None if disagreement is None else round(disagreement, 1)
                ),
                **shape_view,
                **{f"limits_{k}": v for k, v in limits.items()},
            }
            rows.append(row)
            records_handle.write(json.dumps(row, default=str) + "\n")

            say(f"   [{index}] {detection['label']!r}  placement={detection['placement']}"
                f"  pose={orientation.get('pose')}  src={orientation.get('source')}")
            say(f"        bbox=({x1},{y1},{x2},{y2}) {x2-x1}x{y2-y1}"
                f"  mask_area={mask_area if mask_area else '-'}"
                f"  solidity={_round(shape_view.get('mask_solidity'))}"
                f"  colour={source.get('colour','')!r}")
            say(f"        stem_method={source.get('stem_method','none')!r}"
                f"  kp_conf={row['keypoint_confidence']}"
                f"  spread={source.get('stem_spread_deg', 0.0)}deg"
                f"  reason={source.get('unpickable_reason','')!r}")
            say(f"        span={row['span_px']} diag={diagonal:.0f}"
                f"  span_ratio={row['span_ratio']}"
                f"  -> confidence={orientation.get('confidence')}"
                f"  flip={orientation.get('flip_confidence')}")
            say(f"        angle_deg={orientation.get('angle_deg')}"
                f"  angle_plc={detection.get('angle_plc')}"
                f"  notes={orientation.get('notes', [])}")
            say(f"        SILHOUETTE pose={shape_view.get('shape_pose')}"
                f" elong={_round(shape_view.get('shape_elongation'))}"
                f" angle={_round(shape_view.get('shape_angle_deg'), 1)}"
                f" flip_conf={_round(shape_view.get('shape_flip_confidence'))}"
                + (f"   <-- OPPOSITE ENDS ({disagreement:.0f} deg from the stem)"
                   if disagreement is not None and disagreement > OPPOSITE_ENDS_DEG
                   else ""))
            if gate_bypassed:
                say(f"        note: mask_area {mask_area:,} > split gate {gate:,.0f}"
                    f" - this blob is always offered to the splitter")

            if args.masks and mask is not None:
                cv2.imwrite(str(out_dir / f"mask_{path.stem}_f{index}.png"), mask)

        if not args.no_images:
            annotated = frame.copy()
            draw_detections(
                annotated,
                result["detections"],
                primary_center=primary_center,
                flip_warning_below=float(policy.get("min_flip_confidence", 0.40)),
            )
            cv2.imwrite(str(out_dir / f"ann_{path.stem}.jpg"), annotated)
        say()

    summarise(say, rows)

    records_handle.close()
    say(f"records -> {records_path}")
    if not args.no_images:
        say(f"overlays -> {out_dir}/ann_*.jpg")
    say.close()
    return 0


def summarise(say, rows: list[dict]) -> None:
    if not rows:
        return

    say("=" * 100)
    say(f"SUMMARY - {len(rows)} fruit across "
        f"{len({r['frame'] for r in rows})} frame(s)")
    say()

    def tally(key: str) -> str:
        counts: dict = {}
        for row in rows:
            counts[row.get(key)] = counts.get(row.get(key), 0) + 1
        return "  ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=str))

    say(f"  placement:    {tally('placement')}")
    say(f"  pose:         {tally('pose')}")
    say(f"  stem_method:  {tally('stem_method')}")
    say(f"  source:       {tally('orientation_source')}")

    bypassed = [r for r in rows if r["split_gate_bypassed"]]
    if bypassed:
        say(f"  split gate:   {len(bypassed)} of {len(rows)} fruit sit above the "
            f"solidity shortcut and are always split-tested")

    # A stem localisation measured as exactly 0.0 is indistinguishable from one
    # that was never measured, because pose_detector does `stem_quality or 0.5`
    # and 0.0 is falsy. The fruit then carries a middling 0.5 into the policy,
    # which is above min_angle_confidence. Worth counting rather than assuming.
    suspicious = [
        r for r in rows
        if r["keypoint_confidence"] == 0.5 and r["stem_method"] not in ("none", "")
    ]
    if suspicious:
        say(f"  kp_conf==0.5: {len(suspicious)} fruit - exactly 0.5 is what a "
            f"stem_quality of 0.0 becomes in pose_detector; check whether the "
            f"stem was really measured as middling or as worthless")

    comparable = [r for r in rows if r["shape_vs_keypoint_deg"] is not None]
    if comparable:
        opposite = [r for r in comparable if r["shape_vs_keypoint_deg"] > OPPOSITE_ENDS_DEG]
        say()
        say(f"  Silhouette vs stem, on the {len(comparable)} fruit where both "
            f"produced a direction:")
        say(f"    {len(opposite)} disagree by more than {OPPOSITE_ENDS_DEG:.0f} deg "
            f"({100 * len(opposite) / len(comparable):.0f}%) - i.e. they picked "
            f"opposite ends of the fruit")
        say("    Does the silhouette's OWN confidence predict when it agrees?")
        for low, high in ((0.0, 0.2), (0.2, 0.4), (0.4, 1.01)):
            band = [
                r for r in comparable
                if low <= (r.get("shape_flip_confidence") or 0.0) < high
            ]
            if band:
                bad = sum(1 for r in band if r["shape_vs_keypoint_deg"] > OPPOSITE_ENDS_DEG)
                say(f"      flip_conf {low:.1f}-{high:.1f}: {len(band):>3} fruit, "
                    f"{bad} of them opposite")
        say("    If the top band is no better than the bottom one, the silhouette's "
            "confidence is not information and nothing should be gated on it.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the real detector and engine over raw frames and "
                    "report every number behind each verdict."
    )
    parser.add_argument("frames", help="image file, or folder of frames")
    parser.add_argument("--out", default="data/debug/diagnose",
                        help="output folder (default: data/debug/diagnose)")
    parser.add_argument("--config", default=None,
                        help="config file (default: the application's own)")
    parser.add_argument("--masks", action="store_true",
                        help="also write the segmentation mask per fruit")
    parser.add_argument("--no-images", action="store_true",
                        help="skip the annotated overlays, print only")
    return analyse(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
