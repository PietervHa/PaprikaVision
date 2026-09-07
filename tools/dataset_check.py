"""
Dataset check

Validates a YOLO-pose paprika export before you spend a training run on it.

    python -m tools.dataset_check data/datasets/paprika

Every check here exists because the failure it catches is invisible until after
training, when you are left staring at a mediocre metric with no idea which of
a dozen things caused it.

The important one is the keypoint-order check. If `stem_end` and `blossom_end`
got swapped - on some images, or on all of them, by a mis-set tool or a
tired afternoon - the dataset is perfectly self-consistent and trains happily
to a good-looking loss. The model then confidently points every arrow at the
wrong end of the fruit. Nothing except a geometric sanity check against the
fruit's own shape will tell you.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# Targets from docs/ANNOTATION_SPEC.md section 6.
TARGET_STEMLESS = 0.20
TARGET_STANDING = 0.15

# Fallback only, used when the config cannot be read - see
# _standing_span_ratio(). Keep it equal to the shipped
# paprika.min_span_ratio.
DEFAULT_STANDING_SPAN_RATIO = 0.18


def _standing_span_ratio() -> float:
    """The span ratio below which a labelled fruit counts as standing.

    Read from paprika.min_span_ratio rather than pinned here, because this is
    the same test the runtime applies - and the whole value of the number this
    tool prints is that it predicts what the running system will do with the
    dataset. Two copies of the threshold means that the day somebody tunes the
    runtime, this validator quietly starts reporting a standing rate for a
    machine that no longer exists.

    Falls back to the documented default when the config is unreadable: this
    is a dataset validator and must still run on a machine that has the export
    but not the application config.
    """
    try:
        from backend.core.config_loader import cfg

        block = cfg.get("paprika") or {}
        return float(block.get("min_span_ratio", DEFAULT_STANDING_SPAN_RATIO))
    except (Exception, SystemExit) as exc:
        # SystemExit is deliberate: config_loader calls sys.exit() when the
        # config file is missing or unparseable, and a dataset validator must
        # not be killed by that.
        print(
            f"  ! could not read paprika.min_span_ratio ({exc}); using "
            f"{DEFAULT_STANDING_SPAN_RATIO} for the standing check"
        )
        return DEFAULT_STANDING_SPAN_RATIO


class Findings:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def report(self) -> int:
        for message in self.notes:
            print(f"  {message}")
        if self.warnings:
            print("\nWARNINGS")
            for message in self.warnings:
                print(f"  ! {message}")
        if self.errors:
            print("\nERRORS")
            for message in self.errors:
                print(f"  X {message}")
        print()
        if self.errors:
            print("RESULT: not ready to train - fix the errors above.")
            return 1
        if self.warnings:
            print("RESULT: trainable, but read the warnings first.")
            return 0
        print("RESULT: looks good.")
        return 0


def parse_label_file(path: Path) -> list[dict]:
    """Parse one YOLO-pose label file.

    Expected per line, all normalised 0-1:
        class cx cy w h  kx1 ky1 v1  kx2 ky2 v2
    """
    instances = []
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            instances.append({"malformed": f"{path.name}:{line_no} only {len(parts)} fields"})
            continue

        try:
            values = [float(p) for p in parts]
        except ValueError:
            instances.append({"malformed": f"{path.name}:{line_no} non-numeric field"})
            continue

        cls = int(values[0])
        box = values[1:5]
        kp_values = values[5:]

        # Accept both (x, y, v) and (x, y); the former is what we want.
        if len(kp_values) % 3 == 0:
            stride, has_vis = 3, True
        elif len(kp_values) % 2 == 0:
            stride, has_vis = 2, False
        else:
            instances.append({"malformed": f"{path.name}:{line_no} odd keypoint field count"})
            continue

        keypoints = []
        for i in range(0, len(kp_values), stride):
            keypoints.append(
                {
                    "x": kp_values[i],
                    "y": kp_values[i + 1],
                    "v": int(kp_values[i + 2]) if has_vis else None,
                }
            )

        instances.append(
            {"cls": cls, "box": box, "keypoints": keypoints, "has_vis": has_vis,
             "source": f"{path.name}:{line_no}"}
        )
    return instances


def check_split(split_dir: Path, findings: Findings, split_name: str) -> dict:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        findings.error(f"{split_name}: expected images/ and labels/ under {split_dir}")
        return {}

    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        findings.error(f"{split_name}: no images found")
        return {}

    pending_order: list = []
    standing_span_ratio = _standing_span_ratio()

    stats = {
        "images": len(images),
        "instances": 0,
        "missing_labels": 0,
        "empty_labels": 0,
        "classes": Counter(),
        "kp_counts": Counter(),
        "vis_flags": Counter(),
        "no_vis_field": 0,
        "angles": [],
        "standing": 0,
        "stem_occluded": 0,
        "malformed": [],
        "pending_order": pending_order,
    }

    for image_path in images:
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            stats["missing_labels"] += 1
            continue

        instances = parse_label_file(label_path)
        if not instances:
            stats["empty_labels"] += 1
            continue

        for inst in instances:
            if "malformed" in inst:
                stats["malformed"].append(inst["malformed"])
                continue

            stats["instances"] += 1
            stats["classes"][inst["cls"]] += 1
            stats["kp_counts"][len(inst["keypoints"])] += 1

            if not inst["has_vis"]:
                stats["no_vis_field"] += 1
                continue

            keypoints = inst["keypoints"]
            if len(keypoints) < 2:
                continue

            stem, blossom = keypoints[0], keypoints[1]
            for kp in (stem, blossom):
                stats["vis_flags"][kp["v"]] += 1

            if stem["v"] == 1:
                stats["stem_occluded"] += 1

            if stem["v"] == 0 or blossom["v"] == 0:
                continue

            # Normalised coordinates: scale by the box so the span ratio is
            # comparable to the runtime min_span_ratio test.
            bw, bh = inst["box"][2], inst["box"][3]
            dx = (stem["x"] - blossom["x"])
            dy = (stem["y"] - blossom["y"])
            span = math.hypot(dx, dy)
            diagonal = math.hypot(bw, bh)

            if diagonal > 0 and span / diagonal < standing_span_ratio:
                stats["standing"] += 1
                continue

            angle = math.degrees(math.atan2(-dy, dx)) % 360.0
            stats["angles"].append(angle)

            # Keypoint-order check happens later, against the actual image
            # pixels - see check_keypoint_order(). It cannot be done here:
            # a box plus two points carries no shape information at all (the
            # box is axis-aligned and symmetric), so nothing in the label file
            # alone can distinguish the stem end from the blossom end.
            pending_order.append((image_path, inst, angle))

    return stats


def check_keypoint_order(stats: dict, findings: Findings, split_name: str, sample: int = 200) -> None:
    """Detect swapped landmarks by cross-checking against fruit silhouette.

    This is the check worth running before anything else. If `stem_end` and
    `blossom_end` were swapped - by a mis-configured tool, or one tired
    afternoon halfway through the set - the dataset stays perfectly
    self-consistent and trains to a healthy-looking loss. The resulting model
    then points every arrow at the wrong end of the fruit, and no training
    metric will ever tell you.

    The only thing that can tell you is the fruit's own shape, so this reuses
    the exact estimator the runtime uses: segment the fruit, measure which end
    carries the shoulder, and compare that against the direction the labels
    claim. Instances whose silhouette is too symmetric to call are skipped
    rather than counted as votes - they genuinely carry no information.
    """
    pending = stats.get("pending_order") or []
    if not pending:
        return

    try:
        import cv2  # noqa: F401
        from backend.detection.paprika import orientation as orient
    except Exception as exc:
        findings.warn(
            f"{split_name}: skipped the keypoint-order check ({exc}). This is "
            "the check that catches swapped landmarks - worth resolving."
        )
        return

    import cv2
    import random

    if len(pending) > sample:
        random.seed(0)  # deterministic, so re-runs are comparable
        pending = random.sample(pending, sample)

    agree = disagree = skipped = unreadable = 0

    for image_path, inst, label_angle in pending:
        image = cv2.imread(str(image_path))
        if image is None:
            unreadable += 1
            continue

        h, w = image.shape[:2]
        cx, cy, bw, bh = inst["box"]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        mask = orient.segment_fruit(image, (x1, y1, x2, y2))
        if mask is None:
            skipped += 1
            continue

        shape_result = orient.shape_orientation(mask)
        if shape_result.angle_deg is None or shape_result.flip_confidence < 0.35:
            # Too round, or too symmetric to say which end is the shoulder.
            skipped += 1
            continue

        difference = orient.angular_difference(shape_result.angle_deg, label_angle)
        if difference > 120.0:
            disagree += 1
        elif difference < 60.0:
            agree += 1
        else:
            skipped += 1

    total = agree + disagree
    if unreadable and unreadable == len(pending):
        findings.warn(
            f"{split_name}: could not read any image for the keypoint-order "
            "check - are the image files actually present?"
        )
        return

    if total < 15:
        findings.warn(
            f"{split_name}: only {total} fruit had a silhouette clear enough to "
            f"verify keypoint order ({skipped} skipped). Not enough to "
            "conclude anything - check a handful by eye instead."
        )
        return

    agree_rate = agree / total
    if agree_rate < 0.30:
        findings.error(
            f"{split_name}: on {(1 - agree_rate) * 100:.0f}% of {total} checked "
            "fruit, the labelled stem direction points at the NARROW end of the "
            "silhouette. stem_end and blossom_end are almost certainly swapped "
            "(stem_end must be index 0). Fix this before training - the model "
            "will otherwise learn to point every arrow backwards, and no loss "
            "curve will show it."
        )
    elif agree_rate < 0.65:
        findings.warn(
            f"{split_name}: labelled stem direction matches the silhouette on "
            f"only {agree_rate * 100:.0f}% of {total} checked fruit. Expect 80%+. "
            "Some individual images likely have the landmarks reversed."
        )
    else:
        findings.note(
            f"{split_name}: keypoint order verified against silhouette on "
            f"{total} fruit ({agree_rate * 100:.0f}% agree, {skipped} too "
            "symmetric to judge)"
        )


def evaluate(stats: dict, findings: Findings, split_name: str) -> None:
    if not stats:
        return

    total = stats["instances"]
    findings.note(
        f"{split_name}: {stats['images']} images, {total} annotated fruit"
    )

    if stats["missing_labels"]:
        findings.warn(
            f"{split_name}: {stats['missing_labels']} images have no label file "
            "(fine only if they are deliberate empty negatives)"
        )
    if stats["malformed"]:
        findings.error(
            f"{split_name}: {len(stats['malformed'])} malformed label lines, "
            f"first: {stats['malformed'][0]}"
        )

    # --- classes ---
    if len(stats["classes"]) > 1:
        findings.error(
            f"{split_name}: {len(stats['classes'])} classes present {dict(stats['classes'])}; "
            "the spec calls for exactly one ('paprika'). Colour classes will "
            "cut your data per class and teach colour instead of shape."
        )

    # --- keypoint count ---
    counts = dict(stats["kp_counts"])
    if counts and set(counts) != {2}:
        findings.error(
            f"{split_name}: keypoints per fruit is {counts}, expected exactly 2. "
            "Check kpt_shape: [2, 3] in the data yaml."
        )

    # --- visibility flags ---
    if stats["no_vis_field"]:
        findings.error(
            f"{split_name}: {stats['no_vis_field']} instances have no visibility "
            "field. Export with kpt_shape [2, 3], not [2, 2] - without the flag "
            "the engine cannot tell stem-up from stem-down on a standing fruit."
        )

    flags = stats["vis_flags"]
    if flags and flags.get(1, 0) == 0:
        findings.error(
            f"{split_name}: no landmark anywhere is flagged 1 (occluded). This "
            "almost always means occluded landmarks were skipped rather than "
            "positioned. Standing and stem-away fruit will not work - see "
            "ANNOTATION_SPEC section 3."
        )
    elif flags:
        findings.note(
            f"{split_name}: visibility flags "
            f"visible={flags.get(2,0)} occluded={flags.get(1,0)} unlabelled={flags.get(0,0)}"
        )

    # --- keypoint order, against the real pixels ---
    check_keypoint_order(stats, findings, split_name)

    if total == 0:
        return

    # --- balance targets ---
    standing_rate = stats["standing"] / total
    stemless_rate = stats["stem_occluded"] / total
    findings.note(
        f"{split_name}: standing {standing_rate*100:.1f}% "
        f"(target >={TARGET_STANDING*100:.0f}%), "
        f"stem occluded/absent {stemless_rate*100:.1f}% "
        f"(target >={TARGET_STEMLESS*100:.0f}%)"
    )
    if standing_rate < TARGET_STANDING * 0.6:
        findings.warn(
            f"{split_name}: only {standing_rate*100:.1f}% standing fruit. The "
            "model will treat the pose as noise. Feed more through deliberately."
        )
    if stemless_rate < TARGET_STEMLESS * 0.6:
        findings.warn(
            f"{split_name}: only {stemless_rate*100:.1f}% stemless/occluded-stem "
            "fruit. This is the case you specifically need to handle."
        )

    # --- angle coverage ---
    angles = stats["angles"]
    if angles:
        buckets = Counter(int(a // 30) for a in angles)
        empty = [b for b in range(12) if buckets.get(b, 0) == 0]
        thin = [b for b in range(12) if 0 < buckets.get(b, 0) < len(angles) / 40]
        histogram = " ".join(
            f"{b*30:3d}:{buckets.get(b,0):<4d}" for b in range(12)
        )
        findings.note(f"{split_name}: angle histogram (30 deg buckets)\n     {histogram}")
        if empty:
            findings.warn(
                f"{split_name}: no fruit at all in buckets "
                f"{[f'{b*30}-{b*30+30}' for b in empty]}. The model will be blind "
                "at those angles and your val split, drawn from the same "
                "distribution, will not show it. Hand-rotate fruit to fill them."
            )
        elif thin:
            findings.warn(
                f"{split_name}: thin coverage in buckets "
                f"{[f'{b*30}-{b*30+30}' for b in thin]}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="dataset root (contains train/ and val/)")
    args = parser.parse_args()

    root: Path = args.dataset
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    print(f"\nChecking {root}\n")
    findings = Findings()

    found_any = False
    for split in ("train", "val", "valid", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        found_any = True
        evaluate(check_split(split_dir, findings, split), findings, split)

    if not found_any:
        # Flat layout: images/ and labels/ directly under the root.
        if (root / "images").is_dir():
            evaluate(check_split(root, findings, "dataset"), findings, "dataset")
        else:
            findings.error(
                "No train/ or val/ split and no images/ directory found. "
                "Expected a YOLO layout."
            )

    return findings.report()


if __name__ == "__main__":
    sys.exit(main())
