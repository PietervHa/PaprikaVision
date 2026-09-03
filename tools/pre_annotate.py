"""
Pre-annotation for the paprika dataset

    python -m tools.pre_annotate "data/raw" -o data/preannotated

Generates a COCO keypoints JSON (for CVAT/Label Studio) and YOLO-pose labels,
so you annotate by CORRECTING instead of drawing.

What can be automated here, and why
-----------------------------------
The blue belt is a gift. Paprikas are never blue, so fruit and belt separate on
hue alone - measured on the real set the belt sits at hue 100-125 with all fruit
outside it, and the pale edge strips drop out on saturation. That makes the
segmentation, and therefore the bounding box, effectively free and reliable.

The stem is green. On a red, orange or yellow fruit it is therefore just as
separable on hue as the fruit is from the belt, and that gives the `stem_end`
keypoint directly.

What cannot, and why that goes honestly into the review queue
-------------------------------------------------------------
On a GREEN paprika the stem coincides in colour with the fruit, so the colour
route fails by definition. Morphology covers most of those, but not all.

And the shape estimator in orientation.py - the one that would recognise the
stem end by its wider shoulder - does not work on this cultivar. Measured on 87
fruit with a visible stem: the stem end is narrower in 66% of cases, with a
median signal strength of 0.074 and an elongation of just 1.33. A blocky paprika
is simply too round and too symmetric for it. 66% is barely better than tossing
a coin, so that estimator is not used here to guess a keypoint. A guess that
lands in the dataset as an annotation is worse than an empty field, because
nobody checks it again.

Every fruit therefore gets a status, and the review queue tells you exactly
where your eyes are needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detection.paprika import classical

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}

STATUS_AUTO = "auto"
STATUS_STANDING = "auto_standing"
STATUS_MORPH = "auto_via_shape"
STATUS_NO_STEM = "manual_no_stem"
STATUS_TOUCHING = "check_touching"


def analyse_image(path: Path):
    """Analyse one image with exactly the same detection as the application."""
    image = cv2.imread(str(path))
    if image is None:
        return [], None

    fruits = classical.find_fruit(image, max_fruit=12)

    results = []
    for fruit in fruits:
        record = {
            "bbox": fruit.bbox,
            "area": fruit.area,
            "colour": fruit.colour,
            "hue": fruit.hue,
            "centroid": list(fruit.centroid),
            "stem_end": list(fruit.stem_end) if fruit.stem_end else None,
            "blossom_end": list(fruit.blossom_end) if fruit.blossom_end else None,
            "stem_visible": fruit.stem_end is not None,
            "stem_method": fruit.stem_method,
            "standing": fruit.standing,
        }

        if fruit.stem_end is None:
            record["status"] = STATUS_NO_STEM
        elif fruit.standing:
            record["status"] = STATUS_STANDING
        elif fruit.stem_method == "morphology":
            # Morphology reached 86% on green fruit and agreed with the colour
            # method to within 13 px median on non-green fruit. Reliable, but
            # more indirect than colour, so labelled separately - check these on
            # a sample basis rather than trusting them blindly.
            record["status"] = STATUS_MORPH
        else:
            record["status"] = STATUS_AUTO

        results.append(record)

    if len(results) > 1:
        for record in results:
            ax1, ay1, ax2, ay2 = record["bbox"]
            for other in results:
                if other is record:
                    continue
                bx1, by1, bx2, by2 = other["bbox"]
                if ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2:
                    record["status"] = STATUS_TOUCHING
                    break

    return results, image


# ------------------------------------------------------------------ export


def visibility_flag(record: dict, which: str) -> int:
    """COCO: 2 visible, 1 labelled but occluded, 0 not labelled."""
    if record[which] is None:
        return 0
    if record.get("standing"):
        # Exactly the case the spec turns on: on a standing fruit one end is
        # always hidden, but its position is known.
        return 2 if which == "stem_end" else 1
    return 2


def write_coco(records: dict, output: Path, sizes: dict) -> None:
    images, annotations = [], []
    annotation_id = 1

    for index, (name, fruits) in enumerate(sorted(records.items()), start=1):
        width, height = sizes[name]
        images.append({"id": index, "file_name": name, "width": width, "height": height})

        for fruit in fruits:
            x1, y1, x2, y2 = fruit["bbox"]
            keypoints = []
            for which in ("stem_end", "blossom_end"):
                point = fruit[which]
                flag = visibility_flag(fruit, which)
                keypoints += [0, 0, 0] if point is None else [round(point[0], 1), round(point[1], 1), flag]

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": index,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": fruit["area"],
                    "iscrowd": 0,
                    "num_keypoints": sum(1 for i in (2, 5) if keypoints[i] > 0),
                    "keypoints": keypoints,
                    # Non-standard fields, but CVAT and Label Studio preserve
                    # them - so the file itself keeps a record of what was
                    # automatic and what the annotator still has to do.
                    "attributes": {"status": fruit["status"], "colour": fruit["colour"]},
                }
            )
            annotation_id += 1

    coco = {
        "info": {"description": "Paprika pre-annotation - CORRECT these, do not trust blindly"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "paprika",
                "keypoints": ["stem_end", "blossom_end"],
                "skeleton": [[1, 2]],
            }
        ],
    }
    output.write_text(json.dumps(coco, indent=1))


def write_yolo(records: dict, output_dir: Path, sizes: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fruits in records.items():
        width, height = sizes[name]
        lines = []
        for fruit in fruits:
            x1, y1, x2, y2 = fruit["bbox"]
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
            for which in ("stem_end", "blossom_end"):
                point = fruit[which]
                flag = visibility_flag(fruit, which)
                if point is None:
                    parts.append("0.000000 0.000000 0")
                else:
                    parts.append(f"{point[0]/width:.6f} {point[1]/height:.6f} {flag}")
            lines.append(" ".join(parts))
        (output_dir / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")


def write_review_queue(records: dict, output: Path) -> None:
    """Sorted by how much work it is, so you start at the top."""
    priority = {
        STATUS_NO_STEM: 0,
        STATUS_TOUCHING: 1,
        STATUS_STANDING: 2,
        STATUS_MORPH: 3,
        STATUS_AUTO: 4,
    }
    rows = []
    for name, fruits in records.items():
        for index, fruit in enumerate(fruits):
            rows.append(
                {
                    "file": name,
                    "fruit": index,
                    "status": fruit["status"],
                    "colour": fruit["colour"],
                    "stem_found": int(fruit["stem_visible"]),
                    "method": fruit["stem_method"],
                    "action": {
                        STATUS_NO_STEM: "stemless or stem under the fruit - place the calyx yourself",
                        STATUS_TOUCHING: "fruit are touching - split the boxes",
                        STATUS_STANDING: "check whether it really is standing upright",
                        STATUS_MORPH: "stem found via shape - check on a sample basis",
                        STATUS_AUTO: "review only",
                    }[fruit["status"]],
                }
            )
    rows.sort(key=lambda r: (priority[r["status"]], r["file"]))

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_previews(records: dict, source: Path, output_dir: Path, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colours = {
        STATUS_AUTO: (90, 190, 70),
        STATUS_MORPH: (140, 200, 90),
        STATUS_STANDING: (235, 180, 60),
        STATUS_NO_STEM: (60, 60, 220),
        STATUS_TOUCHING: (60, 180, 235),
    }
    for name, fruits in list(records.items())[:limit]:
        image = cv2.imread(str(source / name))
        if image is None:
            continue
        for fruit in fruits:
            x1, y1, x2, y2 = fruit["bbox"]
            colour = colours[fruit["status"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
            cv2.putText(image, fruit["status"], (x1, max(28, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
            if fruit["stem_end"] and fruit["blossom_end"]:
                stem = tuple(int(v) for v in fruit["stem_end"])
                blossom = tuple(int(v) for v in fruit["blossom_end"])
                cv2.arrowedLine(image, blossom, stem, colour, 4, tipLength=0.2)
                cv2.circle(image, stem, 9, (0, 255, 0), -1)
                cv2.circle(image, blossom, 9, (220, 140, 60), -1)
        cv2.imwrite(str(output_dir / (Path(name).stem + ".jpg")), image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="folder of raw images")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/preannotated"))
    parser.add_argument("--previews", type=int, default=25, help="number of check images")
    args = parser.parse_args()

    paths = sorted(p for p in args.source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        print(f"No images found in {args.source}")
        return 1

    print(f"\n{len(paths)} images found in {args.source}\n")

    records, sizes = {}, {}
    statuses, colours = Counter(), Counter()

    for path in paths:
        fruits, image = analyse_image(path)
        if image is None:
            continue
        name = path.name
        records[name] = fruits
        sizes[name] = (image.shape[1], image.shape[0])
        for fruit in fruits:
            statuses[fruit["status"]] += 1
            colours[fruit["colour"]] += 1

    total = sum(statuses.values())
    args.output.mkdir(parents=True, exist_ok=True)

    write_coco(records, args.output / "preannotation_coco.json", sizes)
    write_yolo(records, args.output / "labels", sizes)
    write_review_queue(records, args.output / "review_queue.csv")
    write_previews(records, args.source, args.output / "previews", args.previews)

    print(f"{total} fruit found in {len(records)} images")
    print(f"colours: {dict(colours)}\n")

    print("status                    count    share")
    print("-" * 44)
    labels = {
        STATUS_AUTO: "auto, stem via colour",
        STATUS_MORPH: "auto, stem via shape",
        STATUS_STANDING: "auto, standing upright",
        STATUS_TOUCHING: "touching, needs checking",
        STATUS_NO_STEM: "no stem: by hand",
    }
    for status, label in labels.items():
        count = statuses.get(status, 0)
        print(f"{label:<26}{count:>6}{count/max(1,total)*100:>9.0f}%")

    auto = (statuses.get(STATUS_AUTO, 0) + statuses.get(STATUS_STANDING, 0)
            + statuses.get(STATUS_MORPH, 0))
    print("-" * 44)
    print(f"{'keypoints pre-filled':<26}{auto:>6}{auto/max(1,total)*100:>9.0f}%")
    print(f"{'bounding boxes':<26}{total:>6}{100:>9.0f}%")

    print(f"\nWritten to {args.output}/")
    print("  preannotation_coco.json  import this into CVAT or Label Studio")
    print("  labels/                  YOLO-pose, for training directly")
    print("  review_queue.csv         sorted: what you must do yourself is at the top")
    print("  previews/                check these with your own eyes first")
    print("\nThe boxes are reliable. The keypoints are a PROPOSAL - go through")
    print("them. A mistake that slips into the dataset as an annotation is never")
    print("found again, because the model simply learns it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
