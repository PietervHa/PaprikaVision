"""
Score each estimator separately against the hand labels

    python -m tools.eval_estimators data/raw data/captures
    python -m tools.eval_estimators data/raw --colour green

label_stems --score measures the pipeline: one number for whatever the engine
decided. This measures the ingredients. For every hand-labelled fruit it asks
the keypoints and the silhouette the same question independently, and scores
both against the clicked truth.

Why that is the question worth asking
-------------------------------------
fuse() currently lets the silhouette settle the FLIP and never the AXIS:
keypoints win the direction whenever they produced one. That policy was
written when the silhouette estimator was genuinely bad - it pointed at the
wrong end of the fruit on 63% of a 267-fruit sample, and was most confident
exactly when wrong.

Stripping the stem protrusion before the profile fixed that: 0 flipped of 267,
median error around 5 degrees. So the policy may now be holding back an
estimator that has become good, on precisely the fruit where the keypoints are
weakest - green, where colour cannot find a stem at all.

That is a question about accuracy, and accuracy needs ground truth. Comparing
the two estimators to EACH OTHER cannot answer it: they agree when both are
right and when both are wrong the same way, and this project has been misled by
that comparison more than once. The hand labels are the only independent
reference there is.

Read the per-colour table. If shape's median error on green is at or below the
keypoints', fuse() is costing you accuracy and should let it carry the axis
there. If it is worse, the current policy is correct and this settles it.
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
from backend.utils.paths import project_path  # noqa: E402

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}
CALYX_KEY = "stem_xy"


def load_config(explicit):
    import yaml

    if explicit:
        return yaml.safe_load(Path(explicit).read_text(encoding="utf-8")) or {}
    try:
        from backend.core.config_loader import cfg

        return cfg
    except (Exception, SystemExit):
        default = project_path("config/default.yaml")
        return yaml.safe_load(default.read_text(encoding="utf-8")) or {}


def shape_settings(cfg):
    shape = (cfg.get("paprika") or {}).get("shape") or {}
    belt = shape.get("belt_hue") or [96, 145]
    return {
        "belt_hue": (int(belt[0]), int(belt[1])),
        "saturation_floor": int(shape.get("saturation_floor", 80)),
        "value_floor": int(shape.get("value_floor", 45)),
    }


def error_deg(predicted, truth) -> float:
    return abs(((predicted - truth + 180) % 360) - 180)


def report(name: str, errors: list) -> None:
    if not errors:
        print(f"  {name:<22} no measurements")
        return
    values = np.array(errors)
    print(f"  {name:<22} n={len(values):>4}  median {np.median(values):5.1f}"
          f"  p90 {np.percentile(values, 90):6.1f}"
          f"  <10deg {100 * (values < 10).mean():3.0f}%"
          f"  <20deg {100 * (values < 20).mean():3.0f}%"
          f"  flipped {int((values > 120).sum()):>3}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score the keypoint and silhouette estimators separately "
                    "against the hand labels."
    )
    parser.add_argument("frames", nargs="*",
                        default=[str(project_path("data/raw"))])
    parser.add_argument("--labels", default=str(project_path("data/debug/labels")))
    parser.add_argument("--config", default=None)
    parser.add_argument("--colour", default=None, help="only this colour")
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = shape_settings(cfg)
    labels_path = Path(args.labels).resolve() / "stem_labels.json"
    if not labels_path.exists():
        print(f"No labels at {labels_path}")
        return 1
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    lookup: dict[str, Path] = {}
    for folder in args.frames:
        for candidate in sorted(Path(folder).resolve().rglob("*")):
            if candidate.suffix.lower() in IMAGE_SUFFIXES:
                lookup.setdefault(candidate.name, candidate)

    rows = []
    for name, entries in sorted(labels.items()):
        wanted = [e for e in entries if e.get(CALYX_KEY) and e.get("true_angle_deg") is not None]
        if not wanted or name not in lookup:
            continue
        frame = cv2.imread(str(lookup[name]))
        if frame is None:
            continue

        fruit = classical.find_fruit(frame, **settings)
        for entry in wanted:
            gx, gy = entry["centroid_xy"]
            match = None
            best = None
            for item in fruit:
                x1, y1, x2, y2 = item.bbox
                if not (x1 <= gx <= x2 and y1 <= gy <= y2):
                    continue
                distance = math.hypot(item.centroid[0] - gx, item.centroid[1] - gy)
                if best is None or distance < best:
                    match, best = item, distance
            if match is None:
                continue

            truth = float(entry["true_angle_deg"])
            colour = match.colour or entry.get("colour") or ""
            if args.colour and colour != args.colour:
                continue

            row = {"colour": colour, "stem_method": match.stem_method,
                   "keypoints": None, "shape": None, "fused": None}

            # --- keypoints alone -------------------------------------------
            kp_result = None
            if match.stem_end is not None and match.blossom_end is not None:
                x1, y1, x2, y2 = match.bbox
                diagonal = math.hypot(max(1, x2 - x1), max(1, y2 - y1))
                stem = orient.Keypoint(x=match.stem_end[0], y=match.stem_end[1],
                                       confidence=float(match.stem_quality or 0.5),
                                       visible=True)
                blossom = orient.Keypoint(x=match.blossom_end[0], y=match.blossom_end[1],
                                          confidence=float(match.stem_quality or 0.5),
                                          visible=True)
                kp_result = orient.keypoint_orientation(stem, blossom, diagonal)
                if kp_result.angle_deg is not None:
                    row["keypoints"] = error_deg(kp_result.angle_deg, truth)

            # --- silhouette alone ------------------------------------------
            mask = orient.segment_fruit(
                frame, match.bbox,
                saturation_floor=settings["saturation_floor"],
                belt_hue=settings["belt_hue"],
            )
            shape_result = orient.shape_orientation(mask) if mask is not None else None
            if shape_result is not None and shape_result.angle_deg is not None:
                row["shape"] = error_deg(shape_result.angle_deg, truth)

            # --- what fuse() would return ----------------------------------
            if kp_result is not None:
                fused = orient.fuse(kp_result, shape_result)
                if fused.angle_deg is not None:
                    row["fused"] = error_deg(fused.angle_deg, truth)

            rows.append(row)

    if not rows:
        print("Nothing scored - check the frames folders and the labels path.")
        return 1

    print(f"\n{len(rows)} labelled fruit, each estimator scored independently\n")
    print("ALL")
    for key in ("keypoints", "shape", "fused"):
        report(key, [r[key] for r in rows if r[key] is not None])

    for colour in sorted({r["colour"] for r in rows if r["colour"]}):
        subset = [r for r in rows if r["colour"] == colour]
        print(f"\n{colour.upper()}  ({len(subset)} fruit)")
        for key in ("keypoints", "shape", "fused"):
            report(key, [r[key] for r in subset if r[key] is not None])

    print("\nby how the stem was found:")
    for method in sorted({r["stem_method"] for r in rows}):
        subset = [r for r in rows if r["stem_method"] == method]
        print(f"\n  stem_method = {method}  ({len(subset)} fruit)")
        for key in ("keypoints", "shape", "fused"):
            report("    " + key, [r[key] for r in subset if r[key] is not None])

    print("\n" + "=" * 70)
    print("If shape's median beats keypoints' on a colour, fuse() is holding it")
    print("back there: it currently lets the silhouette settle only the flip,")
    print("never the axis. If shape is worse, the present policy is right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
