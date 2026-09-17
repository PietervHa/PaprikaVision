"""
Build a YOLO-pose training set from hand labels

    python -m tools.export_dataset data/raw
    python -m tools.export_dataset data/raw --out data/datasets/paprika --val 0.2

Merges the landmarks clicked with tools/label_stems.py over the automatic
guesses from tools/pre_annotate.py, and writes a dataset ultralytics can train
on directly.

The merge rule, and why it is this way round
--------------------------------------------
A hand label always wins. The pre-annotator is honest about what it can and
cannot do - it finds the stem by colour, which works on red, orange and yellow
fruit and fails by definition on green, where stem and flesh share a hue. Those
are exactly the fruit that end up hand-labelled, so the two sources are
complementary rather than competing, and where they disagree the human is right
by construction.

Fruit with no landmark from either source are written with visibility 0 rather
than dropped. A fruit whose calyx genuinely cannot be seen is a real case the
model has to handle - that is the whole argument in ANNOTATION_SPEC.md for
keypoints over a separate stem class - and deleting those examples would teach
the model that every fruit has a visible stem.

Matching is by position. A hand label carries the bbox it was clicked in, and
is attached to whichever pre-annotated fruit overlaps it most. Nothing here
relies on detection ORDER, which changes whenever the detector is tuned.

Splitting is by FRAME, never by fruit
-------------------------------------
Two fruit from the same frame share lighting, belt, focus and often the same
physical paprika photographed a moment apart. Splitting by fruit would put near
duplicates on both sides of the train/val line and report a validation score
that flatters the model. Frames are assigned by a hash of the filename, so the
split is stable: re-running after adding more labels moves nobody across.

    <out>/train/images, train/labels    copies of the source frames + labels
    <out>/val/images,   val/labels
    <out>/data.yaml                     ready for: yolo pose train data=...

The split-first layout is the one tools/dataset_check.py validates. Ultralytics
accepts either arrangement, so the tie-breaker is that the dataset can be
checked before a training run rather than after one.

SOURCE FRAMES ARE NEVER MODIFIED - they are read and copied, never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detection.paprika import classical  # noqa: E402
from backend.utils.paths import project_path  # noqa: E402

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}
CALYX_KEY = "stem_xy"
BLOSSOM_KEY = "blossom_xy"

# COCO/YOLO visibility. 2 = visible and labelled, 1 = labelled but occluded,
# 0 = not labelled. Kept identical to tools/pre_annotate.py so a merged row is
# indistinguishable from an automatic one.
VIS_VISIBLE, VIS_OCCLUDED, VIS_ABSENT = 2, 1, 0


def load_config(explicit: str | None) -> dict:
    import yaml

    if explicit:
        return yaml.safe_load(Path(explicit).read_text(encoding="utf-8")) or {}
    try:
        from backend.core.config_loader import cfg

        return cfg
    except (Exception, SystemExit):
        default = project_path("config/default.yaml")
        if not default.exists():
            raise SystemExit(f"No config found - looked for {default}")
        return yaml.safe_load(default.read_text(encoding="utf-8")) or {}


def shape_settings(cfg: dict) -> dict:
    shape = (cfg.get("paprika") or {}).get("shape") or {}
    belt = shape.get("belt_hue") or [96, 145]
    return {
        "belt_hue": (int(belt[0]), int(belt[1])),
        "saturation_floor": int(shape.get("saturation_floor", 80)),
        "value_floor": int(shape.get("value_floor", 45)),
    }


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(1, union)


def is_val(name: str, fraction: float) -> bool:
    """Stable per-frame assignment. The same frame always lands on the same
    side, so adding labels later never reshuffles the split."""
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10_000) / 10_000.0 < fraction


def row(bbox, landmarks, width: int, height: int) -> str:
    """One YOLO-pose row: class, box, then each landmark as x y visibility."""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for point, visibility in landmarks:
        if point is None:
            parts.append(f"0.000000 0.000000 {VIS_ABSENT}")
        else:
            px = min(max(point[0] / width, 0.0), 1.0)
            py = min(max(point[1] / height, 0.0), 1.0)
            parts.append(f"{px:.6f} {py:.6f} {int(visibility)}")
    return " ".join(parts)


def export(args) -> int:
    cfg = load_config(args.config)
    settings = shape_settings(cfg)

    source = Path(args.frames).resolve()
    out_dir = Path(args.out).resolve()
    if not source.exists():
        print(f"Source folder does not exist: {source}")
        return 1
    if out_dir == source or source in out_dir.parents:
        print(f"Refusing to write inside the source folder.\n"
              f"  source: {source}\n  out:    {out_dir}")
        return 1

    labels_path = Path(args.labels).resolve() / "stem_labels.json"
    if not labels_path.exists():
        print(f"No hand labels at {labels_path}.\n"
              f"Run: python -m tools.label_stems {args.frames}")
        return 1
    hand = json.loads(labels_path.read_text(encoding="utf-8"))

    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    written = collections_counter = {"train": 0, "val": 0}
    fruit_total = both = calyx_only = neither = 0
    skipped_no_label = 0

    frames = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    for path in frames:
        entries = hand.get(path.name)
        if not entries:
            skipped_no_label += 1
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  ! could not read {path.name}")
            continue
        height, width = frame.shape[:2]

        detected = [f for f in classical.find_fruit(frame, **settings)
                    if not f.edge_clipped]

        lines = []
        used: set[int] = set()
        for entry in entries:
            box = entry.get("bbox")
            if not box:
                continue
            calyx = entry.get(CALYX_KEY)
            blossom = entry.get(BLOSSOM_KEY)
            if calyx is None and blossom is None and not args.include_unlabelled:
                # Nothing was clicked here at all - an operator skip, not a
                # statement that the fruit has no landmarks.
                continue
            # Attach to the detected fruit it overlaps most, so the bbox in the
            # dataset is the detector's current one rather than a stale copy.
            best, best_iou, best_index = None, 0.0, -1
            for index, item in enumerate(detected):
                overlap = iou(box, item.bbox)
                if overlap > best_iou:
                    best, best_iou, best_index = item, overlap, index
            bbox = tuple(best.bbox) if best is not None and best_iou >= 0.5 else tuple(box)
            if best_index >= 0:
                used.add(best_index)
            # Visibility comes from which mouse button the annotator used.
            # Defaulting to 2 keeps labels taken before the occluded option
            # existed readable, and dataset_check will say so if a set has no
            # occluded landmarks at all.
            landmarks = (
                (calyx, entry.get("stem_vis", VIS_VISIBLE)),
                (blossom, entry.get("blossom_vis", VIS_VISIBLE)),
            )
            lines.append(row(bbox, landmarks, width, height))
            fruit_total += 1
            if calyx and blossom:
                both += 1
            elif calyx:
                calyx_only += 1
            else:
                neither += 1

        if not lines:
            continue

        split = "val" if is_val(path.name, args.val) else "train"
        shutil.copy2(path, out_dir / split / "images" / path.name)
        (out_dir / split / "labels" / (path.stem + ".txt")).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        collections_counter[split] += 1

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "# Written by tools/export_dataset.py\n"
        "#\n"
        "# Landmark order is fixed by docs/ANNOTATION_SPEC.md and baked into\n"
        "# orientation.KEYPOINT_NAMES, the dataset validator and the trained\n"
        "# model. Changing it means re-exporting and retraining.\n"
        f"path: {out_dir.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "kpt_shape: [2, 3]\n"
        "flip_idx: [0, 1]\n"
        "names:\n"
        "  0: paprika\n",
        encoding="utf-8",
    )

    print(f"{len(frames)} frame(s) under {source} (unchanged)")
    if skipped_no_label:
        print(f"{skipped_no_label} had no hand labels and were skipped")
    print(f"\nwritten: {collections_counter['train']} train, "
          f"{collections_counter['val']} val frames")
    print(f"{fruit_total} fruit:")
    print(f"   {both:>4} with both landmarks   <- trainable for orientation")
    print(f"   {calyx_only:>4} calyx only         <- blossom marked absent")
    print(f"   {neither:>4} blossom only")
    print(f"\n-> {out_dir}")
    if calyx_only:
        print(f"\n{calyx_only} fruit still need a blossom. They are usable as they "
              f"stand -\na missing landmark is recorded as not-visible, not "
              f"invented - but the\nmodel learns the flip from pairs, so top them up first:")
        print("   python -m tools.label_stems data/raw --blossom")
    print("\nCheck it, then train:")
    print(f"   python -m tools.dataset_check {out_dir}")
    print(f"   yolo pose train data={data_yaml.as_posix()} model=yolo11n-pose.pt epochs=100")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge hand labels over pre-annotation into a YOLO-pose "
                    "dataset. Source frames are only ever read and copied."
    )
    parser.add_argument("frames", nargs="?", default=str(project_path("data/raw")))
    parser.add_argument("--labels", default=str(project_path("data/debug/labels")),
                        help="folder holding stem_labels.json")
    parser.add_argument("--out", default=str(project_path("data/datasets/paprika")))
    parser.add_argument("--config", default=None)
    parser.add_argument("--val", type=float, default=0.2,
                        help="fraction of FRAMES held out for validation")
    parser.add_argument("--include-unlabelled", action="store_true",
                        help="also export fruit where nothing was clicked, with "
                             "both landmarks marked absent")
    return export(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
