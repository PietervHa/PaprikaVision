"""Turn the rotate-unit labels into a YOLO pose dataset: train, val and a test set.

    python -m tools.export_rotate_dataset --name rotate_v1
    python -m tools.export_rotate_dataset --name rotate_v1 --test 0.15 --val 0.15 --block-minutes 2

Reads data/rotate/labels/rotate_labels.json (made by tools.label_rotate) and writes

    data/datasets/<name>/images/{train,val,test}/   copies of the labelled frames
    data/datasets/<name>/labels/{train,val,test}/   one YOLO pose line per paprika
    data/datasets/<name>/data.yaml                  for training
    data/datasets/<name>/split.json                 which frame went where, and why

The split is by TIME BLOCK, not frame by frame: frames from the same few minutes go to the same
side. Neighbouring frames can show nearly the same scene, and one on each side would make the
test look better than it is. The test frames are never trained on.

Per frame:
    stem      box + two points: point 0 = end of the straight part of the stem, point 1 = base.
              The same order as the belt model (point 1 -> point 0 is the direction), so the
              PaprikaPose code that turns points into an angle works for both.
    no_stem   box, points marked "not visible": the model learns to give no angle there
    empty     no paprika: an image without labels
    skip      left out

An existing dataset folder is never written into: a new name is required, so stale frames
from an earlier export cannot end up in a new training set.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT / "data" / "rotate" / "labels" / "rotate_labels.json"
DATASETS = PROJECT / "data" / "datasets"
STAMP = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")


def frame_time(path: Path) -> datetime:
    """From the file name (Image_2026_09_09_07_54_08_0000), else the file's own time."""
    m = STAMP.search(path.stem)
    if m:
        try:
            return datetime(*(int(v) for v in m.groups()))
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def yolo_line(entry: dict, width: int, height: int) -> str | None:
    status = entry["status"]
    if status == "empty":
        return None
    x1, y1, x2, y2 = entry["box"]
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    bw, bh = (x2 - x1) / width, (y2 - y1) / height
    parts = [0, cx, cy, bw, bh]
    if status == "stem":
        (bx, by), (tx, ty) = entry["base"], entry["tip"]
        parts += [tx / width, ty / height, 2, bx / width, by / height, 2]
    else:                                            # no_stem
        parts += [0, 0, 0, 0, 0, 0]
    return " ".join(str(p) if isinstance(p, int) else f"{p:.6f}" for p in parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="dataset name, e.g. rotate_v1 (must be new)")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--val", type=float, default=0.15, help="share of frames for validation")
    parser.add_argument("--test", type=float, default=0.15, help="share of frames for the test set")
    parser.add_argument("--block-minutes", type=float, default=2.0,
                        help="frames this close in time stay together (a rotate unit takes a new paprika every few s)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = DATASETS / args.name
    if out.exists():
        print(f"{out} exists already - pick a new --name (an export never writes into an old one)")
        return 1
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))["frames"]

    frames = []
    missing = 0
    for key, entry in labels.items():
        if entry.get("status") not in ("stem", "no_stem", "empty"):
            continue
        path = Path(entry["root"]) / key.split("/", 1)[1]
        if not path.exists():
            missing += 1
            continue
        frames.append((frame_time(path), key, path, entry))
    if missing:
        print(f"note: {missing} labelled frames are no longer on disk and are left out")
    if not frames:
        print("nothing to export")
        return 1
    frames.sort()

    # blocks of consecutive frames, cut where the gap or the block length exceeds the limit
    blocks, current = [], [frames[0]]
    for prev, item in zip(frames, frames[1:]):
        if (item[0] - current[0][0]).total_seconds() > args.block_minutes * 60 or \
                (item[0] - prev[0]).total_seconds() > args.block_minutes * 60:
            blocks.append(current)
            current = []
        current.append(item)
    blocks.append(current)

    # whole blocks to test and val until their share is reached, the rest to train
    rng = random.Random(args.seed)
    order = list(range(len(blocks)))
    rng.shuffle(order)
    total = len(frames)
    side_of_block = {}
    counts = Counter()
    for b in order:
        n = len(blocks[b])
        if counts["test"] < args.test * total:
            side = "test"
        elif counts["val"] < args.val * total:
            side = "val"
        else:
            side = "train"
        side_of_block[b] = side
        counts[side] += n

    split = {"name": args.name, "labels": str(Path(args.labels).resolve()), "made": datetime.now().isoformat(timespec="seconds"),
             "block_minutes": args.block_minutes, "seed": args.seed,
             "keypoints": ["end of the straight part of the stem", "stem base"], "frames": {}}
    per_side = defaultdict(Counter)
    for b, block in enumerate(blocks):
        side = side_of_block[b]
        for when, key, path, entry in block:
            width, height = entry.get("size") or (None, None)
            if not width:
                import cv2
                img = cv2.imread(str(path))
                height, width = img.shape[:2]
            name = key.replace("/", "__")
            (out / "images" / side).mkdir(parents=True, exist_ok=True)
            (out / "labels" / side).mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out / "images" / side / name)
            line = yolo_line(entry, width, height)
            (out / "labels" / side / (Path(name).stem + ".txt")).write_text(
                (line + "\n") if line else "", encoding="utf-8")
            split["frames"][key] = {"side": side, "block": b, "time": when.isoformat(), "status": entry["status"],
                                    "angle": entry.get("angle")}
            per_side[side][entry["status"]] += 1

    (out / "data.yaml").write_text(
        f"# rotate-unit stem model, exported {split['made']} from {split['labels']}\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "kpt_shape: [2, 3]   # 0 = end of the straight part of the stem, 1 = stem base\n"
        "flip_idx: [0, 1]    # a mirror image keeps tip and base what they are\n"
        "names:\n  0: paprika\n", encoding="utf-8")
    (out / "split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")

    print(f"{total} frames in {len(blocks)} time blocks -> {out}")
    for side in ("train", "val", "test"):
        c = per_side[side]
        print(f"  {side:5}: {sum(c.values()):4}  (stem {c['stem']}, no_stem {c['no_stem']}, empty {c['empty']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
