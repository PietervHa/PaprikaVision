"""
Export stemless fruit crops for labelling

    python -m tools.export_stemless
    python -m tools.export_stemless data/raw --out data/debug/labelling
    python -m tools.export_stemless --include-stemmed --limit 400

Copies a cropped picture of every fruit the detector could not find a stem on
into two empty folders for you to sort by eye:

    <out>/unsorted/     everything, waiting to be sorted
    <out>/end_on/       drag here: looking down the fruit's axis (blossom scar
                        or calyx facing the camera - standing or upside down)
    <out>/side_on/      drag here: fruit lying on its side, stem simply hidden,
                        facing away or broken off

Why this exists: three different silhouette measures have now separated a
hand-picked set of eight frames perfectly and then misclassified more than half
of the 239 working fruit in the same dataset. Eight examples is not enough to
test a measure - it is only enough to fool one. Sorting a hundred or so crops
turns "this looks like it works" into a number, and the same labels feed the
pose model later, so the effort is not spent twice.

    <out>/index.csv     crop -> source frame, bbox, and the measurements taken
                        at export time, so a label can always be traced back to
                        the frame it came from

SOURCE FRAMES ARE NEVER TOUCHED. This reads them and writes copies elsewhere;
it does not move, rename, modify or delete anything in the source folder, and
it refuses to run if the output would land inside it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detection.paprika import classical  # noqa: E402
from backend.utils.paths import project_path  # noqa: E402

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}

# Breathing room around the bbox. The lobes run right to the silhouette edge,
# and a crop that clips them makes the very thing you are judging harder to
# see. Fraction of the box's own size.
CROP_MARGIN = 0.08


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


def crop_with_margin(frame: np.ndarray, bbox) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * CROP_MARGIN)
    pad_y = int((y2 - y1) * CROP_MARGIN)
    return frame[
        max(0, y1 - pad_y): min(height, y2 + pad_y),
        max(0, x1 - pad_x): min(width, x2 + pad_x),
    ]


def export(args) -> int:
    cfg = load_config(args.config)
    shape = (cfg.get("paprika") or {}).get("shape") or {}
    belt = shape.get("belt_hue") or [96, 145]
    belt_hue = (int(belt[0]), int(belt[1]))
    saturation_floor = int(shape.get("saturation_floor", 80))
    value_floor = int(shape.get("value_floor", 45))

    source = Path(args.frames).resolve()
    out_dir = Path(args.out).resolve()
    if not source.exists():
        print(f"Source folder does not exist: {source}")
        return 1

    # The source frames are the only irreplaceable thing here. Writing anywhere
    # inside them - even a subfolder - risks a later --out or a cleanup step
    # taking originals with it, so refuse outright rather than trust the caller.
    if out_dir == source or source in out_dir.parents:
        print(f"Refusing to write inside the source folder.\n"
              f"  source: {source}\n  out:    {out_dir}\n"
              f"Choose an --out somewhere else; originals are never modified.")
        return 1

    unsorted_dir = out_dir / "unsorted"
    for folder in (unsorted_dir, out_dir / "end_on", out_dir / "side_on"):
        folder.mkdir(parents=True, exist_ok=True)

    frames = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not frames:
        print(f"No images found under {source}")
        return 1

    rows = []
    stemless = stemmed = 0
    for path in frames:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  ! could not read {path.name}")
            continue

        for index, fruit in enumerate(
            classical.find_fruit(
                frame,
                belt_hue=belt_hue,
                saturation_floor=saturation_floor,
                value_floor=value_floor,
            )
        ):
            has_stem = fruit.stem_method not in ("none", "", None)
            # Edge-clipped fruit are excluded: half a paprika cannot be judged
            # end-on or side-on by eye either, so a label on one would be a
            # guess entering the ground truth.
            if fruit.edge_clipped:
                continue
            if has_stem and not args.include_stemmed:
                continue

            crop = crop_with_margin(frame, fruit.bbox)
            if crop.size == 0:
                continue

            name = f"{path.stem}_f{index}{'_stem' if has_stem else ''}.png"
            cv2.imwrite(str(unsorted_dir / name), crop)
            rows.append({
                "crop": name,
                "source_frame": path.name,
                "bbox": " ".join(str(v) for v in fruit.bbox),
                "area_px": fruit.area,
                "stem_method": fruit.stem_method,
                "elongation": round(classical._elongation(fruit.mask), 4),
                "solidity": round(classical._solidity(fruit.mask), 4),
                "colour": fruit.colour,
            })
            if has_stem:
                stemmed += 1
            else:
                stemless += 1
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["crop"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(frames)} frame(s) read from {source} (unchanged)")
    print(f"{len(rows)} crop(s) written to {unsorted_dir}")
    print(f"   {stemless} without a stem" +
          (f", {stemmed} with one (controls)" if stemmed else ""))
    print()
    print("Now sort them by eye into the two folders beside 'unsorted':")
    print(f"   {out_dir / 'end_on'}    looking down the axis - blossom scar or")
    print( "                           calyx facing the camera; the fruit is")
    print( "                           standing or upside down")
    print(f"   {out_dir / 'side_on'}   lying on its side; the stem is hidden,")
    print( "                           facing away, or broken off")
    print()
    print("Anything you are not sure about, leave in 'unsorted' - a guessed")
    print("label is worse than a missing one, because it cannot be spotted later.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy crops of stemless fruit out for labelling. "
                    "Source frames are only ever read."
    )
    parser.add_argument("frames", nargs="?", default=str(project_path("data/raw")),
                        help="folder of source frames (default: data/raw)")
    parser.add_argument("--out", default=str(project_path("data/debug/labelling")),
                        help="output folder (default: data/debug/labelling)")
    parser.add_argument("--config", default=None, help="config file")
    parser.add_argument("--include-stemmed", action="store_true",
                        help="also export fruit that DID get a stem, as controls "
                             "(a visible stem at the rim means side-on, so these "
                             "come pre-labelled and are worth having)")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many crops (0 = no limit)")
    return export(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
