"""
Tune the shape backend against your real belt

    python -m tools.tune_shape data/samples                 # sweep a folder
    python -m tools.tune_shape data/samples --live           # sweep from camera
    python -m tools.tune_shape data/samples --floor 75       # test one value

The `shape` backend thresholds on absolute saturation, so the one number that
decides whether it works on your machine is `paprika.shape.saturation_floor`.
Too low and belt texture fuses into the fruit; too high and a pale yellow
paprika starts dissolving at its edges. The right value depends on your belt,
your lighting and your camera - it cannot be guessed from here.

This sweeps candidate values over real images and reports, for each, how many
fruit were found and how strong the shoulder-versus-tip signal came out. Pick
the value that maximises `flip_conf`, not the one that maximises detections:
finding a blob is easy, and knowing which end the stem is on is the entire job.

Writes annotated montages to data/debug/tune_shape/ so you can see the mask
rather than trusting the numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detection.paprika import orientation as orient  # noqa: E402
from backend.detection.paprika.pose_detector import PaprikaDetector  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEBUG_DIR = Path(__file__).resolve().parents[1] / "data" / "debug" / "tune_shape"


def evaluate_floor(images: list[np.ndarray], floor: int, min_area: int) -> dict:
    detector = PaprikaDetector(
        {
            "backend": "shape",
            "shape": {"saturation_floor": floor, "min_area_px": min_area, "max_area_ratio": 0.7},
        }
    )

    found = 0
    flip_scores: list[float] = []
    elongations: list[float] = []
    angleless = 0

    for image in images:
        for detection in detector.detect(image):
            found += 1
            mask = orient.segment_fruit(image, detection["bbox"], saturation_floor=floor)
            if mask is None:
                angleless += 1
                continue
            result = orient.shape_orientation(mask)
            if result.angle_deg is None:
                angleless += 1
                continue
            flip_scores.append(result.flip_confidence)
            elongations.append(result.elongation)

    return {
        "floor": floor,
        "found": found,
        "per_image": found / max(1, len(images)),
        "flip_mean": float(np.mean(flip_scores)) if flip_scores else 0.0,
        "flip_weak": sum(1 for s in flip_scores if s < 0.4),
        "elongation_mean": float(np.mean(elongations)) if elongations else 0.0,
        "no_angle": angleless,
    }


def save_debug(image: np.ndarray, floor: int, name: str) -> None:
    """Write fruit | mask | overlay side by side, so the mask is inspectable."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    detector = PaprikaDetector(
        {"backend": "shape", "shape": {"saturation_floor": floor, "min_area_px": 3000}}
    )
    detections = detector.detect(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, raw_mask = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if floor > 0:
        raw_mask = cv2.bitwise_and(raw_mask, (hsv[:, :, 1] >= floor).astype(np.uint8) * 255)

    overlay = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        mask = orient.segment_fruit(image, detection["bbox"], saturation_floor=floor)
        if mask is None:
            continue
        result = orient.shape_orientation(mask)
        color = (90, 190, 70) if result.flip_confidence >= 0.4 else (60, 180, 235)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        if result.angle_deg is not None:
            from backend.utils.annotate import draw_orientation_arrow

            draw_orientation_arrow(
                overlay,
                ((x1 + x2) // 2, (y1 + y2) // 2),
                result.angle_deg,
                max(20, min(x2 - x1, y2 - y1) * 0.5),
                color,
            )
            cv2.putText(
                overlay,
                f"{result.angle_deg:.0f}deg f={result.flip_confidence:.2f}",
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    montage = np.hstack([image, cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR), overlay])
    cv2.imwrite(str(DEBUG_DIR / f"{name}_floor{floor:03d}.png"), montage)


def load_images(source: Path, limit: int) -> list[tuple[str, np.ndarray]]:
    if source.is_file():
        image = cv2.imread(str(source))
        return [(source.stem, image)] if image is not None else []

    paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    loaded = []
    for path in paths[:limit]:
        image = cv2.imread(str(path))
        if image is not None:
            loaded.append((path.stem, image))
    return loaded


def grab_live(count: int) -> list[tuple[str, np.ndarray]]:
    from backend.core.camera import Camera
    import time

    camera = Camera(0)
    frames = []
    print(f"Grabbing {count} frames - move fruit through the field of view...")
    for i in range(count):
        time.sleep(0.6)
        frame = camera.get_frame()
        if frame is not None:
            frames.append((f"live{i:02d}", frame))
    camera.release()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", help="image file or folder")
    parser.add_argument("--live", action="store_true", help="grab frames from the camera instead")
    parser.add_argument("--frames", type=int, default=12, help="frames to grab with --live")
    parser.add_argument("--limit", type=int, default=40, help="max images to load from a folder")
    parser.add_argument("--floor", type=int, default=None, help="test a single value instead of sweeping")
    parser.add_argument("--min-area", type=int, default=4000)
    parser.add_argument("--debug-images", type=int, default=3, help="montages to write per value")
    args = parser.parse_args()

    if args.live:
        samples = grab_live(args.frames)
    elif args.source:
        samples = load_images(args.source, args.limit)
    else:
        parser.error("give a source folder/file, or use --live")
        return 1

    if not samples:
        print("No readable images.")
        return 1

    print(f"\nLoaded {len(samples)} image(s)\n")
    images = [image for _, image in samples]

    floors = [args.floor] if args.floor is not None else [30, 45, 60, 75, 90, 110, 130]

    print(f"{'floor':>6} {'fruit':>6} {'per img':>8} {'flip mean':>10} {'weak flip':>10} {'no angle':>9} {'elong':>7}")
    print("-" * 62)

    rows = []
    for floor in floors:
        row = evaluate_floor(images, floor, args.min_area)
        rows.append(row)
        print(
            f"{row['floor']:>6} {row['found']:>6} {row['per_image']:>8.2f} "
            f"{row['flip_mean']:>10.3f} {row['flip_weak']:>10} "
            f"{row['no_angle']:>9} {row['elongation_mean']:>7.2f}"
        )

        for name, image in samples[: args.debug_images]:
            save_debug(image, floor, name)

    best = max(rows, key=lambda r: (r["flip_mean"], r["per_image"]))
    print(
        f"\nStrongest stem-end signal at saturation_floor = {best['floor']} "
        f"(flip mean {best['flip_mean']:.3f}, {best['per_image']:.2f} fruit/image)."
    )
    print(f"Montages (fruit | mask | overlay): {DEBUG_DIR}")
    print(
        "\nLook at the masks before committing. A high flip mean on two badly "
        "segmented fruit beats nothing, but it is not evidence the threshold "
        "is right - it just means those two happened to work."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
