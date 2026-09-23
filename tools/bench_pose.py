"""
Measure what the pose model actually costs, and what would make it cheaper

    python -m tools.bench_pose
    python -m tools.bench_pose --frames data/raw --runs 30
    python -m tools.bench_pose --onnx          also export and time ONNX

Times the model as the application calls it, then times the alternatives, on
this machine. Nothing here is a recommendation until the numbers come out -
inference cost depends on the card, the driver and the image size together,
and none of that can be reasoned about from a spec sheet.

What to look for
----------------
A YOLO11n-pose forward pass is 6.6 GFLOPs. On a card rated near 1.3 TFLOPS that
is under 10 ms of arithmetic, so a measured 60 ms means most of the time is
NOT arithmetic - it is Python, letterboxing, the host-to-device copy and the
Results objects ultralytics builds around every call. Framework overhead is
roughly fixed per call, so it does not shrink when the image does.

That distinction decides which lever is worth pulling:

  overhead-dominated   halving imgsz barely helps. Exporting to ONNX removes
                       most of the per-call Python instead.
  compute-dominated    imgsz is the lever, and it scales with the pixel count -
                       480 is 56% of the work of 640.

If you change imgsz, change paprika.pose.imgsz to match. Inferring at a size
the model never trained on costs accuracy for no reason, and it is a quiet kind
of wrong: nothing fails, the numbers just get worse.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.utils.paths import project_path  # noqa: E402

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}


def load_frames(folder: Path, limit: int) -> list:
    paths = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    frames = []
    for path in paths[:limit]:
        image = cv2.imread(str(path))
        if image is not None:
            frames.append(image)
    return frames


def time_model(model, frames, runs: int, **kwargs) -> tuple:
    """Median and p90 milliseconds per frame.

    Median rather than mean: one scheduler hiccup or a driver reclock skews a
    mean badly at these durations, and the median is what the line experiences.
    """
    # Warm up properly. The first call compiles CUDA kernels and allocates
    # workspace, and on a cold model that alone can take seconds - timing it
    # would say nothing about steady state.
    for _ in range(3):
        model.predict(frames[0], verbose=False, **kwargs)

    samples = []
    for index in range(runs):
        frame = frames[index % len(frames)]
        start = time.perf_counter()
        model.predict(frame, verbose=False, **kwargs)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), np.percentile(samples, 90)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Time the pose model and its alternatives on this machine."
    )
    parser.add_argument("--frames", default=str(project_path("data/raw")))
    parser.add_argument("--model", default=str(project_path("models/paprika_pose.pt")))
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--limit", type=int, default=8,
                        help="how many distinct frames to cycle through")
    parser.add_argument("--onnx", action="store_true",
                        help="also export to ONNX and time it")
    args = parser.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"Needs torch and ultralytics: {exc}")
        return 1

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"No model at {model_path}")
        return 1

    frames = load_frames(Path(args.frames), args.limit)
    if not frames:
        print(f"No frames under {args.frames}")
        return 1

    cuda = torch.cuda.is_available()
    print(f"\ntorch {torch.__version__}   cuda available: {cuda}")
    if cuda:
        print(f"device: {torch.cuda.get_device_name(0)} "
              f"compute {torch.cuda.get_device_capability(0)}")
    print(f"{len(frames)} frame(s), {frames[0].shape[1]}x{frames[0].shape[0]}, "
          f"{args.runs} timed calls each\n")

    model = YOLO(str(model_path))

    print(f"{'setting':<34}{'median':>10}{'p90':>10}")
    print("-" * 54)
    baseline = None
    for imgsz in (640, 512, 480, 416, 320):
        median, p90 = time_model(model, frames, args.runs, imgsz=imgsz)
        if baseline is None:
            baseline = median
        share = f"{100 * median / baseline:.0f}%"
        pixels = f"{100 * (imgsz / 640) ** 2:.0f}%"
        print(f"  imgsz={imgsz:<4} (pixels {pixels:>4} of 640){median:>13.1f}{p90:>10.1f}")

    if cuda:
        median, p90 = time_model(model, frames, args.runs, imgsz=640, half=True)
        print(f"  {'imgsz=640 half=True':<32}{median:>10.1f}{p90:>10.1f}")
        print("     (no fast FP16 below compute 7.0 - expect no gain on Maxwell)")

    if args.onnx:
        print("\nexporting to ONNX (one-off, takes a moment)...")
        try:
            exported = model.export(format="onnx", imgsz=640, simplify=True)
            onnx_model = YOLO(str(exported))
            median, p90 = time_model(onnx_model, frames, args.runs, imgsz=640)
            print(f"  {'ONNX imgsz=640':<32}{median:>10.1f}{p90:>10.1f}")
            print("     onnxruntime-gpu gives this the CUDA provider; without it")
            print("     this is a CPU number and will look slow.")
        except Exception as exc:
            print(f"  export failed: {exc}")

    print("\n" + "=" * 54)
    print("READING THIS")
    print("  If the times barely move as imgsz falls, the cost is per-call")
    print("  overhead, not arithmetic - shrinking the image will not help and")
    print("  ONNX is the lever worth trying.")
    print("  If they scale with the pixel count, compute dominates and imgsz")
    print("  is the lever - but retrain at the size you intend to run.")
    print("\n  Whatever you pick, set paprika.pose.imgsz to match it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
