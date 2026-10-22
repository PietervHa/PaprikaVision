"""
Measure what the pose model actually costs, and what would make it cheaper

    python -m tools.bench_pose
    python -m tools.bench_pose --frames data/raw --runs 30
    python -m tools.bench_pose --onnx          also export and time ONNX
    python -m tools.bench_pose --two-stage     boxes first, then a model on crops
    python -m tools.bench_pose --cudnn-benchmark   as the app runs by default

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


def _shape_settings() -> dict:
    """The shape block from the config, read directly so this tool does not
    pull in the app's config loader and its argv handling."""
    try:
        import yaml

        cfg = yaml.safe_load(
            project_path("config/default.yaml").read_text(encoding="utf-8")
        ) or {}
        paprika = cfg.get("paprika") or {}
        shape = paprika.get("shape") or {}
    except Exception:
        paprika, shape = {}, {}
    return {"shape": shape, "max_detections": int(paprika.get("max_detections", 8))}


def _crop(frame, bbox, margin: float):
    """The box widened by `margin` of its own size on every side, clamped."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    return frame[y1:y2, x1:x2]


def two_stage(model, frames, runs: int, device, full_frame: dict,
              margin: float) -> None:
    """Time "find boxes classically, then run a small model on the crops".

    SPEED ONLY. The pose weights were trained on whole 640 frames and the
    classifier is untrained, so nothing printed here says whether the idea
    would be accurate - only whether it would be fast enough to be worth
    training for.

    Totals are taken per call - boxes and crop model on the same frame added
    together - and the median taken of those sums. Adding two medians would
    pair a fast box search on one frame with a slow crop batch on another.
    """
    from ultralytics import YOLO
    from backend.detection.paprika import classical

    settings = _shape_settings()
    shape = settings["shape"]
    belt_hue = tuple(int(v) for v in (shape.get("belt_hue") or [96, 145]))
    stem_hue = tuple(int(v) for v in (shape.get("stem_hue") or [33, 92]))

    def find_boxes(frame):
        # The same call PaprikaDetector._detect_shape makes, belt mask
        # included, with only the self-check switched off. find_fruit still
        # searches for the stem on every fruit, which a box-only stage would
        # not need - hence an upper bound.
        belt = (classical.belt_mask_raw(frame, belt_hue)
                if bool(shape.get("restrict_to_belt", True)) else None)
        fruits = classical.find_fruit(
            frame,
            belt_hue=belt_hue,
            saturation_floor=int(shape.get("saturation_floor", 80)),
            value_floor=int(shape.get("value_floor", 45)),
            stem_hue=stem_hue,
            min_area=int(shape.get("min_area_px", 4000)),
            max_area_ratio=float(shape.get("max_area_ratio",
                                           classical.DEFAULT_MAX_AREA_RATIO)),
            max_fruit=settings["max_detections"],
            use_roi=bool(shape.get("use_roi", True)),
            selfcheck=False,
            belt_mask=belt,
        )
        return [fruit.bbox for fruit in fruits]

    print("\n" + "=" * 54)
    print("TWO-STAGE - SPEED ONLY")
    print("  None of the crop models below is trained for crops: the pose")
    print("  weights saw whole 640 frames, the classifier is untrained. These")
    print("  are timings, not a statement that either would work.")
    print("=" * 54)

    find_boxes(frames[0])                      # warm-up: first-call allocations
    crops_per_frame = [
        [_crop(frame, bbox, margin) for bbox in find_boxes(frame)]
        for frame in frames
    ]
    crops_per_frame = [[c for c in crops if c.size] for crops in crops_per_frame]
    with_fruit = sum(1 for crops in crops_per_frame if crops)
    total_crops = sum(len(crops) for crops in crops_per_frame)
    print(f"\n{with_fruit} of {len(frames)} frame(s) had fruit, "
          f"{total_crops / len(frames):.1f} crop(s) per frame on average, "
          f"margin {margin:.0%}")
    if not total_crops:
        print("  no fruit found on any frame - nothing to crop, stopping here.")
        return
    if with_fruit < len(frames):
        print("  frames with no fruit cost the crop model nothing below, which")
        print("  flatters the totals - point --frames at a busier folder if so.")

    box_samples = []
    for index in range(runs):
        frame = frames[index % len(frames)]
        start = time.perf_counter()
        find_boxes(frame)
        box_samples.append((time.perf_counter() - start) * 1000.0)

    try:
        classifier = YOLO("yolo11n-cls.yaml")   # built from the yaml, no download
        if device is not None:
            classifier.to(device)
    except Exception as exc:
        classifier = None
        print(f"  yolo11n-cls.yaml could not be built: {exc}")

    configs = [(f"pose weights, crops @ {s}", model, s) for s in (160, 192, 224)]
    if classifier is not None:
        configs += [(f"yolo11n-cls (untrained) @ {s}", classifier, s)
                    for s in (128, 160, 224)]

    sample_batch = next(crops for crops in crops_per_frame if crops)
    crop_samples = {}
    for label, crop_model, size in configs:
        try:
            for _ in range(3):
                crop_model.predict(sample_batch, imgsz=size, verbose=False)
            samples = []
            for index in range(runs):
                crops = crops_per_frame[index % len(frames)]
                start = time.perf_counter()
                if crops:
                    # One batched call per frame, as a second stage would run.
                    crop_model.predict(crops, imgsz=size, verbose=False)
                samples.append((time.perf_counter() - start) * 1000.0)
            crop_samples[label] = samples
        except Exception as exc:
            print(f"  {label}: failed: {exc}")

    print(f"\n{'per frame, median ms':<38}{'stage':>8}{'total':>8}")
    print("-" * 54)
    print(f"  {'a. boxes: find_fruit, UPPER BOUND':<36}"
          f"{statistics.median(box_samples):>8.1f}")
    print("     (also searches for the stem; selfcheck off)")
    for label, samples in crop_samples.items():
        totals = [b + c for b, c in zip(box_samples, samples)]
        print(f"  {label:<36}{statistics.median(samples):>8.1f}"
              f"{statistics.median(totals):>8.1f}")
    print("-" * 54)
    for size in (640, 416):
        if size in full_frame:
            print(f"  {f'full frame, current model @ {size}':<36}"
                  f"{'':>8}{full_frame[size]:>8.1f}")
    print("\n  SPEED ONLY. A two-stage design is worth training for only if a")
    print("  total above is clearly below full frame at the size you would")
    print("  otherwise run - and then it still has to be scored for accuracy.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Time the pose model and its alternatives on this machine."
    )
    parser.add_argument("--frames", default=str(project_path("data/raw")))
    parser.add_argument("--model", default=str(project_path("models/paprika_pose.pt")))
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--limit", type=int, default=8,
                        help="how many distinct frames to cycle through")
    parser.add_argument("--architectures", action="store_true",
                        help="time other pretrained pose architectures at 640. "
                             "An OLDER network can be faster on an older GPU "
                             "even with more FLOPs, because its operations map "
                             "better onto the hardware.")
    parser.add_argument("--onnx", action="store_true",
                        help="also export to ONNX and time it")
    parser.add_argument("--two-stage", action="store_true",
                        help="also time classical box-finding plus a model on "
                             "the crops. SPEED ONLY - nothing it runs is "
                             "trained for crops.")
    parser.add_argument("--crop-margin", type=float, default=0.10,
                        help="--two-stage: widen each box by this share of "
                             "its size on every side before cropping")
    parser.add_argument("--cudnn-benchmark", action="store_true",
                        help="let cuDNN time its algorithms per input shape, "
                             "as paprika.pose.cudnn_benchmark does in the app. "
                             "Off by default here so both tables can be run.")
    parser.add_argument("--device", default=None,
                        help="pin the device: 0, cuda:0 or cpu. Unset lets "
                             "ultralytics choose - which is worth checking.")
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

    # Set explicitly either way, before the first predict, so an "off" run is
    # off whatever torch or ultralytics would otherwise default to. The time_model
    # warm-up then pays the per-shape tuning, not the timed calls.
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    cuda = torch.cuda.is_available()
    print(f"\ntorch {torch.__version__}   cuda available: {cuda}   "
          f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    if cuda:
        print(f"device: {torch.cuda.get_device_name(0)} "
              f"compute {torch.cuda.get_device_capability(0)}")
    print(f"{len(frames)} frame(s), {frames[0].shape[1]}x{frames[0].shape[0]}, "
          f"{args.runs} timed calls each\n")

    model = YOLO(str(model_path))
    if args.device is not None:
        # "0" is what ultralytics and nvidia-smi call the first GPU, and what
        # anyone types. torch wants "cuda:0" and raises on the bare digit, so
        # accept both rather than making the caller remember which tool this is.
        device = args.device.strip()
        if device.isdigit():
            device = f"cuda:{device}"
        model.to(device)
    # Where the weights actually live. Ultralytics chooses a device per call
    # and does not announce it, so a model quietly running on the CPU looks
    # exactly like a slow GPU - and an i7 takes roughly the same time for this
    # model as the numbers we are trying to explain.
    try:
        where = next(model.model.parameters()).device
        print(f"weights are on: {where}")
    except Exception:
        print("weights are on: could not determine")

    # Split the model from the machinery around it. predict() letterboxes on
    # the CPU, copies to the device, runs NMS and builds a Results object with
    # plotting metadata; a bare forward pass does none of that. The difference
    # between these two lines is what an export could plausibly remove, and it
    # is the whole question when the times stop scaling with the image size.
    if True:
        import torch as _torch
        size = 640
        tensor = _torch.zeros(1, 3, size, size)
        try:
            tensor = tensor.to(next(model.model.parameters()).device)
            with _torch.no_grad():
                for _ in range(5):
                    model.model(tensor)
                if cuda:
                    _torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    model.model(tensor)
                if cuda:
                    _torch.cuda.synchronize()
            raw = (time.perf_counter() - start) / 20 * 1000
            print(f"\nbare forward pass at 640 (no pre/post, no Results): "
                  f"{raw:.1f} ms")
            print("   everything above this in the table is framework overhead\n")
        except Exception as exc:
            print(f"\nbare forward pass failed: {exc}\n")

    print(f"{'setting':<34}{'median':>10}{'p90':>10}")
    print("-" * 54)
    baseline = None
    full_frame = {}
    for imgsz in (640, 512, 480, 416, 320):
        median, p90 = time_model(model, frames, args.runs, imgsz=imgsz)
        full_frame[imgsz] = median
        if baseline is None:
            baseline = median
        share = f"{100 * median / baseline:.0f}%"
        pixels = f"{100 * (imgsz / 640) ** 2:.0f}%"
        print(f"  imgsz={imgsz:<4} (pixels {pixels:>4} of 640){median:>13.1f}{p90:>10.1f}")

    if cuda:
        median, p90 = time_model(model, frames, args.runs, imgsz=640, half=True)
        print(f"  {'imgsz=640 half=True':<32}{median:>10.1f}{p90:>10.1f}")
        print("     (no fast FP16 below compute 7.0 - expect no gain on Maxwell)")

    if args.architectures:
        # Downloads each on first use. FLOPs are not the measure here: YOLOv8's
        # plain convolutions are well-trodden ground for cuDNN, while YOLO11's
        # C3k2 and C2PSA blocks were tuned on hardware a decade newer than a
        # Maxwell card. More arithmetic in simpler kernels can finish sooner.
        print("\narchitectures at imgsz=640 (pretrained, untrained on your fruit -")
        print("this measures SPEED only, not accuracy):")
        for name, flops in (("yolo11n-pose.pt", "6.6"), ("yolov8n-pose.pt", "9.2"),
                            ("yolo11s-pose.pt", "23.2")):
            try:
                other = YOLO(name)
                if args.device is not None:
                    device = args.device.strip()
                    other.to(f"cuda:{device}" if device.isdigit() else device)
                median, p90 = time_model(other, frames, args.runs, imgsz=640)
                print(f"  {name:<22} {flops:>5} GFLOPs {median:>9.1f} ms{p90:>10.1f}")
            except Exception as exc:
                print(f"  {name:<22} failed: {exc}")
        print("  If one is clearly faster, retraining on it is a day's work and")
        print("  the dataset is already built.")

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

    if args.two_stage:
        device = None
        if args.device is not None:
            device = args.device.strip()
            device = f"cuda:{device}" if device.isdigit() else device
        two_stage(model, frames, args.runs, device, full_frame, args.crop_margin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())