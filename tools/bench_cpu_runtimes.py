"""Time the pose model on the CPU: PyTorch against OpenVINO and ONNX Runtime.

Written for one decision. The line will probably run on an industrial PC, and
nobody knows yet whether it has an NVIDIA card. Without one the model runs on
the CPU, and PyTorch is rarely the fastest way to run a small network there.
This times the same frames through all three runtimes, so the CPU route can be
judged before anyone specifies hardware or builds a deployment around it.

    python -m tools.bench_cpu_runtimes
    python -m tools.bench_cpu_runtimes --imgsz 416 --frames data/raw

Run it again on the IPC itself once it is available: CPU numbers do not
transfer between machines. SPEED ONLY - an exported model must still be scored
with label_stems before it goes anywhere near the line.

Exports are written under data/debug/bench_cpu/, never into models/, so the
model in service and the tracked models/paprika_pose.onnx stay untouched.
On first run Ultralytics installs openvino / onnx / onnxruntime into the venv.
"""
from __future__ import annotations

import argparse
import platform
import shutil
from pathlib import Path

from backend.utils.paths import project_path
from tools.bench_pose import load_frames, time_model


def cpu_name() -> str:
    try:
        import openvino as ov
        return str(ov.Core().get_property("CPU", "FULL_DEVICE_NAME"))
    except Exception:
        return platform.processor() or "unknown CPU"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=str(project_path("models/paprika_pose.pt")))
    parser.add_argument("--frames", default=str(project_path("data/raw")))
    parser.add_argument("--imgsz", type=int, default=416,
                        help="must match the size the model was trained at")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    frames = load_frames(Path(args.frames), args.limit)
    if not frames:
        print(f"No frames under {args.frames}")
        return 1

    # Export from a copy: Ultralytics writes exports beside the weights, and
    # beside the weights is models/, where an export would overwrite the
    # tracked paprika_pose.onnx.
    work = project_path("data/debug/bench_cpu")
    work.mkdir(parents=True, exist_ok=True)
    weights = work / "paprika_pose.pt"
    shutil.copy2(args.model, weights)

    rows = []
    rows.append(("PyTorch (CPU)",) + time_model(
        YOLO(str(weights)), frames, args.runs, imgsz=args.imgsz, device="cpu"))

    # half=False on purpose: FP16 buys nothing on a CPU and changes what the
    # model outputs, which would make the comparison unfair.
    for name, fmt in (("OpenVINO", "openvino"), ("ONNX Runtime (CPU)", "onnx")):
        try:
            exported = YOLO(str(weights)).export(format=fmt, imgsz=args.imgsz, half=False)
            model = YOLO(str(exported), task="pose")
            rows.append((name,) + time_model(model, frames, args.runs,
                                             imgsz=args.imgsz, device="cpu"))
        except Exception as exc:
            rows.append((name, None, str(exc)))

    print(f"\n{cpu_name()}   torch threads: {torch.get_num_threads()}")
    print(f"{len(frames)} frame(s), imgsz={args.imgsz}, {args.runs} timed calls each\n")
    print(f"{'runtime':<28}{'median':>10}{'p90':>10}")
    print("-" * 48)
    for name, median, p90 in rows:
        if median is None:
            print(f"  {name:<26}failed: {p90}")
        else:
            print(f"  {name:<26}{median:>10.1f}{p90:>10.1f}")

    try:
        import openvino as ov
        devices = ov.Core().available_devices
        print(f"\nOpenVINO devices here: {devices}")
        if any(d.startswith("GPU") for d in devices):
            print("  OpenVINO chooses its own device (AUTO). GPU is Intel's integrated")
            print("  graphics, so the OpenVINO row may have run there, not on the CPU.")
    except Exception:
        pass

    print("\nReference on the Quadro M1200: 45.5 ms at imgsz=416 (cudnn benchmark on).")
    print("SPEED ONLY. Score any exported model with label_stems before using it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
