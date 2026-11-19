"""Train the rotate-unit stem model, apart from the belt model.

    python -m tools.train_rotate --dataset rotate_v1 --name rotate_v1
    python -m tools.train_rotate --dataset rotate_v1 --name rotate_v1_640 --imgsz 640 --batch 8 --epochs 150
    python -m tools.train_rotate --dataset rotate_v1 --name rotate_v1_coco --base yolo11n-pose.pt
    python -m tools.train_rotate --dataset rotate_v1 --name rotate_v1 --dry-run

Trains on data/datasets/<dataset>, then scores the result on that dataset's TEST frames, which
it never trained on, and writes everything under its own name:

    runs/rotate/<name>/          the training run (Ultralytics: curves, validation pictures)
    models/rotate/<name>.pt      the weights
    models/rotate/<name>.onnx    the ONNX export (dynamic size, FP32) for Aurora later
    models/rotate/<name>.json    manifest: dataset, base, arguments, test scores

The belt model (models/paprika_pose.*) is never touched, and nothing is overwritten: a name
that is already used is refused. Compare two trainings with tools.score_rotate on each.

Base weights: by default the belt model - it already knows paprikas and stems; its two points
are simply retrained to their rotate-unit meaning. --base yolo11n-pose.pt starts from the
general Ultralytics pose model instead; worth one comparison run.

Augmentation, chosen for this camera: mirror left/right (the two points keep their meaning),
a little rotation and scale. No mosaic in the last epochs, so the model finishes on frames that
look like the real thing: one paprika, whole.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from tools.score_rotate import load_model, print_result, score_split  # noqa: E402

DATASETS = PROJECT / "data" / "datasets"
MODELS = PROJECT / "models" / "rotate"
RUNS = PROJECT / "runs" / "rotate"
BELT = PROJECT / "models" / "paprika_pose.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="dataset name under data/datasets, e.g. rotate_v1")
    parser.add_argument("--name", required=True, help="name for this training, e.g. rotate_v1 (must be new)")
    parser.add_argument("--base", default=None, help="start weights (default: the belt model, models/paprika_pose.pt)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8, help="8 fits a 4 GB GPU at 640")
    parser.add_argument("--degrees", type=float, default=10.0, help="random rotation, degrees")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="e.g. 0 for the GPU or cpu (default: Ultralytics decides)")
    parser.add_argument("--dry-run", action="store_true", help="check everything, train nothing")
    args = parser.parse_args()

    data_yaml = DATASETS / args.dataset / "data.yaml"
    if not data_yaml.exists():
        print(f"no dataset {data_yaml} - make it with: python -m tools.export_rotate_dataset --name {args.dataset}")
        return 1
    if (RUNS / args.name).exists() or (MODELS / f"{args.name}.pt").exists():
        print(f"the name {args.name} is used already (runs/rotate or models/rotate) - pick a new --name")
        return 1
    base = args.base or (str(BELT) if BELT.exists() else "yolo11n-pose.pt")
    counts = {s: len(list((DATASETS / args.dataset / "images" / s).glob("*"))) for s in ("train", "val", "test")}
    print(f"dataset {args.dataset}: train {counts['train']}, val {counts['val']}, test {counts['test']}")
    print(f"base {base}, imgsz {args.imgsz}, epochs {args.epochs} (patience {args.patience}), batch {args.batch}")
    if args.dry_run:
        print("dry run: nothing trained")
        return 0

    from ultralytics import YOLO
    import ultralytics

    started = time.time()
    model = YOLO(base)
    model.train(
        data=str(data_yaml), imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=args.batch,
        project=str(RUNS), name=args.name, exist_ok=False, device=args.device, workers=args.workers,
        seed=0, deterministic=True, plots=True,
        fliplr=0.5, degrees=args.degrees, scale=0.3, translate=0.1, mosaic=1.0, close_mosaic=15,
    )
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"training ended without {best}")
        return 1

    MODELS.mkdir(parents=True, exist_ok=True)
    weights = MODELS / f"{args.name}.pt"
    shutil.copy2(best, weights)
    onnx = Path(YOLO(str(weights)).export(format="onnx", dynamic=True, imgsz=args.imgsz, simplify=True, half=False))
    if onnx.resolve() != (MODELS / f"{args.name}.onnx").resolve():
        shutil.move(str(onnx), MODELS / f"{args.name}.onnx")
        onnx = MODELS / f"{args.name}.onnx"

    print("\nscoring on the test frames (never trained on):")
    test = score_split(load_model(weights), DATASETS / args.dataset, "test", args.imgsz, 0.25, 0.5, 20.0,
                       PROJECT / "data" / "debug" / "rotate_score" / f"{args.name}_test")
    print_result(weights.name, test)

    manifest = {
        "name": args.name, "made": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": round((time.time() - started) / 60, 1),
        "dataset": args.dataset, "dataset_counts": counts, "base": base, "imgsz": args.imgsz,
        "train_args": {k: v for k, v in vars(args).items() if k not in ("dry_run",)},
        "run_dir": str(run_dir), "weights": str(weights), "onnx": str(onnx),
        "ultralytics": ultralytics.__version__,
        "points": ["end of the straight part of the stem", "stem base"],
        "angle": "point 1 -> point 0, -180..180 degrees, 0 = +X, y down",
        "test": test,
    }
    (MODELS / f"{args.name}.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(f"\nweights  {weights}\nonnx     {onnx}\nmanifest {MODELS / (args.name + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
