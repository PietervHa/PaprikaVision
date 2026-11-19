"""Score a rotate-unit model on frames with a known answer - or draw it on frames without one.

    python -m tools.score_rotate models/rotate/rotate_v1.pt                   the TEST frames of its dataset
    python -m tools.score_rotate models/rotate/rotate_v1.pt --split val
    python -m tools.score_rotate models/rotate/rotate_v1.pt --unlabelled      draw on the frames nobody labelled
    python -m tools.score_rotate models/rotate/rotate_v1.onnx                 the ONNX export gives the same numbers

Against the label, per frame:
    stem      did the model give an angle, and how far off was it: median, p90, within 5/10/15
              degrees, and flips (more than 90 degrees off: the stem on the wrong side)
    no_stem   how often the model gave an angle anyway (it should not)
    empty     how often it saw a paprika anyway

Pictures - label green, model magenta - of the worst frames and of every flip, missed stem and
false angle go to data/debug/rotate_score/<model>_<split>/. With --unlabelled every frame in
data/rotate/raw that has no label is drawn into data/debug/rotate_predict/<model>/, to look at.

The angle is the one the labels and the PLC use: from the stem base (point 1) to the end of the
straight part of the stem (point 0), -180..180 degrees, 0 = +X, y down.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT / "data" / "datasets"
DEBUG = PROJECT / "data" / "debug"
LABELS = PROJECT / "data" / "rotate" / "labels" / "rotate_labels.json"
RAW = PROJECT / "data" / "rotate" / "raw"
SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

GREEN, MAGENTA, RED = (0, 255, 0), (255, 0, 255), (0, 0, 255)


def angle_of(base, tip) -> float:
    return math.degrees(math.atan2(tip[1] - base[1], tip[0] - base[0]))


def angle_error(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# ------------------------------------------------------------------ the model

def load_model(path: Path):
    from ultralytics import YOLO          # only here: the rest of this file runs without it
    return YOLO(str(path), task="pose")


def model_imgsz(path: Path, fallback: int) -> int:
    """The size the model was trained at: from its manifest, else from the .pt itself."""
    manifest = path.with_suffix(".json")
    if manifest.exists():
        size = json.loads(manifest.read_text(encoding="utf-8")).get("imgsz")
        if size:
            return int(size)
    if path.suffix == ".pt":
        try:
            import torch
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            size = (ckpt.get("train_args") or {}).get("imgsz")
            if size:
                return int(size)
        except Exception:
            pass
    return fallback


def predict(model, image, imgsz: int, conf: float, kp_conf: float, min_len_px: float) -> dict:
    """The model's answer for one frame: one paprika (the most confident), its two points, and
    an angle when both points are confident and far enough apart."""
    r = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return {"paprika": False, "has_angle": False}
    i = int(r.boxes.conf.argmax())
    tip, base = (list(map(float, p)) for p in r.keypoints.data[i].tolist())
    length = math.hypot(tip[0] - base[0], tip[1] - base[1])
    has = tip[2] >= kp_conf and base[2] >= kp_conf and length >= min_len_px
    return {"paprika": True, "score": float(r.boxes.conf[i]), "box": [float(v) for v in r.boxes.xyxy[i].tolist()],
            "tip": tip, "base": base, "has_angle": has, "angle": angle_of(base, tip) if has else None,
            "point_conf": min(tip[2], base[2])}


# ------------------------------------------------------------------ truth from a dataset split

def truth_of(label_file: Path, width: int, height: int) -> dict:
    text = label_file.read_text(encoding="utf-8").split() if label_file.exists() else []
    if not text:
        return {"status": "empty"}
    v = [float(x) for x in text[:11]]
    if v[7] == 0 or v[10] == 0:
        return {"status": "no_stem"}
    tip, base = (v[5] * width, v[6] * height), (v[8] * width, v[9] * height)
    return {"status": "stem", "tip": tip, "base": base, "angle": angle_of(base, tip)}


def draw(image, truth: dict | None, pred: dict, caption: str):
    view = image.copy()
    def arrow(base, tip, colour):
        b, t = tuple(int(round(c)) for c in base[:2]), tuple(int(round(c)) for c in tip[:2])
        cv2.arrowedLine(view, b, t, (0, 0, 0), 9, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(view, b, t, colour, 4, cv2.LINE_AA, tipLength=0.25)
    if truth and truth.get("status") == "stem":
        arrow(truth["base"], truth["tip"], GREEN)
    if pred.get("paprika"):
        x1, y1, x2, y2 = (int(round(c)) for c in pred["box"])
        cv2.rectangle(view, (x1, y1), (x2, y2), MAGENTA, 2)
        if pred["has_angle"]:
            arrow(pred["base"], pred["tip"], MAGENTA)
    cv2.rectangle(view, (0, 0), (view.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(view, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def score_split(model, dataset: Path, split: str, imgsz: int, conf: float, kp_conf: float,
                min_len_px: float, out: Path | None, worst: int = 12) -> dict:
    images = sorted(p for p in (dataset / "images" / split).glob("*") if p.suffix.lower() in SUFFIXES)
    rows = []
    for path in images:
        image = cv2.imread(str(path))
        truth = truth_of(dataset / "labels" / split / (path.stem + ".txt"), image.shape[1], image.shape[0])
        pred = predict(model, image, imgsz, conf, kp_conf, min_len_px)
        row = {"frame": path.name, "status": truth["status"], "truth": truth, "pred": pred, "error": None}
        if truth["status"] == "stem" and pred["has_angle"]:
            row["error"] = angle_error(pred["angle"], truth["angle"])
        rows.append(row)

    stem = [r for r in rows if r["status"] == "stem"]
    errors = sorted(r["error"] for r in stem if r["error"] is not None)
    within = lambda limit: 100.0 * sum(e <= limit for e in errors) / len(errors) if errors else 0.0
    result = {
        "split": split, "frames": len(rows), "stem_frames": len(stem),
        "answered_percent": 100.0 * len(errors) / len(stem) if stem else 0.0,
        "median_error": statistics.median(errors) if errors else None,
        "p90_error": errors[min(len(errors) - 1, int(0.9 * len(errors)))] if errors else None,
        "within_5": within(5), "within_10": within(10), "within_15": within(15),
        "flips": sum(e > 90 for e in errors),
        "no_stem_frames": sum(r["status"] == "no_stem" for r in rows),
        "no_stem_with_angle": sum(r["status"] == "no_stem" and r["pred"]["has_angle"] for r in rows),
        "empty_frames": sum(r["status"] == "empty" for r in rows),
        "empty_with_paprika": sum(r["status"] == "empty" and r["pred"]["paprika"] for r in rows),
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        picked = sorted((r for r in stem if r["error"] is not None), key=lambda r: -r["error"])[:worst]
        picked += [r for r in rows if (r["error"] or 0) > 90 and r not in picked]
        picked += [r for r in stem if r["error"] is None]
        picked += [r for r in rows if r["status"] == "no_stem" and r["pred"]["has_angle"]]
        picked += [r for r in rows if r["status"] == "empty" and r["pred"]["paprika"]]
        for r in picked:
            image = cv2.imread(str(dataset / "images" / split / r["frame"]))
            if r["error"] is not None:
                why = f"{r['error']:.1f} deg off"
            elif r["status"] == "stem":
                why = "stem labelled, model gave no angle"
            elif r["status"] == "no_stem":
                why = "no stem labelled, model gave an angle"
            else:
                why = "empty, model saw a paprika"
            cv2.imwrite(str(out / (Path(r["frame"]).stem + ".jpg")),
                        draw(image, r["truth"], r["pred"], f"{why}   (green = label, magenta = model)"),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        (out / "rows.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
        result["pictures"] = str(out)
    return result


def print_result(name: str, r: dict) -> None:
    print(f"{name} on {r['split']}: {r['frames']} frames")
    if r["stem_frames"]:
        print(f"  stem frames {r['stem_frames']}: angle given for {r['answered_percent']:.0f}%")
        if r["median_error"] is not None:
            print(f"    error median {r['median_error']:.1f} deg, p90 {r['p90_error']:.1f} deg, "
                  f"within 5/10/15 deg: {r['within_5']:.0f}% / {r['within_10']:.0f}% / {r['within_15']:.0f}%, "
                  f"flips (> 90 deg): {r['flips']}")
    if r["no_stem_frames"]:
        print(f"  no-stem frames {r['no_stem_frames']}: angle given anyway {r['no_stem_with_angle']}")
    if r["empty_frames"]:
        print(f"  empty frames {r['empty_frames']}: paprika seen anyway {r['empty_with_paprika']}")
    if r.get("pictures"):
        print(f"  pictures: {r['pictures']}")


def draw_unlabelled(model, name: str, imgsz: int, conf: float, kp_conf: float, min_len_px: float,
                    frames: Path, limit: int) -> None:
    labelled = set()
    if LABELS.exists():
        for key, entry in json.loads(LABELS.read_text(encoding="utf-8"))["frames"].items():
            labelled.add(str((Path(entry["root"]) / key.split("/", 1)[1]).resolve()).lower())
    todo = [p for p in sorted(frames.rglob("*")) if p.suffix.lower() in SUFFIXES and str(p.resolve()).lower() not in labelled]
    if limit:
        todo = todo[:limit]
    out = DEBUG / "rotate_predict" / name
    out.mkdir(parents=True, exist_ok=True)
    counts = {"angle": 0, "no angle": 0, "no paprika": 0}
    for path in todo:
        image = cv2.imread(str(path))
        if image is None:
            continue
        pred = predict(model, image, imgsz, conf, kp_conf, min_len_px)
        verdict = "angle" if pred["has_angle"] else "no angle" if pred["paprika"] else "no paprika"
        counts[verdict] += 1
        caption = f"{pred['angle']:.1f} deg" if pred["has_angle"] else verdict
        cv2.imwrite(str(out / (path.stem + ".jpg")), draw(image, None, pred, f"{caption}   (magenta = model)"),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(todo)} unlabelled frames drawn -> {out}")
    print("  " + ", ".join(f"{k} {v}" for k, v in counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="models/rotate/<name>.pt or .onnx")
    parser.add_argument("--dataset", default=None, help="dataset name (default: the one in the model's manifest)")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--unlabelled", action="store_true", help="draw the model on frames nobody labelled")
    parser.add_argument("--frames", default=str(RAW), help="folder for --unlabelled")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=None, help="default: the size the model was trained at")
    parser.add_argument("--conf", type=float, default=0.25, help="paprika detection confidence")
    parser.add_argument("--kp-conf", type=float, default=0.5, help="both points at least this confident for an angle")
    parser.add_argument("--min-length", type=float, default=20.0, help="base to tip at least this many pixels for an angle")
    args = parser.parse_args()

    path = Path(args.model)
    if not path.exists():
        print(f"no model {path}")
        return 1
    imgsz = args.imgsz or model_imgsz(path, 640)
    model = load_model(path)
    name = path.stem + ("_onnx" if path.suffix == ".onnx" else "")
    if args.unlabelled:
        draw_unlabelled(model, name, imgsz, args.conf, args.kp_conf, args.min_length, Path(args.frames), args.limit)
        return 0
    dataset = args.dataset
    if dataset is None and path.with_suffix(".json").exists():
        dataset = json.loads(path.with_suffix(".json").read_text(encoding="utf-8")).get("dataset")
    if dataset is None:
        print("which dataset? give --dataset rotate_v1")
        return 1
    result = score_split(model, DATASETS / dataset, args.split, imgsz, args.conf, args.kp_conf, args.min_length,
                         DEBUG / "rotate_score" / f"{name}_{args.split}")
    print_result(path.name, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
