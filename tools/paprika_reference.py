"""Reference results for the PaprikaPose Aurora filter: this engine, frame by frame.

The filter (PaprikaPose_Filters in the Aurora project) is a C++ port of the pose model
plus this engine's decision rules. This tool runs THIS engine on a folder of frames and
writes, in --out:

    cases/<frame>.bin   the frame as RGB, for the filter's test program
    cases/params.txt    the thresholds from the config, so both sides use the same ones
    reference.csv       this engine's result per fruit

Then, on the PC where the filter is built:

    PaprikaPose_Filters\\test\\bin\\paprika_test.exe PaprikaPose_Filters\\bin\\paprika_onnxruntime.dll ^
        models\\paprika_pose.onnx <out>\\cases <out>\\cpp.csv
    python -m tools.compare_paprika_reference <out>\\reference.csv <out>\\cpp.csv

Give the test program the same .onnx the config uses (paprika.pose.model_path): the
comparison checks the port, not the model.

    python -m tools.paprika_reference data\\datasets\\holdout_live2 --out data\\debug\\paprika_reference
"""
from __future__ import annotations

import argparse
import csv
import math
import struct
import sys
from pathlib import Path

import cv2

from backend.core.paprika_engine import PaprikaEngine
from tools.label_stems import IMAGE_SUFFIXES, load_config

COLUMNS = ["case", "index", "x1", "y1", "x2", "y2", "score",
           "stem_x", "stem_y", "stem_conf", "blossom_x", "blossom_y", "blossom_conf",
           "span_ratio", "pose", "angle_ccw", "confidence", "flip_confidence",
           "edge_clipped", "placement", "source", "notes"]


def _landmark(landmarks: dict, name: str) -> tuple:
    kp = landmarks.get(name)
    if kp is None:
        return 0.0, 0.0, ""
    return float(kp.x), float(kp.y), float(kp.confidence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", help="folder with belt frames (searched recursively)")
    parser.add_argument("--out", default="data/debug/paprika_reference", help="output folder")
    parser.add_argument("--config", default=None, help="config file (default: config/default.yaml)")
    parser.add_argument("--limit", type=int, default=0, help="at most this many frames (0 = all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = PaprikaEngine(cfg)
    engine.load_model()
    detector = engine._detector
    model_path = (cfg.get("paprika", {}).get("pose", {}) or {}).get("model_path", "?")
    print(f"engine model: {model_path}")
    if not str(model_path).lower().endswith(".onnx"):
        print("  note: give paprika_test the .onnx export of this same model")

    frames = sorted(p for p in Path(args.frames).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print(f"no frames under {args.frames}")
        return 1

    out = Path(args.out)
    cases = out / "cases"
    cases.mkdir(parents=True, exist_ok=True)

    params = {
        "imgsz": detector._imgsz,
        "conf": detector.confidence,
        "max_det": detector.max_detections,
        "kp_conf": detector._kp_confidence,
        "min_span_ratio": engine._min_span_ratio,
        "min_usable": engine._min_usable_confidence,
        "min_angle": engine._min_angle_confidence,
        "min_flip": engine._min_flip_confidence,
        "reject_standing": int(engine._reject_standing),
    }
    with open(cases / "params.txt", "w", encoding="ascii") as handle:
        for key, value in params.items():
            handle.write(f"{key} {value}\n")

    rows = []
    for number, path in enumerate(frames, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  could not read {path}")
            continue
        name = path.stem
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with open(cases / f"{name}.bin", "wb") as handle:
            handle.write(struct.pack("<ii", width, height))
            handle.write(rgb.tobytes())

        # The same two steps engine.evaluate runs per fruit, kept apart so the raw
        # landmarks and the decision can both be written down.
        detections = detector.detect(frame)
        if not detections:
            rows.append({"case": name, "index": -1})
        for index, detection in enumerate(detections):
            result = engine._evaluate_one(frame, detection)
            orientation = result.get("orientation") or {}
            x1, y1, x2, y2 = result["bbox"]
            landmarks = detection.get("keypoints") or {}
            sx, sy, sc = _landmark(landmarks, "stem_end")
            bx, by, bc = _landmark(landmarks, "blossom_end")
            diagonal = math.hypot(max(1, x2 - x1), max(1, y2 - y1))
            angle = orientation.get("angle_deg")
            rows.append({
                "case": name, "index": index, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": detection.get("confidence", ""),
                "stem_x": sx, "stem_y": sy, "stem_conf": sc,
                "blossom_x": bx, "blossom_y": by, "blossom_conf": bc,
                "span_ratio": round(float(orientation.get("stem_span_px") or 0.0) / diagonal, 4),
                "pose": orientation.get("pose", ""),
                "angle_ccw": "" if angle is None else angle,
                "confidence": orientation.get("confidence", ""),
                "flip_confidence": orientation.get("flip_confidence", ""),
                "edge_clipped": int(bool(detection.get("edge_clipped"))),
                "placement": result.get("placement", ""),
                "source": orientation.get("source", ""),
                "notes": ";".join(str(n) for n in orientation.get("notes") or []),
            })
        if number % 50 == 0:
            print(f"  {number}/{len(frames)} frames")

    with open(out / "reference.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    fruit = sum(1 for r in rows if r.get("index", -1) != -1)
    print(f"{len(frames)} frame(s), {fruit} fruit -> {out / 'reference.csv'}")
    print(f"cases and thresholds -> {cases}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
