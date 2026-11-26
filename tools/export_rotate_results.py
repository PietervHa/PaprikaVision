"""Every frame's result of a rotate-unit model, in one zip to send back.

    python -m tools.export_rotate_results models/rotate/rotate_kp.onnx
    python -m tools.export_rotate_results models/rotate/rotate_kp.onnx --split val

Per frame of the split: a picture (label green, model magenta, the error in the caption) and a
row in results.csv. Same model, threshold and scoring as tools.score_rotate - only it keeps
every frame, not just the worst. Writes data/debug/rotate_results_<model>_<split>.zip.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2

from tools.score_rotate import DATASETS, DEBUG, SUFFIXES, angle_error, draw, load_model, predict, truth_of


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="models/rotate/rotate_kp.onnx (with rotate_kp.json next to it)")
    ap.add_argument("--dataset", default="rotate_v1")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    args = ap.parse_args()

    path = Path(args.model)
    model = load_model(path)
    own = hasattr(model, "angle_from")
    kp_conf, min_len = (model.threshold, 2.0) if own else (0.5, 20.0)
    folder = DATASETS / args.dataset
    images = sorted(p for p in (folder / "images" / args.split).glob("*") if p.suffix.lower() in SUFFIXES)
    out = DEBUG / f"rotate_results_{path.stem}_{args.split}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for i, image_path in enumerate(images, 1):
            image = cv2.imread(str(image_path))
            truth = truth_of(folder / "labels" / args.split / (image_path.stem + ".txt"), image.shape[1], image.shape[0])
            pred = predict(model, image, 640, 0.25, kp_conf, min_len)
            error = (angle_error(truth["angle"], pred["angle"])
                     if truth["status"] == "stem" and pred["has_angle"] else None)
            verdict = ("ok" if error is not None and error <= 15 else "off" if error is not None
                       else "missed" if truth["status"] == "stem" else "false_angle" if pred["has_angle"] else "ok")
            label_text = f"{truth['angle']:.1f}" if truth["status"] == "stem" else truth["status"]
            model_text = f"{pred['angle']:.1f}" if pred["has_angle"] else "no angle"
            caption = f"{verdict}  label {label_text}  model {model_text}" + (f"  error {error:.1f} deg" if error is not None else "")
            ok, jpg = cv2.imencode(".jpg", draw(image, truth if truth["status"] == "stem" else None, pred, caption),
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            z.writestr(f"{i:03d}_{verdict}_{image_path.stem}.jpg", jpg.tobytes())
            rows.append({"frame": image_path.name, "label": truth["status"],
                         "label_angle": round(truth["angle"], 2) if truth["status"] == "stem" else "",
                         "model_angle": round(pred["angle"], 2) if pred["has_angle"] else "",
                         "error_deg": round(error, 2) if error is not None else "", "verdict": verdict,
                         "point_conf": round(pred.get("point_conf", 0.0), 3)})
            print(f"{i}/{len(images)}  {caption}")
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        z.writestr("results.csv", text.getvalue())
        z.writestr("model.json", json.dumps({"model": path.name, "dataset": args.dataset, "split": args.split,
                                             "point_threshold": kp_conf, "min_stem_px": min_len}, indent=1))
    print(f"\n{len(images)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
