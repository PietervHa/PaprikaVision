"""Export the installed pose model to ONNX, so the CPU path runs the same model

    python -m tools.export_onnx

Reads models/paprika_pose.pt, exports it to models/paprika_pose.onnx at the
size it was trained at, and records the export in models/paprika_pose.json.
tools/retrain runs the same export after it installs a new model, so the .pt
and the .onnx never drift apart: an .onnx left over from an older model loads
without complaint and quietly runs the wrong weights.

Why the export is dynamic
-------------------------
A static export squares every frame up to imgsz x imgsz. The .pt letterboxes
to a rectangle instead - a 900x1092 frame at 416 becomes 352x416 - and only a
dynamic export gets the same. Measured on data/datasets/paprika_v10_416/val:
the static 416x416 export scored worst 21.8 deg and 99% within 20 deg; the
dynamic one reproduced the .pt exactly (n=174, median 3.4, p90 8.3, 91% within
10 deg, 100% within 20 deg, worst 18.0, 0 flipped). check_export refuses a
static result.

Why the size comes from the manifest
------------------------------------
A dynamic export runs at any size, but the model only scores as measured at
the size it was trained at - the one recorded in the manifest. The export is
made at that size and the record written under "onnx" says so; the app reads
it back and refuses to start when paprika.pose.imgsz disagrees. The .pt
carries its own training arguments too; when the two disagree the manifest
belongs to a different model, and this stops rather than guess which is right.

FP32, dynamic height and width, simplified with onnxslim. FP16 buys nothing on
a CPU and changes the outputs, so the export would no longer score like the .pt.

The export is made from a copy in a temporary folder, checked with ONNX
Runtime, and only then swapped into models/ - a failed export leaves the
previous .onnx exactly as it was.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.utils.paths import PROJECT_ROOT, project_path  # noqa: E402
from backend.utils.ultralytics_env import disable_autoinstall  # noqa: E402

disable_autoinstall()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Project-relative where possible, so the manifest survives a moved checkout."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def trained_imgsz(manifest_path: Path) -> int:
    """The imgsz the installed model was trained at, from its manifest."""
    if not manifest_path.exists():
        raise ValueError(
            f"No {manifest_path.name} beside the model, so the trained size is "
            f"unknown. Write one with: python -m tools.retrain --record <run dir>"
        )
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not record.get("imgsz"):
        raise ValueError(f"{manifest_path} has no imgsz")
    return int(record["imgsz"])


def check_export(path: Path) -> None:
    """Refuse an export that is not a 3-channel pose model of dynamic size.

    A static height or width is refused, whatever its value: that is the
    export that letterboxes differently from the .pt and scores worse.
    """
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    shape = list(session.get_inputs()[0].shape)
    expected = "[batch, 3, height, width] with dynamic height and width"
    if len(shape) != 4 or shape[1] != 3:
        raise ValueError(f"exported input shape is {shape}, expected {expected}")
    if shape[0] != 1 and isinstance(shape[0], int):
        raise ValueError(f"exported input shape is {shape}, expected {expected}")
    if isinstance(shape[2], int) or isinstance(shape[3], int):
        raise ValueError(f"exported input shape is {shape}: a static size, expected {expected}")
    task = session.get_modelmeta().custom_metadata_map.get("task")
    if task != "pose":
        raise ValueError(f"exported model says task={task!r}, expected 'pose'")


def export_onnx(pt: Path, onnx_path: Path, imgsz: int) -> dict:
    """Export `pt` to `onnx_path` at `imgsz`, FP32, dynamic, and return the record.

    Writes nothing to `onnx_path` unless the export succeeded and checked out.
    """
    import onnx
    import onnxruntime
    import torch
    import ultralytics
    from ultralytics import YOLO

    pt, onnx_path = Path(pt), Path(onnx_path)
    stored = ((getattr(YOLO(str(pt), task="pose"), "ckpt", None) or {})
              .get("train_args") or {}).get("imgsz")
    if stored is not None and int(stored) != imgsz:
        raise ValueError(
            f"{pt.name} was trained at imgsz {stored} but the manifest says {imgsz}. "
            f"The manifest belongs to a different model - fix it before exporting."
        )

    # ignore_cleanup_errors: Windows can still hold the export open briefly.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        # Ultralytics writes the export beside the weights it was given;
        # beside the installed weights is the live .onnx.
        copy = Path(work) / pt.name
        shutil.copy2(pt, copy)
        exported = Path(YOLO(str(copy), task="pose").export(
            format="onnx", imgsz=imgsz, half=False, dynamic=True,
            simplify=True, batch=1, device="cpu",
        ))
        check_export(exported)
        opset = max((o.version for o in onnx.load(str(exported)).opset_import
                     if o.domain in ("", "ai.onnx")), default=None)
        staging = onnx_path.with_name(onnx_path.name + ".tmp")
        shutil.copy2(exported, staging)
        os.replace(staging, onnx_path)

    return {
        "path": display_path(onnx_path),
        "exported": datetime.now().isoformat(timespec="seconds"),
        "source": display_path(pt),
        "source_sha256": sha256(pt),
        "sha256": sha256(onnx_path),
        "imgsz": imgsz,
        "dynamic": True,
        "precision": "fp32",
        "opset": opset,
        "simplified": True,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
    }


def record_in_manifest(manifest_path: Path, record: dict) -> None:
    """Put the export record under "onnx", keeping everything else as it was."""
    raw = "{}"
    if manifest_path.exists():
        # newline="" so a CRLF manifest is seen as one and written back as one.
        with open(manifest_path, encoding="utf-8", newline="") as handle:
            raw = handle.read()
    data = json.loads(raw)
    data["onnx"] = record
    text = json.dumps(data, indent=2)
    if raw.endswith("\n"):
        text += "\n"
    newline = "\r\n" if "\r\n" in raw else "\n"
    with open(manifest_path, "w", encoding="utf-8", newline=newline) as handle:
        handle.write(text)


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()

    pt = project_path("models/paprika_pose.pt")
    onnx_path = pt.with_suffix(".onnx")
    manifest = pt.with_suffix(".json")
    if not pt.exists():
        print(f"No model at {pt}")
        return 1
    try:
        imgsz = trained_imgsz(manifest)
        print(f"exporting {pt.name} at imgsz {imgsz} (from {manifest.name}), FP32")
        record = export_onnx(pt, onnx_path, imgsz)
    except Exception as exc:
        print(f"\nONNX export failed: {exc}")
        print(f"{onnx_path.name} was left as it was.")
        return 1
    record_in_manifest(manifest, record)
    print(f"\nwrote {onnx_path}")
    print(f"recorded under \"onnx\" in {manifest}")
    print("\nScore it before running it on the line:")
    print("   python -m tools.label_stems <val images> --score   "
          "(with model_path pointing at the .onnx)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
