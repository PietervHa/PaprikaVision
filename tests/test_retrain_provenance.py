"""The manifest beside the model must describe the new export BEFORE scoring.

The detector checks the .onnx against the "onnx" record in the .json beside it
by sha256. Provenance used to be written after the score step, so while
scoring the json still held the previous model's record, the detector refused
the fresh export and the score failed.

Runs retrain.main() end to end with the slow steps stubbed: no GPU, no model
file, no training.
"""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.detection.paprika import pose_detector
from tools import retrain

LABEL = "0 0.5 0.5 0.5 0.5 0.6 0.4 2 0.4 0.6 2\n"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_dataset(root: Path) -> Path:
    dataset = root / "base"
    image = np.full((80, 80, 3), (190, 110, 45), np.uint8)
    for split in ("train", "val"):
        (dataset / split / "images").mkdir(parents=True)
        (dataset / split / "labels").mkdir(parents=True)
        cv2.imwrite(str(dataset / split / "images" / f"{split}1.bmp"), image)
        (dataset / split / "labels" / f"{split}1.txt").write_text(LABEL)
    (dataset / "data.yaml").write_text("path: x\ntrain: train/images\nval: val/images\n")
    return dataset


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """A project in tmp_path with an old .pt/.onnx/.json set already in service."""
    models = tmp_path / "models"
    models.mkdir()
    live = models / "paprika_pose.pt"
    live.write_bytes(b"old weights")
    live.with_suffix(".onnx").write_bytes(b"old onnx")
    old_json = json.dumps({"imgsz": 640, "onnx": {
        "imgsz": 640, "sha256": _sha(live.with_suffix(".onnx"))}})
    live.with_suffix(".json").write_text(old_json)

    monkeypatch.setattr(retrain, "project_path",
                        lambda value, default=None: tmp_path / (value or default or "."))
    monkeypatch.setattr(retrain, "folders_holding_labelled_frames", lambda *a: [])

    def fake_export(pt, onnx_path, imgsz):
        onnx_path.write_bytes(b"new onnx from " + pt.read_bytes())
        return {"imgsz": imgsz, "sha256": _sha(onnx_path)}

    monkeypatch.setattr(retrain, "export_onnx", fake_export)

    seen = {}

    def fake_run(command, dry_run, cwd):
        if command[:3] == ["yolo", "pose", "train"]:
            weights = tmp_path / "runs" / "pose" / "train" / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"new weights")
        elif "tools.label_stems" in command:
            # What the score step's detector does on load. Raises
            # PoseModelConfigError ("sha256 differs") on a stale record.
            seen["score_imgsz"] = pose_detector.onnx_export_imgsz(
                live.with_suffix(".onnx"))
        return 0

    monkeypatch.setattr(retrain, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "retrain", "--name", "vtest", "--base-dataset", str(_base_dataset(tmp_path)),
    ])
    return live, old_json, seen


def test_the_score_step_sees_a_manifest_matching_the_new_export(installed):
    live, _, seen = installed
    assert retrain.main() == 0
    assert seen.get("score_imgsz") == 640, "label_stems was never reached"
    record = json.loads(live.with_suffix(".json").read_text())["onnx"]
    assert record["sha256"] == _sha(live.with_suffix(".onnx"))


def test_the_old_manifest_is_backed_up_and_named_in_go_back(installed, capsys):
    live, old_json, _ = installed
    assert retrain.main() == 0
    backups = list(live.parent.glob("paprika_pose_before_vtest_*.json"))
    assert len(backups) == 1
    assert backups[0].read_text() == old_json
    out = capsys.readouterr().out
    assert f"copy {backups[0].name} models\\paprika_pose.json" in out


def test_go_back_restores_all_three_when_all_were_backed_up():
    lines = "\n".join(retrain.go_back_instructions(
        Path("a.pt"), Path("a.onnx"), Path("a.json")))
    for name, target in (("a.pt", "pt"), ("a.onnx", "onnx"), ("a.json", "json")):
        assert f"copy {name} models\\paprika_pose.{target}" in lines
    assert "export_onnx" not in lines


def test_go_back_re_exports_when_there_is_no_onnx_to_restore():
    lines = "\n".join(retrain.go_back_instructions(Path("a.pt"), None, Path("a.json")))
    assert "copy a.json models\\paprika_pose.json" in lines
    assert "python -m tools.export_onnx" in lines


def test_go_back_without_a_previous_manifest_does_not_keep_the_new_one():
    lines = "\n".join(retrain.go_back_instructions(Path("a.pt"), Path("a.onnx"), None))
    assert "delete models\\paprika_pose.json" in lines
    assert "python -m tools.export_onnx" in lines
    assert "a.onnx" not in lines, "an .onnx without its record would be refused"


def test_go_back_says_nothing_when_there_was_no_model_before():
    assert retrain.go_back_instructions(None, None, None) == []
