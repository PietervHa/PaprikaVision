"""Running the pose model as a .pt or as an .onnx, on the GPU or the CPU.

Which runtime and device the model gets must never be left to Ultralytics: it
guessed task="detect" from an .onnx filename, and tried CUDA for an .onnx on a
machine whose ONNX Runtime has none. And a configuration that can never work -
an export at another imgsz, a GPU that is not there - must stop the app once
rather than fail on every cycle.

No model and no GPU: a stub stands in for ultralytics, and the GPU and the
export's input size are monkeypatched.
"""

import json
import logging
import os
import sys
import types

import numpy as np
import pytest

from backend.core.paprika_engine import PaprikaEngine
from backend.detection.paprika import pose_detector as pd
from backend.detection.paprika.pose_detector import PaprikaDetector, PoseModelConfigError
from backend.utils.ultralytics_env import disable_autoinstall
from tools.export_onnx import record_in_manifest, trained_imgsz

torch = pytest.importorskip("torch")


class StubModel:
    def __init__(self, path="", task=None):
        self.path = path
        self.task = task
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return []


@pytest.fixture
def fake_ultralytics(monkeypatch):
    module = types.ModuleType("ultralytics")
    module.YOLO = StubModel
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    original = torch.backends.cudnn.benchmark
    yield
    torch.backends.cudnn.benchmark = original


def gpus(monkeypatch, torch_count, onnx_cuda=False):
    monkeypatch.setattr(pd, "cuda_device_count", lambda: torch_count)
    monkeypatch.setattr(pd, "onnx_has_cuda", lambda: onnx_cuda)


@pytest.fixture
def onnx_file(tmp_path):
    path = tmp_path / "paprika_pose.onnx"
    path.write_bytes(b"stand-in; the input size is monkeypatched")
    return path


def export_size(monkeypatch, size):
    seen = []

    def fake(path):
        seen.append(path)
        return size

    monkeypatch.setattr(pd, "onnx_input_size", fake)
    return seen


def pose(**pose_cfg):
    return PaprikaDetector({"backend": "pose", "pose": pose_cfg})


# ------------------------------------------------------------------- task

def test_onnx_is_loaded_as_a_pose_model(fake_ultralytics, monkeypatch, onnx_file):
    gpus(monkeypatch, 0)
    export_size(monkeypatch, (416, 416))
    model = pose(model_path=str(onnx_file), imgsz=416)._load_model()
    assert model.task == "pose"


def test_pt_is_loaded_as_a_pose_model(fake_ultralytics, monkeypatch):
    gpus(monkeypatch, 0)
    assert pose()._load_model().task == "pose"


# ----------------------------------------------------------------- device

@pytest.mark.parametrize("setting, suffix, torch_gpus, onnx_cuda, expected", [
    ("auto", ".pt", 0, False, "cpu"),
    ("auto", ".pt", 1, False, "0"),
    # torch sees a card, but the plain onnxruntime package cannot use it.
    ("auto", ".onnx", 1, False, "cpu"),
    ("auto", ".onnx", 1, True, "0"),
    ("cpu", ".pt", 1, False, "cpu"),
    ("cpu", ".onnx", 1, True, "cpu"),
    ("0", ".pt", 1, False, "0"),
    (0, ".pt", 1, False, "0"),          # YAML reads an unquoted 0 as an int
])
def test_device_resolution(fake_ultralytics, monkeypatch, tmp_path,
                           setting, suffix, torch_gpus, onnx_cuda, expected):
    gpus(monkeypatch, torch_gpus, onnx_cuda)
    export_size(monkeypatch, (416, 416))
    path = tmp_path / f"paprika_pose{suffix}"
    path.write_bytes(b"x")
    detector = pose(model_path=str(path), imgsz=416, device=setting)
    assert detector._load_model() is not None
    assert detector._device == expected


def test_missing_device_means_auto():
    assert pose()._device_setting == "auto"


def test_unknown_device_falls_back_to_auto(caplog):
    with caplog.at_level(logging.WARNING):
        detector = pose(device="gpu")
    assert detector._device_setting == "auto"
    assert "gpu" in caplog.text


def test_device_reaches_every_predict(fake_ultralytics, monkeypatch):
    gpus(monkeypatch, 1)
    detector = pose(device="cpu", imgsz=416)
    detector.warm_up((100, 120, 3))
    detector.detect(np.zeros((100, 120, 3), dtype=np.uint8))
    calls = detector._model.calls
    assert len(calls) == 2
    assert all(kwargs["device"] == "cpu" for kwargs in calls)


def test_gpu_index_without_a_gpu_refuses_to_start(fake_ultralytics, monkeypatch):
    gpus(monkeypatch, 0)
    detector = pose(device="0")
    with pytest.raises(PoseModelConfigError, match="no CUDA device 0"):
        detector.load_model()
    assert detector._model is None


def test_gpu_index_for_onnx_needs_onnx_runtime_cuda(fake_ultralytics, monkeypatch, onnx_file):
    gpus(monkeypatch, 1, onnx_cuda=False)
    export_size(monkeypatch, (416, 416))
    detector = pose(model_path=str(onnx_file), imgsz=416, device="0")
    with pytest.raises(PoseModelConfigError, match="ONNX Runtime"):
        detector.load_model()


# ------------------------------------------------------- ONNX input size

def test_onnx_size_mismatch_refuses_to_start(fake_ultralytics, monkeypatch, onnx_file):
    gpus(monkeypatch, 0)
    export_size(monkeypatch, (640, 640))
    detector = pose(model_path=str(onnx_file), imgsz=416)
    with pytest.raises(PoseModelConfigError) as raised:
        detector.load_model()
    assert "640" in str(raised.value) and "416" in str(raised.value)
    assert detector._model is None


def test_onnx_size_mismatch_is_logged_once_not_every_cycle(
        fake_ultralytics, monkeypatch, onnx_file, caplog):
    gpus(monkeypatch, 0)
    export_size(monkeypatch, (640, 640))
    detector = pose(model_path=str(onnx_file), imgsz=416)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    with caplog.at_level(logging.ERROR):
        for _ in range(5):
            assert detector.detect(frame) == []
        detector.warm_up(frame.shape)
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1
    assert not detector.is_ready()


def test_onnx_at_the_configured_size_loads(fake_ultralytics, monkeypatch, onnx_file, caplog):
    gpus(monkeypatch, 0)
    seen = export_size(monkeypatch, (416, 416))
    detector = pose(model_path=str(onnx_file), imgsz=416, device="cpu")
    with caplog.at_level(logging.INFO):
        detector.load_model()
    assert seen == [str(onnx_file)]
    assert detector.is_ready()
    loaded = [r.getMessage() for r in caplog.records if "model loaded" in r.getMessage()]
    assert len(loaded) == 1
    assert str(onnx_file) in loaded[0]
    assert "ONNX Runtime" in loaded[0]
    assert "device cpu (CUDA kept off) | imgsz 416" in loaded[0]


def test_missing_onnx_is_not_a_config_error(fake_ultralytics, monkeypatch, tmp_path):
    gpus(monkeypatch, 0)
    seen = export_size(monkeypatch, (640, 640))
    detector = pose(model_path=str(tmp_path / "absent.onnx"), imgsz=416)
    detector.load_model()          # no raise: someone may be copying it across
    assert seen == []


def test_engine_load_model_reaches_the_detector(fake_ultralytics, monkeypatch, onnx_file):
    gpus(monkeypatch, 0)
    export_size(monkeypatch, (640, 640))
    engine = PaprikaEngine({"paprika": {"backend": "pose", "pose": {
        "model_path": str(onnx_file), "imgsz": 416}}})
    with pytest.raises(PoseModelConfigError):
        engine.load_model()


def test_load_model_is_a_no_op_without_a_model():
    PaprikaDetector({"backend": "shape"}).load_model()


# --------------------------------------------------------- cuDNN benchmark

def test_cudnn_benchmark_is_left_alone_for_onnx(fake_ultralytics, monkeypatch, onnx_file):
    gpus(monkeypatch, 1, onnx_cuda=True)
    export_size(monkeypatch, (416, 416))
    torch.backends.cudnn.benchmark = False
    pose(model_path=str(onnx_file), imgsz=416, cudnn_benchmark=True)._load_model()
    assert torch.backends.cudnn.benchmark is False


def test_cudnn_benchmark_is_left_alone_on_the_cpu(fake_ultralytics, monkeypatch):
    gpus(monkeypatch, 1)
    torch.backends.cudnn.benchmark = False
    pose(device="cpu", cudnn_benchmark=True)._load_model()
    assert torch.backends.cudnn.benchmark is False


# ------------------------------------------------------------ auto-install

def test_autoinstall_is_off_once_the_detector_is_imported():
    assert os.environ.get("YOLO_AUTOINSTALL") == "False"


def test_autoinstall_off_reaches_an_already_imported_ultralytics(monkeypatch):
    # checks.py holds its own copy of the flag, taken at import time.
    utils = types.ModuleType("ultralytics.utils")
    checks = types.ModuleType("ultralytics.utils.checks")
    utils.AUTOINSTALL = checks.AUTOINSTALL = True
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.checks", checks)
    monkeypatch.setenv("YOLO_AUTOINSTALL", "True")
    disable_autoinstall()
    assert os.environ["YOLO_AUTOINSTALL"] == "False"
    assert utils.AUTOINSTALL is False and checks.AUTOINSTALL is False


# ----------------------------------------------------------------- export

def test_trained_imgsz_comes_from_the_manifest(tmp_path):
    manifest = tmp_path / "paprika_pose.json"
    manifest.write_text(json.dumps({"imgsz": 416}), encoding="utf-8")
    assert trained_imgsz(manifest) == 416


def test_trained_imgsz_refuses_without_a_manifest(tmp_path):
    with pytest.raises(ValueError, match="trained size is unknown"):
        trained_imgsz(tmp_path / "paprika_pose.json")


def test_export_record_keeps_the_rest_of_the_manifest(tmp_path):
    manifest = tmp_path / "paprika_pose.json"
    original = {"dataset_name": "v10_416", "imgsz": 416, "replaced": None}
    manifest.write_bytes(json.dumps(original, indent=2).replace("\n", "\r\n").encode())
    record_in_manifest(manifest, {"imgsz": 416, "precision": "fp32"})
    raw = manifest.read_bytes()
    data = json.loads(raw)
    assert {k: data[k] for k in original} == original
    assert data["onnx"] == {"imgsz": 416, "precision": "fp32"}
    assert raw.count(b"\r\n") == raw.count(b"\n")     # line endings kept
