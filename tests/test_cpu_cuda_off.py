"""The CPU path never initialises CUDA, a dynamic ONNX export, and a crash trace.

On a machine with an NVIDIA driver, PyTorch asking for GPUs while the model ran
on the CPU crashed the app minutes later (backend/utils/cuda_env.py). The fix
is only real if it is applied before Ultralytics loads anything, on "cpu"
only, and if resolving "cpu" never asks torch itself.

No GPU, no model file, no CUDA: torch and ONNX Runtime are faked where they
would otherwise be reached, and keep_cuda_off is recorded (tests/conftest.py).
"""

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from backend.detection.paprika import pose_detector as pd
from backend.detection.paprika.pose_detector import PaprikaDetector, PoseModelConfigError
from backend.utils import cuda_env
from tools.export_onnx import check_export

ROOT = Path(__file__).resolve().parents[1]


def pose(**pose_cfg):
    return PaprikaDetector({"backend": "pose", "pose": pose_cfg})


@pytest.fixture
def model_class(monkeypatch, fake_keep_cuda_off):
    """A stand-in for YOLO that notes whether CUDA was already kept off."""

    class StubModel:
        instances = []

        def __init__(self, path="", task=None):
            self.cuda_off_at_load = fake_keep_cuda_off["calls"] > 0
            StubModel.instances.append(self)

        def predict(self, frame, **kwargs):
            return []

    module = types.ModuleType("ultralytics")
    module.YOLO = StubModel
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return StubModel


def gpus(monkeypatch, torch_count, onnx_cuda=False):
    monkeypatch.setattr(pd, "cuda_device_count", lambda: torch_count)
    monkeypatch.setattr(pd, "onnx_has_cuda", lambda: onnx_cuda)


def model_file(tmp_path, suffix):
    path = tmp_path / f"paprika_pose{suffix}"
    path.write_bytes(b"stand-in model")
    return path


def write_record(path, **record):
    path.with_suffix(".json").write_text(
        json.dumps({"imgsz": 416, "onnx": record}), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------- the helper

@pytest.fixture
def fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(cuda_env, "_kept_off", False)
    return torch


def test_keep_cuda_off_makes_torch_report_no_cuda(fake_torch):
    assert not cuda_env.cuda_kept_off()
    cuda_env.keep_cuda_off()
    assert fake_torch.cuda.is_available() is False
    assert fake_torch.cuda.device_count() == 0
    assert cuda_env.cuda_kept_off()


def test_keep_cuda_off_twice_is_harmless(fake_torch):
    cuda_env.keep_cuda_off()
    cuda_env.keep_cuda_off()
    assert fake_torch.cuda.is_available() is False
    assert fake_torch.cuda.device_count() == 0


# --------------------------------------------------------- when it applies

@pytest.mark.parametrize("suffix", [".onnx", ".pt"])
def test_cpu_keeps_cuda_off_before_the_model_loads(
        model_class, monkeypatch, tmp_path, fake_keep_cuda_off, suffix):
    gpus(monkeypatch, 1, onnx_cuda=True)
    monkeypatch.setattr(pd, "onnx_input_size", lambda path: (416, 416))
    detector = pose(model_path=str(model_file(tmp_path, suffix)), imgsz=416, device="cpu")
    assert detector._load_model() is not None
    assert fake_keep_cuda_off["calls"] == 1
    assert model_class.instances[-1].cuda_off_at_load


def test_auto_without_a_gpu_keeps_cuda_off(model_class, monkeypatch, fake_keep_cuda_off):
    gpus(monkeypatch, 0)
    detector = pose(device="auto")
    detector._load_model()
    assert detector._device == "cpu"
    assert fake_keep_cuda_off["calls"] == 1


@pytest.mark.parametrize("setting", ["0", "auto"])
def test_a_gpu_device_leaves_cuda_alone(model_class, monkeypatch, fake_keep_cuda_off, setting):
    gpus(monkeypatch, 1)
    # Both would reach the real torch, and through it a real GPU if there is one.
    monkeypatch.setattr(PaprikaDetector, "_set_cudnn_benchmark", lambda self: None)
    monkeypatch.setattr(PaprikaDetector, "_device_name", lambda self: "cuda:0")
    detector = pose(device=setting)
    detector._load_model()
    assert detector._device == "0"
    assert fake_keep_cuda_off["calls"] == 0


@pytest.mark.parametrize("suffix", [".onnx", ".pt"])
def test_resolving_cpu_never_asks_about_gpus(model_class, monkeypatch, tmp_path, suffix):
    def asked():
        raise AssertionError("asked about GPUs on device cpu")

    monkeypatch.setattr(pd, "cuda_device_count", asked)
    monkeypatch.setattr(pd, "onnx_has_cuda", asked)
    monkeypatch.setattr(pd, "onnx_input_size", lambda path: (416, 416))
    detector = pose(model_path=str(model_file(tmp_path, suffix)), imgsz=416, device="cpu")
    assert detector._load_model() is not None


def test_loaded_line_says_cuda_is_kept_off(model_class, monkeypatch, caplog):
    gpus(monkeypatch, 1)
    with caplog.at_level("INFO"):
        pose(device="cpu").load_model()
    assert "device cpu (CUDA kept off)" in caplog.text


# ------------------------------------------- dynamic export: size from record

@pytest.fixture
def dynamic_onnx(model_class, monkeypatch, tmp_path):
    gpus(monkeypatch, 0)
    monkeypatch.setattr(pd, "onnx_input_size", lambda path: None)
    return model_file(tmp_path, ".onnx")


def test_dynamic_export_at_the_configured_size_loads(dynamic_onnx):
    write_record(dynamic_onnx, imgsz=416, sha256=sha(dynamic_onnx))
    detector = pose(model_path=str(dynamic_onnx), imgsz=416, device="cpu")
    detector.load_model()
    assert detector.is_ready()


def test_dynamic_export_at_another_size_refuses_to_start(dynamic_onnx):
    write_record(dynamic_onnx, imgsz=640, sha256=sha(dynamic_onnx))
    detector = pose(model_path=str(dynamic_onnx), imgsz=416, device="cpu")
    with pytest.raises(PoseModelConfigError) as raised:
        detector.load_model()
    assert "640" in str(raised.value) and "416" in str(raised.value)
    assert detector._model is None


def test_dynamic_export_without_a_record_refuses_to_start(dynamic_onnx):
    detector = pose(model_path=str(dynamic_onnx), imgsz=416, device="cpu")
    with pytest.raises(PoseModelConfigError, match="no export record"):
        detector.load_model()


def test_dynamic_export_with_a_stale_record_refuses_to_start(dynamic_onnx):
    write_record(dynamic_onnx, imgsz=416, sha256="0" * 64)
    detector = pose(model_path=str(dynamic_onnx), imgsz=416, device="cpu")
    with pytest.raises(PoseModelConfigError, match="sha256 differs"):
        detector.load_model()


# ----------------------------------------------------------- check_export

def fake_onnxruntime(monkeypatch, shape, task="pose"):
    class Session:
        def __init__(self, path, providers=None):
            pass

        def get_inputs(self):
            return [types.SimpleNamespace(shape=shape)]

        def get_modelmeta(self):
            return types.SimpleNamespace(custom_metadata_map={"task": task})

    module = types.ModuleType("onnxruntime")
    module.InferenceSession = Session
    monkeypatch.setitem(sys.modules, "onnxruntime", module)


@pytest.mark.parametrize("shape", [
    ["batch", 3, "height", "width"],
    [1, 3, "height", "width"],
])
def test_check_export_accepts_a_dynamic_pose_model(monkeypatch, shape):
    fake_onnxruntime(monkeypatch, shape)
    check_export(Path("model.onnx"))


@pytest.mark.parametrize("shape", [
    [1, 3, 416, 416],                   # static: letterboxes unlike the .pt
    [1, 3, "height", 416],
    ["batch", 1, "height", "width"],    # not 3 channels
    [2, 3, "height", "width"],
    [3, "height", "width"],
])
def test_check_export_refuses_other_shapes(monkeypatch, shape):
    fake_onnxruntime(monkeypatch, shape)
    with pytest.raises(ValueError, match="input shape"):
        check_export(Path("model.onnx"))


def test_check_export_refuses_a_non_pose_model(monkeypatch):
    fake_onnxruntime(monkeypatch, ["batch", 3, "height", "width"], task="detect")
    with pytest.raises(ValueError, match="task='detect'"):
        check_export(Path("model.onnx"))


# ------------------------------------------------------------ crash trace

def test_crash_trace_enables_faulthandler_and_appends(tmp_path):
    # In a child: faulthandler is process-wide, and pytest has its own.
    child = (
        "import faulthandler, sys\n"
        "from pathlib import Path\n"
        "from backend.main import _enable_crash_trace\n"
        "first = _enable_crash_trace(Path(sys.argv[1]))\n"
        "second = _enable_crash_trace(Path(sys.argv[1]))\n"
        "assert first == second\n"
        "print(faulthandler.is_enabled())\n"
    )
    out = subprocess.run([sys.executable, "-c", child, str(tmp_path)], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "True"
    text = (tmp_path / "native_crash.log").read_text(encoding="utf-8")
    assert text.count("=== started ") == 2
