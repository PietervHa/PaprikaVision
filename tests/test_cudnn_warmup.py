"""cuDNN benchmark mode and a warm-up on the real frame shape.

Benchmark mode tunes per input shape, so the startup warm-up must predict on a
frame the size the line sends - a square imgsz x imgsz blank letterboxes to a
different tensor and the first triggered cycle would pay the tuning.

No model and no GPU: a stub stands in for ultralytics. Setting the cuDNN flag
needs torch but not CUDA.
"""

import sys
import types

import pytest

from backend.core.paprika_engine import PaprikaEngine
from backend.detection.paprika.pose_detector import PaprikaDetector

torch = pytest.importorskip("torch")


class StubModel:
    def __init__(self, path="", task=None):
        self.calls = []
        # What the flag was when the model was built - i.e. before any predict.
        self.benchmark_at_load = torch.backends.cudnn.benchmark

    def predict(self, frame, **kwargs):
        self.calls.append((frame.shape, kwargs))
        return []


@pytest.fixture
def fake_ultralytics(monkeypatch):
    module = types.ModuleType("ultralytics")
    module.YOLO = StubModel
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    original = torch.backends.cudnn.benchmark
    yield
    torch.backends.cudnn.benchmark = original


@pytest.mark.parametrize("pose_cfg, expected", [
    ({}, True),
    ({"cudnn_benchmark": True}, True),
    ({"cudnn_benchmark": False}, False),
])
def test_benchmark_flag_follows_config_before_the_model_loads(
        fake_ultralytics, pose_cfg, expected):
    torch.backends.cudnn.benchmark = not expected
    detector = PaprikaDetector({"backend": "pose", "pose": pose_cfg})
    model = detector._load_model()
    assert model.benchmark_at_load is expected


def test_loading_alone_does_not_warm_up_on_a_square(fake_ultralytics):
    detector = PaprikaDetector({"backend": "pose", "pose": {"imgsz": 416}})
    assert detector._load_model().calls == []


def test_warm_up_uses_the_real_frame_shape(fake_ultralytics):
    detector = PaprikaDetector({"backend": "pose", "pose": {"imgsz": 416}})
    detector.warm_up((1092, 900, 3))
    (shape, kwargs), = detector._model.calls
    assert shape == (1092, 900, 3)
    assert kwargs["imgsz"] == 416


def test_engine_warm_up_reaches_the_detector(fake_ultralytics):
    engine = PaprikaEngine({"paprika": {"backend": "pose"}})
    engine.warm_up((1024, 1280))
    (shape, _), = engine._detector._model.calls
    assert shape == (1024, 1280, 3)


def test_warm_up_is_a_no_op_without_a_model(fake_ultralytics):
    detector = PaprikaDetector({"backend": "shape"})
    detector.warm_up((1092, 900, 3))
    assert detector._model is None


def test_a_failing_warm_up_does_not_raise(fake_ultralytics, monkeypatch):
    def boom(self, frame, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(StubModel, "predict", boom)
    detector = PaprikaDetector({"backend": "pose"})
    detector.warm_up((1092, 900, 3))
    assert detector._model is not None
