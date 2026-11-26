"""The rotate engine (MDE's own stem model) - the parts that would not crash, only give a wrong answer.

The model here is a tiny ONNX graph that always answers the same two points, so every number the
engine produces can be checked by hand.
"""
import json
import math

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
cv2 = pytest.importorskip("cv2")

from onnx import TensorProto, helper  # noqa: E402

from backend.core.rotate_engine import RotateEngine  # noqa: E402
from backend.detection.paprika.pose_detector import PoseModelConfigError  # noqa: E402

IN_W, IN_H = 160, 128
# in input pixels: point 0 = end of the stem, point 1 = stem base (confidences 0.9 and 0.8)
POINTS = [[80.0, 30.0, 0.9], [80.0, 90.0, 0.8]]


def _model(tmp_path, threshold=0.5, name="rotate_kp"):
    """An "mde-keypoints-1" model answering POINTS whatever the frame (the input still flows in)."""
    zero = helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0)
    times = helper.make_node("Mul", ["mean", "zero"], ["nothing"])
    plus = helper.make_node("Add", ["points_const", "nothing"], ["points"])
    graph = helper.make_graph(
        [zero, times, plus], "fixed_points",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, IN_H, IN_W])],
        [helper.make_tensor_value_info("points", TensorProto.FLOAT, [1, 2, 3])],
        initializer=[helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
                     helper.make_tensor("points_const", TensorProto.FLOAT, [1, 2, 3], sum(POINTS, []))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, str(path))
    (tmp_path / f"{name}.json").write_text(json.dumps({"format": "mde-keypoints-1", "threshold": threshold,
                                                      "angle": {"from": 1, "to": 0}}))
    return path


def _engine(path, **rotate):
    frame = rotate.pop("frame", {})
    return RotateEngine({"paprika": {"engine": "rotate", "rotate": {"model_path": str(path), **rotate}, "frame": frame}})


def test_points_angles_and_the_letterbox(tmp_path):
    engine = _engine(_model(tmp_path))
    frame = np.zeros((256, 400, 3), np.uint8)          # wider than the model: letterboxed with padding at top/bottom
    r = engine.evaluate(frame)
    p = r["primary"]
    assert r["status"] == "OK" and p["placement"] == "place"
    # scale 0.4, 12.8 px padding on top: end (80, 30) -> (200, 43), base (80, 90) -> (200, 193)
    assert p["keypoints"]["stem_end"]["x"] == pytest.approx(200.0) and p["keypoints"]["stem_end"]["y"] == pytest.approx(43.0)
    assert p["keypoints"]["stem_base"]["y"] == pytest.approx(193.0)
    # the stem points straight up the image
    assert p["angle_rotate"] == pytest.approx(-90.0)                 # rotate unit / Aurora: y down
    assert p["orientation"]["angle_deg"] == pytest.approx(90.0)      # this app's screen convention: y up
    assert p["angle_plc"] == pytest.approx(-90.0) and p["label"] == "stem -90 deg"
    assert r["timing"]["backend"] == "mde-keypoints" and r["timing"]["inference_ms"] is not None


@pytest.mark.parametrize("offset, invert, expected", [(30.0, False, -60.0), (30.0, True, 60.0),
                                                      (-100.0, False, 170.0), (100.0, False, 10.0)])
def test_the_plc_angle_takes_offset_and_invert_and_stays_in_plus_minus_180(tmp_path, offset, invert, expected):
    engine = _engine(_model(tmp_path), frame={"angle_offset_deg": offset, "angle_invert": invert})
    assert engine.evaluate(np.zeros((256, 400, 3), np.uint8))["primary"]["angle_plc"] == pytest.approx(expected)


def test_threshold_from_the_json_and_from_the_config(tmp_path):
    frame = np.zeros((256, 400, 3), np.uint8)
    strict = _engine(_model(tmp_path, threshold=0.85))         # the base (0.8) is below it
    r = strict.evaluate(frame)
    assert r["status"] == "NOK" and r["failure_reason"] == "no_stem_found"
    assert r["primary"]["angle_plc"] is None and r["primary"]["orientation"]["pose"] == "stem_not_found"
    assert r["primary"]["label"] == "no stem" and r["primary"]["keypoints"]["stem_base"]["visible"] is False
    assert strict.status()["point_threshold"] == 0.85 and strict.status()["settings_from"] == "rotate_kp.json"
    relaxed = _engine(tmp_path / "rotate_kp.onnx", point_threshold=0.5)
    assert relaxed.evaluate(frame)["status"] == "OK" and relaxed.status()["settings_from"] == "config"


def test_a_missing_model_is_reported_not_crashed_on(tmp_path):
    engine = _engine(tmp_path / "nowhere.onnx")
    assert not engine.is_ready()
    r = engine.evaluate(np.zeros((100, 100, 3), np.uint8))
    assert r["status"] == "NOK" and r["failure_reason"] == "detector_unavailable" and "nowhere.onnx" in r["error"]
    with pytest.raises(PoseModelConfigError):              # main.py: "Refusing to start", one clear line
        engine.load_model()


def test_the_config_chooses_the_engine(tmp_path, monkeypatch):
    from backend.core import vision
    monkeypatch.setattr(vision, "cfg", {"paprika": {"engine": "rotate", "rotate": {"model_path": str(_model(tmp_path))}}})
    assert type(vision._make_engine(None)).__name__ == "RotateEngine"


def test_the_overlay_draws_it(tmp_path):
    from backend.utils import annotate
    frame = np.zeros((256, 400, 3), np.uint8)
    r = _engine(_model(tmp_path)).evaluate(frame)
    out = annotate.draw_detections(frame.copy(), r["detections"])
    drawn = out if isinstance(out, np.ndarray) else frame
    assert drawn.any()                                     # something was drawn
    # the arrow runs up the image from the stem's middle (x = 200): pixels above the middle, not below
    column = drawn[:, 195:206].sum(axis=(1, 2))
    middle = r["primary"]["center"][1]
    assert column[:middle - 10].sum() > column[middle + 10:].sum()
    assert math.isfinite(r["primary"]["orientation"]["angle_deg"])


def test_no_box_for_the_rotate_unit_but_still_for_the_belt(tmp_path):
    from backend.utils import annotate
    frame = np.zeros((256, 400, 3), np.uint8)
    r = _engine(_model(tmp_path)).evaluate(frame)
    assert r["primary"]["draw_box"] is False
    drawn = annotate.draw_detections(frame.copy(), r["detections"])
    x1, y1, x2, y2 = r["primary"]["bbox"]
    assert not drawn[y2, x1:x2].any() and not drawn[y1 + 30:y2, x2].any()    # no rectangle edges
    belt = {"bbox": [50, 60, 150, 160], "center": [100, 110], "placement": "place", "label": "x",
            "orientation": {"angle_deg": 0.0}}
    drawn = annotate.draw_detections(np.zeros((256, 400, 3), np.uint8), [belt])
    assert drawn[160, 60:140].any()                                          # the belt keeps its box
