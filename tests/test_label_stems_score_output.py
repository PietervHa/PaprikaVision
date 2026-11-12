"""label_stems --score output must not change when its report code is shared.

compare_aurora --score-aurora scores Aurora with the same report function that
--score uses for our engine, so the two numbers are produced by one piece of
code. Lifting that function out of score() must not change a single byte of
what --score prints; GOLDEN below was captured from the version before the
refactor, on this fixture, with a fake engine so no model is needed.
"""

import argparse
import json

import cv2
import numpy as np
import pytest

from backend.core import paprika_engine
from tools import label_stems

# Frame number -> what the fake engine reports. The frame number is written
# into pixel (0, 0) so evaluate() can tell the frames apart.
DETECTIONS = {
    1: [
        {"bbox": [10, 10, 60, 60], "center": [35, 35], "placement": "place",
         "orientation": {"angle_deg": 95.0, "source": "keypoints",
                         "notes": ["width_signal=0.3", "agree", "x"]}},
        {"bbox": [70, 10, 120, 60], "center": [95, 35], "placement": "place",
         "orientation": {"angle_deg": 300.0, "source": "fused", "notes": []}},
    ],
    2: [
        {"bbox": [10, 10, 60, 60], "center": [35, 35], "placement": "reject",
         "orientation": {"angle_deg": None, "source": "none", "notes": []}},
        {"bbox": [70, 10, 120, 60], "center": [95, 35], "placement": "unknown",
         "orientation": {"angle_deg": 12.5, "source": "shape", "notes": ["weak"]}},
    ],
    3: [],
}

LABELS = {
    "frame_1.bmp": [
        {"centroid_xy": [33.0, 36.0], "stem_xy": [33.0, 20.0], "true_angle_deg": 90.0},
        {"centroid_xy": [96.0, 34.0], "stem_xy": [96.0, 50.0], "true_angle_deg": 120.0},
    ],
    "frame_2.bmp": [
        {"centroid_xy": [30.0, 30.0], "stem_xy": [40.0, 30.0], "true_angle_deg": 0.0},
        {"centroid_xy": [90.0, 30.0], "stem_xy": [80.0, 30.0], "true_angle_deg": 180.0},
        {"centroid_xy": [200.0, 200.0], "stem_xy": None, "note": "no_stem_visible"},
    ],
    "frame_3.bmp": [
        {"centroid_xy": [30.0, 30.0], "stem_xy": [30.0, 10.0], "true_angle_deg": 90.0},
    ],
}

GOLDEN = "\n".join([
    '',
    '5 labelled fruit scored against the detector',
    '',
    '  ALL                        n=   5  median 167.5 deg  p90  177.5  <10deg  33%  <20deg  33%  flipped 2   (2 gave no angle)',
    '',
    '  NOT DETECTED               n=   1   (1 gave no angle)',
    '  place                      n=   2  median  92.5 deg  p90  162.5  <10deg  50%  <20deg  50%  flipped 1',
    '  reject                     n=   1   (1 gave no angle)',
    '  unknown                    n=   1  median 167.5 deg  p90  167.5  <10deg   0%  <20deg   0%  flipped 1',
    '',
    '  estimator: fused           n=   1  median 180.0 deg  p90  180.0  <10deg   0%  <20deg   0%  flipped 1',
    '  estimator: keypoints       n=   1  median   5.0 deg  p90    5.0  <10deg 100%  <20deg 100%  flipped 0',
    '  estimator: none            n=   1   (1 gave no angle)',
    '  estimator: shape           n=   1  median 167.5 deg  p90  167.5  <10deg   0%  <20deg   0%  flipped 1',
    '',
    'WORST 2 PLACED FRUIT - these reached the actuator:',
    '    180.0 deg off   frame_1.bmp                                said  300.0, truth  120.0   fused  []',
    "      5.0 deg off   frame_1.bmp                                said   95.0, truth   90.0   keypoints  ['agree', 'x']",
    '',
    '==================================================================',
    'THE NUMBER THAT MATTERS - fruit the machine actually placed:',
    '   n=2   median 92.5 deg   p90 162.5   worst 180.0',
    '   within 10 deg:  50%   (1 fruit worse)',
    '   within 20 deg:  50%   (1 fruit worse)',
    '   within 30 deg:  50%   (1 fruit worse)',
    '   within 45 deg:  50%   (1 fruit worse)',
    '',
    '   A placed fruit that is far out is the failure this system exists',
    '   to prevent - it reaches the actuator with nothing flagged.',
    '',
])


class FakeEngine:
    def __init__(self, cfg, app_state=None):
        pass

    def load_model(self):
        pass

    def evaluate(self, frame):
        return {"detections": [dict(d) for d in DETECTIONS[int(frame[0, 0, 0])]]}


@pytest.fixture
def scored(tmp_path, monkeypatch, capsys):
    frames = tmp_path / "frames"
    frames.mkdir()
    for number in DETECTIONS:
        image = np.zeros((240, 240, 3), np.uint8)
        image[0, 0, 0] = number
        cv2.imwrite(str(frames / f"frame_{number}.bmp"), image)
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "stem_labels.json").write_text(json.dumps(LABELS), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("paprika: {}\n", encoding="utf-8")

    monkeypatch.setattr(paprika_engine, "PaprikaEngine", FakeEngine)
    args = argparse.Namespace(frames=str(frames), out=str(labels),
                              config=str(config), worst=10)
    code = label_stems.score(args)
    return code, capsys.readouterr().out


def test_score_output_is_byte_for_byte_unchanged(scored):
    code, out = scored
    assert code == 0
    assert out == GOLDEN
