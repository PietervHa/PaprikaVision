"""tools.compare_aurora: our engine against Aurora Vision, offline.

A fake engine and a small fake results.csv; no model, no GPU.
"""

import csv
import json
import math

import cv2
import numpy as np
import pytest

from backend.detection.paprika import orientation as orient
from tools import compare_aurora as ca
from tools import label_stems


# ------------------------------------------------------------------- conversion


@pytest.mark.parametrize("aurora, ours", [
    (0.0, 0.0), (90.0, 270.0), (-90.0, 90.0), (180.0, 180.0), (-180.0, 180.0),
    (45.5, 314.5), (-135.0, 135.0),
])
def test_aurora_to_ours(aurora, ours):
    assert ca.aurora_to_ours(aurora) == pytest.approx(ours)


@pytest.mark.parametrize("aurora", [-179.0, -120.0, -45.0, -1.0, 1.0, 30.0, 100.0, 179.0])
def test_conversion_matches_vector_to_angle(aurora):
    """Aurora's angle is a y-down image vector; ours is vector_to_angle of it."""
    rad = math.radians(aurora)
    expected = orient.vector_to_angle(math.cos(rad), math.sin(rad))
    assert orient.angular_difference(ca.aurora_to_ours(aurora), expected) < 1e-9


def test_conversion_never_returns_360():
    assert 0.0 <= ca.aurora_to_ours(1e-17) < 360.0


# ---------------------------------------------------------------------- parsing


def _write(path, text):
    path.write_bytes(text.encode("utf-8"))
    return path


def test_parse_line_windows_path_and_comma_decimals():
    row = ca.parse_line(r"C:\Aurora\Images\Image_7.bmp;2;0;50,5;40,25;-90,5;0,93;0;1;1")
    assert row.image == "Image_7.bmp"
    assert (row.camera_id, row.index) == ("2", 0)
    assert (row.x, row.y, row.angle, row.confidence) == (50.5, 40.25, -90.5, 0.93)
    assert (row.method, row.stem_source, row.pushed) == (0, 1, 1)
    assert row.verdict == ca.DIRECTION_GIVEN


def test_parse_line_nothing_found_and_nine_columns():
    row = ca.parse_line("D:/x/Image_1.bmp;1;-1;;;;;;")
    assert row.index == -1 and row.x is None and row.pushed is None


@pytest.mark.parametrize("method, verdict", [(0, ca.DIRECTION_GIVEN), (1, ca.DIRECTION_GIVEN),
                                             (4, ca.DIRECTION_GIVEN), (2, ca.AXIS),
                                             (3, ca.UNDET)])
def test_verdict_per_method(method, verdict):
    assert ca.parse_line(f"I.bmp;1;0;1;1;10;0.5;{method};0").verdict == verdict


def test_undetermined_may_leave_the_angle_empty():
    row = ca.parse_line("I.bmp;1;0;10;10;;;3;0")
    assert row.method == 3 and row.angle is None


@pytest.mark.parametrize("line", [
    "I.bmp;1;0;10;10",                    # too few fields
    "I.bmp;1;0;10;10;200;0.5;0;1",        # angle out of range
    "I.bmp;1;0;10;10;20;0.5;7;1",         # unknown method
    "I.bmp;1;0;10;10;20;0.5;0;1;2",       # pushed not 0/1
    "I.bmp;1;0;nan;10;20;0.5;0;1",        # not a number
    "I.bmp;1;0;10;10;;0.5;0;1",           # direction method without an angle
])
def test_parse_line_rejects(line):
    with pytest.raises(ValueError):
        ca.parse_line(line)


def test_parse_results_counts_malformed_and_drops_partial_frame(tmp_path):
    path = _write(tmp_path / "results.csv",
                  "Image_1.bmp;1;0;10;10;0;0.9;0;1\r\n"
                  "garbage\r\n"
                  "Image_2.bmp;1;0;10;10;0;0.9;0;1\r\n"
                  "Image_2.bmp;1;1;30;30;0;0.")           # Aurora still writing
    result = ca.parse_results(path)
    assert [r.image for r in result.rows] == ["Image_1.bmp"]
    assert result.malformed and result.malformed[0][0] == 2
    assert result.partial_last_line == "Image_2.bmp;1;1;30;30;0;0."
    assert result.dropped_frame == "Image_2.bmp"


def test_parse_results_keeps_a_finished_last_line(tmp_path):
    path = _write(tmp_path / "results.csv", "Image_1.bmp;1;0;10;10;0;0.9;0;1\n")
    result = ca.parse_results(path)
    assert len(result.rows) == 1 and result.partial_last_line is None
    assert not result.has_pushed


# --------------------------------------------------------------------- matching


def _det(bbox, angle, placement="place", source="keypoints"):
    x1, y1, x2, y2 = bbox
    return {"bbox": list(bbox), "center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "placement": placement,
            "orientation": {"angle_deg": angle, "source": source, "confidence": 0.9}}


def test_pair_up_nearest_centre_and_one_to_one():
    near = _det((20, 20, 60, 60), 0.0)                # centre (40, 40)
    far = _det((0, 0, 100, 100), 0.0)                 # centre (50, 50)
    first = ca.parse_line("I.bmp;1;0;42;42;0;0.9;0;1")
    second = ca.parse_line("I.bmp;1;1;44;44;0;0.9;0;1")
    outside = ca.parse_line("I.bmp;1;2;300;300;0;0.9;0;1")
    pairs, lonely, free = ca.pair_up([first, second, outside], [far, near])
    assert pairs == [(first, near), (second, far)]
    assert lonely == [outside] and free == []


def test_axis_only_compares_modulo_180_and_undetermined_not_at_all():
    detection = _det((0, 0, 100, 100), 85.0)
    axis = ca.pair_record("c", "f", ca.parse_line("I.bmp;1;0;50;50;90;0.9;2;0"), detection)
    assert axis["compared_as"] == "axis"
    assert axis["diff_deg"] == pytest.approx(5.0)          # 85 vs 270, as an axis
    assert axis["full_diff_deg"] == pytest.approx(175.0)
    undetermined = ca.pair_record("c", "f", ca.parse_line("I.bmp;1;0;50;50;90;0.9;3;0"),
                                  detection)
    assert undetermined["diff_deg"] is None and undetermined["compared_as"] is None


@pytest.mark.parametrize("diff, name", [(0, "<10"), (9.99, "<10"), (10, "10-20"),
                                        (20, "20-45"), (45, "45-120"), (120, "45-120"),
                                        (120.1, ">120"), (180, ">120")])
def test_buckets(diff, name):
    assert ca.bucket(diff) == name


# ---------------------------------------------------------------------- compare


# Frame number (pixel 0,0) -> what the fake engine finds.
DETECTIONS = {
    1: [_det((10, 10, 60, 60), 95.0), _det((150, 150, 200, 200), 10.0, "unknown")],
    2: [_det((10, 10, 60, 60), 180.0),                       # Aurora 0 -> 180 off
        _det((70, 10, 120, 60), 85.0),                       # Aurora axis 90 -> 5 off
        _det((10, 100, 60, 150), 30.0, "reject"),            # Aurora undetermined
        _det((70, 100, 120, 150), None, "review")],          # Aurora m4, ours withheld
    3: [_det((10, 10, 60, 60), 45.0)],
    5: [],
}

RESULTS = (
    "C:\\Aurora\\cam1\\Image_1.bmp;1;0;35;35;-90;0,9;0;1;1\n"     # ours 95, Aurora 90
    "Image_2.bmp;1;0;35;35;0;0.9;1;2;1\n"                         # 180 off -> review
    "Image_2.bmp;1;1;95;35;-90;0.7;2;0;0\n"                       # axis, 5 off
    "Image_2.bmp;1;2;35;125;;;3;0;0\n"                            # undetermined
    "Image_2.bmp;1;3;95;125;45;0.6;4;0;1\n"                       # ours withheld
    "Image_2.bmp;1;4;300;300;10;0.8;0;1;1\n"                      # only Aurora
    "Image_3.bmp;1;-1;;;;;;;\n"                                   # Aurora found nothing
    "this line is broken\n"
    "Image_4.bmp;1;0;35;35;0;0.9;0;1;0\n"                         # image missing
)


class FakeEngine:
    def evaluate(self, frame):
        return {"detections": [dict(d) for d in DETECTIONS[int(frame[0, 0, 0])]]}


def _camera(root, name="cam1", results=RESULTS):
    folder = root / name
    folder.mkdir(parents=True)
    for number in (1, 2, 3, 5):
        image = np.full((240, 240, 3), 40, np.uint8)
        image[0, 0] = number
        cv2.imwrite(str(folder / f"Image_{number}.bmp"), image)
    (folder / "results.csv").write_text(results, encoding="utf-8")
    return folder


@pytest.fixture
def run(tmp_path):
    _camera(tmp_path / "in")
    out = tmp_path / "out"
    summary = ca.compare(ca.find_cameras([tmp_path / "in"]), FakeEngine(), out, 30.0)
    return tmp_path, out, summary


def test_counts(run):
    _, _, summary = run
    s = summary["cameras"]["cam1"]
    assert summary["overall"]["matched"] == s["matched"] == 5
    assert s["frames_in_csv"] == 4
    assert s["frames_compared"] == 3
    assert s["frames_missing"] == 1 and s["rows_missing_image"] == 1
    assert s["images_not_in_csv"] == 1                       # Image_5.bmp
    assert s["csv_malformed"] == 1
    assert s["unmatched_ours"] == 2                          # frame 1 extra, frame 3
    assert s["unmatched_aurora"] == 1
    assert s["matched_by_aurora_verdict"] == {ca.DIRECTION_GIVEN: 3, ca.AXIS: 1, ca.UNDET: 1}
    assert s["our_angle_withheld"] == 1


def test_difference_buckets(run):
    s = run[2]["cameras"]["cam1"]["difference_deg"]
    assert s["direction given (0,1,4)"] == {"n": 2, "<10": 1, "10-20": 0, "20-45": 0,
                                            "45-120": 0, ">120": 1}
    assert s["axis only (2), modulo 180"]["<10"] == 1
    assert set(s["by_method"]) == {"0 axis+stem", "1 stem only", "2 axis only, modulo 180"}
    assert s["by_stem_source (direction given)"]["2 green fallback"][">120"] == 1


def test_placement_tables(run):
    s = run[2]["cameras"]["cam1"]
    table = s["placement_vs_aurora"]
    assert table["place"] == {ca.DIRECTION_GIVEN: 2, ca.AXIS: 1, ca.NO_AURORA: 1}
    assert table["reject"] == {ca.UNDET: 1}
    assert table["review"] == {ca.DIRECTION_GIVEN: 1}
    assert table["unknown"] == {ca.NO_AURORA: 1}
    assert table[ca.NOT_DETECTED] == {ca.DIRECTION_GIVEN: 1}
    assert s["placement_vs_pushed"]["place"] == {"pushed": 2, "not pushed": 1}
    assert s["placement_vs_pushed"][ca.NOT_DETECTED] == {"pushed": 1}


def test_no_push_table_without_the_column(tmp_path):
    nine = "\n".join(line.rsplit(";", 1)[0] for line in RESULTS.splitlines()) + "\n"
    _camera(tmp_path / "in", results=nine)
    summary = ca.compare(ca.find_cameras([tmp_path / "in"]), FakeEngine(),
                         tmp_path / "out", 30.0)
    assert "placement_vs_pushed" not in summary["overall"]


def test_review_folder_holds_only_frames_over_the_threshold(run):
    tmp_path, out, summary = run
    assert summary["overall"]["review_frames"] == 1
    frames = out / "frames" / "cam1"
    assert sorted(p.name for p in frames.iterdir()) == ["Image_2.bmp", "results.csv"]
    original = tmp_path / "in" / "cam1" / "Image_2.bmp"
    assert (frames / "Image_2.bmp").read_bytes() == original.read_bytes()
    # label_stems walks the frames folder recursively: only originals there.
    assert [p.name for p in label_stems.gather_frames(out / "frames")] == ["Image_2.bmp"]

    overlay = cv2.imread(str(out / "overlays" / "cam1" / "Image_2.overlay.jpg"))
    assert overlay is not None and overlay.shape == (240, 240, 3)
    record = json.loads((out / "overlays" / "cam1" / "Image_2.json").read_text())
    assert [p["over_threshold"] for p in record["pairs"]] == [True, False, False, False]

    lines = ca.parse_results(frames / "results.csv")
    assert {r.image for r in lines.rows} == {"Image_2.bmp"} and len(lines.rows) == 5


def test_rerun_does_not_duplicate_review_lines(run):
    tmp_path, out, _ = run
    ca.compare(ca.find_cameras([tmp_path / "in"]), FakeEngine(), out, 30.0)
    assert len(ca.parse_results(out / "frames" / "cam1" / "results.csv").rows) == 5


def test_unmatched_csv_and_summary_json(run):
    _, out, summary = run
    with (out / "unmatched.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert sorted((r["frame"], r["side"]) for r in rows) == [
        ("Image_1.bmp", "ours"), ("Image_2.bmp", "aurora"), ("Image_3.bmp", "ours")]
    assert set(rows[0]) == {"frame", "camera", "side", "x_px", "y_px", "detail"}
    on_disk = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["overall"] == summary["overall"]
    assert on_disk["problems"]["cam1"]["frames_missing"] == ["Image_4.bmp"]


def test_engine_failure_is_not_counted_as_unmatched(tmp_path):
    _camera(tmp_path / "in")

    class Broken:
        def evaluate(self, frame):
            return {"detections": [], "failure_reason": "detector_error"}

    summary = ca.compare(ca.find_cameras([tmp_path / "in"]), Broken(), tmp_path / "out")
    s = summary["overall"]
    assert s["frames_engine_failed"] == 3 and s["frames_compared"] == 0
    assert s["unmatched_aurora"] == 0


def test_refuses_output_overlapping_a_source(tmp_path):
    folder = _camera(tmp_path / "in")
    assert ca.main([str(folder), "--out", str(folder / "review")]) == 1


# ----------------------------------------------------------------- score Aurora


def test_score_aurora_uses_the_label_stems_report(tmp_path, capsys, monkeypatch):
    folder = tmp_path / "frames" / "cam1"
    folder.mkdir(parents=True)
    (folder / "results.csv").write_text(
        "Image_1.bmp;1;0;36;36;-95;0.9;0;1;1\n"     # nearest -> ours 95, truth 90: 5 off
        "Image_1.bmp;1;1;55;55;0;0.9;0;1;1\n"       # same box, farther from centroid
        "Image_1.bmp;1;2;95;35;180;0.7;2;0;0\n"     # axis: 180 vs truth 0
        "Image_1.bmp;1;3;35;125;;;3;0;0\n",         # undetermined
        encoding="utf-8")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "stem_labels.json").write_text(json.dumps({"Image_1.bmp": [
        {"bbox": [10, 10, 60, 60], "centroid_xy": [35, 35], "stem_xy": [35, 20],
         "true_angle_deg": 90.0},
        {"bbox": [70, 10, 120, 60], "centroid_xy": [95, 35], "stem_xy": [110, 35],
         "true_angle_deg": 0.0},
        {"bbox": [10, 100, 60, 150], "centroid_xy": [35, 125], "stem_xy": [50, 110],
         "true_angle_deg": 45.0},
        {"bbox": [150, 150, 200, 200], "centroid_xy": [175, 175], "stem_xy": [185, 175],
         "true_angle_deg": 10.0},
        {"bbox": [0, 0, 5, 5], "centroid_xy": [2, 2], "stem_xy": None},
    ]}), encoding="utf-8")

    calls = []
    original = label_stems.report
    monkeypatch.setattr(label_stems, "report",
                        lambda name, subset: (calls.append(name), original(name, subset)))

    assert ca.score_aurora([tmp_path / "frames"], labels) == 0
    out = capsys.readouterr().out
    assert "ALL (full angle)" in calls                # the shared function did the work
    assert ("  ALL (full angle)           n=   4  median  92.5 deg  p90  162.5"
            "  <10deg  50%  <20deg  50%  flipped 1   (2 gave no angle)") in out
    assert "  NOT DETECTED               n=   1   (1 gave no angle)" in out
    assert "  undetermined               n=   1   (1 gave no angle)" in out
    assert "  axis only, modulo 180      n=   1  median   0.0 deg" in out
    assert "  axis only, full angle      n=   1  median 180.0 deg" in out
    assert "   n=1   median 5.0 deg   p90 5.0   worst 5.0" in out
