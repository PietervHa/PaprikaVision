"""tools.aurora_watch: the live comparison, against a fake Aurora.

A fake engine and a temp folder where a fake "Aurora" appends lines and writes
images between polls - including half an image and half a line. A fake clock
drives the watcher step by step; no GPU, no model, no real waiting.
"""

import csv
import json
import os
import random

import cv2
import numpy as np
import pytest

from tools import aurora_watch as aw
from tools import compare_aurora as ca
from tests.test_compare_aurora import FakeEngine as BatchEngine, _camera, _det


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class Engine:
    """One fruit per frame; its angle per frame number (pixel 0,0)."""

    def __init__(self, angles=None, clock=None, delay=0.0, on_frame=None):
        self.angles = angles or {}
        self.clock, self.delay, self.on_frame = clock, delay, on_frame
        self.seen = []

    def evaluate(self, frame):
        number = int(frame[0, 0, 0])
        self.seen.append(number)
        if self.clock is not None:
            self.clock.advance(self.delay)
        if self.on_frame:
            self.on_frame(number)
        return {"detections": [_det((10, 10, 60, 60), self.angles.get(number, 95.0))]}


def line(number, angle=-90, index=0):
    """Aurora's line for the fruit Engine finds: -90 here is 90 ours, 5 from 95."""
    return f"Image_{number}.bmp;1;{index};35;35;{angle};0.9;0;1;1\r\n"


class FakeAurora:
    def __init__(self, root, camera="Cam1"):
        self.folder = root / camera
        self.folder.mkdir(parents=True)
        self.csv = self.folder / "results.csv"
        self.csv.write_bytes(b"")

    @staticmethod
    def bmp(number):
        image = np.full((120, 120, 3), 40, np.uint8)
        image[0, 0] = number
        return cv2.imencode(".bmp", image)[1].tobytes()

    def image(self, number, partial=False):
        data = self.bmp(number)
        (self.folder / f"Image_{number}.bmp").write_bytes(data[:len(data) // 2] if partial
                                                          else data)

    def append(self, text):
        with self.csv.open("ab") as handle:
            handle.write(text.encode("utf-8"))

    def frames(self, *numbers):
        for number in numbers:
            self.image(number)
            self.append(line(number))


def make(root, out, engine, clock, **kw):
    kw.setdefault("summary_every", 1000)
    kw.setdefault("image_timeout", 60)
    messages = []
    watcher = aw.Watcher(root, engine, out, poll=1, settle=3, clock=clock,
                         sleep=clock.advance, echo=messages.append, **kw)
    watcher.messages = messages
    return watcher


def drain(watcher, clock, steps=8):
    for _ in range(steps):
        watcher.step()
        clock.advance(1)


def counts(watcher, camera="Cam1"):
    return watcher.cameras[camera].counts


# -------------------------------------------------------------- same as batch


def snapshot(folder):
    return {str(p.relative_to(folder)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_same_results_as_batch_and_source_untouched(tmp_path):
    _camera(tmp_path / "in")
    before = snapshot(tmp_path / "in")
    batch = ca.compare(ca.find_cameras([tmp_path / "in"]), BatchEngine(),
                       tmp_path / "batch", 30.0)

    clock = Clock()
    watcher = make(tmp_path / "in", tmp_path / "watch", BatchEngine(), clock,
                   image_timeout=5, sample=1)
    drain(watcher, clock, 15)
    live = watcher.finish()

    expected = dict(batch["overall"], images_not_in_csv=0)   # not tracked live
    assert live["overall"] == expected
    assert live["cameras"]["cam1"] == dict(batch["cameras"]["cam1"], images_not_in_csv=0)
    assert live["problems"]["cam1"]["frames_missing"] == ["Image_4.bmp"]
    assert json.loads((tmp_path / "watch" / "summary.json").read_text())["overall"] == expected

    for name in ("frames/cam1/Image_2.bmp", "frames/cam1/results.csv"):
        assert (tmp_path / "watch" / name).read_bytes() == \
            (tmp_path / "batch" / name).read_bytes()
    pairs = [json.loads((tmp_path / d / "overlays/cam1/Image_2.json").read_text())["pairs"]
             for d in ("batch", "watch")]
    assert pairs[0] == pairs[1]
    assert (tmp_path / "watch/overlays/cam1/Image_2.overlay.jpg").is_file()

    def rows(path):
        with path.open(newline="", encoding="utf-8") as handle:
            return sorted(tuple(r.values()) for r in csv.DictReader(handle))
    assert rows(tmp_path / "watch/unmatched.csv") == rows(tmp_path / "batch/unmatched.csv")

    assert snapshot(tmp_path / "in") == before          # nothing written, moved, deleted


# ------------------------------------------------------------ lines and images


def test_half_line_and_half_image_wait(tmp_path):
    aurora, clock = FakeAurora(tmp_path / "in"), Clock()
    watcher = make(tmp_path / "in", tmp_path / "out", Engine(), clock)

    aurora.image(1, partial=True)
    aurora.append("Image_1.bmp;1;0;35;35;-9")            # cut inside the angle
    drain(watcher, clock, 6)
    cam = watcher.cameras["Cam1"]
    assert cam.open is None and not cam.queue and cam.buffer

    aurora.append("0;0.9;0;1;1\r\n")
    drain(watcher, clock, 6)                             # lines settled, image half
    assert counts(watcher)["frames_compared"] == 0 and len(cam.queue) == 1

    aurora.image(1)
    drain(watcher, clock, 3)
    assert counts(watcher)["frames_compared"] == 1
    assert cam.tally[("pair", "<10", ca.DIRECTION_GIVEN, 0, 1, "place", False, 1)] == 1
    assert cam.offset == aurora.csv.stat().st_size


def test_frame_is_complete_when_the_next_image_appears(tmp_path):
    aurora, clock = FakeAurora(tmp_path / "in"), Clock()
    engine = Engine()
    watcher = make(tmp_path / "in", tmp_path / "out", engine, clock)
    aurora.frames(1, 2)
    drain(watcher, clock, 2)                             # well inside --settle
    assert engine.seen == [1]
    assert watcher.cameras["Cam1"].open.name == "Image_2.bmp"
    drain(watcher, clock, 4)
    assert engine.seen == [1, 2]


def test_missing_image_is_counted_and_later_frames_go_on(tmp_path):
    aurora, clock = FakeAurora(tmp_path / "in"), Clock()
    engine = Engine()
    watcher = make(tmp_path / "in", tmp_path / "out", engine, clock, image_timeout=3)
    aurora.append(line(7) + line(7, index=1))
    aurora.frames(8)
    drain(watcher, clock, 10)
    c = counts(watcher)
    assert (c["frames_missing"], c["rows_missing_image"], c["frames_compared"]) == (1, 2, 1)
    assert engine.seen == [8]
    assert watcher.cameras["Cam1"].problems["frames_missing"] == ["Image_7.bmp"]


def test_late_lines_are_counted_not_merged(tmp_path):
    aurora, clock = FakeAurora(tmp_path / "in"), Clock()
    watcher = make(tmp_path / "in", tmp_path / "out", Engine(), clock)
    aurora.frames(1, 2)
    aurora.append(line(1, index=1) + "garbage\r\n")
    drain(watcher, clock, 8)
    c = counts(watcher)
    assert (c["frames_compared"], c["late_lines"], c["csv_malformed"], c["csv_lines"]) == \
        (2, 1, 1, 4)
    assert c["aurora_peppers"] == 2


# ---------------------------------------------------------------- review/sample


def test_review_and_every_nth_agreeing_frame_sampled(tmp_path):
    aurora, clock = FakeAurora(tmp_path / "in"), Clock()
    engine = Engine(angles={3: 270.0})                   # frame 3: 180 off
    watcher = make(tmp_path / "in", tmp_path / "out", engine, clock, sample=2)
    aurora.frames(1, 2, 3, 4, 5)
    drain(watcher, clock, 10)

    out = tmp_path / "out"
    assert sorted(p.name for p in (out / "frames/Cam1").iterdir()) == \
        ["Image_3.bmp", "results.csv"]
    # Agreeing frames are 1, 2, 4, 5: every 2nd is 2 and 5.
    assert sorted(p.name for p in (out / "sample/frames/Cam1").iterdir()) == \
        ["Image_2.bmp", "Image_5.bmp", "results.csv"]
    assert sorted(p.name for p in (out / "sample/overlays/Cam1").glob("*.jpg")) == \
        ["Image_2.overlay.jpg", "Image_5.overlay.jpg"]
    sampled = ca.parse_results(out / "sample/frames/Cam1/results.csv")
    assert sorted(r.image for r in sampled.rows) == ["Image_2.bmp", "Image_5.bmp"]
    assert (out / "sample/frames/Cam1/Image_2.bmp").read_bytes() == FakeAurora.bmp(2)
    summary = watcher.finish()
    assert summary["watch"]["cameras"]["Cam1"]["sample_frames"] == 2
    assert summary["overall"]["review_frames"] == 1


# ------------------------------------------------------- shrink, replace, start


def test_shrunk_or_replaced_results_csv_starts_over(tmp_path):
    aurora, clock = FakeAurora(tmp_path / "in"), Clock()
    engine = Engine()
    watcher = make(tmp_path / "in", tmp_path / "out", engine, clock)
    aurora.frames(1, 2, 3)
    drain(watcher, clock, 8)
    assert engine.seen == [1, 2, 3]

    aurora.csv.write_bytes(line(1).encode())             # shorter: Aurora restarted
    drain(watcher, clock, 8)
    assert engine.seen == [1, 2, 3, 1]                   # same name again is no late line

    # Replaced by a longer file with another first line.
    aurora.csv.write_bytes((line(2, angle=-91) + line(3) + line(4)).encode())
    aurora.image(4)
    drain(watcher, clock, 8)
    assert engine.seen == [1, 2, 3, 1, 2, 3, 4]
    assert counts(watcher)["csv_resets"] == 2 and counts(watcher)["late_lines"] == 0


def test_restart_neither_redoes_nor_skips(tmp_path):
    aurora, clock = FakeAurora(tmp_path / "in"), Clock()
    first = Engine()
    watcher = make(tmp_path / "in", tmp_path / "out", first, clock)
    aurora.frames(1, 2)
    aurora.append(line(3))                               # read, but its image is late
    drain(watcher, clock, 8)
    watcher.finish()
    assert first.seen == [1, 2]

    aurora.image(3)
    aurora.frames(4)
    second = Engine()
    again = make(tmp_path / "in", tmp_path / "out", second, clock)
    drain(again, clock, 8)
    assert second.seen == [3, 4]
    summary = again.finish()
    assert summary["overall"]["frames_compared"] == 4
    assert summary["overall"]["matched"] == 4


def test_from_now_starts_at_the_end_and_skips_the_cut_frame(tmp_path):
    aurora, clock = FakeAurora(tmp_path / "in"), Clock()
    aurora.frames(1, 2)                                  # frame 2 may have more lines
    engine = Engine()
    watcher = make(tmp_path / "in", tmp_path / "out", engine, clock, from_now=True)
    aurora.append(line(2, index=1))
    aurora.frames(3)
    drain(watcher, clock, 8)
    assert engine.seen == [3]
    assert counts(watcher)["skipped_at_start"] == 1


# ---------------------------------------------------------------- stop and lag


def test_stop_finishes_the_current_frame_and_writes_the_summary(tmp_path):
    aurora, clock = FakeAurora(tmp_path / "in"), Clock()
    holder = {}
    engine = Engine(on_frame=lambda n: holder["w"].request_stop())
    watcher = holder["w"] = make(tmp_path / "in", tmp_path / "out", engine, clock)
    aurora.frames(1, 2)
    assert watcher.run() == 0
    assert engine.seen == [1]
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["overall"]["frames_compared"] == 1
    state = json.loads((tmp_path / "out" / aw.STATE_NAME).read_text())
    assert state["cameras"]["Cam1"]["offset"] == len(line(1))
    assert state["cameras"]["Cam1"]["last_frame"] == "Image_1.bmp"


def test_slow_engine_reports_lag_and_drops_nothing(tmp_path):
    aurora, clock = FakeAurora(tmp_path / "in"), Clock()
    engine = Engine(clock=clock, delay=5.0)              # 5 s a frame, polls every 1 s
    watcher = make(tmp_path / "in", tmp_path / "out", engine, clock)
    aurora.frames(*range(1, 11))
    drain(watcher, clock, 4)
    summary = watcher.summary()
    lag = summary["watch"]["cameras"]["Cam1"]
    assert 0 < len(engine.seen) < 10
    assert lag["frames_waiting"] == 10 - len(engine.seen)
    assert lag["oldest_waiting_s"] > 5
    assert "waiting (lag" in watcher.status_line(summary)
    drain(watcher, clock, 20)
    assert engine.seen == list(range(1, 11))


# ------------------------------------------------------------------ the pieces


def test_tally_gives_the_same_summary_as_summarise():
    rng = random.Random(7)
    records, unmatched = [], []
    for i in range(600):
        method = rng.choice([0, 1, 2, 3, 4])
        angle = "" if method == 3 and rng.random() < 0.5 else f"{rng.uniform(-180, 180):.1f}"
        pushed = rng.choice(["", "0", "1"])
        row = ca.parse_line(f"I.bmp;1;0;50;50;{angle};0.9;{method};{rng.choice([0, 1, 2])};"
                            f"{pushed}")
        detection = _det((0, 0, 100, 100), rng.choice([None, rng.uniform(0, 360)]),
                         rng.choice(["place", "reject", "review", "unknown"]))
        if i % 5 == 0:
            unmatched.append(ca.unmatched_record("c", "f", "aurora", row))
        elif i % 7 == 0:
            unmatched.append(ca.unmatched_record("c", "f", "ours", detection))
        else:
            records.append(ca.pair_record("c", "f", row, detection))
    tally = __import__("collections").Counter()
    aw.tally_frame(tally, records, unmatched)
    base = {"frames_compared": 3}
    assert aw.summary_from_tally(tally, base) == ca.summarise(records, unmatched, base)
    # And through the state file's JSON.
    stored = json.loads(json.dumps([[list(k), n] for k, n in tally.items()]))
    back = __import__("collections").Counter({tuple(k): n for k, n in stored})
    assert aw.summary_from_tally(back, base) == ca.summarise(records, unmatched, base)


@pytest.mark.skipif(os.name != "nt", reason="share flags are a Windows matter")
def test_open_shared_lets_aurora_delete_a_file_we_are_reading(tmp_path):
    path = tmp_path / "Image_1.bmp"
    path.write_bytes(b"BM1234")
    with aw.open_shared(path) as handle:
        assert handle.read(2) == b"BM"
        os.remove(path)                                  # Aurora's rotation must work
    assert not path.exists()


def test_open_shared_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        aw.open_shared(tmp_path / "nope.bmp")


def test_refuses_output_inside_the_watched_folder(tmp_path):
    FakeAurora(tmp_path / "in")
    args = ["--watch", str(tmp_path / "in"), "--out", str(tmp_path / "in" / "review")]
    assert aw.main(args) == 1
    assert ca.main(args) == 1                            # compare_aurora hands it over
    assert not (tmp_path / "in" / "review").exists()
