"""Hotkey capture of the live frame.

This runs on the keyboard thread of a machine that is processing fruit. It has
to be impossible for it to raise, impossible for it to block, and impossible
for a missing dependency to take the line down - the frame is worth having, but
never at that price.
"""

import json
import time

import cv2
import numpy as np
import pytest

from backend.core.manual_capture import ManualCapture


@pytest.fixture
def frame():
    image = np.full((120, 160, 3), (200, 120, 40), np.uint8)
    cv2.circle(image, (80, 60), 40, (40, 40, 200), -1)
    return image


def _capture(tmp_path, frame=None, result=None, **block):
    block.setdefault("directory", str(tmp_path))
    block.setdefault("min_interval_s", 0.0)
    return ManualCapture(
        lambda: frame,
        (lambda: result) if result is not None else None,
        block,
    )


def _settle(capture):
    time.sleep(0.5)
    capture.stop()


def test_saves_the_raw_frame_losslessly(tmp_path, frame):
    """The whole point is that the file can be re-run and trained on. JPEG
    shifted the reported stem angle by up to 4.1 degrees in measurement, which
    is comparable to the 6 degree placement gate."""
    capture = _capture(tmp_path, frame)
    capture.start()
    assert capture.capture() is True
    _settle(capture)

    images = [p for p in tmp_path.iterdir() if p.suffix == ".png"]
    assert len(images) == 1
    assert np.array_equal(cv2.imread(str(images[0])), frame), "must be byte-identical"


def test_filename_carries_the_verdict(tmp_path, frame):
    """So the interesting captures can be found without opening sidecars."""
    result = {"primary": {"placement": "reorient", "orientation": {"pose": "lying"}}}
    capture = _capture(tmp_path, frame, result)
    capture.start()
    capture.capture()
    _settle(capture)

    name = next(p.name for p in tmp_path.iterdir() if p.suffix == ".png")
    assert "reorient_lying" in name


def test_sidecar_holds_the_overlay_rather_than_burning_it_in(tmp_path, frame):
    result = {"status": "NOK", "primary": {"placement": "review",
                                           "orientation": {"pose": "lying"}}}
    capture = _capture(tmp_path, frame, result)
    capture.start()
    capture.capture(note="angle points the wrong way")
    _settle(capture)

    sidecar = json.loads(
        next(p for p in tmp_path.iterdir() if p.suffix == ".json").read_text()
    )
    assert sidecar["result"]["primary"]["placement"] == "review"
    assert sidecar["note"] == "angle points the wrong way"


def test_a_fruit_with_no_verdict_is_still_worth_keeping(tmp_path, frame):
    """A fruit the detector missed entirely has no result at all, and is
    exactly the kind of failure worth capturing."""
    capture = _capture(tmp_path, frame)
    capture.start()
    assert capture.capture() is True
    _settle(capture)
    assert any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_no_frame_available_is_refused_not_crashed(tmp_path):
    capture = _capture(tmp_path, frame=None)
    capture.start()
    assert capture.capture() is False
    capture.stop()


def test_a_held_key_writes_one_file_not_six(tmp_path, frame):
    capture = _capture(tmp_path, frame, min_interval_s=5.0)
    capture.start()
    accepted = sum(1 for _ in range(10) if capture.capture())
    _settle(capture)
    assert accepted == 1


def test_it_never_raises_from_the_key_handler(tmp_path):
    """An exception on the keyboard thread would kill the hotkey for the rest
    of the session, silently."""
    def explode():
        raise RuntimeError("camera gone")

    capture = ManualCapture(explode, None, {"directory": str(tmp_path),
                                            "min_interval_s": 0.0})
    capture.start()
    assert capture.capture() is False
    capture.stop()


def test_an_unwritable_directory_disables_cleanly(tmp_path):
    capture = _capture(tmp_path / "x", None, directory="/proc/nope/x")
    assert capture.start() is False
    assert capture.enabled is False


def test_can_be_switched_off(tmp_path, frame):
    capture = _capture(tmp_path, frame, enabled=False)
    assert capture.start() is False
    assert capture.capture() is False


def test_status_reports_where_frames_go(tmp_path, frame):
    capture = _capture(tmp_path, frame)
    status = capture.status()
    assert status["hotkey"]
    assert str(tmp_path) in status["dir"]
