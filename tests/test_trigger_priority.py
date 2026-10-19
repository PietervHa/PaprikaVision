"""Triggers must not queue behind the overlay preview on the model lock.

The overlay worker and a PLC trigger run the same engine, and the pose
detector holds one lock around the model. Triggered cycles measured 103-157 ms
against a forward pass of ~88 ms; the suspected gap was a trigger waiting for a
preview inference to finish. hmi.overlay_pause_after_trigger_s keeps the
overlay off the model while the line is triggering.

No model and no GPU: a fake engine sleeps inside a lock the way the pose
detector does, so contention is real and the tests stay fast.
"""

import threading
import time

import numpy as np
import pytest

from backend.core import overlay_worker as ow
from backend.core import vision
from backend.core.paprika_engine import PaprikaEngine
from backend.core.state import AppState
from backend.detection.paprika.pose_detector import PaprikaDetector

INFER_S = 0.03          # stands in for the forward pass; long enough to contend
FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeEngine:
    """One model behind one lock, as PaprikaDetector._detect_pose has it."""

    def __init__(self, fail: bool = False):
        self._model_lock = threading.Lock()
        self._record_lock = threading.Lock()
        self._fail = fail
        self.calls = []          # (thread name, monotonic start of inference)

    def is_ready(self):
        return True

    def evaluate(self, frame):
        if self._fail:
            raise RuntimeError("engine failed")
        start = time.perf_counter()
        with self._model_lock:
            acquired = time.perf_counter()
            with self._record_lock:
                self.calls.append((threading.current_thread().name, time.monotonic()))
            time.sleep(INFER_S)
        lock_wait_ms = (acquired - start) * 1000
        detection = {"bbox": [1, 1, 10, 10], "center": [5, 5], "placement": "place",
                     "orientation": {"pose": "lying"}}
        return {
            "status": "OK",
            "detections": [detection],
            "primary": detection,
            "processing_time_ms": (time.perf_counter() - start) * 1000,
            "timing": {"lock_wait_ms": lock_wait_ms},
        }

    def overlay_starts(self, after: float = 0.0):
        with self._record_lock:
            return [t for name, t in self.calls
                    if name == "overlay_worker" and t >= after]


class FakeCamera:
    def get_frame(self):
        return FRAME.copy()


@pytest.fixture
def rig(monkeypatch):
    """Start an overlay worker against a fake engine, wired into run_vision."""
    workers = []

    def build(pause_s: float, engine=None):
        engine = engine or FakeEngine()
        monkeypatch.setattr(ow, "cfg", {
            # Interval 0: the overlay inferences back to back, the worst case
            # for contention.
            "hmi": {"overlay_interval_s": 0.0,
                    "overlay_pause_after_trigger_s": pause_s},
            "debug": {},
        })
        app_state = AppState()
        monkeypatch.setattr(vision, "_engine", engine)
        monkeypatch.setattr(vision, "_app_state", app_state)
        worker = ow.OverlayWorker(FakeCamera(), app_state, engine)
        assert worker.start()
        workers.append(worker)
        _wait_for(lambda: engine.overlay_starts())
        return engine, app_state, worker

    yield build
    for worker in workers:
        worker.stop()


def _wait_for(condition, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached in time")


def trigger(timeout: float = 2.0) -> dict:
    """One triggered cycle through the real run_vision callback path."""
    done = threading.Event()
    box = []

    def callback(result):
        box.append(result)
        done.set()

    vision.run_vision(FRAME, callback=callback)
    assert done.wait(timeout), "trigger never completed"
    return box[0]


# ------------------------------------------------------------------- hold-off

def test_no_overlay_inference_inside_hold_off(rig):
    engine, app_state, worker = rig(pause_s=0.5)

    trigger()
    finished = time.monotonic()
    time.sleep(0.3)

    assert engine.overlay_starts(after=finished) == []
    assert worker.is_paused()
    assert worker.status()["paused"] is True
    # Nothing left over to draw: the preview from before the trigger is gone,
    # not merely ageing towards overlay_stale_after_s.
    snapshot = app_state.get_overlay(max_age_s=10.0, now=time.time())
    assert snapshot["stale"] and snapshot["detections"] == []


def test_overlay_resumes_after_hold_off(rig):
    pause_s = 0.3
    engine, app_state, worker = rig(pause_s=pause_s)

    trigger()
    trigger_end = app_state.last_trigger_end
    _wait_for(lambda: engine.overlay_starts(after=trigger_end), timeout=2.0)

    first = min(engine.overlay_starts(after=trigger_end))
    # Not before the hold-off ended (small allowance for clock granularity),
    # and not long after it either.
    assert first >= trigger_end + pause_s - 0.02
    assert first <= trigger_end + pause_s + 0.5
    assert not worker.is_paused()
    _wait_for(lambda: not app_state.get_overlay(now=time.time())["stale"])


def test_zero_restores_previous_behaviour(rig):
    """0 = the overlay keeps running and the trigger takes its turn on the lock."""
    engine, app_state, worker = rig(pause_s=0.0)

    started = time.monotonic()
    waits = [trigger()["timing"]["lock_wait_ms"] for _ in range(5)]

    assert not worker.is_paused()
    assert engine.overlay_starts(after=started), "overlay stopped running"
    # The contention this change exists to remove. With the overlay holding
    # the lock back to back, a trigger lands mid-inference almost every time;
    # all five landing in the gaps would be well under a 1-in-1000 event.
    assert max(waits) > 5.0


def test_trigger_never_waits_on_the_lock_while_paused(rig):
    engine, app_state, worker = rig(pause_s=1.0)

    # The first may land mid-preview: an overlay inference already running
    # cannot be pre-empted. Every one after it is inside the hold-off.
    trigger()
    waits = []
    for _ in range(5):
        time.sleep(0.05)
        waits.append(trigger()["timing"]["lock_wait_ms"])

    assert max(waits) < 5.0, waits


def test_trigger_in_flight_holds_overlay_even_before_it_finishes(rig):
    engine, app_state, worker = rig(pause_s=0.5)

    app_state.trigger_started()
    try:
        assert worker.is_paused()
        marked = time.monotonic()
        time.sleep(0.2)
        # One overlay inference may have been mid-flight at the mark; none
        # may START after it.
        assert engine.overlay_starts(after=marked + INFER_S) == []
    finally:
        app_state.trigger_finished()


def test_failed_trigger_does_not_leave_overlay_held(monkeypatch):
    app_state = AppState()
    monkeypatch.setattr(vision, "_engine", FakeEngine(fail=True))
    monkeypatch.setattr(vision, "_app_state", app_state)

    result = trigger()

    assert result["failure_reason"] == "vision_exception"
    assert app_state.triggers_in_flight == 0
    # Held for the pause after the failure, like any other trigger, then free.
    assert app_state.overlay_held(0.2)
    assert not app_state.overlay_held(0.2, now=time.monotonic() + 0.3)


def test_overlay_held_is_off_at_zero():
    app_state = AppState()
    app_state.trigger_started()
    assert not app_state.overlay_held(0.0)
    assert app_state.overlay_held(3.0)
    app_state.trigger_finished()


# --------------------------------------------------------------------- timing

TIMING_KEYS = {"backend", "lock_wait_ms", "inference_ms", "detect_ms",
               "post_ms", "total_ms"}


def _engine(backend: str) -> PaprikaEngine:
    return PaprikaEngine({"paprika": {"backend": backend}})


def test_timing_present_on_simulator_result():
    result = _engine("simulator").evaluate(np.zeros((480, 640, 3), np.uint8))

    timing = result["timing"]
    assert set(timing) == TIMING_KEYS
    assert timing["backend"] == "simulator"
    # No model, no lock: absent, not zero.
    assert timing["lock_wait_ms"] is None
    assert timing["inference_ms"] is None
    assert timing["detect_ms"] is not None
    assert timing["detect_ms"] + timing["post_ms"] == pytest.approx(
        timing["total_ms"], abs=0.02)
    # Additive: the existing key is still there, and is the same number.
    assert timing["total_ms"] == result["processing_time_ms"]


def test_timing_present_when_nothing_is_found():
    result = _engine("shape").evaluate(np.zeros((240, 320, 3), np.uint8))

    assert result["failure_reason"] == "no_paprika_detected"
    assert set(result["timing"]) == TIMING_KEYS
    assert result["timing"]["backend"] == "shape"
    assert result["timing"]["total_ms"] == result["processing_time_ms"]


def test_timing_present_on_no_frame():
    result = _engine("shape").evaluate(None)
    assert result["failure_reason"] == "no_frame"
    assert set(result["timing"]) == TIMING_KEYS
    assert result["timing"]["total_ms"] == result["processing_time_ms"] == 0.0


class StubModel:
    """Enough of an ultralytics model for _detect_pose: no boxes found."""

    def predict(self, frame, **kwargs):
        time.sleep(0.005)
        return []


def test_pose_timing_measures_the_lock_wait():
    """The measurement itself: a held model lock shows up as lock_wait_ms."""
    detector = PaprikaDetector({"backend": "pose"})
    detector._model = StubModel()
    timing = {}

    holder_has_lock = threading.Event()

    def hold():
        with detector._model_lock:
            holder_has_lock.set()
            time.sleep(0.08)

    holder = threading.Thread(target=hold)
    holder.start()
    holder_has_lock.wait(1.0)
    detector.detect(FRAME, timing=timing)
    holder.join()

    assert timing["lock_wait_ms"] >= 50.0
    assert timing["inference_ms"] >= 4.0
    assert timing["detect_ms"] >= timing["lock_wait_ms"] + timing["inference_ms"] - 0.1


def test_pose_engine_result_carries_inference_split():
    engine = _engine("pose")
    engine._detector._model = StubModel()

    result = engine.evaluate(FRAME)
    timing = result["timing"]

    assert timing["backend"] == "pose"
    assert timing["lock_wait_ms"] is not None and timing["lock_wait_ms"] < 5.0
    assert timing["inference_ms"] >= 4.0
    assert timing["post_ms"] is not None
    assert timing["total_ms"] == result["processing_time_ms"]
