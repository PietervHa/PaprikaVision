"""
Overlay Worker

Continuously evaluates the camera feed so the HMI can draw a live orientation
overlay, independent of PLC-triggered inspections.

Why this is a separate thread and not part of the frame generator
-----------------------------------------------------------------
The MJPEG generator is pinned to ~30 fps. A pose inference is 20-80 ms on CPU
and the shape backend is a few ms, but neither is free, and running detection
inline would couple the video frame rate to inference latency - so the live
view would visibly stutter whenever the model slowed down. Instead this thread
scans on its own clock and the stream redraws the most recent completed result
at full speed.

The overlay is display-only. It never increments a counter, never writes a
result, and never reaches the PLC. That path stays exclusively driven by the
trigger, so what the operator sees can never be mistaken for what the machine
acted on.

Failure capture
---------------
Optionally (debug.save_failures, off by default) this thread also drops the RAW
frame behind any non-placeable verdict into data/debug/failures/. That is the
one artefact you cannot reconstruct afterwards: an HMI screenshot shows the
verdict, but tools/diagnose_frames.py needs the input that produced it. Saving
here rather than in the trigger path is deliberate - failures are noticed by
somebody watching the live view, and this thread is what they are watching.

It stays display-only in the sense that matters: it writes to data/debug and
nowhere else, touches no counter, and cannot reach the PLC.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from typing import Optional

import cv2

from backend.core.config_loader import cfg
from backend.utils.logger import get_logger
from backend.utils.paths import project_path

log = get_logger(__name__)



class _FailureCapture:
    """Writes raw frames of non-placeable verdicts, off the overlay thread.

    Encoding and writing a frame is milliseconds, but this runs on a machine
    vision loop and milliseconds are the currency. More to the point, a disk
    that stalls - a full volume, a network share, a virus scanner waking up -
    would stall the overlay with it. So the overlay thread only ever hands a
    frame to a bounded queue and moves on; a writer thread does the work, and
    when the queue is full the frame is DROPPED rather than waited on.

    Dropping is the right failure mode here. This is a diagnostic aid: missing
    one frame out of a run costs nothing, while blocking the loop to keep it
    costs exactly what the separate thread was built to avoid. Drops are
    counted and reported in status() so a suspiciously empty folder has an
    explanation.

    Three limits keep it from filling the disk, and the second one matters more
    than it looks: a belt that stops with one rejected fruit under the camera
    produces the same failure at the overlay rate, forever. Without a minimum
    interval you would come back to forty thousand copies of one pepper.
    """

    def __init__(self, block: dict) -> None:
        self.enabled = bool(block.get("save_failures", False))
        self._dir = project_path(block.get("failures_dir"), "data/debug/failures")
        self._max_per_run = int(block.get("max_failures_per_run", 200))
        self._min_interval_s = float(block.get("min_interval_s", 2.0))
        self._save_json = bool(block.get("save_result_json", True))
        # PNG by default, and measured rather than assumed. These frames are an
        # INPUT to segmentation, and re-encoding moves the answer: across ten
        # fruit, JPEG at quality 95 shifted the reported stem angle by up to
        # 4.1 degrees and at quality 80 by 8.1, while PNG shifted it by exactly
        # zero on every one. Against a max_stem_spread_deg of 6.0 that is not a
        # rounding difference - a frame saved as JPEG can cross the stability
        # threshold purely by being saved, which would have you investigating
        # the compression instead of the fault. Roughly 1.2 MB a frame against
        # 160 KB is a fair price for a diagnostic that is off by default and
        # capped at max_failures_per_run.
        fmt = str(block.get("image_format", "png")).strip().lower().lstrip(".")
        self._suffix = "jpg" if fmt in ("jpg", "jpeg") else "png"
        self._jpeg_quality = int(block.get("jpeg_quality", 95))

        # Four is enough to ride out a brief disk hiccup and small enough that
        # a sustained one is noticed as drops instead of hidden as latency.
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_save = 0.0
        # Four counters rather than one, because they fail differently and the
        # difference is the whole diagnosis. `accepted` is what passed the rate
        # limit and the cap; `written` is what actually reached the disk. An
        # earlier version incremented one counter at accept time and called it
        # "written", so a run whose every write threw still reported success -
        # precisely the reassurance a diagnostic must never give.
        self._accepted = 0
        self._written = 0
        self._dropped = 0
        self._errors = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if not self.enabled or (self._thread is not None and self._thread.is_alive()):
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # A debug aid must never stop the machine from running.
            log.error("Failure capture disabled - cannot create %s: %s", self._dir, exc)
            self.enabled = False
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="failure_capture", daemon=True
        )
        self._thread.start()
        log.info(
            "Failure capture on -> %s (max %d per run, min %.1fs apart)",
            self._dir, self._max_per_run, self._min_interval_s,
        )

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        if self._accepted or self._dropped or self._errors:
            log.info(
                "Failure capture: %d frame(s) written to %s, %d dropped, %d failed",
                self._written, self._dir, self._dropped, self._errors,
            )

    # ------------------------------------------------------------- producing

    @staticmethod
    def _first_failure(result: dict) -> Optional[dict]:
        """The detection that makes this frame worth keeping, or None.

        Frames with no detections at all are deliberately NOT captured. An
        empty belt is the normal state, so treating "found nothing" as a
        failure would save the whole shift. A fruit that was missed entirely is
        a real failure and this will not catch it - that one needs a frame
        saved by hand.
        """
        primary = result.get("primary")
        if isinstance(primary, dict) and primary.get("placement") not in (None, "place"):
            return primary
        for detection in result.get("detections") or []:
            if detection.get("placement") != "place":
                return detection
        return None

    def offer(self, frame, result: dict) -> None:
        """Called on the overlay thread. Must stay cheap and must never raise."""
        if not self.enabled or frame is None:
            return
        failure = self._first_failure(result)
        if failure is None:
            return

        now = time.time()
        with self._lock:
            # Capped on what was accepted, not on what landed: a failing disk
            # must not be allowed to retry forever behind a cap that never
            # advances.
            if self._accepted >= self._max_per_run:
                return
            if now - self._last_save < self._min_interval_s:
                return
            # Claimed before the write happens: two cycles must not both slip
            # past the interval check while the first is still queued.
            self._last_save = now
            self._accepted += 1

        placement = str(failure.get("placement") or "unknown")
        pose = str((failure.get("orientation") or {}).get("pose") or "none")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        stem = f"{stamp}_{placement}_{pose}"

        try:
            self._queue.put_nowait((stem, frame, result))
        except queue.Full:
            with self._lock:
                self._accepted -= 1
                self._dropped += 1

    # --------------------------------------------------------------- writing

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            stem, frame, result = item
            try:
                self._write(stem, frame, result)
                with self._lock:
                    self._written += 1
            except Exception as exc:
                with self._lock:
                    self._errors += 1
                # Never let a write failure kill the thread: the next frame may
                # well succeed, and a silently dead capture thread is worse
                # than a logged failed write.
                log.error("Failure capture write failed (%s): %s", stem, exc)

    def _write(self, stem: str, frame, result: dict) -> None:
        path = self._dir / f"{stem}.{self._suffix}"
        params = []
        if self._suffix == "jpg":
            # Only reached when image_format is set to jpg explicitly. Quality
            # stays high because the alternative is measurably worse - see the
            # note on _suffix above.
            params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        if not cv2.imwrite(str(path), frame, params):
            raise OSError(f"cv2.imwrite returned False for {path}")

        if self._save_json:
            # default=str so an unexpected numpy scalar or dataclass in the
            # result can never cost us the frame that goes with it.
            (self._dir / f"{stem}.json").write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            # Absolute, so "where are my frames" is answerable from /status
            # without knowing what directory the process was launched from.
            "dir": str(self._dir),
            "accepted": self._accepted,
            "written": self._written,
            "dropped": self._dropped,
            "errors": self._errors,
            "max_per_run": self._max_per_run,
        }


class OverlayWorker:
    def __init__(self, camera, app_state, engine) -> None:
        self._camera = camera
        self._app_state = app_state
        self._engine = engine

        hmi_cfg = cfg.get("hmi") if isinstance(cfg.get("hmi"), dict) else {}
        self._interval_s = max(0.0, float(hmi_cfg.get("overlay_interval_s", 0.15)))

        debug_cfg = cfg.get("debug") if isinstance(cfg.get("debug"), dict) else {}
        self._capture = _FailureCapture(debug_cfg)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_error = ""

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.is_running():
            return True
        if self._engine is None or not self._engine.is_ready():
            log.warning("Overlay not started: engine is not ready")
            return False

        self._stop_event.clear()
        self._capture.start()
        self._thread = threading.Thread(target=self._run, name="overlay_worker", daemon=True)
        self._thread.start()
        log.info("Overlay worker started (interval %.2fs)", self._interval_s)
        return True

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # Bounded join - the worker may be mid-inference, and blocking a web
            # handler until that finishes would hang the HMI.
            thread.join(timeout=2.0)
        self._thread = None
        self._capture.stop()
        self._app_state.clear_overlay()
        log.info("Overlay worker stopped")

    def toggle(self) -> bool:
        if self.is_running():
            self.stop()
            return False
        return self.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            cycle_start = time.perf_counter()
            try:
                frame = self._camera.get_frame()
                if frame is None:
                    self._stop_event.wait(0.2)
                    continue

                result = self._engine.evaluate(frame)
                # Before the overlay is published, and on the untouched frame:
                # nothing here draws on it, and the annotated copy the HMI
                # builds is useless as an input to re-run detection on.
                self._capture.offer(frame, result)
                self._app_state.set_overlay(
                    detections=result.get("detections") or [],
                    primary=result.get("primary"),
                    scan_ms=float(result.get("processing_time_ms", 0.0)),
                    ts=time.time(),
                )
                self._last_error = ""
            except Exception as exc:
                self._last_error = str(exc)
                log.error("Overlay cycle failed: %s", exc)
                self._stop_event.wait(0.5)
                continue

            remaining = self._interval_s - (time.perf_counter() - cycle_start)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def status(self) -> dict:
        return {
            "running": self.is_running(),
            "interval_s": self._interval_s,
            "error": self._last_error,
            "failure_capture": self._capture.status(),
        }