"""
Keep the frame in front of you, on a keypress

When a fruit is judged wrongly on the line, the thing worth keeping is the
frame that produced it. By the time anyone reaches a terminal the belt has
moved on and that frame is gone, so the failure gets described from memory
instead of reproduced - and a failure that cannot be reproduced cannot be
fixed. This puts it one key away.

What it saves, and why in this form
-----------------------------------
The RAW frame, exactly as the detector received it: no overlay, no annotation.
An overlaid frame is a picture of an answer and cannot be re-run, re-labelled
or trained on. The overlay is worth keeping too, so it goes in the sidecar JSON
as data rather than being burnt into the pixels.

PNG, not JPEG, and that is measured rather than assumed. These frames are an
input to segmentation and to a model, and re-encoding moves the answer: over
ten fruit, JPEG at quality 95 shifted the reported stem angle by up to 4.1
degrees and at quality 80 by 8.1, while PNG shifted it by exactly zero on every
one. Against a max_stem_spread_deg of 6.0 that is not a rounding difference.
About 1.2 MB a frame against 160 KB is a fair price for training data.

The filename carries the verdict - `20260918_141530_123_place_lying.png` - so
the interesting ones can be found without opening the sidecars.

Writing happens on a background thread behind a bounded queue. Saving a 1.2 MB
PNG takes tens of milliseconds and this runs on a machine vision loop; more to
the point, a disk that stalls would otherwise stall whatever called it. When
the queue is full the frame is dropped and counted, because missing one capture
costs nothing and blocking the line costs what the thread was built to avoid.

Feeding them back
-----------------
    python -m tools.label_stems data/captures
    python -m tools.label_stems data/captures --blossom
    python -m tools.export_dataset data/captures --out data/datasets/paprika_v2

Captures land in their own folder rather than in data/raw, so a batch collected
to fix one problem stays identifiable. Point the labeller at it directly.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2

from backend.utils.logger import get_logger
from backend.utils.paths import project_path

log = get_logger(__name__)


class ManualCapture:
    """Saves the current frame when a hotkey is pressed.

    Args:
        frame_provider:  returns the current raw frame, or None. Normally
                         `camera.get_frame`.
        result_provider: returns the latest verdict as a dict, or None. Used
                         only to name the file and fill the sidecar, so a
                         capture still works when nothing has been evaluated -
                         a fruit the detector missed entirely is exactly the
                         kind worth keeping, and it has no verdict at all.
        block:           the config block, normally cfg["capture"].
    """

    def __init__(
        self,
        frame_provider: Callable[[], object],
        result_provider: Optional[Callable[[], Optional[dict]]] = None,
        block: Optional[dict] = None,
    ) -> None:
        block = block or {}
        self._frame_provider = frame_provider
        self._result_provider = result_provider

        self.enabled = bool(block.get("enabled", True))
        self.hotkey = str(block.get("hotkey", "c"))
        self._dir = project_path(block.get("directory"), "data/captures")
        self._save_json = bool(block.get("save_result_json", True))
        fmt = str(block.get("image_format", "png")).strip().lower().lstrip(".")
        self._suffix = "jpg" if fmt in ("jpg", "jpeg") else "png"
        self._jpeg_quality = int(block.get("jpeg_quality", 95))
        # Two presses inside this window count as one. A keyboard hotkey
        # repeats while held and an operator reaching across a machine will
        # not press cleanly, so without this a single intent writes six copies
        # of the same frame.
        self._min_interval_s = float(block.get("min_interval_s", 0.75))

        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._hook = None
        self._lock = threading.Lock()
        self._last_press = 0.0
        self._saved = 0
        self._dropped = 0
        self._errors = 0

    # ------------------------------------------------------------- lifecycle

    def start(self) -> bool:
        """Register the hotkey and start the writer. Never raises."""
        if not self.enabled:
            return False
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.error("Manual capture disabled - cannot create %s: %s", self._dir, exc)
            self.enabled = False
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="manual_capture", daemon=True
        )
        self._thread.start()

        try:
            import keyboard

            self._hook = keyboard.add_hotkey(self.hotkey, self.capture)
        except Exception as exc:
            # A missing hotkey must not take the line down. capture() still
            # works, so the HMI or a test can call it directly.
            log.error(
                "Manual capture: could not bind hotkey '%s' (%s). "
                "Capturing is still available programmatically.",
                self.hotkey, exc,
            )
            return False

        log.info(
            "Manual capture ready: press '%s' to save the current frame to %s",
            self.hotkey, self._dir,
        )
        return True

    def stop(self) -> None:
        if self._hook is not None:
            try:
                import keyboard

                keyboard.remove_hotkey(self._hook)
            except Exception:
                pass
            self._hook = None
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        if self._saved or self._dropped or self._errors:
            log.info(
                "Manual capture: %d saved, %d dropped, %d failed",
                self._saved, self._dropped, self._errors,
            )

    # --------------------------------------------------------------- capture

    def capture(self, note: str = "") -> bool:
        """Queue the current frame. Safe to call from a key handler.

        Returns True if it was accepted. Cheap and never raises: this runs on
        the keyboard thread and an exception there would kill the hotkey for
        the rest of the session, silently.
        """
        try:
            if not self.enabled:
                return False

            now = time.time()
            with self._lock:
                if now - self._last_press < self._min_interval_s:
                    return False
                self._last_press = now

            frame = self._frame_provider()
            if frame is None:
                log.warning("Manual capture: no frame available right now")
                return False

            result = None
            if self._result_provider is not None:
                try:
                    result = self._result_provider()
                except Exception:
                    result = None

            primary = (result or {}).get("primary") or {}
            placement = str(primary.get("placement") or "none")
            pose = str((primary.get("orientation") or {}).get("pose") or "none")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            stem = f"{stamp}_{placement}_{pose}"

            # A copy, because the provider may hand back the buffer the capture
            # thread is about to overwrite.
            try:
                self._queue.put_nowait((stem, frame.copy(), result, note))
            except queue.Full:
                with self._lock:
                    self._dropped += 1
                log.warning("Manual capture: writer busy, frame dropped")
                return False
            log.info("Manual capture: queued %s", stem)
            return True
        except Exception as exc:
            log.error("Manual capture failed: %s", exc)
            return False

    # --------------------------------------------------------------- writing

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            stem, frame, result, note = item
            try:
                self._write(stem, frame, result, note)
                with self._lock:
                    self._saved += 1
                log.info("Manual capture: saved %s.%s", stem, self._suffix)
            except Exception as exc:
                with self._lock:
                    self._errors += 1
                # Never let a bad write kill the thread - the next frame may
                # well succeed, and a silently dead writer is worse than a
                # logged failure.
                log.error("Manual capture: write failed (%s): %s", stem, exc)

    def _write(self, stem: str, frame, result: Optional[dict], note: str) -> None:
        path = self._dir / f"{stem}.{self._suffix}"
        params = []
        if self._suffix == "jpg":
            params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        if not cv2.imwrite(str(path), frame, params):
            raise OSError(f"cv2.imwrite returned False for {path}")

        if self._save_json:
            sidecar = {
                "captured_at": datetime.now().isoformat(timespec="seconds"),
                "image": path.name,
                "note": note,
                "result": result,
            }
            # default=str so an unexpected numpy scalar in the result can never
            # cost us the frame that goes with it.
            (self._dir / f"{stem}.json").write_text(
                json.dumps(sidecar, indent=2, default=str), encoding="utf-8"
            )

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "hotkey": self.hotkey,
            "dir": str(self._dir),
            "saved": self._saved,
            "dropped": self._dropped,
            "errors": self._errors,
        }
