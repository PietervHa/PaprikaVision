"""
Camera Management Module

Provides the Camera class for asynchronous frame capture, supporting various
rotations and flipping based on configuration and application state.

Recovery behaviour
-------------------
A camera can fail in three different ways when it disappears mid-capture,
and each needs its own detection strategy:

  1. read() returns (False, None) - the easy case.
  2. read() blocks forever (observed with DirectShow on Windows once the
     USB device vanishes) - a strategy that only reacts *after* read()
     returns never runs in this case.
  3. read() keeps returning (True, <same frame as before>) - some
     DirectShow drivers keep serving the last buffered frame instead of
     failing once the device is gone. A strategy that only checks "did we
     get *a* frame recently" is blind to this, since frames keep "arriving".

To handle all three, capture is split into two parts:

  - _CaptureWorker: a small dedicated thread that does nothing but call
    cap.read() in a loop and publish the latest successful frame, the time
    it arrived, and whether the content has ever been *observed to change*
    (case 3). If its read() call hangs (case 2), this thread simply sits
    there - it is never joined or waited on.

  - Camera._update (the supervisor): watches these signals from the
    *current* worker. If no frame has arrived recently (case 1/2), or
    frames are arriving but their content hasn't changed in a while
    (case 3), the camera is considered stalled/disconnected. The
    supervisor then opens a brand new VideoCapture and spins up a brand
    new worker for it, and simply abandons the old (possibly still
    blocked) worker as a daemon thread - it is never relied upon again.
    Reconnect attempts use a short exponential backoff so a camera that
    comes back quickly is picked up almost instantly, while a camera
    that stays absent doesn't get hammered with open attempts. Every
    (re)started worker gets a STALE_TIMEOUT grace period to deliver its
    first frame before it's judged stalled - without this, a worker that
    simply hasn't warmed up yet (normal autoexposure/init delay) would be
    torn down and replaced before it ever got a chance to work.

A subtlety with case 3: on some Windows/DirectShow setups, when the
physical device is gone, cv2.VideoCapture(index).isOpened() still returns
True and the very *first* read() after opening succeeds too (serving one
cached/placeholder frame), even though nothing further will ever change.
If "connected" were set the moment a single frame arrives, this produces
a false "Camera recovered" every reconnect cycle, followed ~FROZEN_TIMEOUT
seconds later by another "stalled" - an endless open/close loop that never
reflects reality. To guard against this, a worker is only allowed to be
treated as healthy once it has been *observed* to deliver two genuinely
different frames (a real sensor's per-pixel noise makes two bit-identical
frames from a live feed essentially impossible, so this costs at most a
frame or two of extra latency on a real camera, while a driver replaying
one static buffer can never satisfy it).

The frozen-frame check (case 3) compares raw frame content byte-for-byte.
A live sensor essentially never produces two bit-identical frames in a
row (sensor noise alone makes that astronomically unlikely), even when
pointed at a completely static scene, so this is a safe signal - it only
fires when the driver is truly re-serving the same buffer. FROZEN_TIMEOUT
is set well above STALE_TIMEOUT to give a wide margin.

Device selection
-----------------
On Windows, opening a camera by plain numeric index (cv2.VideoCapture(0))
is fragile: when the configured USB camera is unplugged, the OS renumbers
the remaining devices, so index 0 can silently start pointing at a
completely different camera (e.g. a laptop's built-in webcam). That
webcam is a real, live camera, so it passes every health check above and
gets reported as "connected" - just the wrong one.

To avoid this, when camera.device_name is set, _resolve_camera_index()
re-enumerates the currently connected cameras by friendly name (via
pygrabber, Windows-only) on *every* (re)connect attempt and only opens the
device whose name matches. If no matching device is currently connected,
no capture is opened at all - the app correctly waits, instead of
grabbing whatever else happens to be sitting at index 0.

Trade-off: because a thread blocked inside a C-level call can't be forced
to stop from Python, an abandoned worker may leak until its read() call
eventually returns or the process exits. worker.stop() does a best-effort
cap.release() to try to unblock it, which works on most platforms/backends
but isn't guaranteed. This is still far better than the whole app being
unable to recover at all.
"""

import cv2
import numpy as np
import threading
import time
import sys
from backend.core.config_loader import cfg
from backend.utils.logger import get_logger

log = get_logger(__name__)
TARGET = 1 / 30  # Target time per frame for ~30 FPS

# Cached across calls so a missing pygrabber install (or a lookup failure)
# is only logged once instead of spamming every reconnect attempt.
_device_name_lookup_warned = False

# How long we tolerate zero fresh frames before treating the camera as
# stalled/disconnected. Set comfortably above the sub-second read hiccups
# seen in practice (single slow frame, self-recovers) so those don't
# trigger needless reconnects, while still reacting quickly to a real
# disconnect.
STALE_TIMEOUT = 1.5

# How long we tolerate the frame content staying byte-for-byte identical
# before treating that as a stall too (driver serving a cached last frame
# instead of failing). Kept well above STALE_TIMEOUT: it only needs to be
# long enough that no normal camera would ever legitimately sit there,
# not short enough to react instantly.
FROZEN_TIMEOUT = 4.0

# Backoff bounds (in seconds) between reopen attempts while the camera is
# down. Starts fast so a quick replug is picked up almost immediately, and
# backs off so a camera that stays unplugged doesn't get hammered with
# open attempts.
RECONNECT_MIN_INTERVAL = 0.25
RECONNECT_MAX_INTERVAL = 2.0


def _resolve_camera_index(cam_cfg):
    """
    Decides which numeric camera index to open this attempt.

    If cam_cfg["device_name"] is set (Windows only), the current list of
    connected camera devices is re-enumerated by name on *every* call -
    indices are not cached, because they can shift as devices are plugged
    and unplugged. Returns the index of the device whose friendly name
    contains device_name (case-insensitive).

    If no currently-connected device matches, returns None. Callers must
    treat None as "don't open anything" rather than falling back to a
    bare numeric index - falling back is exactly what lets an unrelated
    camera (e.g. a laptop's built-in webcam) get silently opened once the
    real device disappears and indices renumber.

    If device_name isn't set, or we're not on Windows, or the lookup
    itself fails (e.g. pygrabber not installed), falls back to the plain
    numeric cam_cfg["index"] - the old, simple behaviour.
    """
    global _device_name_lookup_warned
    device_name = (cam_cfg.get("device_name") or "").strip()
    if not device_name or sys.platform != "win32":
        return cam_cfg["index"]

    try:
        from pygrabber.dshow_graph import FilterGraph
        devices = FilterGraph().get_input_devices()
    except Exception as exc:
        if not _device_name_lookup_warned:
            log.error(
                "Could not enumerate camera devices to resolve "
                "camera.device_name=%r (%s). Falling back to numeric "
                "camera.index=%s, which may open the wrong camera (e.g. a "
                "laptop's built-in webcam) if the USB camera renumbers. "
                "Install pygrabber to fix this: pip install pygrabber",
                device_name, exc, cam_cfg["index"],
            )
            _device_name_lookup_warned = True
        return cam_cfg["index"]

    for idx, name in enumerate(devices):
        if device_name.lower() in name.lower():
            return idx

    return None


# Properties applied by _apply_manual_settings, in the order they're set.
# Order matters: auto-exposure must be switched off *before* a manual
# exposure value will stick, and the same goes for auto-white-balance
# before a manual white balance temperature.
_MANUAL_PROPS = (
    # (config key, cv2 property, human label)
    ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, "auto_exposure"),
    ("exposure", cv2.CAP_PROP_EXPOSURE, "exposure"),
    ("gain", cv2.CAP_PROP_GAIN, "gain"),
    ("auto_white_balance", cv2.CAP_PROP_AUTO_WB, "auto_white_balance"),
    ("white_balance", cv2.CAP_PROP_WB_TEMPERATURE, "white_balance"),
    ("brightness", cv2.CAP_PROP_BRIGHTNESS, "brightness"),
    ("contrast", cv2.CAP_PROP_CONTRAST, "contrast"),
)


def _apply_manual_settings(cap, cam_cfg):
    """Pin exposure/gain/white-balance to fixed values so the driver's
    auto-exposure can't drift between (or during) runs.

    Why this exists: with auto-exposure free-running, the same physical
    scene produces different pixel values from one cycle to the next. That
    feeds straight into the blackhat threshold in text_locator.py (an
    absolute intensity cutoff) and into recognition confidence, so OCR
    accuracy moves for reasons that have nothing to do with the product or
    the code. Measured drift on this rig has been large enough to swamp the
    code changes being A/B tested - accuracy moved from 92.5% to 75% within
    a single uninterrupted session on one setup - which makes locking these
    a prerequisite for any meaningful before/after comparison, not just a
    nice-to-have.

    Every setting is optional: a key left out of config (or set to null) is
    simply not touched, so this can be adopted one property at a time.

    IMPORTANT - the values are NOT portable. cv2.CAP_PROP_AUTO_EXPOSURE in
    particular has no agreed meaning across backends: DirectShow, V4L2 and
    individual driver builds each interpret it differently (0/1 vs
    0.25/0.75 vs 1/3 all occur in the wild). That's why the "off" value is
    config-driven rather than hardcoded, and why this function reads every
    property back after writing it and logs both numbers. A silent mismatch
    between requested and actual is the normal failure mode here - OpenCV
    returns True from cap.set() for properties the driver quietly ignores -
    so the read-back log is the only reliable evidence that a setting
    actually took.
    """
    manual_cfg = cam_cfg.get("manual_settings") or {}
    if not manual_cfg.get("enabled", False):
        log.debug(
            "Camera manual_settings disabled - exposure/gain left on driver auto. "
            "Expect frame-to-frame brightness drift; see camera.manual_settings in config."
        )
        return

    applied = []
    mismatched = []

    for key, prop, label in _MANUAL_PROPS:
        requested = manual_cfg.get(key)
        if requested is None:
            continue
        try:
            cap.set(prop, float(requested))
            actual = cap.get(prop)
        except Exception as exc:
            log.warning("Camera setting %s could not be applied: %s", label, exc)
            continue

        # Compare loosely - drivers routinely quantise a requested value to
        # whatever step they actually support, which is fine. What matters
        # is catching the case where the value didn't move at all.
        if abs(actual - float(requested)) > max(1e-3, abs(float(requested)) * 0.25):
            mismatched.append(f"{label}: requested={requested} actual={actual}")
        else:
            applied.append(f"{label}={actual}")

    if applied:
        log.info("Camera manual settings applied: %s", ", ".join(applied))
    if mismatched:
        log.warning(
            "Camera manual settings NOT honoured by the driver (requested vs actual): %s. "
            "This usually means the property is unsupported on this backend, or that "
            "camera.manual_settings.auto_exposure needs a different value for this driver "
            "(common alternatives: 0.25, 0, 1, 3). Exposure may still be drifting.",
            "; ".join(mismatched),
        )


class _CaptureWorker(threading.Thread):
    """
    Owns exactly one cv2.VideoCapture instance and does nothing but read
    from it in a loop, publishing (frame, timestamp, content_changed_at,
    verified_live) to whoever asks via latest(). Deliberately dumb: it has
    no opinion about reconnecting - if its read() call hangs (device
    disappeared mid-capture), this thread just sits there. The supervisor
    is the one that decides when to stop trusting a worker and start a
    fresh one.
    """
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        # When this worker was created - used by the supervisor to grant a
        # startup grace period before its first frame has to have arrived.
        self.started_at = time.monotonic()
        self._stop_requested = False
        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0
        self._last_raw_frame = None
        self._content_changed_at = self.started_at
        # Only flips to True once we've seen the frame content genuinely
        # change at least once. A worker whose device is actually gone but
        # whose driver keeps re-serving one cached frame will never reach
        # this - which is exactly the point (see module docstring).
        self._verified_live = False

    def run(self):
        while not self._stop_requested:
            t0 = time.perf_counter()
            try:
                ret, frame = self.cap.read()
            except Exception as exc:
                log.error("Camera read raised an exception: %s", exc, exc_info=True)
                ret, frame = False, None

            if ret and frame is not None:
                now = time.monotonic()
                with self._lock:
                    # Only the *second and later* frames can prove the feed
                    # is actually live - the first frame is just a baseline
                    # to compare against, not evidence of anything by itself.
                    if self._last_raw_frame is not None and not np.array_equal(frame, self._last_raw_frame):
                        self._content_changed_at = now
                        self._verified_live = True
                    self._last_raw_frame = frame
                    self._frame = frame
                    self._timestamp = now

            # If read() blocked (e.g. device gone), this sleep is simply
            # never reached until it unblocks - that's fine, the
            # supervisor isn't waiting on this thread.
            elapsed = time.perf_counter() - t0
            remaining = TARGET - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def latest(self):
        """
        Thread-safe read of (frame, timestamp of last successful read,
        timestamp the frame content last actually changed, whether the
        feed has ever been observed to change at all).
        """
        with self._lock:
            return self._frame, self._timestamp, self._content_changed_at, self._verified_live

    def stop(self):
        """
        Best-effort shutdown. If this worker's read() call is currently
        blocked, releasing the capture from another thread will often (but
        not always, depending on backend/OS) cause it to unblock with an
        error. Either way, we don't wait for it.
        """
        self._stop_requested = True
        try:
            self.cap.release()
        except Exception as exc:
            log.warning("Error releasing superseded camera handle: %s", exc)


class Camera:
    """
    Asynchronous camera interface that captures frames in a background thread.
    """
    def __init__(self, index=0, app_state=None):
        self.app_state = app_state
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self._flip = cfg["camera"]["flip"]

        # Connection health, readable from any thread (plain bool reads/
        # writes are atomic under the GIL, so no extra lock is needed here).
        self.connected = False
        # Set from any thread to request an immediate reconnect attempt.
        self._force_reconnect = threading.Event()
        # Guards against stacking multiple concurrent reconnect attempts.
        self._spawning = threading.Event()

        self.cap = self._open_new_capture()
        self._worker = _CaptureWorker(self.cap)
        self._worker.start()

        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _open_new_capture(self):
        """
        Opens a brand new VideoCapture using the configured index/resolution
        (or, when camera.device_name is set, the specific USB camera
        resolved by name - see _resolve_camera_index). Does not touch any
        existing capture/worker - callers own that.
        """
        cam_cfg = cfg["camera"]
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

        resolved_index = _resolve_camera_index(cam_cfg)
        if resolved_index is None:
            # The named USB camera isn't among the currently connected
            # devices. Deliberately return an unopened VideoCapture instead
            # of falling back to a numeric index - opening *something* here
            # is exactly how a laptop's built-in webcam gets grabbed by
            # accident once the USB camera disappears and indices shift.
            log.warning(
                "USB camera device_name=%r not found among connected "
                "cameras; waiting for it to be plugged in.",
                cam_cfg.get("device_name", ""),
            )
            return cv2.VideoCapture()

        cap = cv2.VideoCapture(resolved_index, backend)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

        if cap.isOpened():
            # Applied after the capture is open (properties don't stick on a
            # closed device) and re-applied on every reopen, since a
            # reconnect resets the driver back to its own defaults - a lock
            # that silently lapses after a camera hiccup would be worse than
            # no lock at all, because the drift would look like a code
            # regression.
            _apply_manual_settings(cap, cam_cfg)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(
                "Camera opened successfully: index=%s resolution=%sx%s",
                resolved_index,
                width,
                height,
            )
        else:
            log.error("Camera failed to open: index=%s", resolved_index)

        return cap

    def _start_new_worker(self):
        """
        Kicks off opening a fresh capture + worker on a disposable helper
        thread, so that even a hanging VideoCapture *constructor* can never
        block the supervisor loop. Skips if a spawn is already in flight.
        """
        if self._spawning.is_set():
            return
        self._spawning.set()
        threading.Thread(target=self._spawn_worker, daemon=True).start()

    def _spawn_worker(self):
        try:
            # IMPORTANT:
            # Release the previous capture BEFORE opening the replacement.
            #
            # Opening a second VideoCapture while the old worker can still be
            # inside cap.read() may cause two native capture handles to compete
            # for the same DirectShow camera. This can result in:
            #
            #   isOpened() == True
            #   -> no real frames
            #   -> repeated reconnects
            #   -> OpenCV native/C++ exception
            #
            old_worker = self._worker

            # Detach the old worker from the supervisor first so no other
            # supervisor iteration can treat it as the active worker.
            self._worker = None
            self.cap = None

            if old_worker is not None:
                old_worker.stop()

            # Only after the old capture has been released do we open another one.
            cap = self._open_new_capture()

            worker = _CaptureWorker(cap)
            worker.start()

            self._worker = worker
            self.cap = cap

        except Exception as exc:
            log.error(
                "Camera reconnect attempt failed: %s",
                exc,
                exc_info=True,
            )
        finally:
            self._spawning.clear()

    def _update(self):
        """
        Supervisor loop. Never touches a VideoCapture directly - it only
        reads timestamps published by the current worker and decides
        whether to replace that worker. This means it can never itself get
        stuck, regardless of what the underlying camera/driver does.
        """
        backoff = RECONNECT_MIN_INTERVAL
        next_reconnect_attempt = 0.0
        stalled_logged = False

        while self.running:
            t0 = time.perf_counter()

            if self._force_reconnect.is_set():
                self._force_reconnect.clear()
                self._start_new_worker()
                backoff = RECONNECT_MIN_INTERVAL
                next_reconnect_attempt = 0.0

            worker = self._worker
            now = time.monotonic()
            if worker is not None:
                frame, ts, content_changed_at, verified_live = worker.latest()
                worker_age = now - worker.started_at
            else:
                frame, ts, content_changed_at, verified_live = None, 0.0, 0.0, False
                worker_age = float("inf")

            has_recent_read = frame is not None and (now - ts) < STALE_TIMEOUT
            not_frozen = (now - content_changed_at) < FROZEN_TIMEOUT

            # verified_live is the key guard against case 3 (see module
            # docstring): a worker whose device is actually gone but whose
            # driver keeps re-serving one cached frame will have
            # has_recent_read=True forever, but will never earn
            # verified_live, so it can never be reported as healthy here.
            if has_recent_read and not_frozen and verified_live:
                # Healthy: current worker has delivered a recent, genuinely
                # changing frame.
                if not self.connected:
                    log.info("Camera recovered")
                self.connected = True
                stalled_logged = False
                backoff = RECONNECT_MIN_INTERVAL
                next_reconnect_attempt = 0.0
                try:
                    # Flip the frame (1 = horizontal, 0 = vertical, -1 = both)
                    processed = cv2.flip(frame, self._flip)
                    rotation = self.app_state.get_camera_rotation() if self.app_state else 0
                    if rotation == 1:
                        processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation == 2:
                        processed = cv2.rotate(processed, cv2.ROTATE_180)
                    elif rotation == 3:
                        processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    # rotation == 0: no rotation
                    with self.lock:
                        self.latest_frame = processed
                except Exception as exc:
                    log.error("Frame post-processing failed: %s", exc, exc_info=True)

            elif worker_age < STALE_TIMEOUT:
                # Worker is still warming up: wait for it to deliver a frame.
                pass

            else:
                # Genuinely stalled/disconnected: either no recent frame at
                # all, or the driver is re-serving the same cached frame
                # (some DirectShow drivers do this instead of failing) and
                # has never proven itself live.
                if self.connected:
                    self.connected = False
                with self.lock:
                    self.latest_frame = None
                if not stalled_logged:
                    if not has_recent_read:
                        log.warning(
                            "Camera feed stalled (no frame for >%.1fs); attempting recovery",
                            STALE_TIMEOUT,
                        )
                    else:
                        log.warning(
                            "Camera feed frozen (no genuine frame change for >%.1fs); attempting recovery",
                            FROZEN_TIMEOUT,
                        )
                    stalled_logged = True

                if now >= next_reconnect_attempt:
                    log.info("Attempting to reconnect camera...")
                    self._start_new_worker()
                    backoff = min(backoff * 2, RECONNECT_MAX_INTERVAL)
                    next_reconnect_attempt = now + backoff

            elapsed = time.perf_counter() - t0
            remaining = TARGET - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def get_frame(self):
        """
        Retrieves a thread-safe copy of the latest captured frame.
        """
        with self.lock:
            if not self.connected or self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def is_connected(self):
        """
        Returns whether the camera is currently producing fresh, genuinely
        changing frames. Useful for surfacing a "camera disconnected /
        reconnecting..." state in the UI.
        """
        return self.connected

    def reconnect(self):
        """
        Requests an immediate reconnect attempt, bypassing the current
        backoff wait. Safe to call from any thread (e.g. a manual "retry"
        button).
        """
        self._force_reconnect.set()

    def release(self):
        """
        Stops the update thread and releases the camera hardware.
        """
        log.info("Camera release called")
        self.running = False
        if self._worker is not None:
            self._worker.stop()