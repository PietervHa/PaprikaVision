"""
Main Entry Point

Starts the camera, the vision engine, the web/HMI server and the PLC trigger
listener, then blocks on a manual trigger loop.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import threading
import time

import uvicorn

from backend.core.image_source import ImageFolderSource, build_source
from backend.core.config_loader import cfg
from backend.core.manual_capture import ManualCapture
from backend.core.state import AppState
from backend.core.tcp_trigger_server import TCPTriggerServer
from backend.core.vision import bind_app_state, get_engine, run_vision
from backend.output.result_writer import save_result
from backend.utils.logger import UVICORN_LOG_CONFIG, get_logger, setup_logging
from frontend.web import create_app

log = get_logger(__name__)
_vision_busy = threading.Event()


def _process_vision_result(result, trigger_time, app_state):
    """Record a completed cycle: state, counters, database, JSONL."""
    try:
        cycle_time_ms = round((time.perf_counter() - trigger_time) * 1000, 1)
        primary = result.get("primary") or {}
        placement = str(primary.get("placement", ""))

        result_dict = {**result, "cycle_time_ms": cycle_time_ms}
        app_state.update_result(result_dict)
        orientation = primary.get("orientation") or {}
        app_state.increment_counter(
            result.get("status", "NOK"), placement, str(orientation.get("pose", ""))
        )

        try:
            save_result(result_dict)
        except Exception as exc:
            log.error("Failed to write result: %s", exc)

        # The split behind processing_time_ms. lock_wait_ms is the time this
        # trigger queued behind an overlay preview on the model; it is what
        # hmi.overlay_pause_after_trigger_s exists to remove. detect_ms is the
        # detector stage on every backend - for pose, lock wait plus inference;
        # for shape, the classical search - and detect_ms + post_ms is the
        # engine's whole processing_time_ms.
        #
        # "-" means not measured, never 0: lock_wait_ms and infer_ms are "-"
        # on shape and simulator, which have no model and no lock, and every
        # field is "-" on a cycle that never reached the engine (no camera
        # frame, a timeout).
        timing = result.get("timing") or {}

        def _ms(key):
            value = timing.get(key)
            return "-" if value is None else f"{float(value):.1f}"

        log.info(
            "CYCLE: status=%s placement=%s angle=%s pose=%s conf=%.2f "
            "detections=%d cycle_ms=%s lock_wait_ms=%s infer_ms=%s detect_ms=%s "
            "post_ms=%s reason=%s",
            result.get("status"),
            placement or "-",
            primary.get("angle_plc"),
            orientation.get("pose", "-"),
            float(orientation.get("confidence", 0.0)),
            len(result.get("detections", [])),
            cycle_time_ms,
            _ms("lock_wait_ms"),
            _ms("inference_ms"),
            _ms("detect_ms"),
            _ms("post_ms"),
            result.get("failure_reason", "-"),
        )
    except Exception as exc:
        log.error("Failed to process vision result: %s", exc)


def _live_result(app_state) -> dict:
    """Whatever the machine last actually decided, from either path.

    The overlay worker and the trigger loop write to different places and only
    one of them is running at a time. Preferring the overlay when it is fresh
    covers frame-stepping in the HMI; falling back to latest_result covers a
    live line.
    """
    import time as _time

    overlay = app_state.get_overlay(now=_time.time())
    if not overlay.get("stale") and overlay.get("detections"):
        return {
            "status": "OK",
            "detections": overlay.get("detections") or [],
            "primary": overlay.get("primary"),
            "processing_time_ms": overlay.get("scan_ms"),
            "source": "overlay",
        }
    result = dict(app_state.get_snapshot().get("result") or {})
    result["source"] = "trigger"
    return result

WARM_UP_FRAME_TIMEOUT_S = 10.0


def _warm_up_on_first_frame(camera, app_state) -> None:
    """Warm the model up on the shape of the frames the line will actually send.

    cuDNN benchmark mode tunes per input shape, so the warm-up has to see the
    real frame size, and only the source knows it: the folder's images, or
    the camera after flip and HMI rotation. Runs before the trigger server and
    the HMI start, so no cycle can reach a model that is still tuning.
    """
    engine = get_engine()
    if engine is None:
        return
    deadline = time.monotonic() + WARM_UP_FRAME_TIMEOUT_S
    frame = camera.get_frame()
    while frame is None and time.monotonic() < deadline:
        time.sleep(0.1)
        frame = camera.get_frame()
    if frame is not None:
        engine.warm_up(frame.shape)
        return

    camera_cfg = cfg.get("camera") if isinstance(cfg.get("camera"), dict) else {}
    if isinstance(camera, ImageFolderSource) or not camera_cfg.get("width"):
        log.warning(
            "No frame within %.0f s to warm the pose model up on - the first "
            "cycle will pay the warm-up", WARM_UP_FRAME_TIMEOUT_S,
        )
        return
    # The configured size, as camera.py would deliver it: a 90 or 270 degree
    # rotation swaps width and height.
    height, width = int(camera_cfg.get("height", 1024)), int(camera_cfg["width"])
    if app_state.get_camera_rotation() % 2:
        height, width = width, height
    log.warning(
        "No camera frame within %.0f s - warming the pose model up on the "
        "configured %dx%d", WARM_UP_FRAME_TIMEOUT_S, width, height,
    )
    engine.warm_up((height, width))


def vision_trigger_loop(camera, app_state, capture=None):
    import keyboard

    while True:
        try:
            keyboard.wait("q")
            if _vision_busy.is_set():
                continue

            frame = camera.get_frame()
            if frame is None:
                log.warning("No frame available")

            trigger_time = time.perf_counter()

            def _callback(result):
                try:
                    _process_vision_result(result, trigger_time, app_state)
                finally:
                    _vision_busy.clear()

            _vision_busy.set()
            run_vision(frame, callback=_callback)
        except Exception as exc:
            _vision_busy.clear()
            log.error("Trigger loop error: %s", exc)


def main():
    setup_logging()
    app_state = AppState()
    bind_app_state(app_state)

    # Camera or a folder of raw images, see camera.source in the config.
    camera = build_source(app_state)
    _warm_up_on_first_frame(camera, app_state)

    web_cfg = cfg["web"]
    app = create_app(camera, app_state)
    threading.Thread(
        target=lambda: uvicorn.run(
            app,
            host=web_cfg["host"],
            port=web_cfg["port"],
            log_config=UVICORN_LOG_CONFIG,
        ),
        daemon=True,
    ).start()

    display_host = "127.0.0.1" if web_cfg["host"] in ("0.0.0.0", "127.0.0.1", "localhost") else web_cfg["host"]
    log.info("HMI: http://%s:%s", display_host, web_cfg["port"])

    # Hotkey capture of the current frame, for fruit the detector gets wrong.
    # Reads the latest result rather than re-running vision, so pressing it
    # never costs the line a cycle.
    capture = ManualCapture(
        frame_provider=camera.get_frame,
        # The overlay first. latest_result is only written by the trigger loop,
        # so while an operator is stepping through frames in the HMI - which is
        # exactly when they notice a bad detection and reach for the key - it
        # holds an empty dict, and every sidecar came out saying
        # {"detections": [], "primary": None}. The whole diagnostic half of a
        # capture was being thrown away at the moment it was most wanted.
        result_provider=lambda: _live_result(app_state),
        block=cfg.get("capture") if isinstance(cfg.get("capture"), dict) else {},
    )
    capture.start()

    TCPTriggerServer(
        camera=camera,
        run_vision_fn=run_vision,
        process_result_fn=lambda result, t: _process_vision_result(result, t, app_state),
    ).start()

    try:
        vision_trigger_loop(camera, app_state, capture)
    except KeyboardInterrupt:
        # Ctrl-C lands inside keyboard.wait() and dumped a stack trace on every
        # shutdown. Stopping the app is not an error.
        log.info("Shutting down")
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
