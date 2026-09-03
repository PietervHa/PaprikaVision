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

from backend.core.camera import Camera
from backend.core.config_loader import cfg
from backend.core.state import AppState
from backend.core.tcp_trigger_server import TCPTriggerServer
from backend.core.vision import bind_app_state, run_vision
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
        app_state.increment_counter(result.get("status", "NOK"), placement)

        try:
            save_result(result_dict)
        except Exception as exc:
            log.error("Failed to write result: %s", exc)

        orientation = primary.get("orientation") or {}
        log.info(
            "CYCLE: status=%s placement=%s angle=%s pose=%s conf=%.2f "
            "detections=%d cycle_ms=%s reason=%s",
            result.get("status"),
            placement or "-",
            primary.get("angle_plc"),
            orientation.get("pose", "-"),
            float(orientation.get("confidence", 0.0)),
            len(result.get("detections", [])),
            cycle_time_ms,
            result.get("failure_reason", "-"),
        )
    except Exception as exc:
        log.error("Failed to process vision result: %s", exc)


def vision_trigger_loop(camera, app_state):
    """Manual trigger: press Q to run one cycle, same as the old project."""
    import keyboard

    log.info("Press Q to trigger a cycle. Ctrl+C to exit.")

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

    camera = Camera(0, app_state=app_state)

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

    TCPTriggerServer(
        camera=camera,
        run_vision_fn=run_vision,
        process_result_fn=lambda result, t: _process_vision_result(result, t, app_state),
    ).start()

    vision_trigger_loop(camera, app_state)


if __name__ == "__main__":
    main()
