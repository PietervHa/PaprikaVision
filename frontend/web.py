"""
Web / HMI layer

FastAPI app serving the operator screen, the MJPEG stream with the orientation
overlay drawn into it, and the control endpoints.

The overlay is composited server-side into the JPEG rather than drawn as an
HTML layer over the video. That is a deliberate trade: a browser overlay would
be crisper and scale with the window, but it cannot stay registered to the
video, because the boxes and the frame they describe would arrive over two
different channels with two different latencies. Drawing into the pixels makes
the arrow and the fruit inseparable by construction - and on a machine where
somebody is checking whether the arrow points at the stem, that guarantee is
worth more than the sharper text.
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.core import db, vision
from backend.core.config_loader import cfg
from backend.core.overlay_worker import OverlayWorker
from backend.utils.annotate import draw_detections, draw_hud
from backend.utils.logger import get_logger

log = get_logger(__name__)


class VisionModeBody(BaseModel):
    vision_mode: str = ""


class MaintenanceBody(BaseModel):
    maintenance_mode: bool = False
    username: str = ""
    password: str = ""


def create_app(camera, app_state) -> FastAPI:
    app = FastAPI(title="PaprikaVisionMDE")

    db.init_db()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    hmi_cfg = cfg.get("hmi", {})
    JPEG_QUALITY = int(hmi_cfg.get("stream_quality", 75))
    DRAW_KEYPOINTS = bool(hmi_cfg.get("draw_keypoints", True))
    DRAW_HUD = bool(hmi_cfg.get("draw_hud", True))
    OVERLAY_STALE_S = float(hmi_cfg.get("overlay_stale_after_s", 2.0))

    engine = vision.get_engine()
    overlay = OverlayWorker(camera, app_state, engine)
    overlay.start()

    _NO_STORE_HEADERS = {
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    }

    def _is_maintenance_access(request: Request) -> bool:
        if not app_state.get_maintenance_mode():
            return False
        token = request.cookies.get("maintenance_session", "")
        return bool(token) and token == app_state.get_maintenance_session_token()

    # ------------------------------------------------------------------ stream

    def generate_frames():
        if not hmi_cfg.get("enable_video_feed", True):
            import numpy as np

            blank = np.zeros((480, 640, 3), dtype="uint8")
            cv2.putText(blank, "Video feed disabled", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            while True:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                time.sleep(1.0)

        FRAME_INTERVAL_MS = 33.33

        while True:
            frame_start = time.time()

            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            snapshot = app_state.get_overlay(max_age_s=OVERLAY_STALE_S, now=time.time())
            detections = snapshot["detections"]

            frame_for_stream = frame
            if detections or DRAW_HUD:
                # get_frame() already returns a copy, but copy again so nothing
                # downstream can be surprised by a mutated shared frame.
                frame_for_stream = frame.copy()

            if detections:
                primary = snapshot.get("primary") or {}
                draw_detections(
                    frame_for_stream,
                    detections,
                    primary_center=primary.get("center"),
                    show_keypoints=DRAW_KEYPOINTS,
                )

            if DRAW_HUD:
                lines = []
                if is_folder:
                    info = camera.status()
                    lines.append(
                        f"[{info['index']+1}/{info['count']}] {info['name']}"
                        + ("  (vast)" if info["hold"] else "")
                    )
                if not overlay.is_running():
                    lines.append("Live overlay stopped")
                elif snapshot["stale"]:
                    age = snapshot["age_s"]
                    lines.append(
                        f"No recent result ({age:.0f}s)" if age else "Overlay warming up"
                    )
                else:
                    lines.append(f"{len(detections)} paprika | {snapshot['scan_ms']:.0f} ms")
                    primary = snapshot.get("primary")
                    if primary:
                        angle = primary.get("angle_plc")
                        angle_text = f"{angle:.0f} deg" if angle is not None else "no angle"
                        lines.append(f"{primary.get('placement','?')} | {angle_text}")
                draw_hud(frame_for_stream, lines)

            _, buffer = cv2.imencode(".jpg", frame_for_stream,
                                     [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

            remaining_ms = FRAME_INTERVAL_MS - (time.time() - frame_start) * 1000
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000)

    # --------------------------------------------------------------- endpoints

    @app.get("/")
    def index():
        return FileResponse(
            Path(__file__).resolve().parent / "templates" / "hmi.html",
            headers=_NO_STORE_HEADERS,
        )

    @app.get("/video_feed")
    def video_feed():
        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ------------------------------------------------------- bestandsbron
    # Alleen aanwezig als de bron een map is. Zo hoeft de HMI niets te weten
    # over welke bron actief is: de knoppen verschijnen als het endpoint
    # bestaat en blijven weg als er een camera hangt.
    is_folder = hasattr(camera, "status") and hasattr(camera, "next")

    @app.get("/source")
    def source_status():
        if not is_folder:
            return {"source": "camera"}
        return camera.status()

    @app.post("/source/next")
    def source_next(request: Request):
        if not is_folder:
            return JSONResponse(status_code=400, content={"error": "geen bestandsbron"})
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"success": False})
        return {"success": True, "name": camera.next(), **camera.status()}

    @app.post("/source/previous")
    def source_previous(request: Request):
        if not is_folder:
            return JSONResponse(status_code=400, content={"error": "geen bestandsbron"})
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"success": False})
        return {"success": True, "name": camera.previous(), **camera.status()}

    @app.post("/source/hold")
    def source_hold(request: Request):
        if not is_folder:
            return JSONResponse(status_code=400, content={"error": "geen bestandsbron"})
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"success": False})
        held = camera.set_hold(not camera.status()["hold"])
        return {"success": True, "hold": held, **camera.status()}

    @app.get("/status")
    def get_status(request: Request, response: Response):
        response.headers.update(_NO_STORE_HEADERS)
        is_maintenance = _is_maintenance_access(request)
        return {
            "vision_mode": app_state.get_vision_mode(),
            "maintenance_mode": is_maintenance,
            "username": app_state.get_maintenance_session_user() if is_maintenance else None,
            "machine_id": cfg.get("machine_id", ""),
            "engine": engine.status() if engine else {},
            "overlay": overlay.status(),
        }

    @app.get("/result")
    def get_result():
        snapshot = app_state.get_snapshot()
        live = app_state.get_overlay(max_age_s=OVERLAY_STALE_S, now=time.time())
        return {
            "result": snapshot["result"],
            "counters": snapshot["counters"],
            "vision_mode": snapshot["vision_mode"],
            # The HMI shows the live view as the headline number, because that
            # is what the operator is looking at while adjusting the machine.
            # The triggered result stays available alongside it for the record.
            "live": {
                "primary": live["primary"],
                "count": len(live["detections"]),
                "stale": live["stale"],
                "scan_ms": live["scan_ms"],
            },
        }

    @app.post("/vision_mode")
    def set_vision_mode(body: VisionModeBody, request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"success": False})
        app_state.set_vision_mode(body.vision_mode.strip().lower())
        return {"success": True, "vision_mode": app_state.get_vision_mode()}

    @app.post("/overlay/toggle")
    def toggle_overlay(request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"success": False})
        running = overlay.toggle()
        return {"success": True, "running": running}

    @app.post("/camera_rotation")
    def rotate_camera(request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"success": False})
        app_state.rotate_camera()
        rotation = app_state.get_camera_rotation()
        # Rotation is applied inside Camera before the engine ever sees a frame,
        # so every reported angle shifts with it. That is fine while framing the
        # camera and actively wrong if it happens after the PLC offset has been
        # commissioned - hence the log line, and hence this staying gated behind
        # maintenance mode.
        if rotation:
            log.warning(
                "Camera rotation set to %d x 90deg - all reported angles are now "
                "offset by %d deg. Re-check paprika.frame.angle_offset_deg.",
                rotation, rotation * 90,
            )
        return {
            "success": True,
            "rotation": rotation,
            "angle_shift_deg": rotation * 90,
        }

    @app.post("/reset_counters")
    def reset_counters(request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"success": False})
        app_state.reset_counters()
        return {"success": True}

    @app.post("/maintenance_mode")
    def set_maintenance(body: MaintenanceBody, request: Request, response: Response):
        """Per-user maintenance login, carried over unchanged in behaviour.

        The lockout logic is worth keeping verbatim rather than simplifying:
        an unknown username still pays for a scrypt hash so it takes about as
        long as a real one, and it can never accumulate strikes, so only
        failures against an account that actually exists can lock anything.
        """
        from backend.core import auth

        if not body.maintenance_mode:
            app_state.set_maintenance_mode(False)
            app_state.set_maintenance_session_token("")
            app_state.set_maintenance_session_user("")
            response.delete_cookie("maintenance_session", path="/")
            return {"maintenance_mode": False}

        username = body.username.strip()
        client_ip = request.client.host if request.client else None

        user = db.get_user(username) if username else None
        if user is None:
            auth.verify_password(body.password, auth.DUMMY_PASSWORD_HASH)
            db.record_login(username or "(empty)", False, client_ip)
            return JSONResponse(
                status_code=403,
                content={"error": "unknown_user", "message": "No account exists for that username."},
            )

        recent_failures = db.get_recent_failures_since_last_success(
            username, limit=auth.MAX_FAILED_ATTEMPTS
        )
        unlock_at = auth.lockout_until(recent_failures)
        if unlock_at is not None:
            db.record_login(username, False, client_ip)
            return JSONResponse(
                status_code=403,
                content={
                    "error": "locked_out",
                    "message": (
                        "Account locked after repeated failed attempts. "
                        f"Try again after {unlock_at.strftime('%H:%M:%S')}."
                    ),
                    "locked_until": unlock_at.isoformat(),
                },
            )

        password_ok = auth.verify_password(body.password, user["password_hash"])
        db.record_login(username, password_ok, client_ip)

        if not password_ok:
            return JSONResponse(
                status_code=403,
                content={"error": "invalid_password", "message": "Incorrect password."},
            )

        session_token = secrets.token_urlsafe(24)
        app_state.set_maintenance_mode(True)
        app_state.set_maintenance_session_token(session_token)
        app_state.set_maintenance_session_user(username)
        response.set_cookie(
            "maintenance_session",
            session_token,
            httponly=True,
            samesite="lax",
            max_age=60 * 30,
            path="/",
        )
        return {"maintenance_mode": True, "username": username}

    @app.get("/login_log")
    def login_log(request: Request):
        if not _is_maintenance_access(request):
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        return {"logins": db.get_recent_logins(limit=20)}

    return app
