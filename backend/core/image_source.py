"""
File source: run the application on raw images instead of the camera

Drop-in replacement for Camera. Same methods (`get_frame`, `is_connected`,
`reconnect`, `release`), so nothing above it needs to know where a frame comes
from - the HMI, the overlay thread, the PLC trigger and the engine all work
unchanged.

Why this is more than a testing convenience
-------------------------------------------
On a moving belt every frame shows a different paprika, so there is nothing to
compare against: if a result changes, you cannot tell whether that came from
your setting or from the next fruit. With a fixed folder the image is identical,
so any difference is always attributable to what you changed. That turns tuning
thresholds and fixing the angle convention into a measurement rather than a
feeling.

`loop` and `hold` exist for the same reason. With `hold` a single image stays up
until you click on, which is what you need to study one difficult fruit; `loop`
plays the folder like a belt, which is what you need to see whether something
holds across the whole set.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.core.config_loader import cfg
from backend.utils.logger import get_logger

log = get_logger(__name__)

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class ImageFolderSource:
    """Plays a folder of images as if it were a camera.

    Args:
        folder:      folder of images (read recursively).
        interval_s:  seconds per image. 0 or `hold=True` holds the current one.
        loop:        start over at the end.
        hold:        do not advance automatically; only via `next()`/`previous()`.
        app_state:   for the rotation button in the HMI, same as Camera.
    """

    def __init__(
        self,
        folder: str | Path,
        interval_s: float = 2.0,
        loop: bool = True,
        hold: bool = False,
        app_state=None,
    ) -> None:
        self.app_state = app_state
        self._folder = Path(folder)
        self._interval_s = max(0.0, float(interval_s))
        self._loop = bool(loop)
        self._hold = bool(hold)

        self._paths: list[Path] = sorted(
            p for p in self._folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        )

        self._lock = threading.Lock()
        self._index = 0
        self._last_advance = time.time()
        self._cached_index = -1
        self._cached_frame: Optional[np.ndarray] = None
        self._exhausted = False

        flip_cfg = cfg.get("camera", {}).get("flip", None)
        self._flip = flip_cfg

        if not self._paths:
            log.error("No images found in %s", self._folder)
        else:
            log.info(
                "File source: %d images from %s (interval %.1fs, loop=%s, hold=%s)",
                len(self._paths), self._folder, self._interval_s, self._loop, self._hold,
            )

    # ------------------------------------------------------------ navigation

    def count(self) -> int:
        return len(self._paths)

    def current_name(self) -> str:
        with self._lock:
            if not self._paths:
                return ""
            return self._paths[self._index % len(self._paths)].name

    def next(self) -> str:
        with self._lock:
            if self._paths:
                self._index = (self._index + 1) % len(self._paths)
                self._last_advance = time.time()
            return self._paths[self._index].name if self._paths else ""

    def previous(self) -> str:
        with self._lock:
            if self._paths:
                self._index = (self._index - 1) % len(self._paths)
                self._last_advance = time.time()
            return self._paths[self._index].name if self._paths else ""

    def seek(self, index: int) -> str:
        with self._lock:
            if self._paths:
                self._index = int(index) % len(self._paths)
                self._last_advance = time.time()
            return self._paths[self._index].name if self._paths else ""

    def set_hold(self, hold: bool) -> bool:
        with self._lock:
            self._hold = bool(hold)
            self._last_advance = time.time()
            return self._hold

    # ----------------------------------------------------- Camera-compatible

    def get_frame(self) -> Optional[np.ndarray]:
        """Current image. Advances by itself once the interval has elapsed.

        Always returns a copy, just like Camera: the overlay thread and the HMI
        stream get the same frame, and neither may draw into the other's image.
        """
        with self._lock:
            if not self._paths:
                return None

            if not self._hold and self._interval_s > 0:
                if time.time() - self._last_advance >= self._interval_s:
                    nxt = self._index + 1
                    if nxt >= len(self._paths) and not self._loop:
                        self._exhausted = True
                    else:
                        self._index = nxt % len(self._paths)
                    self._last_advance = time.time()

            index = self._index
            path = self._paths[index]

            # Cached by index: with hold, or a slow interval, the video stream
            # requests the same file ~30x per second, and reading it from disk
            # every time is pure wasted work.
            if index == self._cached_index and self._cached_frame is not None:
                return self._cached_frame.copy()

        frame = cv2.imread(str(path))
        if frame is None:
            log.warning("Could not read image: %s", path)
            return None

        if self._flip is not None:
            try:
                frame = cv2.flip(frame, int(self._flip))
            except Exception:
                pass

        if self.app_state is not None:
            rotation = self.app_state.get_camera_rotation()
            for _ in range(rotation):
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        with self._lock:
            self._cached_index = index
            self._cached_frame = frame

        return frame.copy()

    def is_connected(self) -> bool:
        return bool(self._paths) and not self._exhausted

    def reconnect(self) -> None:
        """Re-read the folder, so new images are picked up without a restart."""
        with self._lock:
            self._paths = sorted(
                p for p in self._folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
            )
            self._index = 0
            self._cached_index = -1
            self._cached_frame = None
            self._exhausted = False
        log.info("File source reloaded: %d images", len(self._paths))

    def release(self) -> None:
        with self._lock:
            self._cached_frame = None

    # ------------------------------------------------------------------ HMI

    def status(self) -> dict:
        with self._lock:
            return {
                "source": "folder",
                "folder": str(self._folder),
                "count": len(self._paths),
                "index": self._index,
                "name": self._paths[self._index].name if self._paths else "",
                "hold": self._hold,
                "loop": self._loop,
                "interval_s": self._interval_s,
            }


def build_source(app_state):
    """Builds the frame source from config. Camera or folder.

    With `source: folder` and a missing folder it falls back to the camera
    rather than crashing - a wrong path in a config file should not bring a
    machine to a stop.
    """
    camera_cfg = cfg.get("camera") if isinstance(cfg.get("camera"), dict) else {}
    source = str(camera_cfg.get("source", "camera")).strip().lower()

    if source == "folder":
        folder_cfg = camera_cfg.get("folder") if isinstance(camera_cfg.get("folder"), dict) else {}
        path = Path(str(folder_cfg.get("path", "data/raw")))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path

        if path.is_dir():
            return ImageFolderSource(
                folder=path,
                interval_s=float(folder_cfg.get("interval_s", 2.0)),
                loop=bool(folder_cfg.get("loop", True)),
                hold=bool(folder_cfg.get("hold", False)),
                app_state=app_state,
            )
        log.error("camera.folder.path does not exist: %s - falling back to camera", path)

    from backend.core.camera import Camera

    return Camera(int(camera_cfg.get("index", 0)), app_state=app_state)
