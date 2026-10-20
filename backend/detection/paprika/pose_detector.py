"""
Paprika Detector

Finds paprikas in a frame and hands each one to the orientation engine.

Three backends, selected by `paprika.backend` in config:

  "shape"      Classical segmentation. No model, no training, no GPU. Finds
               saturated blobs against the belt and orients them from
               silhouette alone. Works TODAY, before a single image has been
               annotated - which is the point: the HMI, the PLC handshake, the
               overlay and the angle convention can all be commissioned and
               signed off while the dataset is still being collected. It is
               also a genuine fallback if the model ever fails to load.

  "pose"       YOLOv8/YOLO11 pose model predicting one box plus the two
               landmarks per fruit. The production backend once trained.

  "simulator"  Synthetic fruit sweeping through 360 degrees, ignoring the
               camera entirely. For exercising the PLC integration on a bench
               with no camera and no product.

All three return the identical contract, so switching backend is a config edit
and nothing downstream can tell the difference.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.detection.paprika import classical
from backend.detection.paprika import orientation as orient
from backend.detection.paprika.orientation import Keypoint
from backend.utils.logger import get_logger
from backend.detection.paprika.classical import (
    EDGE_MARGIN_PX, REASON_EDGE_CLIPPED, colour_name,
)
from backend.utils.paths import project_path
from backend.utils.ultralytics_env import disable_autoinstall

# Before anything can import ultralytics: the line must never pip-install.
disable_autoinstall()

# Share of the bounding-box perimeter that may lie on the frame border before
# the fruit counts as incomplete. Calibrated against classical.edge_cut_ratio
# on 170 fruit: see _bbox_edge_cut.
BBOX_EDGE_CUT_THRESHOLD = 0.23

log = get_logger(__name__)

VALID_BACKENDS = ("shape", "pose", "simulator")


class PoseModelConfigError(RuntimeError):
    """The pose model can never run as configured, so the app must not start.

    Distinct from a model that merely failed to load. A missing file is
    survivable - someone may be copying weights across, and the HMI shows "not
    ready" - but an .onnx exported at another imgsz, or a GPU that is not
    there, fails identically on every cycle. Before this existed that meant an
    error in the log every 0.15 s from the overlay, and an app that looked like
    it was running.
    """


def onnx_input_size(path) -> Optional[tuple]:
    """(height, width) an ONNX export was built for, or None if it is dynamic.

    An ONNX export has its input size baked in; predict letterboxes every frame
    to exactly that, whatever imgsz says. Read through ONNX Runtime on the CPU
    because that is the runtime the app uses anyway - nothing extra to install.
    """
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    height, width = shape[2], shape[3]
    if not isinstance(height, int) or not isinstance(width, int):
        return None
    return height, width


def onnx_has_cuda() -> bool:
    """Whether the installed ONNX Runtime can run on an NVIDIA GPU at all.

    The plain onnxruntime package is CPU-only. Without this check Ultralytics
    asks for a CUDA session whenever torch sees a GPU, warns, and falls back.
    """
    try:
        import onnxruntime

        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


def cuda_device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


class PaprikaDetector:
    """Detects paprikas and returns box + landmarks for each.

    Args:
        cfg_block: the `paprika` section of the config.
    """

    def __init__(self, cfg_block: dict) -> None:
        cfg_block = cfg_block if isinstance(cfg_block, dict) else {}
        self._cfg = cfg_block

        backend = str(cfg_block.get("backend", "shape")).strip().lower()
        if backend not in VALID_BACKENDS:
            log.warning("Unknown paprika backend '%s'; falling back to 'shape'", backend)
            backend = "shape"
        self.backend = backend

        self.confidence = float(cfg_block.get("confidence", 0.45))
        self.max_detections = int(cfg_block.get("max_detections", 8))

        shape_cfg = cfg_block.get("shape") if isinstance(cfg_block.get("shape"), dict) else {}
        self._saturation_floor = int(shape_cfg.get("saturation_floor", 80))
        belt = shape_cfg.get("belt_hue") or [96, 145]
        self._belt_hue = (int(belt[0]), int(belt[1]))
        self._value_floor = int(shape_cfg.get("value_floor", 45))
        stem = shape_cfg.get("stem_hue") or [33, 92]
        self._stem_hue = (int(stem[0]), int(stem[1]))
        self._use_roi = bool(shape_cfg.get("use_roi", True))
        self._selfcheck = bool(shape_cfg.get("stem_selfcheck", True))
        self._restrict_to_belt = bool(shape_cfg.get("restrict_to_belt", True))
        self._belt_warned = False
        self._min_area_px = int(shape_cfg.get("min_area_px", 4000))
        self._max_area_ratio = float(
            shape_cfg.get("max_area_ratio", classical.DEFAULT_MAX_AREA_RATIO)
        )

        pose_cfg = cfg_block.get("pose") if isinstance(cfg_block.get("pose"), dict) else {}
        self._model_path = str(project_path(pose_cfg.get("model_path"), "models/paprika_pose.pt"))
        self._imgsz = int(pose_cfg.get("imgsz", 640))
        self._kp_confidence = float(pose_cfg.get("keypoint_confidence", 0.30))
        self._cudnn_benchmark = bool(pose_cfg.get("cudnn_benchmark", True))
        self._is_onnx = self._model_path.lower().endswith(".onnx")

        # "auto", "cpu", or a CUDA index. YAML reads an unquoted 0 as an int.
        device = str(pose_cfg.get("device", "auto")).strip().lower()
        if device not in ("auto", "cpu") and not device.isdigit():
            log.warning("Unknown paprika.pose.device '%s'; using 'auto'", device)
            device = "auto"
        self._device_setting = device
        # What every predict gets: "cpu" or a CUDA index. Resolved at load.
        self._device: Optional[str] = None

        self._model = None
        self._model_lock = threading.Lock()
        self._model_failed = False
        self._config_error: Optional[str] = None

        self._sim_start = time.time()

        log.info("Paprika detector backend: %s", self.backend)

    # ------------------------------------------------------------- lifecycle

    def is_ready(self) -> bool:
        if self.backend in ("shape", "simulator"):
            return True
        return self._load_model() is not None

    def load_model(self) -> None:
        """Load the model now, at startup, and refuse if it can never run.

        Raises PoseModelConfigError for a model that would fail every cycle.
        A model that is merely missing stays non-fatal; see the error class.
        """
        if self.backend != "pose":
            return
        with self._model_lock:
            self._load_model()
        if self._config_error:
            raise PoseModelConfigError(self._config_error)

    def _load_model(self):
        """Lazily load the pose model, mirroring the old project's YOLO pattern.

        Fails once: the error is logged here and every later call returns None
        without trying again, so a broken model costs one log line rather than
        one per cycle.
        """
        if self._model is not None:
            return self._model
        if self._model_failed:
            return None

        try:
            self._device = self._resolve_device()
            if self._is_onnx and Path(self._model_path).exists():
                self._check_onnx_size()
            # CUDA + PyTorch only. Decided from the config rather than from
            # whether a GPU is visible: without CUDA the flag does nothing.
            if not self._is_onnx and self._device_setting != "cpu":
                self._set_cudnn_benchmark()
            from ultralytics import YOLO

            # task is never left to Ultralytics. For an .onnx it guesses from
            # the filename, and "paprika_pose.onnx" was guessed as "detect":
            # every fruit came back undetected, with only a warning.
            self._model = YOLO(self._model_path, task="pose")
            log.info(
                "Paprika pose model loaded: %s | %s | device %s | imgsz %d",
                self._model_path, self._runtime_name(), self._device_name(), self._imgsz,
            )
            return self._model
        except PoseModelConfigError as exc:
            log.error("%s", exc)
            self._config_error = str(exc)
            self._model_failed = True
            return None
        except Exception as exc:
            log.error("Failed to load paprika pose model '%s': %s", self._model_path, exc)
            self._model_failed = True
            return None

    def _resolve_device(self) -> str:
        """Turn paprika.pose.device into what predict gets: "cpu" or an index.

        Never left to Ultralytics. "auto" is the GPU when there is one the
        runtime can use and the CPU otherwise - for an .onnx that means ONNX
        Runtime must have a CUDA provider, not just torch seeing a card. An
        explicit index with no such GPU is a config error: quietly running on
        the CPU instead would hide a machine that is not what the config says.
        """
        usable = cuda_device_count() if not self._is_onnx or onnx_has_cuda() else 0
        if self._device_setting == "cpu":
            return "cpu"
        if self._device_setting == "auto":
            return "0" if usable else "cpu"
        if int(self._device_setting) >= usable:
            runtime = "ONNX Runtime" if self._is_onnx else "PyTorch"
            raise PoseModelConfigError(
                f"paprika.pose.device is '{self._device_setting}' but {runtime} has "
                f"no CUDA device {self._device_setting} on this machine. "
                f'Set paprika.pose.device to "cpu" or "auto".'
            )
        return self._device_setting

    def _check_onnx_size(self) -> None:
        """Refuse an .onnx exported at a different size than paprika.pose.imgsz."""
        size = onnx_input_size(self._model_path)
        if size is not None and size != (self._imgsz, self._imgsz):
            raise PoseModelConfigError(
                f"{self._model_path} was exported for {size[1]}x{size[0]} input but "
                f"paprika.pose.imgsz is {self._imgsz}. An ONNX export has a fixed "
                f"input size: re-export it (python -m tools.export_onnx) or set "
                f"paprika.pose.imgsz to {size[0]}."
            )

    def _runtime_name(self) -> str:
        try:
            if self._is_onnx:
                import onnxruntime

                provider = "CPUExecutionProvider" if self._device == "cpu" else "CUDAExecutionProvider"
                return f"ONNX Runtime {onnxruntime.__version__} ({provider})"
            import torch

            return f"PyTorch {torch.__version__}"
        except Exception:
            return "ONNX Runtime" if self._is_onnx else "PyTorch"

    def _device_name(self) -> str:
        if self._device in (None, "cpu"):
            return "cpu"
        try:
            import torch

            return f"cuda:{self._device} ({torch.cuda.get_device_name(int(self._device))})"
        except Exception:
            return f"cuda:{self._device}"

    def _set_cudnn_benchmark(self) -> None:
        """Let cuDNN time its algorithms per input shape instead of guessing.

        Off, its heuristics pick slow kernels for some shapes on the Quadro
        M1200: 99 ms at imgsz 512 against 59 ms with this on (see
        paprika.pose.cudnn_benchmark). The flag is process-wide, not per
        model, and has no effect without CUDA. Set before the first predict,
        since tuning happens on the first call for each shape.
        """
        try:
            import torch

            torch.backends.cudnn.benchmark = self._cudnn_benchmark
            log.info("cuDNN benchmark mode: %s", "on" if self._cudnn_benchmark else "off")
        except Exception as exc:
            log.warning("Could not set cuDNN benchmark mode: %s", exc)

    def warm_up(self, frame_shape) -> None:
        """Load the model and warm it up for frames of `frame_shape` (h, w, ...).

        Takes the real frame shape because benchmark mode tunes per input
        tensor shape, and predict letterboxes a 900x1092 frame at imgsz 416 to
        352x416 - a square imgsz x imgsz blank would tune a shape the line
        never sends, and the first triggered cycle would pay the tuning.
        No-op on the backends with no model.
        """
        if self.backend != "pose":
            return
        with self._model_lock:
            model = self._load_model()
            if model is not None:
                self._warm_up(model, frame_shape)

    def _warm_up(self, model, frame_shape) -> None:
        """Run one throwaway inference so the first real frame is not the slow one.

        A cold CUDA model compiles kernels, allocates workspace and initialises
        cuDNN on its first call - seconds, not milliseconds - and with
        benchmark mode on it also times algorithms for the input shape. Without
        this that cost lands on whichever fruit happens to arrive first after a
        restart, which on a trigger-driven line means a missed piece rather
        than a slow one. Doing it at startup moves the cost to somewhere nobody
        is waiting.

        Deliberately forgiving: a warm-up that fails has not broken anything
        the next real call will not also hit and report properly, and refusing
        to start over a throwaway inference would be worse than a slow frame.
        """
        try:
            height, width = int(frame_shape[0]), int(frame_shape[1])
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            start = time.perf_counter()
            model.predict(blank, imgsz=self._imgsz, conf=self.confidence,
                          device=self._device, verbose=False)
            log.info(
                "Paprika pose model warmed up for %dx%d frames in %.0f ms",
                width, height, (time.perf_counter() - start) * 1000.0,
            )
        except Exception as exc:
            log.warning("Paprika pose warm-up skipped: %s", exc)

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "ready": self.is_ready(),
            "model_path": self._model_path if self.backend == "pose" else "",
            "confidence": self.confidence,
        }

    # -------------------------------------------------------------- dispatch

    def detect(self, frame: np.ndarray, timing: Optional[dict] = None) -> list[dict]:
        """Return raw detections: [{bbox, confidence, keypoints}].

        `keypoints` maps landmark name -> Keypoint, and may be empty (the shape
        backend never produces landmarks). Orientation is NOT computed here -
        that belongs to the engine, so all three backends stay comparable.

        `timing`, when given, is filled in with where the time went. It is the
        caller's dict rather than an attribute on the detector because the
        overlay and a trigger run this concurrently on one instance, and a
        shared "last timing" would hand one of them the other's numbers.
        lock_wait_ms and inference_ms stay None on backends with no model and
        no lock - absent, not zero.
        """
        if timing is not None:
            timing.setdefault("lock_wait_ms", None)
            timing.setdefault("inference_ms", None)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        if self.backend == "pose":
            return self._detect_pose(frame, timing)
        start = time.perf_counter()
        if self.backend == "simulator":
            detections = self._detect_simulator(frame)
        else:
            detections = self._detect_shape(frame)
        if timing is not None:
            timing["detect_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return detections

    # ----------------------------------------------------------- shape route

    def _detect_shape(self, frame: np.ndarray) -> list[dict]:
        """Classical detection: fruit by hue, stem by colour or morphology.

        Emits real keypoints, not just a box. That is the difference between a
        backend that finds something and a backend the machine can run on:
        without stem_end and blossom_end the engine has no source for the angle
        at all, and rightly reports "unknown" on every fruit.

        Colour-independent in the fruit detection - nothing selects on paprika
        colour, it only filters the belt out.
        """
        belt = None
        if self._restrict_to_belt:
            belt = classical.belt_mask_raw(frame, self._belt_hue)
            if belt is None and not self._belt_warned:
                log.warning(
                    "No belt recognised in the frame - searching the whole image. "
                    "Check the camera framing and paprika.shape.belt_hue."
                )
                self._belt_warned = True

        fruits = classical.find_fruit(
            frame,
            belt_hue=self._belt_hue,
            saturation_floor=self._saturation_floor,
            value_floor=self._value_floor,
            stem_hue=self._stem_hue,
            min_area=self._min_area_px,
            max_area_ratio=self._max_area_ratio,
            max_fruit=self.max_detections,
            use_roi=self._use_roi,
            selfcheck=self._selfcheck,
            belt_mask=belt,
        )

        detections: list[dict] = []
        for fruit in fruits:
            landmarks: dict[str, Keypoint] = {}
            if fruit.stem_end is not None and fruit.blossom_end is not None:
                # The keypoint confidence IS the stem localisation quality.
                # Feeding a fixed 0.85 here was what let a loosely pinned stem
                # on a green fruit reach the PLC looking as certain as a stem
                # separated cleanly by colour.
                #
                # `fruit.stem_quality` is a plain float (default 0.0), always
                # assigned before stem_end/blossom_end are, so it is never
                # genuinely absent here - `or 0.5` was not filling in a
                # missing value, it was overwriting a real one. The stability
                # self-check legitimately reports exactly 0.0 for a stem
                # whose direction moved too much under a lighting change to
                # trust at all (see _stem_selfcheck), and `x or 0.5` treats
                # that 0.0 as falsy and silently promotes it to a middling
                # 0.5 - turning "reject this, it is unstable" into "somewhat
                # confident", which is backwards. Use the value as computed.
                quality = float(fruit.stem_quality)
                landmarks["stem_end"] = Keypoint(
                    fruit.stem_end[0], fruit.stem_end[1], quality, True
                )
                landmarks["blossom_end"] = Keypoint(
                    fruit.blossom_end[0],
                    fruit.blossom_end[1],
                    quality,
                    # On a standing fruit the blossom end is underneath.
                    # Marking it occluded is exactly what the engine needs to
                    # tell stem-up from stem-down.
                    not fruit.standing,
                )

            x1, y1, x2, y2 = fruit.bbox
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    # No model score available, so these are honestly fixed:
                    # found is found. An invented variable score would imply a
                    # confidence that nothing supports.
                    "confidence": round(0.55 + 0.35 * float(fruit.stem_quality or 0.0), 3),
                    "keypoints": landmarks,
                    "area_px": fruit.area,
                    "stem_method": fruit.stem_method,
                "stem_area_ratio": float(fruit.stem_area_ratio),
                "stem_selfcheck": str(fruit.stem_selfcheck),
                    "stem_quality": round(float(fruit.stem_quality or 0.0), 3),
                    "stem_spread_deg": float(fruit.stem_spread_deg or 0.0),
                    "colour": fruit.colour,
                    "unpickable_reason": fruit.unpickable_reason,
                    "edge_clipped": fruit.edge_clipped,
                }
            )
        return detections

    # ------------------------------------------------------------ pose route

    @staticmethod
    def _bbox_edge_cut(bbox, frame_shape, margin: int = EDGE_MARGIN_PX) -> float:
        """How much of a box sits on the frame border, without needing a mask.

        classical.edge_cut_ratio measures the fruit's own outline against the
        border, which is the better measure and needs a segmentation the pose
        backend does not produce. This is its bbox-only stand-in: the share of
        the box perimeter lying on the border.

        Calibrated against the real measure on 170 fruit from 99 belt frames.
        Fruit the classical detector calls clipped score 0.231 to 0.406 here;
        fruit it calls clean score 0.000 to 0.269. At 0.23 this catches 27 of
        27 clipped and wrongly flags 4 of 143 clean - and a false flag costs a
        reorient, while a miss puts a half-seen fruit on the line with a
        confident angle taken from landmarks that are partly off the frame.
        """
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        on_border = 0
        if x1 <= margin:
            on_border += box_h
        if y1 <= margin:
            on_border += box_w
        if x2 >= width - margin:
            on_border += box_h
        if y2 >= height - margin:
            on_border += box_w
        return on_border / float(2 * (box_w + box_h))

    @staticmethod
    def _colour_in_box(frame: np.ndarray, bbox) -> str:
        """The fruit's colour, read straight from the pixels in its box.

        The pose model has no concept of colour - it is trained on one class -
        so without this every pose detection carried colour="". That is not
        cosmetic: every per-colour diagnostic reads this field, and with it
        empty a model failing on one colour and fine on the others looks
        exactly like a model failing at random. Three rounds of chasing a
        false "standing" verdict ended the moment the colours were checked by
        hand and all three turned out to be yellow.

        Median hue over the saturated, non-belt pixels in the box, through the
        same colour_name() thresholds the classical path uses, so both
        backends name colours identically.
        """
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return ""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Saturated, bright, and not the belt - the fruit is what is left.
        fruit = (saturation > 80) & (value > 60) & ~((hue >= 96) & (hue <= 145))
        if int(fruit.sum()) < 100:
            return ""
        return colour_name(float(np.median(hue[fruit])))

    def _detect_pose(self, frame: np.ndarray, timing: Optional[dict] = None) -> list[dict]:
        # The lock wait is timed apart from the inference because they have
        # different fixes. The overlay worker runs this same detector, so a
        # trigger arriving mid-preview queues here behind a whole forward pass,
        # and processing_time_ms alone cannot tell that from a slow model.
        wait_start = time.perf_counter()
        with self._model_lock:
            acquired = time.perf_counter()
            model = self._load_model()
            if model is None:
                return []
            infer_start = time.perf_counter()
            # Warmed up for the startup frame shape only. A frame of another
            # shape - the camera rotated 90 degrees in the HMI, or a folder
            # image of a different size - makes cuDNN benchmark mode retune on
            # this call: a one-off cost of a slow cycle, then full speed.
            results = model.predict(
                frame, imgsz=self._imgsz, conf=self.confidence,
                device=self._device, verbose=False,
            )
            infer_end = time.perf_counter()
        if timing is not None:
            timing["lock_wait_ms"] = round((acquired - wait_start) * 1000, 2)
            timing["inference_ms"] = round((infer_end - infer_start) * 1000, 2)
            timing["detect_ms"] = round((infer_end - wait_start) * 1000, 2)

        detections: list[dict] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)
            if boxes is None:
                continue

            box_xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.empty((0, 4))
            box_conf = boxes.conf.cpu().numpy() if len(boxes) else np.empty((0,))

            kp_xy = None
            kp_conf = None
            if keypoints is not None and getattr(keypoints, "xy", None) is not None:
                kp_xy = keypoints.xy.cpu().numpy()
                if getattr(keypoints, "conf", None) is not None:
                    kp_conf = keypoints.conf.cpu().numpy()

            for i in range(len(box_xyxy)):
                x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy[i])
                landmarks: dict[str, Keypoint] = {}

                if kp_xy is not None and i < len(kp_xy):
                    for k, name in enumerate(orient.KEYPOINT_NAMES):
                        if k >= len(kp_xy[i]):
                            break
                        kx, ky = float(kp_xy[i][k][0]), float(kp_xy[i][k][1])
                        kc = float(kp_conf[i][k]) if kp_conf is not None and i < len(kp_conf) else 1.0

                        # Ultralytics emits (0, 0) for a landmark it declined to
                        # place. Treated as absent rather than as a real point
                        # at the image origin, which would otherwise produce a
                        # confident angle pointing at the top-left corner.
                        if kx == 0.0 and ky == 0.0:
                            continue

                        landmarks[name] = Keypoint(
                            x=kx,
                            y=ky,
                            confidence=kc,
                            # Below the landmark threshold means "I think it is
                            # here but I cannot see it" - the occluded case the
                            # annotation spec trains for. Still positioned, just
                            # not visible, which is what tells the engine the
                            # fruit may be standing stem-down.
                            visible=kc >= self._kp_confidence,
                        )

                clipped = self._bbox_edge_cut(
                    (x1, y1, x2, y2), frame.shape
                ) > BBOX_EDGE_CUT_THRESHOLD
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        # Without this the engine never sees a reason to call
                        # the fruit incomplete: _UNPICKABLE_POSES keys off
                        # "edge_clipped", the pose path never set it, and a
                        # fruit running off the frame was measured as though it
                        # were whole - landmarks partly outside the image and
                        # all.
                        # unpickable_reason is the field the engine reads:
                        # _UNPICKABLE_POSES is keyed by its VALUE. Setting only
                        # "edge_clipped" left the flag inert - nothing in the
                        # codebase consumes that key - so a fruit running off
                        # the frame kept being measured as a whole one and
                        # placed with a confident angle, from landmarks partly
                        # outside the image.
                        "unpickable_reason": (
                            REASON_EDGE_CLIPPED if clipped else ""
                        ),
                        "edge_clipped": clipped,
                        "colour": self._colour_in_box(frame, (x1, y1, x2, y2)),
                        "confidence": round(float(box_conf[i]), 3) if i < len(box_conf) else 0.0,
                        "keypoints": landmarks,
                        "area_px": max(0, (x2 - x1) * (y2 - y1)),
                    }
                )

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[: self.max_detections]

    # ------------------------------------------------------- simulator route

    def _detect_simulator(self, frame: np.ndarray) -> list[dict]:
        """One synthetic fruit rotating at 30 deg/s, ignoring the camera.

        Exists so the PLC handshake and the angle convention can be verified on
        a bench: the angle is known exactly, so any mismatch at the PLC end is
        unambiguously a convention error rather than a detection error.
        """
        h, w = frame.shape[:2]
        angle = (time.time() - self._sim_start) * 30.0 % 360.0

        cx, cy = w / 2.0, h / 2.0
        half_len = min(w, h) * 0.18
        radians = math.radians(angle)
        dx, dy = math.cos(radians) * half_len, -math.sin(radians) * half_len

        stem = (cx + dx, cy + dy)
        blossom = (cx - dx, cy - dy)

        pad = min(w, h) * 0.09
        x1 = int(min(stem[0], blossom[0]) - pad)
        y1 = int(min(stem[1], blossom[1]) - pad)
        x2 = int(max(stem[0], blossom[0]) + pad)
        y2 = int(max(stem[1], blossom[1]) + pad)

        return [
            {
                "bbox": [max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)],
                "confidence": 0.99,
                "keypoints": {
                    "stem_end": Keypoint(stem[0], stem[1], 0.99, True),
                    "blossom_end": Keypoint(blossom[0], blossom[1], 0.99, True),
                },
                "area_px": max(0, (x2 - x1) * (y2 - y1)),
                "simulated": True,
            }
        ]