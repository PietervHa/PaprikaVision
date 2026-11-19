"""Label the stem of the paprika in rotate-unit frames, for the rotate-unit model.

    python -m tools.label_rotate                      every frame in data/rotate/raw not labelled yet
    python -m tools.label_rotate --pick 400           400 frames spread over data/rotate/raw
    python -m tools.label_rotate "D:\\other\\folder"    frames from somewhere else
    python -m tools.label_rotate --review                               page through what is labelled
    python -m tools.label_rotate --summary                              counts per status and per folder

What is labelled
----------------
The direction in which the stem LEAVES the fruit: from the BASE of the stem, where it meets
the fruit, to the END OF ITS STRAIGHT PART, just before it starts to bend (on a straight stem:
the cut end). -180..180 degrees, 0 = +X, y down - the convention the PLC already gets from
Aurora's StemDirection. The curled end of a stem is left out on purpose: where a hook ends up
is close to random, the direction the stem leaves the fruit is not. One paprika per frame.

Controls
--------
    click, click   first the stem base, then the end of its straight part  -> saved, next frame
    drag           draw the box around the paprika and its stem, when the automatic one is wrong
    a              back to the automatic box
    n              no stem visible: it points at or away from the camera, or is hidden
    e              empty: no paprika in the rotate unit
    s              skip: unusable or unsure (not used for training)
    u              back to the previous frame, to label it again
    q  or  Esc     save and quit

Put the frames in data/rotate/raw, like the belt's data/raw - one subfolder per day or per
rotate unit keeps that information, since the file names carry no camera ID. Every frame is
saved the moment it is labelled, so stopping and starting again is safe:
labelled frames are skipped. Several folders (days, rotate units) can go into the same
labels file one after the other.

What gets stored
----------------
    data/rotate/labels/rotate_labels.json     (change with --out)

Points and boxes are in full-frame pixel coordinates. SOURCE FRAMES ARE NEVER MODIFIED:
they are only read, and the tool refuses to write its labels inside the source folder.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT / "data" / "rotate" / "labels" / "rotate_labels.json"
DEFAULT_FRAMES = PROJECT / "data" / "rotate" / "raw"   # like data/raw for the belt
SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
WINDOW = "label_rotate"
STATUSES = ("stem", "no_stem", "empty", "skip")
HEADER_H = 58             # text bar above the image, in window pixels
MIN_STEM_PX = 15          # base and tip closer than this (full-frame pixels) is a misclick, not a stem


# ------------------------------------------------------------------ screen

def screen_area():
    """Free screen area (left, top, width, height) in real pixels, or None if unknown.
    On Windows the process is first made DPI aware: otherwise Windows enlarges the window by the
    display scaling (125 %, 150 %) and it no longer fits on the screen."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        rect = wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)   # SPI_GETWORKAREA
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    except Exception:
        return None


def open_window(width: int, height: int, area) -> None:
    """A resizable window that keeps the aspect ratio; clicks stay in image coordinates."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, width, height)
    if area:
        cv2.moveWindow(WINDOW, area[0] + max(0, (area[2] - width) // 2), area[1] + 10)


def header(width: int, lines: list, colours: list) -> np.ndarray:
    bar = np.zeros((HEADER_H, width, 3), np.uint8)
    for i, (text, colour) in enumerate(zip(lines, colours)):
        cv2.putText(bar, text, (8, 17 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.47, colour, 1, cv2.LINE_AA)
    return bar


# ------------------------------------------------------------------ automatic box

def auto_box(bgr: np.ndarray):
    """Box around the paprika: the largest coloured region (not blue, not dark, not grey) plus
    the regions touching it - the other half behind the stamp, and the stem. None if nothing
    paprika-sized is in view. Checked on 80 frames of 9 Sept 2026: boxes fit fruit and stem; the
    two empty rotate units got a small box on the stamp, so small boxes are flagged."""
    h, w = bgr.shape[:2]
    scale = 400.0 / max(h, w)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (hue >= 85) & (hue <= 135)
    mask = ((sat > 90) & (val > 50) & ~blue).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < 0.01 * mask.size:
        return None
    region = (lab == biggest).astype(np.uint8)
    reach = np.ones((25, 25), np.uint8)              # about the width of the stamp's arm
    for _ in range(3):
        near = cv2.dilate(region, reach)
        touching = set(np.unique(lab[(near > 0) & (lab > 0)])) - {0}
        grown = np.isin(lab, list(touching)) & (stats[lab, cv2.CC_STAT_AREA] >= 6)
        grown = grown.astype(np.uint8)
        if grown.sum() == region.sum():
            break
        region = np.maximum(region, grown)
    ys, xs = np.nonzero(region)
    pad = 3
    return [max(0, int((xs.min() - pad) / scale)), max(0, int((ys.min() - pad) / scale)),
            min(w, int((xs.max() + 1 + pad) / scale)), min(h, int((ys.max() + 1 + pad) / scale))]


def looks_empty(box, shape) -> bool:
    if box is None:
        return True
    return (box[2] - box[0]) * (box[3] - box[1]) < 0.06 * shape[0] * shape[1]


def stem_angle(base, tip) -> float:
    """Aurora's convention: -180..180, 0 = +X, y down."""
    return math.degrees(math.atan2(tip[1] - base[1], tip[0] - base[0]))


# ------------------------------------------------------------------ frames and labels

def gather(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES)


def key_of(root: Path, frame: Path) -> str:
    return f"{root.name}/{frame.relative_to(root).as_posix()}"


def spread(frames: list, n: int) -> list:
    if n <= 0 or n >= len(frames):
        return list(frames)
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def load_labels(path: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("frames", {})
        return data
    return {"version": 1, "angle": "stem base -> end of the straight part of the stem, -180..180, 0 = +X, y down",
            "frames": {}}


def save_labels(path: Path, labels: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ the labeller (no window here)

class Labeller:
    """Clicks and keys for one list of frames; the window itself is in run(). Kept apart so the
    logic can be tested without a screen."""

    def __init__(self, root: Path, frames: list[Path], labels: dict, out: Path, view_scale: float):
        self.root, self.frames, self.labels, self.out = root, frames, labels, out
        self.view_scale = view_scale
        self.index = 0
        self.history: list[int] = []
        self.done = False
        self._load()

    # state of the current frame
    def _load(self):
        while self.index < len(self.frames) and key_of(self.root, self.frames[self.index]) in self.labels["frames"]:
            self.index += 1
        if self.index >= len(self.frames):
            self.done = True
            return
        self.image = cv2.imread(str(self.frames[self.index]), cv2.IMREAD_COLOR)
        if self.image is None:
            self._store({"status": "skip", "reason": "unreadable"})
            return
        self.box = auto_box(self.image)
        self.auto = self.box
        self.points: list[list[int]] = []
        self.drag_from = None
        self.drag_to = None
        self.message = ""

    def _store(self, entry: dict):
        frame = self.frames[self.index]
        entry.update({"root": str(self.root), "size": None if self.image is None else
                      [int(self.image.shape[1]), int(self.image.shape[0])],
                      "labelled": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.labels["frames"][key_of(self.root, frame)] = entry
        save_labels(self.out, self.labels)
        self.history.append(self.index)
        self.index += 1
        self._load()

    def _full(self, x, y):
        return [int(round(x / self.view_scale)), int(round(y / self.view_scale))]

    # input
    def mouse(self, event, x, y, flags=0):
        if self.done:
            return
        y -= HEADER_H                      # the text bar sits above the image
        if event == cv2.EVENT_LBUTTONDOWN and y < 0:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_from, self.drag_to = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_from is not None:
            self.drag_to = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_from is not None:
            (x0, y0), self.drag_from = self.drag_from, None
            if abs(x - x0) > 8 or abs(y - y0) > 8:            # a drag: the box
                a, b = self._full(min(x0, x), min(y0, y)), self._full(max(x0, x), max(y0, y))
                self.box = [a[0], a[1], b[0], b[1]]
                self.message = "box set"
                return
            if self.box is None:                               # a click: a point, but a box comes first
                self.message = "draw a box around the paprika first (drag), or e = empty"
                return
            self.points.append(self._full(x, y))
            if len(self.points) == 2:
                base, tip = self.points
                if math.hypot(tip[0] - base[0], tip[1] - base[1]) < MIN_STEM_PX:
                    self.points = []
                    self.message = "tip too close to the base - click the base again"
                    return
                # the box must hold the whole stem: the pale cut end is often outside the automatic one
                h, w = self.image.shape[:2]
                margin = 10
                box = [max(0, min(self.box[0], base[0] - margin, tip[0] - margin)),
                       max(0, min(self.box[1], base[1] - margin, tip[1] - margin)),
                       min(w, max(self.box[2], base[0] + margin, tip[0] + margin)),
                       min(h, max(self.box[3], base[1] + margin, tip[1] + margin))]
                self._store({"status": "stem", "box": box, "base": base, "tip": tip,
                             "angle": round(stem_angle(base, tip), 2)})

    def key(self, k: int):
        if self.done:
            return
        k = k & 0xFF
        if k in (ord("q"), 27):
            self.done = True
        elif k == ord("n"):
            if self.box is None:
                self.message = "no stem, but draw the box around the paprika first (drag)"
            else:
                self._store({"status": "no_stem", "box": self.box})
        elif k == ord("e"):
            self._store({"status": "empty", "box": None})
        elif k == ord("s"):
            self._store({"status": "skip"})
        elif k == ord("a"):
            self.box, self.points, self.message = self.auto, [], "automatic box"
        elif k == ord("u"):
            if self.history:
                back = self.history.pop()
                self.labels["frames"].pop(key_of(self.root, self.frames[back]), None)
                save_labels(self.out, self.labels)
                self.index = back
                self._load()
            elif self.points:
                self.points = []

    # drawing
    def render(self) -> np.ndarray:
        s = self.view_scale
        view = cv2.resize(self.image, (int(self.image.shape[1] * s), int(self.image.shape[0] * s)),
                          interpolation=cv2.INTER_AREA)
        if self.box is not None:
            x1, y1, x2, y2 = (int(v * s) for v in self.box)
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if self.drag_from is not None and self.drag_to is not None and self.drag_to != self.drag_from:
            cv2.rectangle(view, self.drag_from, (self.drag_to[0], max(0, self.drag_to[1])), (0, 255, 255), 1)
        if self.points:
            base = tuple(int(v * s) for v in self.points[0])
            cv2.circle(view, base, 7, (0, 0, 0), -1)
            cv2.circle(view, base, 5, (255, 255, 0), -1)
        done = sum(1 for f in self.frames if key_of(self.root, f) in self.labels["frames"])
        step = ("click the END of the STRAIGHT part of the stem (before it bends)" if self.points
                else "click the BASE of the stem (where it meets the fruit)")
        if looks_empty(self.box, self.image.shape) and not self.message:
            self.message = "looks empty: e = empty, or drag a box if there is a paprika"
        lines = [f"{done}/{len(self.frames)}   {self.frames[self.index].name}   -   {step}",
                 "drag = box   a = auto box   n = no stem   e = empty   s = skip   u = back   q = quit",
                 self.message]
        bar = header(view.shape[1], lines, [(255, 255, 255), (180, 180, 180), (0, 255, 255)])
        return np.vstack([bar, view])


def draw_label(image: np.ndarray, entry: dict, scale: float) -> np.ndarray:
    view = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    if entry.get("box"):
        x1, y1, x2, y2 = (int(v * scale) for v in entry["box"])
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if entry.get("status") == "stem":
        base = tuple(int(v * scale) for v in entry["base"])
        tip = tuple(int(v * scale) for v in entry["tip"])
        cv2.arrowedLine(view, base, tip, (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(view, base, tip, (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.2)
    return view


# ------------------------------------------------------------------ modes

def view_scale_for(args, shape, area=None) -> float:
    if args.scale:
        return args.scale
    if area:
        return min(1.0, 0.9 * area[2] / shape[1], (0.78 * area[3] - HEADER_H) / shape[0])
    return min(1.0, args.max_width / shape[1], args.max_height / shape[0])


def run_label(args) -> int:
    root = Path(args.frames).resolve()
    out = Path(args.out).resolve()
    if root == out.parent or root in out.parents:
        print(f"Refusing: the labels file {out} would be inside the source folder {root}.")
        return 2
    frames = spread(gather(root), args.pick)
    if not frames:
        print(f"no frames under {root}")
        return 1
    labels = load_labels(out)
    first = cv2.imread(str(frames[0]))
    area = screen_area()
    lab = Labeller(root, frames, labels, out, view_scale_for(args, first.shape, area))
    if lab.done:
        print(f"all {len(frames)} frames are labelled already ({out})")
        return 0
    canvas = lab.render()
    open_window(canvas.shape[1], canvas.shape[0], area)
    cv2.setMouseCallback(WINDOW, lambda e, x, y, f, p: lab.mouse(e, x, y, f))
    while not lab.done:
        cv2.imshow(WINDOW, lab.render())
        k = cv2.waitKey(20)
        if k != -1:
            lab.key(k)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    summary(labels, out)
    return 0


def run_review(args) -> int:
    out = Path(args.out).resolve()
    labels = load_labels(out)
    keys = sorted(k for k, e in labels["frames"].items() if e.get("status") != "skip")
    if not keys:
        print(f"nothing labelled in {out}")
        return 0
    i = 0
    area = screen_area()
    opened = False
    print("review: space/n = next, p = previous, d = delete label (labelled again next time), q = quit")
    while 0 <= i < len(keys):
        entry = labels["frames"][keys[i]]
        path = Path(entry["root"]) / keys[i].split("/", 1)[1]
        image = cv2.imread(str(path))
        if image is None:
            print(f"cannot read {path}")
            i += 1
            continue
        view = draw_label(image, entry, view_scale_for(args, image.shape, area))
        text = f"{i + 1}/{len(keys)}   {keys[i]}   {entry['status']}" + (f"   {entry['angle']:.1f} deg" if "angle" in entry else "")
        view = np.vstack([header(view.shape[1], [text, "space/n = next   p = previous   d = delete label   q = quit"],
                                 [(255, 255, 255), (180, 180, 180)]), view])
        if not opened:
            open_window(view.shape[1], view.shape[0], area)
            opened = True
        cv2.imshow(WINDOW, view)
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("p"):
            i = max(0, i - 1)
        elif k == ord("d"):
            labels["frames"].pop(keys[i])
            save_labels(out, labels)
            keys.pop(i)
        else:
            i += 1
    cv2.destroyAllWindows()
    return 0


def summary(labels: dict, out: Path) -> None:
    frames = labels["frames"]
    by_status = Counter(e.get("status") for e in frames.values())
    by_folder = Counter(k.split("/", 1)[0] for k in frames)
    print(f"{len(frames)} frames in {out}")
    print("  " + ", ".join(f"{s} {by_status.get(s, 0)}" for s in STATUSES))
    for folder, n in sorted(by_folder.items()):
        print(f"  {folder}: {n}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", nargs="?", default=str(DEFAULT_FRAMES),
                        help="folder with rotate-unit frames, searched recursively (default data/rotate/raw)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="labels file")
    parser.add_argument("--pick", type=int, default=0, help="label this many frames, spread over the folder")
    parser.add_argument("--review", action="store_true", help="page through the labels")
    parser.add_argument("--summary", action="store_true", help="counts per status and per folder")
    parser.add_argument("--scale", type=float, default=None,
                        help="image size on screen, e.g. 0.5 (default: about 3/4 of the screen height)")
    parser.add_argument("--max-width", type=int, default=1100, help="used when the screen size is unknown")
    parser.add_argument("--max-height", type=int, default=760, help="used when the screen size is unknown")
    args = parser.parse_args()
    if args.summary:
        summary(load_labels(Path(args.out).resolve()), Path(args.out).resolve())
        return 0
    if args.review:
        return run_review(args)
    if not Path(args.frames).is_dir():
        print(f"no folder {args.frames} - put the rotate frames in data/rotate/raw, or give a folder")
        return 1
    return run_label(args)


if __name__ == "__main__":
    sys.exit(main())
