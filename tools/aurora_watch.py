"""
Compare with Aurora Vision live, next to the production line

    python -m tools.compare_aurora --watch \\\\AURORA-PC\\Shadow
    python -m tools.compare_aurora --watch \\\\AURORA-PC\\Shadow --sample 20
    python -m tools.compare_aurora --watch \\\\AURORA-PC\\Shadow --from-now

The batch tool (tools/compare_aurora.py) runs on copied folders. This follows
Aurora's own folder while Aurora runs the machine: the Cam<ID> subfolders, each
with Image_*.bmp and the results.csv Aurora appends to (same format). Every
frame goes through the batch tool's own per-frame path (process_frame) and
review output, into the same folder layout.

Aurora's folder is only ever read
---------------------------------
Nothing is written, moved or deleted there. Files are opened read-only, for as
long as one read takes, and on Windows with every share flag (read, write,
delete), so Aurora can always write, rename or delete a file we have open. The
folder is never listed per poll: only results.csv and the one image a frame
names are looked at. --out may not overlap it.

How a frame is followed
-----------------------
results.csv is tailed from the last byte read; only lines ending in a line
break count, so a half-written last line waits. If the file shrinks, or its
first line changes (replaced), reading starts again at its beginning.

A frame's lines are complete when a line for another image follows, or when
nothing was added for --settle seconds (and no half line is pending). Its image
is ready when its size and time stamp held still for one poll, the bytes read
match that size and the BMP header's own size, and it decodes. An image that
is not ready within --image-timeout seconds is counted (missing, or unreadable
when it existed) and the frame skipped - never silently.

Lines for an image that already had its frame (not consecutive) are counted as
late_lines and not used.

Output (in --out, default data/datasets/aurora_review/)
------------------------------------------------------
    as batch: summary.json, unmatched.csv, frames/<camera>/, overlays/<camera>/
    sample/frames/<camera>/, sample/overlays/<camera>/   with --sample N
    watch_state.json                                       progress, for restarts

--sample N keeps every N-th frame per camera where both agree (at least one
pair compared, none over --threshold): original, Aurora's lines and overlay.
Review frames are every disagreement and the sample is 1 in N of agreements;
weigh them by that when scoring the two together.

summary.json is rewritten every --summary-every seconds and on exit, with the
batch layout plus a "watch" block (queue, lag, resets, late lines). The engine
runs frame by frame in order; when it falls behind, frames queue up and the
status line shows how many wait and for how long. None are dropped.

Restart and stop
----------------
watch_state.json holds per camera the byte offset of the first line not yet
done, the last frame done and the running totals, written after every frame.
A restart continues from there. Without it the first run starts at the
beginning of each results.csv, or with --from-now at its current end. Ctrl+C
finishes the current frame, then writes state and summary; a second Ctrl+C
stops at once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.utils.paths import project_path  # noqa: E402
from tools import compare_aurora as ca  # noqa: E402

STATE_NAME = "watch_state.json"
STATE_VERSION = 1
MAX_READ = 4 << 20              # bytes of results.csv read per poll per camera
MAX_QUEUE = 500                 # complete frames held in memory per camera
PROBLEM_LIST_MAX = 50           # names kept per problem list (counts are exact)
RECENT_NAMES = 5000             # frame names remembered to recognise late lines
TAIL_BYTES = 64 << 10           # how far back --from-now looks for a line end
HEAD_BYTES = 4096               # the first line, as the file's fingerprint

WATCH_KEYS = ("sample_frames", "late_lines", "csv_resets", "skipped_at_start",
              "output_errors")


# -------------------------------------------------------------- shared reading


def open_shared(path):
    """Open read-only without ever getting in Aurora's way.

    Python's open() on Windows does not share delete: while it is open Aurora
    could not delete or rename the file. CreateFileW with all three share
    flags can. Elsewhere plain open() already behaves that way.
    """
    if os.name != "nt":
        return open(path, "rb")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    generic_read, share_all, open_existing, normal = 0x80000000, 0x7, 3, 0x80
    handle = create(str(path), generic_read, share_all, None, open_existing, normal, None)
    if handle is None or handle == wintypes.HANDLE(-1).value:
        error = ctypes.get_last_error()
        raise OSError(None, ctypes.FormatError(error), str(path), error)
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError:
        kernel32.CloseHandle(handle)
        raise
    return os.fdopen(fd, "rb")


def read_image(path: Path, size: int):
    """(bytes, frame) when the image is there in full, else (None, None)."""
    with open_shared(path) as handle:
        data = handle.read()
    if len(data) != size:
        return None, None
    if data[:2] == b"BM" and len(data) >= 6:
        if len(data) < int.from_bytes(data[2:6], "little"):
            return None, None
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return (data, frame) if frame is not None else (None, None)


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def write_json_atomic(path: Path, obj, indent=2) -> bool:
    """Replace path in one step, so a reader never sees half a file.

    False when it stays locked (someone has it open); the caller retries later.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=indent), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
    except OSError:
        pass
    return False


# ---------------------------------------------------------------- running totals


def tally_frame(tally: Counter, records: list[dict], unmatched: list[dict]) -> None:
    """Count one frame's pairs by everything summarise() looks at.

    A day on the line is millions of pairs; this keeps a few hundred counters
    instead, and summary_from_tally turns them into the batch summary.
    """
    for r in records:
        aurora, ours = r["aurora"], r["ours"]
        tally[("pair", None if r["diff_deg"] is None else ca.bucket(r["diff_deg"]),
               aurora["verdict"], aurora["method"], aurora["stem_source"],
               ours["placement"],
               aurora["verdict"] != ca.UNDET and ours["angle_deg"] is None,
               aurora["pushed"])] += 1
    for u in unmatched:
        if u["side"] == "ours":
            tally[("ours", u["detail"])] += 1
        else:
            tally[("aurora", u["verdict"], u["pushed"])] += 1


def _bucket_counts(items) -> dict:
    counts = {name: 0 for name in ca.BUCKETS}
    for name, n in items:
        counts[name] += n
    return {"n": sum(counts.values()), **counts}


def summary_from_tally(tally: Counter, counts: dict) -> dict:
    """ca.summarise, from counters instead of every pair. Same output."""
    pairs = [(k, n) for k, n in tally.items() if k[0] == "pair"]
    ours = [(k, n) for k, n in tally.items() if k[0] == "ours"]
    aurora = [(k, n) for k, n in tally.items() if k[0] == "aurora"]
    # pair key: ("pair", bucket, verdict, method, stem_source, placement, withheld, pushed)
    compared = [(k, n) for k, n in pairs if k[1] is not None]
    direction = [(k, n) for k, n in compared if k[2] == ca.DIRECTION_GIVEN]
    axis = [(k, n) for k, n in compared if k[2] == ca.AXIS]

    by_method = {}
    for method in sorted(ca.METHOD_NAMES):
        subset = [(k[1], n) for k, n in compared if k[3] == method]
        if subset:
            suffix = ", modulo 180" if method == ca.AXIS_ONLY else ""
            by_method[f"{method} {ca.METHOD_NAMES[method]}{suffix}"] = _bucket_counts(subset)
    by_source = {}
    for source in sorted(ca.STEM_SOURCES):
        subset = [(k[1], n) for k, n in direction if k[4] == source]
        if subset:
            by_source[f"{source} {ca.STEM_SOURCES[source]}"] = _bucket_counts(subset)

    vs_aurora: dict[str, Counter] = {}
    for k, n in pairs:
        vs_aurora.setdefault(k[5], Counter())[k[2]] += n
    for k, n in ours:
        vs_aurora.setdefault(k[1], Counter())[ca.NO_AURORA] += n
    for k, n in aurora:
        vs_aurora.setdefault(ca.NOT_DETECTED, Counter())[k[1]] += n

    summary = {
        **counts,
        "matched": sum(n for _, n in pairs),
        "unmatched_ours": sum(n for _, n in ours),
        "unmatched_aurora": sum(n for _, n in aurora),
        "matched_by_aurora_verdict": {v: sum(n for k, n in pairs if k[2] == v)
                                      for v in ca.VERDICTS},
        "our_angle_withheld": sum(n for k, n in pairs if k[6]),
        "difference_deg": {
            "direction given (0,1,4)": _bucket_counts((k[1], n) for k, n in direction),
            "axis only (2), modulo 180": _bucket_counts((k[1], n) for k, n in axis),
            "by_method": by_method,
            "by_stem_source (direction given)": by_source,
        },
        "placement_vs_aurora": {k: dict(v) for k, v in sorted(vs_aurora.items())},
    }

    pushed_known = [(k, n) for k, n in pairs if k[7] is not None]
    lonely_known = [(k, n) for k, n in aurora if k[2] is not None]
    if pushed_known or lonely_known:
        vs_pushed: dict[str, Counter] = {}
        for k, n in pushed_known:
            vs_pushed.setdefault(k[5], Counter())["pushed" if k[7] else "not pushed"] += n
        for k, n in lonely_known:
            vs_pushed.setdefault(ca.NOT_DETECTED, Counter())[
                "pushed" if k[2] else "not pushed"] += n
        summary["placement_vs_pushed"] = {k: dict(v) for k, v in sorted(vs_pushed.items())}
    return summary


# ------------------------------------------------------------------- per camera


@dataclass
class FrameGroup:
    """One frame's lines, plus the stats of other lines in its byte range."""
    name: str
    key: str
    start: int                      # byte offset of its first line
    end: int                        # byte offset after its last line
    seen: float                     # clock when its first line was read
    generation: int
    rows: list = field(default_factory=list)
    lines: int = 0
    malformed: list = field(default_factory=list)
    late: int = 0
    skip: bool = False
    first_look: float | None = None
    last_look: float | None = None
    last_stat: tuple | None = None
    existed: bool = False


@dataclass
class Camera:
    name: str
    folder: Path
    offset: int = 0                 # everything before this byte is done
    read_pos: int = 0               # bytes read from results.csv so far
    buffer: bytes = b""             # a line not finished yet
    fingerprint: str | None = None
    fp_len: int = 0
    generation: int = 0
    queue: deque = field(default_factory=deque)
    open: FrameGroup | None = None
    carry: dict | None = None       # stats of lines before any open frame
    last_growth: float = 0.0
    recent: dict = field(default_factory=dict)
    skip_key: str | None = None
    last_frame: str | None = None
    agree: int = 0
    counts: Counter = field(default_factory=Counter)
    tally: Counter = field(default_factory=Counter)
    problems: dict = field(default_factory=dict)
    reachable: bool = True

    def __post_init__(self):
        for key in ca.COUNT_KEYS + WATCH_KEYS:
            self.counts.setdefault(key, 0)
        for key in ("malformed_lines", "frames_missing", "frames_unreadable",
                    "frames_engine_failed", "late_lines"):
            self.problems.setdefault(key, [])

    def note(self, kind: str, item) -> None:
        entries = self.problems[kind]
        entries.append(item)
        del entries[:-PROBLEM_LIST_MAX]

    def remember(self, key: str) -> None:
        self.recent[key] = None
        while len(self.recent) > RECENT_NAMES:
            self.recent.pop(next(iter(self.recent)))

    def waiting(self) -> int:
        return len(self.queue) + (1 if self.open else 0)

    def to_state(self) -> dict:
        return {
            "folder": str(self.folder), "offset": self.offset,
            "fingerprint": self.fingerprint, "fp_len": self.fp_len,
            "last_frame": self.last_frame, "agree": self.agree,
            "counts": dict(self.counts),
            "tally": [[list(k), n] for k, n in self.tally.items()],
            "problems": self.problems,
        }

    @classmethod
    def from_state(cls, name: str, folder: Path, state: dict) -> "Camera":
        cam = cls(name, folder)
        cam.offset = cam.read_pos = int(state.get("offset", 0))
        cam.fingerprint = state.get("fingerprint")
        cam.fp_len = int(state.get("fp_len", 0))
        cam.last_frame = state.get("last_frame")
        cam.agree = int(state.get("agree", 0))
        cam.counts.update(state.get("counts") or {})
        cam.tally = Counter({tuple(k): n for k, n in state.get("tally") or []})
        for kind, entries in (state.get("problems") or {}).items():
            cam.problems[kind] = list(entries)
        if cam.last_frame:
            cam.remember(cam.last_frame.lower())
        return cam


def _head_fingerprint(handle) -> tuple[str | None, int]:
    """sha256 of results.csv's first line, and its length; (None, 0) if unfinished."""
    handle.seek(0)
    head = handle.read(HEAD_BYTES)
    end = head.find(b"\n")
    if end < 0:
        return None, 0
    return hashlib.sha256(head[:end + 1]).hexdigest(), end + 1


# ---------------------------------------------------------------------- watcher


class Watcher:
    """Follows Aurora's folder and compares frame by frame. See the module doc.

    clock and sleep are injectable so tests can drive it step by step.
    """

    def __init__(self, root, engine, out_dir, threshold: float = 30.0, poll: float = 1.0,
                 settle: float = 3.0, sample: int = 0, summary_every: float = 60.0,
                 image_timeout: float = 15.0, from_now: bool = False,
                 settings: dict | None = None, clock=time.monotonic, sleep=time.sleep,
                 echo=print):
        self.root = Path(root)
        self.engine = engine
        self.out_dir = Path(out_dir)
        self.threshold = float(threshold)
        self.poll = float(poll)
        self.settle = float(settle)
        self.sample = int(sample)
        self.summary_every = float(summary_every)
        self.image_timeout = float(image_timeout)
        self.from_now = from_now
        self.settings = settings or {}
        self.clock = clock
        self.sleep = sleep
        self.echo = echo
        self.stop = False
        self.started = datetime.now().isoformat(timespec="seconds")
        self.cameras: dict[str, Camera] = {}
        self.unmatched_rows: list[dict] = []
        self._unmatched_fresh = True
        self._next_discover = 0.0
        self._next_summary = self.clock() + self.summary_every
        self._warned: set[str] = set()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out_dir / STATE_NAME
        self._load_state()
        # Now, so --from-now means the moment the watch starts.
        self.discover()
        self._next_discover = self.clock() + max(self.summary_every, self.poll)

    # ------------------------------------------------------------------ state

    def _load_state(self) -> None:
        if not self.state_path.exists():
            self._saved_cameras = {}
            return
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION:
            raise SystemExit(f"{self.state_path}: unknown state version {state.get('version')}")
        if _same_path(state.get("root", ""), self.root) is False:
            raise SystemExit(
                f"{self.state_path} belongs to {state.get('root')}, not {self.root}.\n"
                f"Use another --out, or remove that file to start over.")
        self.started = state.get("started", self.started)
        self._saved_cameras = state.get("cameras") or {}
        self._unmatched_fresh = False
        self.echo(f"resuming from {self.state_path}")

    def save_state(self) -> None:
        state = {
            "version": STATE_VERSION,
            "root": str(self.root),
            "started": self.started,
            "saved": datetime.now().isoformat(timespec="seconds"),
            "threshold_deg": self.threshold,
            "cameras": {name: cam.to_state() for name, cam in self.cameras.items()},
        }
        if not write_json_atomic(self.state_path, state, indent=None):
            self._warn("state", f"! could not write {self.state_path} - a restart may "
                                f"redo frames since the last save")

    def _warn(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.echo(message)

    # ---------------------------------------------------------------- cameras

    def discover(self) -> None:
        try:
            if (self.root / ca.RESULTS_NAME).is_file():
                found = [self.root]
            else:
                found = sorted(d for d in self.root.iterdir()
                               if d.is_dir() and (d / ca.RESULTS_NAME).is_file())
        except OSError as exc:
            self._warn("root", f"! cannot read {self.root}: {exc}")
            return
        self._warned.discard("root")
        first = not self.cameras
        for folder in found:
            name = folder.name
            if name in self.cameras:
                continue
            saved = self._saved_cameras.get(name)
            if saved is not None:
                cam = Camera.from_state(name, folder, saved)
            else:
                cam = Camera(name, folder)
                # Only cameras there at the start: one that appears later is new.
                if self.from_now and first:
                    self._start_at_end(cam)
            cam.last_growth = self.clock()
            self.cameras[name] = cam
            where = f"byte {cam.offset}" if cam.offset else "the beginning"
            self.echo(f"{name}: watching {folder / ca.RESULTS_NAME} from {where}")

    def _start_at_end(self, cam: Camera) -> None:
        """--from-now: after the last full line, skipping the frame it belongs to."""
        try:
            with open_shared(cam.folder / ca.RESULTS_NAME) as handle:
                cam.fingerprint, cam.fp_len = _head_fingerprint(handle)
                size = handle.seek(0, os.SEEK_END)
                start = max(0, size - TAIL_BYTES)
                handle.seek(start)
                tail = handle.read(size - start)
        except OSError as exc:
            self._warn(f"tail:{cam.name}", f"! {cam.name}: {exc}")
            return
        end = tail.rfind(b"\n")
        if end < 0:
            return
        cam.offset = cam.read_pos = start + end + 1
        last = _decode(tail[:end]).splitlines()
        if last and ";" in last[-1]:
            # Its frame may continue after the offset: skip that one partial frame.
            cam.skip_key = PureWindowsPath(last[-1].split(";")[0].strip()).name.lower()

    # ------------------------------------------------------------------ tail

    def _reset(self, cam: Camera, why: str) -> None:
        self.echo(f"{cam.name}: {ca.RESULTS_NAME} {why} - reading it from the beginning")
        cam.counts["csv_resets"] += 1
        cam.generation += 1
        if cam.open:
            cam.queue.append(cam.open)
            cam.open = None
        cam.offset = cam.read_pos = 0
        cam.buffer = b""
        cam.carry = None
        cam.fingerprint, cam.fp_len = None, 0
        cam.recent.clear()
        cam.skip_key = None

    def tail(self, cam: Camera, now: float) -> None:
        path = cam.folder / ca.RESULTS_NAME
        try:
            size = path.stat().st_size
            if size < cam.read_pos:
                self._reset(cam, "shrank")
            if size == cam.read_pos or len(cam.queue) >= MAX_QUEUE:
                cam.reachable = True
                return
            with open_shared(path) as handle:
                head, length = _head_fingerprint(handle)
                if cam.fingerprint and cam.read_pos and (
                        length < cam.fp_len or head != cam.fingerprint):
                    self._reset(cam, "was replaced")
                if cam.fingerprint is None:
                    cam.fingerprint, cam.fp_len = head, length
                handle.seek(cam.read_pos)
                data = handle.read(min(size - cam.read_pos, MAX_READ))
        except OSError as exc:
            if cam.reachable:
                self.echo(f"! {cam.name}: cannot read {path}: {exc}")
            cam.reachable = False
            return
        cam.reachable = True
        if not data:
            return
        cam.last_growth = now
        base = cam.read_pos - len(cam.buffer)
        cam.read_pos += len(data)
        data = cam.buffer + data
        last = data.rfind(b"\n")
        if last < 0:
            cam.buffer = data
            return
        cam.buffer = data[last + 1:]
        position = base
        for raw in data[:last + 1].split(b"\n")[:-1]:
            end = position + len(raw) + 1
            text = _decode(raw)
            if position == 0:
                text = text.lstrip("﻿")
            self._line(cam, text, position, end, now)
            position = end

    def _attach(self, cam: Camera, start: int, end: int, lines=0, malformed=None,
                late=0) -> None:
        target = cam.open
        if target is None:
            if cam.carry is None:
                cam.carry = {"start": start, "lines": 0, "malformed": [], "late": 0}
            target = cam.carry
            target["lines"] += lines
            target["late"] += late
            if malformed:
                target["malformed"].append(malformed)
            return
        target.lines += lines
        target.late += late
        target.end = end
        if malformed:
            target.malformed.append(malformed)

    def _line(self, cam: Camera, text: str, start: int, end: int, now: float) -> None:
        if not text.strip():
            self._attach(cam, start, end)
            return
        try:
            row = ca.parse_line(text)
        except (ValueError, OverflowError) as exc:
            self._attach(cam, start, end, lines=1, malformed=(start, str(exc)))
            return
        key = row.image.lower()
        if cam.open is not None and cam.open.key == key:
            cam.open.rows.append(row)
            cam.open.lines += 1
            cam.open.end = end
            return
        if key in cam.recent:
            # Its frame was already taken; merging now would split one frame in two.
            self._attach(cam, start, end, lines=1, late=1)
            cam.note("late_lines", row.image)
            return
        if cam.open is not None:
            self._close(cam, now)
        group = FrameGroup(name=row.image, key=key, start=start, end=end, seen=now,
                           generation=cam.generation, rows=[row], lines=1)
        if cam.carry:
            group.start = cam.carry["start"]
            group.lines += cam.carry["lines"]
            group.late += cam.carry["late"]
            group.malformed = cam.carry["malformed"] + group.malformed
            cam.carry = None
        if cam.skip_key is not None:
            group.skip = cam.skip_key == key
            cam.skip_key = None
        cam.open = group
        cam.remember(key)

    def _close(self, cam: Camera, now: float) -> None:
        group, cam.open = cam.open, None
        cam.queue.append(group)
        if not group.skip:
            self._look(cam, group, now)     # first look now, so a backlog is ready later

    # ---------------------------------------------------------------- frames

    def _look(self, cam: Camera, group: FrameGroup, now: float) -> tuple | None:
        try:
            st = (cam.folder / group.name).stat()
            stat = (st.st_size, st.st_mtime_ns)
            group.existed = True
        except OSError:
            stat = None
        if group.first_look is None:
            group.first_look = now
        if stat != group.last_stat or group.last_look is None:
            group.last_stat, group.last_look = stat, now
        return stat

    def _try_head(self, cam: Camera, now: float) -> bool:
        """Do the camera's oldest frame if it can be done now. True when done."""
        group = cam.queue[0]
        if group.skip:
            cam.counts["skipped_at_start"] += 1
            self._commit(cam, group, counted=False)
            return True
        before = (group.last_stat, group.last_look)
        stat = self._look(cam, group, now)
        steady = (stat is not None and before[0] == stat and before[1] is not None
                  and now - before[1] >= self.poll)
        if steady:
            path = cam.folder / group.name
            try:
                data, frame = read_image(path, stat[0])
            except OSError:
                data = frame = None
            if frame is not None:
                self._process(cam, group, path, frame, data)
                return True
        if now - group.first_look >= self.image_timeout:
            rows = len(group.rows)
            if group.existed:
                cam.counts["frames_unreadable"] += 1
                cam.note("frames_unreadable", group.name)
            else:
                cam.counts["frames_missing"] += 1
                cam.counts["rows_missing_image"] += rows
                cam.note("frames_missing", group.name)
            self._commit(cam, group)
            return True
        return False

    def _output(self, cam: Camera, what: str, write) -> None:
        try:
            write()
        except OSError as exc:
            cam.counts["output_errors"] += 1
            self.echo(f"! {cam.name}: could not write {what}: {exc}")

    def _process(self, cam: Camera, group: FrameGroup, path: Path, frame, data) -> None:
        result = ca.process_frame(self.engine, cam.name, path.name, frame, group.rows,
                                  self.threshold)
        if result is None:
            cam.counts["frames_engine_failed"] += 1
            cam.note("frames_engine_failed", group.name)
            self._commit(cam, group)
            return
        cam.counts["frames_compared"] += 1
        cam.counts["aurora_peppers"] += result["peppers"]
        cam.counts["our_detections"] += result["detections"]
        records, lonely, over = result["records"], result["unmatched"], result["over"]
        tally_frame(cam.tally, records, lonely)
        self.unmatched_rows.extend(lonely)
        lines = {path.name: [r.line for r in group.rows]}

        def save(out_dir):
            ca.write_review(out_dir, cam.name, path, frame, records, lonely,
                            self.threshold, data=data)
            ca.write_review_results(out_dir / "frames" / cam.name / ca.RESULTS_NAME, lines)

        if over:
            cam.counts["review_frames"] += 1
            cam.counts["review_pairs"] += len(over)
            self._output(cam, f"review of {path.name}", lambda: save(self.out_dir))
        elif any(r["diff_deg"] is not None for r in records):
            cam.agree += 1
            if self.sample and cam.agree % self.sample == 0:
                cam.counts["sample_frames"] += 1
                self._output(cam, f"sample of {path.name}",
                             lambda: save(self.out_dir / "sample"))
        self._commit(cam, group)

    def _commit(self, cam: Camera, group: FrameGroup, counted: bool = True) -> None:
        cam.queue.popleft()
        if counted:
            cam.counts["frames_in_csv"] += 1
        cam.counts["csv_lines"] += group.lines
        cam.counts["csv_malformed"] += len(group.malformed)
        cam.counts["late_lines"] += group.late
        for offset, reason in group.malformed:
            cam.note("malformed_lines", {"byte": offset, "reason": reason})
        if group.generation == cam.generation:
            cam.offset = group.end
        cam.last_frame = group.name
        self.flush_unmatched()
        self.save_state()

    # ---------------------------------------------------------------- output

    def flush_unmatched(self) -> None:
        if not self.unmatched_rows and not self._unmatched_fresh:
            return
        path = self.out_dir / "unmatched.csv"
        fresh = self._unmatched_fresh or not path.exists()
        try:
            with path.open("w" if fresh else "a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if fresh:
                    writer.writerow(["frame", "camera", "side", "x_px", "y_px", "detail"])
                for u in self.unmatched_rows:
                    writer.writerow([u["frame"], u["camera"], u["side"], u["x_px"],
                                     u["y_px"], u["detail"]])
        except OSError as exc:
            # Kept and tried again after the next frame: someone has it open.
            self._warn("unmatched", f"! could not write {path}: {exc}")
            return
        self._warned.discard("unmatched")
        self.unmatched_rows.clear()
        self._unmatched_fresh = False

    def _camera_watch(self, cam: Camera, now: float) -> dict:
        oldest = cam.queue[0].seen if cam.queue else (cam.open.seen if cam.open else None)
        size = None
        try:
            size = (cam.folder / ca.RESULTS_NAME).stat().st_size
        except OSError:
            pass
        return {
            "folder": str(cam.folder),
            "reachable": cam.reachable,
            "frames_done": cam.counts["frames_in_csv"] + cam.counts["skipped_at_start"],
            "frames_waiting": cam.waiting(),
            "oldest_waiting_s": None if oldest is None else round(now - oldest, 1),
            "unread_bytes": None if size is None else max(0, size - cam.read_pos),
            "last_frame": cam.last_frame,
            **{key: cam.counts[key] for key in WATCH_KEYS},
        }

    def summary(self) -> dict:
        now = self.clock()
        per_camera, problems, watch = {}, {}, {}
        totals, tally = Counter(), Counter()
        for name, cam in self.cameras.items():
            counts = {key: cam.counts[key] for key in ca.COUNT_KEYS}
            per_camera[name] = summary_from_tally(cam.tally, counts)
            totals.update(counts)
            tally.update(cam.tally)
            problems[name] = {"folder": str(cam.folder), **cam.problems,
                              "partial_line_bytes": len(cam.buffer)}
            watch[name] = self._camera_watch(cam, now)
        overall = summary_from_tally(tally, {key: totals[key] for key in ca.COUNT_KEYS})
        return {
            "created": datetime.now().isoformat(timespec="seconds"),
            "mode": "watch",
            "threshold_deg": self.threshold,
            "engine": self.settings,
            "watch": {
                "root": str(self.root),
                "started": self.started,
                "sample_every": self.sample,
                "images_not_in_csv": "not tracked in watch mode",
                "cameras": watch,
            },
            "cameras": per_camera,
            "overall": overall,
            "problems": problems,
        }

    def write_summary(self) -> dict:
        summary = self.summary()
        if not write_json_atomic(self.out_dir / "summary.json", summary):
            self.echo(f"! could not write {self.out_dir / 'summary.json'} - next time")
        return summary

    def status_line(self, summary: dict) -> str:
        parts = []
        for name, w in summary["watch"]["cameras"].items():
            text = f"{name} {w['frames_done']} done, {w['frames_waiting']} waiting"
            if w["oldest_waiting_s"] is not None and w["frames_waiting"]:
                text += f" (lag {w['oldest_waiting_s']:.0f}s)"
            if w["unread_bytes"]:
                text += f" +{w['unread_bytes'] // 1024} KB unread"
            if not w["reachable"]:
                text += " UNREACHABLE"
            parts.append(text)
        o = summary["overall"]
        sample = sum(w["sample_frames"] for w in summary["watch"]["cameras"].values())
        parts.append(f"review {o['review_frames']}, sample {sample}, "
                     f"missing {o['frames_missing']}, unreadable {o['frames_unreadable']}")
        return f"{datetime.now():%H:%M:%S} " + " | ".join(parts or ["no cameras yet"])

    # ------------------------------------------------------------------ loop

    def step(self) -> bool:
        """One poll: tail every camera, then frames for at most one poll's time.

        True when at least one frame was done (the loop then polls again
        without sleeping).
        """
        now = self.clock()
        if now >= self._next_discover:
            self.discover()
            self._next_discover = now + max(self.summary_every, self.poll)
        for cam in self.cameras.values():
            self.tail(cam, now)
            if (cam.open is not None and not cam.buffer
                    and now - cam.last_growth >= self.settle):
                self._close(cam, now)

        deadline = now + self.poll
        done = 0
        while not self.stop:
            progressed = False
            for cam in sorted((c for c in self.cameras.values() if c.queue),
                              key=lambda c: c.queue[0].seen):
                if self._try_head(cam, self.clock()):
                    progressed = True
                    done += 1
                if self.stop or self.clock() >= deadline:
                    break
            if not progressed or self.clock() >= deadline:
                break

        if self.clock() >= self._next_summary:
            self.echo(self.status_line(self.write_summary()))
            self._next_summary = self.clock() + self.summary_every
        return done > 0

    def request_stop(self, *_) -> None:
        if self.stop:
            raise KeyboardInterrupt
        self.stop = True
        self.echo("stopping after the current frame (Ctrl+C again to quit now)")

    def finish(self) -> dict:
        self.flush_unmatched()
        self.save_state()
        summary = self.write_summary()
        self.echo(self.status_line(summary))
        return summary

    def run(self) -> int:
        main_thread = threading.current_thread() is threading.main_thread()
        previous = signal.signal(signal.SIGINT, self.request_stop) if main_thread else None
        try:
            while not self.stop:
                if not self.step() and not self.stop:
                    self.sleep(self.poll)
        finally:
            if main_thread:
                signal.signal(signal.SIGINT, previous)
        summary = self.finish()
        if summary["cameras"]:
            ca.print_summary("ALL CAMERAS (watch)", summary["overall"], self.threshold)
        self.echo(f"-> {self.out_dir / 'summary.json'}")
        return 0


# ------------------------------------------------------------------------- main


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(str(path))).rstrip("\\/")


def _same_path(a, b) -> bool:
    return _norm(a) == _norm(b)


def overlaps(a, b) -> bool:
    a, b = _norm(a), _norm(b)
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.compare_aurora",
        description="Compare with Aurora Vision live, following its folder while "
                    "the line runs. Aurora's folder is only ever read.")
    parser.add_argument("--watch", required=True, metavar="ROOT",
                        help="Aurora's folder: Cam<ID> subfolders with Image_*.bmp and "
                             "results.csv (or one such camera folder)")
    parser.add_argument("--out", default=str(project_path(ca.DEFAULT_OUT)),
                        help=f"review folder (default: {ca.DEFAULT_OUT})")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="review pairs differing by more than this many degrees "
                             "(default 30)")
    parser.add_argument("--config", default=None, help="config file")
    parser.add_argument("--poll", type=float, default=1.0,
                        help="seconds between looks at results.csv (default 1)")
    parser.add_argument("--settle", type=float, default=3.0,
                        help="a frame's lines are complete after this many seconds "
                             "without new lines (default 3)")
    parser.add_argument("--image-timeout", type=float, default=15.0,
                        help="give up on a frame's image after this many seconds "
                             "(default 15)")
    parser.add_argument("--sample", type=int, default=0,
                        help="also keep every N-th frame where both agree (0 = off)")
    parser.add_argument("--summary-every", type=float, default=60.0,
                        help="rewrite summary.json and print a status every this many "
                             "seconds (default 60)")
    parser.add_argument("--from-now", action="store_true",
                        help="without a state file, start at the current end of each "
                             "results.csv instead of its beginning")
    args = parser.parse_args(argv)

    for name in ("poll", "settle", "image_timeout", "summary_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be more than 0")
    if args.sample < 0:
        parser.error("--sample must be 0 or more")

    root = Path(args.watch)
    if not root.is_dir():
        print(f"Not a folder: {root}")
        return 1
    out_dir = Path(os.path.abspath(args.out))
    if overlaps(out_dir, root):
        print(f"Refusing: the output and the watched folder overlap.\n"
              f"  watched: {root}\n  out:     {out_dir}")
        return 1

    engine, settings = ca.build_engine(args.config)
    watcher = Watcher(root, engine, out_dir, threshold=args.threshold, poll=args.poll,
                      settle=args.settle, sample=args.sample,
                      summary_every=args.summary_every, image_timeout=args.image_timeout,
                      from_now=args.from_now, settings=settings)
    return watcher.run()


if __name__ == "__main__":
    raise SystemExit(main())
