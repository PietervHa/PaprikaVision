"""
Compare this app with the production Aurora Vision program, offline

    python -m tools.compare_aurora D:/aurora_copy                  compare
    python -m tools.compare_aurora D:/aurora_copy --threshold 20   stricter review
    python -m tools.compare_aurora data/datasets/aurora_review/frames \
        --score-aurora --labels data/debug/aurora_labels           score Aurora

Input is what Aurora logs on the production PC, copied off it: per belt camera
a folder of Image_*.bmp and the results.csv Aurora wrote next to them. Point
the tool at one camera folder, or at a folder whose subfolders are camera
folders. Nothing here talks to the line, and source frames are only read.

results.csv
-----------
No header, ';' separated, one line per pepper:

    image_file;camera_id;pepper_index;x_px;y_px;orientation_deg;confidence;method;stem_source[;pushed]

    image_file       may be a full Windows path - matched on the file name only
    pepper_index     -1 with empty fields: Aurora analysed the frame, found nothing
    x_px, y_px       pepper mass centre, in pixels of the saved image
    orientation_deg  stem direction, -180..180, 0 = +X, y DOWN (clockwise positive)
    method           0 axis+stem  1 stem only  2 axis only  3 undetermined  4 axis+shape
    stem_source      0 none  1 Aurora DL model  2 green fallback
    pushed           optional, 0/1: whether Aurora sent the pepper to the PLC

Decimals may use '.' or ','. A last line without a line end is skipped together
with the rest of its frame, because Aurora may still be writing it. Malformed
lines and lines whose image is missing are counted and reported, never fatal.

What Aurora's methods mean here
-------------------------------
The CSV says whether Aurora produced a direction, not whether it acted on it:
pushes to the PLC are only known when the pushed column is there. So nothing
here calls an Aurora result "placed".

    direction given  methods 0, 1, 4 - compared as directions, 0-180 deg
                     (4 has no stem and is also reported on its own)
    axis only        method 2 - no stem end, compared as an axis, 0-90 deg
    undetermined     method 3 - not compared, counted

Output (default data/datasets/aurora_review/)
---------------------------------------------
    summary.json                            per camera and overall
    unmatched.csv                           peppers only one side found
    frames/<camera>/Image_x.bmp             originals of frames to review
    frames/<camera>/results.csv             Aurora's lines for those frames
    overlays/<camera>/Image_x.overlay.jpg   both arrows, labelled
    overlays/<camera>/Image_x.json          every pair in that frame

A frame goes to review only when a matched pair differs by more than
--threshold degrees. Unmatched peppers alone do not qualify: Aurora analyses
only the top layer, so fruit we see underneath are expected to be missing from
its results.

frames/ holds originals and Aurora's lines and nothing else, so it can be
labelled as a holdout set and both systems scored on it. Label into a separate
--out: Aurora's Image_*.bmp names can collide with frames already labelled.

    python -m tools.label_stems data/datasets/aurora_review/frames --out data/debug/aurora_labels
    python -m tools.label_stems data/datasets/aurora_review/frames --out data/debug/aurora_labels --score
    python -m tools.compare_aurora data/datasets/aurora_review/frames --score-aurora --labels data/debug/aurora_labels

It sits under data/datasets, which retrain skips when it collects labelled
frames, so the review set never ends up in training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detection.paprika import orientation as orient  # noqa: E402
from backend.utils.paths import project_path  # noqa: E402
from tools import label_stems  # noqa: E402

RESULTS_NAME = "results.csv"
DEFAULT_OUT = "data/datasets/aurora_review"

METHOD_NAMES = {0: "axis+stem", 1: "stem only", 2: "axis only",
                3: "undetermined", 4: "axis+shape"}
STEM_SOURCES = {0: "none", 1: "aurora DL", 2: "green fallback"}
DIRECTION_METHODS = (0, 1, 4)
AXIS_ONLY = 2
UNDETERMINED = 3

DIRECTION_GIVEN = "direction given"
AXIS = "axis only"
UNDET = "undetermined"
VERDICTS = (DIRECTION_GIVEN, AXIS, UNDET)
NO_AURORA = "no Aurora pepper"          # column: our fruit Aurora did not report
NOT_DETECTED = "NOT DETECTED"           # row: Aurora's pepper we did not find

BUCKETS = ("<10", "10-20", "20-45", "45-120", ">120")

OURS_COLOUR = (60, 220, 60)             # BGR green
AURORA_COLOUR = (255, 60, 255)          # BGR magenta


def aurora_to_ours(aurora_deg: float) -> float:
    """Aurora's stem direction in this app's convention.

    Aurora: -180..180, 0 = +X, y down, so positive turns clockwise on screen.
    Ours (orientation.vector_to_angle): 0..360, 0 = +X, counter-clockwise.
    Same zero, opposite sense: negate and wrap.
    """
    angle = (-float(aurora_deg)) % 360.0
    # A tiny positive input wraps to exactly 360.0 in floating point.
    return 0.0 if angle >= 360.0 else angle


def bucket(diff: float) -> str:
    """Name of the difference bucket; >120 is the same "flipped" as label_stems."""
    if diff < 10:
        return "<10"
    if diff < 20:
        return "10-20"
    if diff < 45:
        return "20-45"
    if diff <= 120:
        return "45-120"
    return ">120"


# ------------------------------------------------------------------ results.csv


@dataclass
class AuroraRow:
    image: str
    camera_id: str
    index: int
    x: float | None = None
    y: float | None = None
    angle: float | None = None          # Aurora's own convention
    confidence: float | None = None
    method: int | None = None
    stem_source: int | None = None
    pushed: int | None = None
    line: str = ""

    @property
    def verdict(self) -> str:
        if self.method in DIRECTION_METHODS:
            return DIRECTION_GIVEN
        if self.method == AXIS_ONLY:
            return AXIS
        return UNDET


@dataclass
class ResultsFile:
    rows: list[AuroraRow] = field(default_factory=list)
    lines: int = 0
    malformed: list[tuple[int, str]] = field(default_factory=list)
    partial_last_line: str | None = None
    dropped_frame: str | None = None

    @property
    def has_pushed(self) -> bool:
        return any(r.pushed is not None for r in self.rows)


def _number(text: str) -> float:
    text = text.strip().replace(",", ".")
    if not text:
        raise ValueError("empty field")
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text!r} is not a number")
    return value


def _whole(text: str, name: str, allowed) -> int:
    value = _number(text)
    if value != int(value) or int(value) not in allowed:
        raise ValueError(f"{name} {text.strip()!r} is not one of {sorted(allowed)}")
    return int(value)


def parse_line(line: str) -> AuroraRow:
    """One results.csv line; raises ValueError if it cannot be used."""
    fields = line.strip().split(";")
    if len(fields) < 9 or (len(fields) > 10 and any(f.strip() for f in fields[10:])):
        raise ValueError(f"{len(fields)} fields, expected 9 or 10")
    image = PureWindowsPath(fields[0].strip()).name
    if not image:
        raise ValueError("no image file")
    index = _number(fields[2])
    if index != int(index) or index < -1:
        raise ValueError(f"pepper_index {fields[2].strip()!r}")
    row = AuroraRow(image=image, camera_id=fields[1].strip(), index=int(index),
                    line=line.strip())
    if len(fields) >= 10 and fields[9].strip():
        row.pushed = _whole(fields[9], "pushed", {0, 1})
    if row.index < 0:
        return row                    # analysed, nothing found; the rest is empty

    row.x, row.y = _number(fields[3]), _number(fields[4])
    row.method = _whole(fields[7], "method", set(METHOD_NAMES))
    row.stem_source = _whole(fields[8], "stem_source", set(STEM_SOURCES))
    if fields[6].strip():
        row.confidence = _number(fields[6])
    # An undetermined pepper has no direction, and Aurora may leave it empty.
    if row.method != UNDETERMINED or fields[5].strip():
        row.angle = _number(fields[5])
        if not -180.0 <= row.angle <= 180.0:
            raise ValueError(f"orientation {row.angle} outside -180..180")
    return row


def parse_results(path: Path) -> ResultsFile:
    """Read results.csv, keeping what is usable and counting what is not."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    text = text.lstrip("\ufeff")
    lines = text.splitlines()
    result = ResultsFile()

    # Aurora writes while the line runs. A last line without its line end may
    # be cut anywhere - including inside a number, where it would still parse,
    # just to the wrong value - so it is never used. The frame it belongs to is
    # dropped as a whole: the lines before it may be only part of that frame.
    if lines and not text.endswith(("\n", "\r")):
        result.partial_last_line = lines.pop()
        name = result.partial_last_line.split(";")[0].strip()
        if ";" in result.partial_last_line and name:
            result.dropped_frame = PureWindowsPath(name).name

    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        result.lines += 1
        try:
            result.rows.append(parse_line(line))
        except (ValueError, OverflowError) as exc:
            result.malformed.append((number, str(exc)))

    if result.dropped_frame:
        dropped = result.dropped_frame.lower()
        result.rows = [r for r in result.rows if r.image.lower() != dropped]
    return result


def find_cameras(paths) -> list[tuple[str, Path]]:
    """(name, folder) for every camera folder: one holding a results.csv."""
    cameras: list[tuple[str, Path]] = []
    for given in paths:
        root = Path(given).resolve()
        if not root.is_dir():
            raise SystemExit(f"Not a folder: {root}")
        if (root / RESULTS_NAME).is_file():
            found = [root]
        else:
            found = sorted(d for d in root.iterdir()
                           if d.is_dir() and (d / RESULTS_NAME).is_file())
        if not found:
            raise SystemExit(f"No {RESULTS_NAME} in {root} or in its subfolders")
        cameras.extend((folder.name, folder) for folder in found)

    names = Counter(name for name, _ in cameras)
    seen: Counter = Counter()
    unique = []
    for name, folder in cameras:
        seen[name] += 1
        unique.append((name if names[name] == 1 else f"{name}_{seen[name]}", folder))
    return unique


# --------------------------------------------------------------------- matching


def pair_up(peppers: list[AuroraRow], detections: list[dict]):
    """Pair Aurora's peppers with our detections.

    label_stems.match does the matching, so the rule is the one --score uses:
    Aurora's mass centre inside our box, nearest box centre wins. A detection
    is used once; a second Aurora centre in the same box is reported unmatched.

    Returns (pairs, unmatched Aurora peppers, unmatched detections).
    """
    free = list(detections)
    pairs, lonely = [], []
    for pepper in peppers:
        found = label_stems.match({"centroid_xy": (pepper.x, pepper.y)}, free)
        if found is None:
            lonely.append(pepper)
            continue
        free = [d for d in free if d is not found]
        pairs.append((pepper, found))
    return pairs, lonely, free


def pair_record(camera: str, frame: str, pepper: AuroraRow, detection: dict) -> dict:
    orientation = detection.get("orientation") or {}
    ours = orientation.get("angle_deg")
    aurora = aurora_to_ours(pepper.angle) if pepper.angle is not None else None
    diff = full = None
    compared_as = None
    if ours is not None and aurora is not None and pepper.verdict != UNDET:
        full = round(orient.angular_difference(ours, aurora), 2)
        if pepper.verdict == AXIS:
            diff, compared_as = round(orient.axis_difference(ours, aurora), 2), "axis"
        else:
            diff, compared_as = full, "direction"
    return {
        "camera": camera,
        "frame": frame,
        "aurora": {
            "x": pepper.x, "y": pepper.y,
            "angle_deg": pepper.angle,
            "angle_ours_convention": None if aurora is None else round(aurora, 2),
            "confidence": pepper.confidence,
            "method": pepper.method,
            "method_name": METHOD_NAMES.get(pepper.method),
            "stem_source": pepper.stem_source,
            "stem_source_name": STEM_SOURCES.get(pepper.stem_source),
            "verdict": pepper.verdict,
            "pushed": pepper.pushed,
        },
        "ours": {
            "bbox": detection.get("bbox"),
            "center": detection.get("center"),
            "angle_deg": ours,
            "placement": detection.get("placement"),
            "source": orientation.get("source"),
            "confidence": orientation.get("confidence"),
        },
        "diff_deg": diff,
        "full_diff_deg": full,
        "compared_as": compared_as,
    }


def unmatched_record(camera: str, frame: str, side: str, item) -> dict:
    if side == "ours":
        x, y = item.get("center") or (None, None)
        return {"camera": camera, "frame": frame, "side": "ours", "x_px": x, "y_px": y,
                "detail": item.get("placement")}
    return {"camera": camera, "frame": frame, "side": "aurora", "x_px": item.x,
            "y_px": item.y, "detail": f"method {item.method} {item.verdict}",
            "verdict": item.verdict, "pushed": item.pushed}


# ---------------------------------------------------------------------- overlay


def _text(canvas, text, origin, colour, scale) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(scale * 1.6)))
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    x = int(min(max(2, origin[0]), max(2, canvas.shape[1] - width - 2)))
    y = int(min(max(height + 2, origin[1]), canvas.shape[0] - 4))
    cv2.putText(canvas, text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, records: list[dict], threshold: float) -> np.ndarray:
    """Both arrows on the frame, each labelled with whose it is.

    Aurora's arrow is drawn from its RAW number in its own convention (y down),
    not through aurora_to_ours. If the conversion were ever wrong, the arrows
    on screen would still be right and the printed difference would visibly
    disagree with them, instead of the overlay repeating the mistake.
    """
    canvas = frame.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    height, width = canvas.shape[:2]
    scale = max(0.5, width / 1600)

    for record in records:
        ours, aurora = record["ours"], record["aurora"]
        flagged = record["diff_deg"] is not None and record["diff_deg"] > threshold
        thick = 3 if flagged else 1
        x1, y1, x2, y2 = ours["bbox"]
        length = max(40.0, 0.6 * max(x2 - x1, y2 - y1))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), OURS_COLOUR, 1)

        cx, cy = ours["center"]
        if ours["angle_deg"] is not None:
            rad = math.radians(ours["angle_deg"])
            tip = (int(cx + length * math.cos(rad)), int(cy - length * math.sin(rad)))
            cv2.arrowedLine(canvas, (int(cx), int(cy)), tip, OURS_COLOUR, thick + 1,
                            cv2.LINE_AA, tipLength=0.2)
            _text(canvas, f"OURS {ours['angle_deg']:.0f} {ours['placement']}",
                  (tip[0] + 6, tip[1]), OURS_COLOUR, scale)
        else:
            _text(canvas, f"OURS no angle {ours['placement']}",
                  (int(cx) + 6, int(cy) - 10), OURS_COLOUR, scale)

        ax, ay = aurora["x"], aurora["y"]
        start = (int(ax), int(ay))
        cv2.circle(canvas, start, 5, AURORA_COLOUR, -1)
        tag = f"m{aurora['method']} s{aurora['stem_source']}"
        if aurora["angle_deg"] is None or aurora["method"] == UNDETERMINED:
            _text(canvas, f"AURORA undetermined {tag}", (start[0] + 6, start[1] + 24),
                  AURORA_COLOUR, scale)
        else:
            rad = math.radians(aurora["angle_deg"])
            tip = (int(ax + length * math.cos(rad)), int(ay + length * math.sin(rad)))
            if aurora["method"] == AXIS_ONLY:
                tail = (int(ax - length * math.cos(rad)), int(ay - length * math.sin(rad)))
                cv2.line(canvas, tail, tip, AURORA_COLOUR, thick + 1, cv2.LINE_AA)
                tag += " axis"
            else:
                cv2.arrowedLine(canvas, start, tip, AURORA_COLOUR, thick + 1,
                                cv2.LINE_AA, tipLength=0.2)
            _text(canvas, f"AURORA {aurora['angle_deg']:.0f} "
                          f"(={aurora['angle_ours_convention']:.0f}) {tag}",
                  (tip[0] + 6, tip[1] + 24), AURORA_COLOUR, scale)

        if record["diff_deg"] is not None:
            _text(canvas, f"diff {record['diff_deg']:.0f}", (x1, y2 + int(28 * scale)),
                  (0, 0, 255) if flagged else (255, 255, 255), scale)

    legend = [
        ("green  = OURS    angle 0=+X, counter-clockwise", OURS_COLOUR),
        ("magenta = AURORA raw angle 0=+X, y down (=ours)", AURORA_COLOUR),
        (f"thick = differs more than {threshold:g} deg", (255, 255, 255)),
    ]
    for index, (line, colour) in enumerate(legend):
        _text(canvas, line, (10, int((index + 1) * 30 * scale) + 6), colour, scale)
    return canvas


# ---------------------------------------------------------------- review output


def write_review(out_dir: Path, camera: str, image_path: Path, frame: np.ndarray,
                 records: list[dict], unmatched: list[dict], threshold: float) -> None:
    frames_dir = out_dir / "frames" / camera
    overlays_dir = out_dir / "overlays" / camera
    frames_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    # The original, byte for byte and under its own name, so it can be
    # labelled and scored like any other frame.
    shutil.copy2(image_path, frames_dir / image_path.name)
    cv2.imwrite(str(overlays_dir / f"{image_path.stem}.overlay.jpg"),
                draw_overlay(frame, records, threshold))
    flagged = [dict(r, over_threshold=r["diff_deg"] is not None and r["diff_deg"] > threshold)
               for r in records]
    record = {
        "frame": image_path.name,
        "camera": camera,
        "source": str(image_path),
        "threshold_deg": threshold,
        "pairs": flagged,
        "unmatched": unmatched,
    }
    (overlays_dir / f"{image_path.stem}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")


def write_review_results(path: Path, frames: dict[str, list[str]]) -> None:
    """Aurora's lines for the reviewed frames, so --score-aurora can run on them.

    Merged with what an earlier run left: lines for frames reviewed again now
    are replaced, the rest kept, so re-running never duplicates a pepper.
    """
    if not frames:
        return
    replaced = {name.lower() for name in frames}
    keep = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name = PureWindowsPath(line.split(";")[0].strip()).name.lower()
            if line.strip() and name not in replaced:
                keep.append(line)
    lines = keep + [line for group in frames.values() for line in group]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_unmatched(path: Path, unmatched: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "camera", "side", "x_px", "y_px", "detail"])
        for u in unmatched:
            writer.writerow([u["frame"], u["camera"], u["side"], u["x_px"], u["y_px"],
                             u["detail"]])


# ---------------------------------------------------------------------- summary


def _buckets(diffs: list[float]) -> dict:
    counts = {name: 0 for name in BUCKETS}
    for diff in diffs:
        counts[bucket(diff)] += 1
    return {"n": len(diffs), **counts}


def summarise(pairs: list[dict], unmatched: list[dict], counts: dict) -> dict:
    compared = [p for p in pairs if p["diff_deg"] is not None]
    direction = [p for p in compared if p["aurora"]["verdict"] == DIRECTION_GIVEN]
    axis = [p for p in compared if p["aurora"]["verdict"] == AXIS]

    by_method = {}
    for method in sorted(METHOD_NAMES):
        subset = [p["diff_deg"] for p in compared if p["aurora"]["method"] == method]
        if subset:
            suffix = ", modulo 180" if method == AXIS_ONLY else ""
            by_method[f"{method} {METHOD_NAMES[method]}{suffix}"] = _buckets(subset)
    by_source = {}
    for source in sorted(STEM_SOURCES):
        subset = [p["diff_deg"] for p in direction if p["aurora"]["stem_source"] == source]
        if subset:
            by_source[f"{source} {STEM_SOURCES[source]}"] = _buckets(subset)

    vs_aurora: dict[str, Counter] = {}
    for p in pairs:
        vs_aurora.setdefault(p["ours"]["placement"], Counter())[p["aurora"]["verdict"]] += 1
    for u in unmatched:
        if u["side"] == "ours":
            vs_aurora.setdefault(u["detail"], Counter())[NO_AURORA] += 1
        else:
            vs_aurora.setdefault(NOT_DETECTED, Counter())[u["verdict"]] += 1

    summary = {
        **counts,
        "matched": len(pairs),
        "unmatched_ours": sum(1 for u in unmatched if u["side"] == "ours"),
        "unmatched_aurora": sum(1 for u in unmatched if u["side"] == "aurora"),
        "matched_by_aurora_verdict": {v: sum(1 for p in pairs if p["aurora"]["verdict"] == v)
                                      for v in VERDICTS},
        # Aurora gave something to compare, but our engine withheld its angle.
        "our_angle_withheld": sum(1 for p in pairs if p["aurora"]["verdict"] != UNDET
                                  and p["ours"]["angle_deg"] is None),
        "difference_deg": {
            "direction given (0,1,4)": _buckets([p["diff_deg"] for p in direction]),
            "axis only (2), modulo 180": _buckets([p["diff_deg"] for p in axis]),
            "by_method": by_method,
            "by_stem_source (direction given)": by_source,
        },
        "placement_vs_aurora": {k: dict(v) for k, v in sorted(vs_aurora.items())},
    }

    pushed_known = [p for p in pairs if p["aurora"]["pushed"] is not None]
    lonely_known = [u for u in unmatched if u["side"] == "aurora" and u["pushed"] is not None]
    if pushed_known or lonely_known:
        vs_pushed: dict[str, Counter] = {}
        for p in pushed_known:
            column = "pushed" if p["aurora"]["pushed"] else "not pushed"
            vs_pushed.setdefault(p["ours"]["placement"], Counter())[column] += 1
        for u in lonely_known:
            column = "pushed" if u["pushed"] else "not pushed"
            vs_pushed.setdefault(NOT_DETECTED, Counter())[column] += 1
        summary["placement_vs_pushed"] = {k: dict(v) for k, v in sorted(vs_pushed.items())}
    return summary


def _print_table(title: str, table: dict, columns) -> None:
    print(f"  {title}")
    print(f"    {'ours':<16}" + "".join(f"{c:>18}" for c in columns))
    for row, values in table.items():
        print(f"    {str(row):<16}" + "".join(f"{values.get(c, 0):>18}" for c in columns))


def print_summary(title: str, s: dict, threshold: float) -> None:
    print(f"\n=== {title} ===")
    print(f"  frames compared {s['frames_compared']}   (in csv {s['frames_in_csv']}, "
          f"image missing {s['frames_missing']}, unreadable {s['frames_unreadable']}, "
          f"engine failed {s['frames_engine_failed']}; "
          f"images not in csv {s['images_not_in_csv']})")
    print(f"  csv: {s['csv_lines']} lines, {s['csv_malformed']} malformed, "
          f"{s['rows_missing_image']} with image missing, "
          f"{s['csv_partial_last_line']} partial last line(s) skipped")
    print(f"  Aurora peppers {s['aurora_peppers']}   our detections {s['our_detections']}"
          f"   matched {s['matched']}   unmatched: ours {s['unmatched_ours']}, "
          f"Aurora {s['unmatched_aurora']}")
    verdicts = s["matched_by_aurora_verdict"]
    print(f"  matched, by Aurora: direction given {verdicts[DIRECTION_GIVEN]}, "
          f"axis only {verdicts[AXIS]}, undetermined {verdicts[UNDET]} (not compared);"
          f"  our angle withheld {s['our_angle_withheld']}")

    print(f"\n  {'difference (deg)':<34}{'n':>6}" + "".join(f"{b:>11}" for b in BUCKETS))
    diffs = s["difference_deg"]
    lines = [(k, diffs[k]) for k in ("direction given (0,1,4)", "axis only (2), modulo 180")]
    lines += [(f"method {k}", v) for k, v in diffs["by_method"].items()]
    lines += [(f"stem {k}", v) for k, v in diffs["by_stem_source (direction given)"].items()]
    for name, b in lines:
        cells = "".join(
            f"{b[c]:>5} {100 * b[c] / b['n']:3.0f}% " if b["n"] else f"{'-':>11}"
            for c in BUCKETS)
        print(f"  {name:<34}{b['n']:>6}{cells}")

    print()
    _print_table("our placement vs Aurora", s["placement_vs_aurora"],
                 VERDICTS + (NO_AURORA,))
    if "placement_vs_pushed" in s:
        _print_table("our placement vs Aurora's push", s["placement_vs_pushed"],
                     ("pushed", "not pushed"))
    print(f"\n  review: {s['review_frames']} frame(s), {s['review_pairs']} pair(s) "
          f"over {threshold:g} deg")


# ---------------------------------------------------------------------- compare


COUNT_KEYS = ("frames_in_csv", "frames_compared", "frames_missing", "frames_unreadable",
              "frames_engine_failed",
              "images_not_in_csv", "csv_lines", "csv_malformed", "csv_partial_last_line",
              "rows_missing_image", "aurora_peppers", "our_detections",
              "review_frames", "review_pairs")


def compare(cameras: list[tuple[str, Path]], engine, out_dir: Path, threshold: float = 30.0,
            limit: int = 0, settings: dict | None = None) -> dict:
    """Run the engine on every frame Aurora logged, pair, summarise, write review."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_pairs: list[dict] = []
    all_unmatched: list[dict] = []
    per_camera: dict[str, dict] = {}
    problems: dict[str, dict] = {}

    for camera, folder in cameras:
        results = parse_results(folder / RESULTS_NAME)
        by_frame: dict[str, list[AuroraRow]] = {}
        for row in results.rows:
            by_frame.setdefault(row.image, []).append(row)
        on_disk = {p.name.lower(): p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in label_stems.IMAGE_SUFFIXES}
        logged = {name.lower() for name in by_frame}
        if results.dropped_frame:
            logged.add(results.dropped_frame.lower())

        counts = Counter({key: 0 for key in COUNT_KEYS})
        counts["frames_in_csv"] = len(by_frame)
        counts["images_not_in_csv"] = sum(1 for name in on_disk if name not in logged)
        counts["csv_lines"] = results.lines + (1 if results.partial_last_line else 0)
        counts["csv_malformed"] = len(results.malformed)
        counts["csv_partial_last_line"] = 1 if results.partial_last_line else 0
        missing, unreadable, failed = [], [], []
        pairs: list[dict] = []
        unmatched: list[dict] = []
        review_lines: dict[str, list[str]] = {}

        frames = list(by_frame.items())
        if limit:
            frames = frames[:limit]
        print(f"{camera}: {len(frames)} frame(s) from {folder}")
        for number, (name, rows) in enumerate(frames, 1):
            path = on_disk.get(name.lower())
            if path is None:
                missing.append(name)
                counts["rows_missing_image"] += len(rows)
                continue
            frame = cv2.imread(str(path))
            if frame is None:
                unreadable.append(name)
                continue
            evaluated = engine.evaluate(frame)
            if evaluated.get("failure_reason") in ("detector_unavailable", "detector_error"):
                # Not "we found nothing": every Aurora pepper would count as
                # unmatched and read like a detector that misses fruit.
                failed.append(name)
                continue
            counts["frames_compared"] += 1
            detections = evaluated["detections"]
            peppers = [r for r in rows if r.index >= 0]
            counts["aurora_peppers"] += len(peppers)
            counts["our_detections"] += len(detections)

            matched, lonely_aurora, lonely_ours = pair_up(peppers, detections)
            records = [pair_record(camera, path.name, p, d) for p, d in matched]
            lonely = ([unmatched_record(camera, path.name, "ours", d) for d in lonely_ours]
                      + [unmatched_record(camera, path.name, "aurora", p)
                         for p in lonely_aurora])
            pairs.extend(records)
            unmatched.extend(lonely)

            over = [r for r in records if r["diff_deg"] is not None and r["diff_deg"] > threshold]
            if over:
                counts["review_frames"] += 1
                counts["review_pairs"] += len(over)
                write_review(out_dir, camera, path, frame, records, lonely, threshold)
                review_lines[path.name] = [r.line for r in rows]
            if number % 100 == 0:
                print(f"  {number}/{len(frames)}")

        counts["frames_missing"] = len(missing)
        counts["frames_unreadable"] = len(unreadable)
        counts["frames_engine_failed"] = len(failed)
        write_review_results(out_dir / "frames" / camera / RESULTS_NAME, review_lines)
        per_camera[camera] = summarise(pairs, unmatched, dict(counts))
        problems[camera] = {
            "folder": str(folder),
            "malformed_lines": [{"line": n, "reason": why} for n, why in results.malformed],
            "partial_last_line": results.partial_last_line,
            "dropped_incomplete_frame": results.dropped_frame,
            "frames_missing": missing,
            "frames_unreadable": unreadable,
            "frames_engine_failed": failed,
        }
        all_pairs.extend(pairs)
        all_unmatched.extend(unmatched)

    totals = Counter()
    for s in per_camera.values():
        totals.update({key: s[key] for key in COUNT_KEYS})
    overall = summarise(all_pairs, all_unmatched, {key: totals[key] for key in COUNT_KEYS})

    for camera, s in per_camera.items():
        print_summary(f"camera {camera}", s, threshold)
        p = problems[camera]
        for entry in p["malformed_lines"][:10]:
            print(f"  ! line {entry['line']}: {entry['reason']}")
        if p["dropped_incomplete_frame"]:
            print(f"  ! last line unfinished - skipped frame {p['dropped_incomplete_frame']}")
        if p["frames_missing"]:
            shown = ", ".join(p["frames_missing"][:10])
            more = len(p["frames_missing"]) - 10
            print(f"  ! image missing: {shown}" + (f" (+{more} more)" if more > 0 else ""))
    if len(per_camera) > 1:
        print_summary("ALL CAMERAS", overall, threshold)

    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "threshold_deg": threshold,
        "engine": settings or {},
        "cameras": per_camera,
        "overall": overall,
        "problems": problems,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_unmatched(out_dir / "unmatched.csv", all_unmatched)
    print(f"\n-> {out_dir / 'summary.json'}")
    print(f"-> {out_dir / 'unmatched.csv'}")
    if overall["review_frames"]:
        print(f"-> {out_dir / 'frames'}   (originals to review / label)")
        print(f"-> {out_dir / 'overlays'} (both arrows + JSON per frame)")
    return summary


def engine_settings(cfg: dict) -> dict:
    """What was run, recorded with the results."""
    paprika = cfg.get("paprika") or {}
    backend = paprika.get("backend")
    block = (paprika.get(backend) or {}) if backend else {}
    policy = paprika.get("policy") or {}
    return {
        "backend": backend,
        "model_path": block.get("model_path"),
        "device": block.get("device"),
        "min_angle_confidence": policy.get("min_angle_confidence"),
    }


# ------------------------------------------------------------------ score Aurora


def match_label(entry: dict, peppers: list[AuroraRow]) -> AuroraRow | None:
    """The Aurora pepper a labelled fruit belongs to.

    label_stems.match turned round. There the labelled centroid has to land in
    our detection's box; Aurora reports a point and no box, so here Aurora's
    point has to land in the box stored with the label, and where several do,
    the one nearest the labelled centroid wins.
    """
    box = entry.get("bbox")
    if not box:
        return None
    x1, y1, x2, y2 = box
    gx, gy = entry["centroid_xy"]
    best, best_distance = None, None
    for pepper in peppers:
        if not (x1 <= pepper.x <= x2 and y1 <= pepper.y <= y2):
            continue
        distance = math.hypot(pepper.x - gx, pepper.y - gy)
        if best_distance is None or distance < best_distance:
            best, best_distance = pepper, distance
    return best


def score_aurora(folders, labels_dir: Path, worst: int = 10) -> int:
    """Score Aurora against stem_labels.json, with label_stems' own scoring code.

    The truth angle, the error formula (label_stems.angle_error) and the report
    lines (label_stems.report, report_within) are the ones --score uses for our
    engine, so the two reports can be read side by side.

    "error" is the full-angle error for every pepper with an angle - method 2
    included: if Aurora uses that angle anyway, 180 degrees out is a real flip.
    Method 2 is also reported modulo 180, as an axis.
    """
    labels_path = Path(labels_dir).resolve() / "stem_labels.json"
    labels = label_stems.load_labels(labels_path)
    if not labels:
        print(f"No labels in {labels_path} - run label_stems on the frames first.")
        return 1

    by_frame: dict[str, list[AuroraRow]] = {}
    ambiguous: set[str] = set()
    for camera, folder in find_cameras(folders):
        results = parse_results(folder / RESULTS_NAME)
        if results.malformed or results.partial_last_line:
            print(f"({camera}: {len(results.malformed)} malformed line(s), "
                  f"{1 if results.partial_last_line else 0} partial last line skipped)")
        here: dict[str, list[AuroraRow]] = {}
        for row in results.rows:
            here.setdefault(row.image.lower(), []).append(row)
        for key, rows in here.items():
            if key in by_frame:
                ambiguous.add(key)
            by_frame[key] = rows

    rows: list[dict] = []
    missing = skipped_ambiguous = no_box = 0
    for name, entries in sorted(labels.items()):
        wanted = [e for e in entries if e.get("stem_xy")]
        if not wanted:
            continue
        key = name.lower()
        if key in ambiguous:
            skipped_ambiguous += 1
            continue
        if key not in by_frame:
            missing += 1
            continue
        peppers = [p for p in by_frame[key] if p.index >= 0]
        for entry in wanted:
            if not entry.get("bbox"):
                no_box += 1
                continue
            pepper = match_label(entry, peppers)
            if pepper is None:
                rows.append({"verdict": NOT_DETECTED, "method": None, "stem_source": None,
                             "pushed": None, "error": None, "axis_error": None})
                continue
            truth = entry["true_angle_deg"]
            reported = error = axis_error = None
            if pepper.angle is not None and pepper.method != UNDETERMINED:
                reported = aurora_to_ours(pepper.angle)
                error = label_stems.angle_error(reported, truth)
                if pepper.method == AXIS_ONLY:
                    axis_error = min(error, 180 - error)
            rows.append({
                "verdict": pepper.verdict, "method": pepper.method,
                "stem_source": pepper.stem_source, "pushed": pepper.pushed,
                "error": error, "axis_error": axis_error,
                "frame": name, "reported": reported, "raw": pepper.angle, "truth": truth,
            })

    if missing:
        print(f"({missing} labelled frame(s) not in any {RESULTS_NAME})")
    if skipped_ambiguous:
        print(f"({skipped_ambiguous} labelled frame(s) skipped: the name is in more than "
              f"one camera's {RESULTS_NAME})")
    if no_box:
        print(f"({no_box} label(s) without a bbox - cannot be matched)")
    if not rows:
        print("Nothing to score.")
        return 1

    report = label_stems.report
    print(f"\n{len(rows)} labelled fruit scored against Aurora\n")
    report("ALL (full angle)", rows)
    print()
    for verdict in (NOT_DETECTED,) + VERDICTS:
        subset = [r for r in rows if r["verdict"] == verdict]
        if not subset:
            continue
        if verdict == AXIS:
            report("axis only, full angle", subset)
            report("axis only, modulo 180", [{"error": r["axis_error"]} for r in subset])
        else:
            report(verdict, subset)
    print()
    for method in sorted(METHOD_NAMES):
        subset = [r for r in rows if r["method"] == method]
        if subset:
            report(f"method {method} {METHOD_NAMES[method]}", subset)
    print()
    for source in sorted(STEM_SOURCES):
        subset = [r for r in rows if r["stem_source"] == source]
        if subset:
            report(f"stem: {STEM_SOURCES[source]}", subset)
    if any(r["pushed"] is not None for r in rows):
        print()
        report("pushed", [r for r in rows if r["pushed"] == 1])
        report("not pushed", [r for r in rows if r["pushed"] == 0])

    given = [r for r in rows if r["verdict"] == DIRECTION_GIVEN and r["error"] is not None]
    listed = sorted(given, key=lambda r: -r["error"])[:worst]
    if listed and worst:
        print(f"\nWORST {len(listed)} FRUIT AURORA GAVE A DIRECTION FOR:")
        for r in listed:
            print(f"   {r['error']:6.1f} deg off   {r['frame']:<42} "
                  f"said {r['reported']:6.1f} (raw {r['raw']:6.1f}), truth {r['truth']:6.1f}"
                  f"   method {r['method']}  stem {STEM_SOURCES.get(r['stem_source'])}")

    print("\n" + "=" * 66)
    if given:
        print("AURORA - fruit it gave a direction for (methods 0, 1, 4):")
        label_stems.report_within([r["error"] for r in given])
    else:
        print("Aurora gave a direction for none of the labelled fruit.")
    axis = [r for r in rows if r["verdict"] == AXIS and r["error"] is not None]
    if axis:
        print("\nAURORA - axis only (method 2), modulo 180:")
        label_stems.report_within([r["axis_error"] for r in axis])
        print("AURORA - axis only (method 2), full angle - if the angle is used anyway:")
        label_stems.report_within([r["error"] for r in axis])
    pushed = [r for r in rows if r["pushed"] == 1 and r["error"] is not None]
    if pushed:
        print("\nAURORA - fruit it pushed to the PLC:")
        label_stems.report_within([r["error"] for r in pushed])
    print("\nOur engine on the same frames and labels:")
    print("   python -m tools.label_stems <frames> --out <labels folder> --score")
    return 0


# ------------------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare this app with Aurora Vision on frames Aurora logged. "
                    "Source frames are only ever read."
    )
    parser.add_argument("folders", nargs="+",
                        help="camera folder(s) holding Image_*.bmp and results.csv, "
                             "or folder(s) whose subfolders are camera folders")
    parser.add_argument("--out", default=str(project_path(DEFAULT_OUT)),
                        help=f"review folder (default: {DEFAULT_OUT})")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="review pairs differing by more than this many "
                             "degrees (default 30)")
    parser.add_argument("--config", default=None, help="config file")
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N frames per camera (0 = all)")
    parser.add_argument("--score-aurora", action="store_true",
                        help="score Aurora's angles against stem_labels.json "
                             "instead of comparing")
    parser.add_argument("--labels", default=str(project_path("data/debug/labels")),
                        help="with --score-aurora: folder holding stem_labels.json "
                             "(default: data/debug/labels)")
    parser.add_argument("--worst", type=int, default=10,
                        help="with --score-aurora, list this many worst fruit (0 = none)")
    args = parser.parse_args(argv)

    if args.score_aurora:
        return score_aurora(args.folders, Path(args.labels), args.worst)

    cameras = find_cameras(args.folders)
    out_dir = Path(args.out).resolve()
    for _, folder in cameras:
        if out_dir == folder or folder in out_dir.parents or out_dir in folder.parents:
            print(f"Refusing: the output and a source folder overlap.\n"
                  f"  source: {folder}\n  out:    {out_dir}")
            return 1

    from backend.core.paprika_engine import PaprikaEngine

    cfg = label_stems.load_config(args.config)
    settings = engine_settings(cfg)
    print("engine: " + ", ".join(f"{k}={v}" for k, v in settings.items()))
    engine = PaprikaEngine(cfg)
    # As in label_stems --score: a model that cannot run would report every
    # fruit unmatched, which reads like a bad model rather than a broken config.
    engine.load_model()
    compare(cameras, engine, out_dir, args.threshold, args.limit, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
