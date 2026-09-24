"""
Orientation confusion matrix

    python -m tools.eval_confusion data/debug/greencheck/records.jsonl
    python -m tools.eval_confusion data/debug/before/records.jsonl data/debug/after/records.jsonl

Compares a debug records.jsonl (one line per detection, written by the same
debug-dump path that produces data/debug/<run>/report.txt) against the hand
labels in data/debug/labels/stem_labels.json, and prints a colour x
stem_method confusion matrix of the outcome.

Why this exists
----------------
The report.txt files already show OPPOSITE ENDS per detection, but that is a
self-consistency check (stem angle vs silhouette angle) - it has no idea what
the fruit's orientation actually is. This script is the first thing here that
compares a prediction against ground truth, which is what "did tightening the
green threshold help or hurt" actually needs answered as a number instead of
a feeling from skimming a few hundred report lines.

Outcome buckets, per matched detection
---------------------------------------
correct    predicted angle within ANGLE_TOLERANCE_DEG of the true angle
opposite   predicted angle within ANGLE_TOLERANCE_DEG of the true angle + 180
           (stem end and blossom end swapped - the specific failure mode
           dataset_check.py's docstring warns a bad keypoint order produces)
other      predicted an angle, but neither end matches (a genuinely wrong
           orientation guess, not just a flip)
no_call    the engine did not commit to an angle (placement is unknown,
           review, reorient, or reject) - counted separately because "did not
           guess" and "guessed wrong" are different failures with different
           fixes

Matching a prediction record to a label
-----------------------------------------
Same rule as tools/export_dataset.py: match by (frame, best bbox IoU), never
by list position, because detection order is not stable across runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.utils.paths import project_path  # noqa: E402

ANGLE_TOLERANCE_DEG = 25.0
NO_CALL_PLACEMENTS = {"unknown", "review", "reorient", "reject"}

DEFAULT_LABELS = "data/debug/labels/stem_labels.json"


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(1, union)


def angle_diff(a: float, b: float) -> float:
    """Smallest difference between two angles in degrees, 0-180."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def load_labels(path: Path) -> dict[str, list[dict]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def match(records: list[dict], labels: dict[str, list[dict]]) -> list[dict]:
    """Yield one row per record that overlaps a hand label for its frame,
    each carrying the record plus the matched label's colour and true angle.
    """
    matched = []
    for rec in records:
        frame_labels = labels.get(rec.get("frame"))
        if not frame_labels:
            continue
        best, best_iou = None, 0.0
        for lbl in frame_labels:
            if "true_angle_deg" not in lbl:
                # Labeler marked this fruit unpickable (see its "note") -
                # no ground-truth angle to compare against.
                continue
            score = iou(rec["bbox"], lbl["bbox"])
            if score > best_iou:
                best, best_iou = lbl, score
        if best is None or best_iou < 0.3:
            continue
        matched.append(
            {
                "colour": best.get("colour", rec.get("colour", "?")),
                "stem_method": rec.get("stem_method", "?"),
                "placement": rec.get("placement", "?"),
                "true_angle_deg": best["true_angle_deg"],
                "angle_deg": rec.get("angle_deg"),
            }
        )
    return matched


def classify(row: dict) -> str:
    if row["placement"] in NO_CALL_PLACEMENTS or row["angle_deg"] is None:
        return "no_call"
    diff = angle_diff(row["angle_deg"], row["true_angle_deg"])
    if diff <= ANGLE_TOLERANCE_DEG:
        return "correct"
    if diff >= 180.0 - ANGLE_TOLERANCE_DEG:
        return "opposite"
    return "other"


OUTCOMES = ["correct", "opposite", "other", "no_call"]


def print_matrix(title: str, matched: list[dict], key: str) -> None:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {o: 0 for o in OUTCOMES})
    for row in matched:
        counts[row[key]][classify(row)] += 1

    print(f"\n{title}")
    header = f"{'':16}" + "".join(f"{o:>10}" for o in OUTCOMES) + f"{'n':>8}{'acc%':>8}"
    print(header)
    for group in sorted(counts):
        c = counts[group]
        n = sum(c.values())
        decided = n - c["no_call"]
        acc = 100.0 * c["correct"] / decided if decided else float("nan")
        row_str = f"{group:16}" + "".join(f"{c[o]:>10}" for o in OUTCOMES)
        row_str += f"{n:>8}{acc:>8.1f}"
        print(row_str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", help="records.jsonl file(s) to evaluate")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="hand-label json (default: %(default)s)")
    args = parser.parse_args()

    labels_path = project_path(args.labels)
    record_paths = [project_path(p) for p in args.records]

    labels = load_labels(labels_path)
    records = load_records(record_paths)
    matched = match(records, labels)

    if not matched:
        print("No predictions matched a hand-labeled frame - check --labels and the records path.")
        return

    unmatched_labels = sum(len(v) for v in labels.values()) - len(matched)
    print(f"Matched {len(matched)} detections against hand labels "
          f"({unmatched_labels} labeled fruit had no corresponding prediction).")

    print_matrix("By colour", matched, "colour")
    print_matrix("By stem_method", matched, "stem_method")

    overall = defaultdict(int)
    for row in matched:
        overall[classify(row)] += 1
    n = sum(overall.values())
    decided = n - overall["no_call"]
    acc = 100.0 * overall["correct"] / decided if decided else float("nan")
    print(f"\nOverall: {dict(overall)}  accuracy (of decided)={acc:.1f}%  n={n}")


if __name__ == "__main__":
    main()
