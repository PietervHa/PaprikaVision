"""
Label the true stem position, so angle accuracy can be measured

    python -m tools.label_stems data/raw
    python -m tools.label_stems data/raw --colour green --limit 200
    python -m tools.label_stems --score

One click per fruit, on the base of the stem where it meets the body. The true
angle follows from that point and the fruit's own centroid, which is the same
centroid the estimator uses, so the comparison is about the stem and nothing
else.

Why this exists
---------------
Every accuracy number in this project so far has been the estimator checked
against another estimator. That catches gross failures - it is how the
width-profile flip was found, 63% of fruit pointing at the wrong end - but it
is blind to the case where both routes are wrong the same way, and it cannot
measure a 20-degree error at all. Placement accuracy is the whole product: the
fruit has to come out facing a particular way. That needs a number that does
not come from the software being judged.

It also replaces judgement by eye, which has been wrong repeatedly here - twice
in one session I read a correct detection as a failure and a wrong one as fine.

What gets stored
----------------
    <out>/stem_labels.json      one entry per labelled fruit

Points are stored in FULL-FRAME pixel coordinates, deliberately, not as an
index into a detection list. Detections move when the detector is tuned, and
the entire purpose of these labels is to compare one version of the detector
against another. A label tied to a detection index would silently re-point at a
different fruit the first time anything changed; a point on the belt stays
where it is forever.

Labelling is resumable - a frame already carrying labels is skipped - so this
can be done in several sittings, and a second pass can add frames to an
existing set.

SOURCE FRAMES ARE NEVER MODIFIED. They are opened read-only and everything is
written elsewhere; the tool refuses to run if the output would land inside the
source folder.

Controls
--------
    left click   mark the base of the stem, where it meets the fruit
    n            no stem visible on this fruit (recorded as stemless)
    s            skip - unsure, or the crop is unusable
    u            undo the last fruit and label it again
    q  or  Esc   save and quit

Scoring
-------
    python -m tools.label_stems --score

Runs the engine over the labelled frames and reports the angle error against
the labels: overall, and split by verdict and estimator. The number that
matters most is printed on its own - how far out the fruit it PLACED were,
because those are the ones the machine acted on.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detection.paprika import classical  # noqa: E402
from backend.detection.paprika import orientation as orient  # noqa: E402
from backend.utils.paths import project_path  # noqa: E402

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}

# The crop is scaled to this height for clicking. A stem base can be pinned to
# a few pixels on a 700px view and not on a 200px one, and the label is only
# ever as good as the click.
VIEW_HEIGHT = 700

# Margin around the bbox, as a fraction of its size. The stem is the thing
# being clicked and it often sits right on the silhouette edge, so a crop cut
# exactly to the box can clip the very feature the label is about.
CROP_MARGIN = 0.12

NO_STEM = "no_stem_visible"


def load_config(explicit: str | None) -> dict:
    import yaml

    if explicit:
        return yaml.safe_load(Path(explicit).read_text(encoding="utf-8")) or {}
    try:
        from backend.core.config_loader import cfg

        return cfg
    except (Exception, SystemExit):
        default = project_path("config/default.yaml")
        if not default.exists():
            raise SystemExit(f"No config found - looked for {default}")
        return yaml.safe_load(default.read_text(encoding="utf-8")) or {}


def shape_settings(cfg: dict) -> dict:
    shape = (cfg.get("paprika") or {}).get("shape") or {}
    belt = shape.get("belt_hue") or [96, 145]
    return {
        "belt_hue": (int(belt[0]), int(belt[1])),
        "saturation_floor": int(shape.get("saturation_floor", 80)),
        "value_floor": int(shape.get("value_floor", 45)),
    }


def gather_frames(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def load_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{path} is not readable as JSON ({exc}). Move it aside rather than "
            f"letting this overwrite work that may still be good."
        )


def save_labels(path: Path, labels: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written via a temporary file: an interrupted write in the middle of a
    # labelling session would otherwise destroy every label taken so far.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


# --------------------------------------------------------------------- labelling


class _Clicker:
    """Collects one click, in view coordinates."""

    def __init__(self) -> None:
        self.point: tuple[int, int] | None = None

    def __call__(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x, y)


def draw_instructions(view: np.ndarray, caption: str) -> np.ndarray:
    panel = view.copy()
    lines = [caption, "click the STEM BASE   n = no stem   s = skip   u = undo   q = save+quit"]
    for index, text in enumerate(lines):
        y = 26 + index * 26
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
    return panel


def label(args) -> int:
    cfg = load_config(args.config)
    settings = shape_settings(cfg)

    source = Path(args.frames).resolve()
    out_dir = Path(args.out).resolve()
    if not source.exists():
        print(f"Source folder does not exist: {source}")
        return 1
    if out_dir == source or source in out_dir.parents:
        print(f"Refusing to write inside the source folder.\n"
              f"  source: {source}\n  out:    {out_dir}")
        return 1

    labels_path = out_dir / "stem_labels.json"
    labels = load_labels(labels_path)
    frames = gather_frames(source)
    if not frames:
        print(f"No images found under {source}")
        return 1

    already = sum(len(v) for v in labels.values())
    print(f"{len(frames)} frame(s) under {source}")
    print(f"{already} fruit already labelled in {labels_path}")

    try:
        cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        print("This needs a desktop OpenCV build - cv2.imshow is unavailable.\n"
              "opencv-python-headless has no GUI; install opencv-python instead.")
        return 1

    clicker = _Clicker()
    cv2.setMouseCallback("label", clicker)

    # (frame_name, fruit_record) queued up so undo can step back across frames.
    history: list[tuple[str, dict]] = []
    labelled = 0
    quit_now = False

    for path in frames:
        if quit_now:
            break
        if path.name in labels and labels[path.name]:
            continue                                   # resumable: already done
        if args.limit and labelled >= args.limit:
            break

        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  ! could not read {path.name}")
            continue

        fruit = classical.find_fruit(frame, **settings)
        if args.colour:
            fruit = [f for f in fruit if f.colour == args.colour]
        fruit = [f for f in fruit if not f.edge_clipped]
        if not fruit:
            labels.setdefault(path.name, [])
            continue

        height, width = frame.shape[:2]
        entries: list[dict] = []
        index = 0
        while index < len(fruit):
            item = fruit[index]
            x1, y1, x2, y2 = item.bbox
            pad_x = int((x2 - x1) * CROP_MARGIN)
            pad_y = int((y2 - y1) * CROP_MARGIN)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                index += 1
                continue

            scale = VIEW_HEIGHT / crop.shape[0]
            view = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LINEAR)
            # The centroid is drawn because the true angle is measured FROM it.
            # Seeing it makes an implausible label obvious while clicking
            # rather than weeks later in a scoring run.
            centre_view = (
                int((item.centroid[0] + x1 - cx1) * scale),
                int((item.centroid[1] + y1 - cy1) * scale),
            )
            cv2.circle(view, centre_view, 6, (255, 200, 0), -1)

            caption = (f"{path.name}   fruit {index + 1}/{len(fruit)}   "
                       f"{item.colour}   [{labelled} labelled]")
            clicker.point = None
            action = None
            while action is None:
                cv2.imshow("label", draw_instructions(view, caption))
                key = cv2.waitKey(20) & 0xFF
                if clicker.point is not None:
                    action = "click"
                elif key in (ord("q"), 27):
                    action, quit_now = "quit", True
                elif key == ord("n"):
                    action = "none"
                elif key == ord("s"):
                    action = "skip"
                elif key == ord("u"):
                    action = "undo"

            if action == "quit":
                break
            if action == "skip":
                index += 1
                continue
            if action == "undo":
                if entries:
                    entries.pop()
                    index = max(0, index - 1)
                    labelled = max(0, labelled - 1)
                elif history:
                    prev_name, _ = history.pop()
                    labels.pop(prev_name, None)
                    labelled = max(0, labelled - 1)
                    print(f"  undone: {prev_name} - re-run to label it again")
                continue

            record = {
                "bbox": [int(v) for v in item.bbox],
                "centroid_xy": [float(item.centroid[0] + x1), float(item.centroid[1] + y1)],
                "colour": item.colour,
            }
            if action == "none":
                record["stem_xy"] = None
                record["note"] = NO_STEM
            else:
                vx, vy = clicker.point
                # Back to full-frame coordinates, which is what gets stored.
                record["stem_xy"] = [float(cx1 + vx / scale), float(cy1 + vy / scale)]
                sx, sy = record["stem_xy"]
                gx, gy = record["centroid_xy"]
                record["true_angle_deg"] = round(orient.vector_to_angle(sx - gx, sy - gy), 2)
            entries.append(record)
            history.append((path.name, record))
            labelled += 1
            index += 1

        labels[path.name] = entries
        save_labels(labels_path, labels)          # after every frame, not at the end

    cv2.destroyAllWindows()
    save_labels(labels_path, labels)
    total = sum(len(v) for v in labels.values())
    with_stem = sum(1 for v in labels.values() for r in v if r.get("stem_xy"))
    print(f"\n{labelled} fruit labelled this session")
    print(f"{total} in the file, {with_stem} of them with a stem marked")
    print(f"-> {labels_path}")
    print("\nScore the detector against them with:")
    print("   python -m tools.label_stems --score")
    return 0


# ----------------------------------------------------------------------- scoring


def match(labelled: dict, detections: list[dict]) -> dict | None:
    """Find the detection this label belongs to.

    Matched by position, not by index: a label exists to compare one version of
    the detector against another, and detection ordering is not stable across
    versions. The labelled centroid lands inside the right fruit's box, and
    where boxes overlap the nearest centre wins.
    """
    gx, gy = labelled["centroid_xy"]
    best, best_distance = None, None
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        if not (x1 <= gx <= x2 and y1 <= gy <= y2):
            continue
        cx, cy = detection["center"]
        distance = math.hypot(cx - gx, cy - gy)
        if best_distance is None or distance < best_distance:
            best, best_distance = detection, distance
    return best


def score(args) -> int:
    from backend.core.paprika_engine import PaprikaEngine

    cfg = load_config(args.config)
    out_dir = Path(args.out).resolve()
    labels = load_labels(out_dir / "stem_labels.json")
    if not labels:
        print(f"No labels in {out_dir / 'stem_labels.json'} - run the labeller first.")
        return 1

    source = Path(args.frames).resolve()
    engine = PaprikaEngine(cfg)
    rows = []
    missing = 0
    for name, entries in sorted(labels.items()):
        wanted = [e for e in entries if e.get("stem_xy")]
        if not wanted:
            continue
        path = source / name
        if not path.exists():
            hits = list(source.rglob(name))
            if not hits:
                missing += 1
                continue
            path = hits[0]
        frame = cv2.imread(str(path))
        if frame is None:
            missing += 1
            continue
        detections = engine.evaluate(frame)["detections"]
        for entry in wanted:
            found = match(entry, detections)
            if found is None:
                rows.append({"placement": "NOT DETECTED", "source": None, "error": None})
                continue
            angle = (found.get("orientation") or {}).get("angle_deg")
            error = None
            if angle is not None:
                error = abs(((angle - entry["true_angle_deg"] + 180) % 360) - 180)
            rows.append({
                "placement": found["placement"],
                "source": (found.get("orientation") or {}).get("source"),
                "error": error,
            })

    if missing:
        print(f"({missing} labelled frame(s) could not be found under {source})")
    if not rows:
        print("Nothing to score.")
        return 1

    def report(name: str, subset: list[dict]) -> None:
        measured = [r["error"] for r in subset if r["error"] is not None]
        silent = len(subset) - len(measured)
        line = f"  {name:<26} n={len(subset):>4}"
        if measured:
            values = np.array(measured)
            line += (f"  median {np.median(values):5.1f} deg"
                     f"  p90 {np.percentile(values, 90):6.1f}"
                     f"  <10deg {100 * (values < 10).mean():3.0f}%"
                     f"  <20deg {100 * (values < 20).mean():3.0f}%"
                     f"  flipped {int((values > 120).sum())}")
        if silent:
            line += f"   ({silent} gave no angle)"
        print(line)

    print(f"\n{len(rows)} labelled fruit scored against the detector\n")
    report("ALL", rows)
    print()
    for placement in sorted({r["placement"] for r in rows}):
        report(placement, [r for r in rows if r["placement"] == placement])
    print()
    for src in sorted({r["source"] for r in rows if r["source"]}):
        report(f"estimator: {src}", [r for r in rows if r["source"] == src])

    placed = [r["error"] for r in rows
              if r["placement"] == "place" and r["error"] is not None]
    print("\n" + "=" * 66)
    if placed:
        values = np.array(placed)
        print("THE NUMBER THAT MATTERS - fruit the machine actually placed:")
        print(f"   n={len(values)}   median {np.median(values):.1f} deg   "
              f"p90 {np.percentile(values, 90):.1f}   worst {values.max():.1f}")
        for limit in (10, 20, 30, 45):
            print(f"   within {limit:>2} deg: {100 * (values < limit).mean():3.0f}%"
                  f"   ({int((values >= limit).sum())} fruit worse)")
        print("\n   A placed fruit that is far out is the failure this system exists")
        print("   to prevent - it reaches the actuator with nothing flagged.")
    else:
        print("No placed fruit among the labels yet.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Click the stem base on each fruit, then score the detector "
                    "against those labels. Source frames are only ever read."
    )
    parser.add_argument("frames", nargs="?", default=str(project_path("data/raw")),
                        help="folder of frames (default: data/raw)")
    parser.add_argument("--out", default=str(project_path("data/debug/labels")),
                        help="where labels are stored (default: data/debug/labels)")
    parser.add_argument("--config", default=None, help="config file")
    parser.add_argument("--colour", default=None,
                        help="only label fruit of this colour, e.g. green")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many fruit (0 = no limit)")
    parser.add_argument("--score", action="store_true",
                        help="score the detector against existing labels")
    args = parser.parse_args()
    return score(args) if args.score else label(args)


if __name__ == "__main__":
    raise SystemExit(main())
