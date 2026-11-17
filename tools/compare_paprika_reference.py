"""Compare the PaprikaPose Aurora filter (its test program) with this engine, fruit by fruit.

    python -m tools.compare_paprika_reference data\\debug\\paprika_reference\\reference.csv cpp.csv

reference.csv comes from tools.paprika_reference, cpp.csv from the filter's
test\\bin\\paprika_test.exe run on the cases that tool exported. The two must agree:
same fruit per frame, the same boxes and landmarks to within a pixel, the same angle
and the same decision. Two things the filter deliberately does not do are reported
apart rather than as failures:

  - end-on re-check: lying fruit whose landmarks are 0.18-0.22 of the box diagonal
    apart. This engine looks again with a segmentation; the filter does not.
  - groove fallback: fruit below min_usable_confidence that this engine re-measures
    from the grooves; the filter leaves them on "review".

Exit code 0 when everything else agrees, 1 when it does not.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict

BOX_TOL_PX = 1.0
KP_TOL_PX = 1.0
ANGLE_TOL_DEG = 0.5
CONF_TOL = 0.01


def _load(path: str) -> dict[str, list[dict]]:
    frames: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frames.setdefault(row["case"], [])
            if row["index"] != "-1":
                frames[row["case"]].append(row)
    return frames


def _f(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def _iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = (_f(a, k) for k in ("x1", "y1", "x2", "y2"))
    bx1, by1, bx2, by2 = (_f(b, k) for k in ("x1", "y1", "x2", "y2"))
    iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _not_ported(ref: dict) -> str:
    notes = (ref.get("notes") or "").lower()
    if "standing_from_silhouette" in notes or "end_on_p=" in notes:
        return "end-on re-check"
    if "low_confidence_fallback" in notes or "groove" in notes:
        return "groove fallback"
    if ref.get("source") and ref.get("source") not in ("keypoints", "classical"):
        return f"source {ref['source']}"
    return ""


def _withheld_angle(ref: dict):
    """The engine removes the angle of review/unknown fruit and keeps it in the notes."""
    match = re.search(r"angle_withheld=(-?\d+)deg", ref.get("notes") or "")
    return float(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", help="reference.csv from tools.paprika_reference")
    parser.add_argument("cpp", help="CSV written by the filter's paprika_test.exe")
    parser.add_argument("--show", type=int, default=10, help="how many differences to list")
    args = parser.parse_args()

    ref, cpp = _load(args.reference), _load(args.cpp)
    cases = sorted(set(ref) | set(cpp))
    worst = defaultdict(float)
    problems: list[str] = []
    expected: list[str] = []
    matched = 0

    for case in cases:
        r_rows, c_rows = ref.get(case, []), cpp.get(case, [])
        if case not in cpp:
            problems.append(f"{case}: missing in the C++ results")
            continue
        if case not in ref:
            problems.append(f"{case}: missing in the reference")
            continue
        used: set[int] = set()
        for r in r_rows:
            best, pick = 0.0, None
            for j, c in enumerate(c_rows):
                if j not in used and (v := _iou(r, c)) > best:
                    best, pick = v, j
            if pick is None or best < 0.5:
                problems.append(f"{case}: fruit at {r['x1']},{r['y1']} found by the engine only")
                continue
            used.add(pick)
            c = c_rows[pick]
            matched += 1
            box = max(abs(_f(r, k) - _f(c, k)) for k in ("x1", "y1", "x2", "y2"))
            worst["box px"] = max(worst["box px"], box)
            kp = 0.0
            for name in ("stem", "blossom"):
                if _f(r, f"{name}_x") or _f(r, f"{name}_y"):
                    kp = max(kp, math.hypot(_f(r, f"{name}_x") - _f(c, f"{name}_x"),
                                            _f(r, f"{name}_y") - _f(c, f"{name}_y")))
            worst["landmark px"] = max(worst["landmark px"], kp)
            conf = abs(_f(r, "confidence") - _f(c, "confidence"))
            worst["confidence"] = max(worst["confidence"], conf)
            ra, ca = r.get("angle_ccw", ""), c.get("angle_ccw", "")
            withheld = _withheld_angle(r) if not ra and ca else None
            if withheld is not None and _angle_diff(withheld, float(ca)) <= 1.0:
                ra = ca          # same angle, only hidden by the engine (rounded to 1 degree there)
            angle = _angle_diff(float(ra), float(ca)) if ra and ca else 0.0
            worst["angle deg"] = max(worst["angle deg"], angle)

            reason = _not_ported(r)
            issues = []
            if box > BOX_TOL_PX: issues.append(f"box {box:.1f}px")
            if kp > KP_TOL_PX: issues.append(f"landmark {kp:.1f}px")
            if bool(ra) != bool(ca): issues.append(f"angle {ra or '-'} vs {ca or '-'}")
            elif angle > ANGLE_TOL_DEG: issues.append(f"angle {angle:.2f} deg")
            if r["pose"] != c["pose"]: issues.append(f"pose {r['pose']} vs {c['pose']}")
            if r["placement"] != c["placement"]: issues.append(f"placement {r['placement']} vs {c['placement']}")
            elif conf > CONF_TOL: issues.append(f"confidence {conf:.3f}")
            if issues:
                line = f"{case} fruit {r['index']}: " + ", ".join(issues)
                (expected if reason else problems).append(line + (f"  [{reason}: not in the filter]" if reason else ""))
        for j, c in enumerate(c_rows):
            if j not in used:
                problems.append(f"{case}: fruit at {c['x1']},{c['y1']} found by the filter only")

    print(f"{len(cases)} frame(s), {matched} fruit matched")
    for key, value in worst.items():
        print(f"  largest {key:12} difference: {value:.4f}")
    if expected:
        print(f"\n{len(expected)} difference(s) from steps the filter does not have (expected):")
        for line in expected[: args.show]:
            print("  " + line)
    if problems:
        print(f"\n{len(problems)} real difference(s):")
        for line in problems[: args.show]:
            print("  " + line)
        print("\nNOT OK - the filter does not reproduce the engine on these frames.")
        return 1
    print("\nOK - the filter reproduces the engine on these frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
