"""
Score each estimator separately against the hand labels

    python -m tools.eval_estimators data/raw data/captures
    python -m tools.eval_estimators data/raw --colour green

label_stems --score measures the pipeline: one number for whatever the engine
decided. This measures the ingredients. For every hand-labelled fruit it asks
the keypoints and the silhouette the same question independently, and scores
both against the clicked truth.

Why that is the question worth asking
-------------------------------------
fuse() currently lets the silhouette settle the FLIP and never the AXIS:
keypoints win the direction whenever they produced one. That policy was
written when the silhouette estimator was genuinely bad - it pointed at the
wrong end of the fruit on 63% of a 267-fruit sample, and was most confident
exactly when wrong.

Stripping the stem protrusion before the profile fixed that: 0 flipped of 267,
median error around 5 degrees. So the policy may now be holding back an
estimator that has become good, on precisely the fruit where the keypoints are
weakest - green, where colour cannot find a stem at all.

That is a question about accuracy, and accuracy needs ground truth. Comparing
the two estimators to EACH OTHER cannot answer it: they agree when both are
right and when both are wrong the same way, and this project has been misled by
that comparison more than once. The hand labels are the only independent
reference there is.

Read the per-colour table. If shape's median error on green is at or below the
keypoints', fuse() is costing you accuracy and should let it carry the axis
there. If it is worse, the current policy is correct and this settles it.
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
from backend.detection.paprika import end_on  # noqa: E402
from backend.detection.paprika import orientation as orient  # noqa: E402
from backend.utils.paths import project_path  # noqa: E402

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}
CALYX_KEY = "stem_xy"


def load_config(explicit):
    import yaml

    if explicit:
        return yaml.safe_load(Path(explicit).read_text(encoding="utf-8")) or {}
    try:
        from backend.core.config_loader import cfg

        return cfg
    except (Exception, SystemExit):
        default = project_path("config/default.yaml")
        return yaml.safe_load(default.read_text(encoding="utf-8")) or {}


def shape_settings(cfg):
    shape = (cfg.get("paprika") or {}).get("shape") or {}
    belt = shape.get("belt_hue") or [96, 145]
    return {
        "belt_hue": (int(belt[0]), int(belt[1])),
        "saturation_floor": int(shape.get("saturation_floor", 80)),
        "value_floor": int(shape.get("value_floor", 45)),
    }


def policy_settings(cfg):
    policy = ((cfg.get("paprika") or {}).get("policy") or {})
    return {
        "min_groove_coherence": float(policy.get("min_groove_coherence", 0.35)),
    }


def error_deg(predicted, truth) -> float:
    """Direction error, 0-180. Which end the stem is on counts."""
    return abs(((predicted - truth + 180) % 360) - 180)


def axis_error_deg(predicted, truth) -> float:
    """AXIS error, 0-90. Being 180 degrees out scores zero here.

    The grooves cannot tell which end carries the stem - they give an
    orientation, not a direction - so scoring them as a direction would report
    a coin flip and say nothing about the quantity they actually measure.

    Delegates to orient.axis_difference so this offline scorer and the
    runtime's groove_crosscheck() can never drift onto two different
    definitions of "how far apart are two axes" - see that function's
    docstring.
    """
    return orient.axis_difference(predicted, truth)


def report(name: str, errors: list) -> None:
    if not errors:
        print(f"  {name:<22} no measurements")
        return
    values = np.array(errors)
    print(f"  {name:<22} n={len(values):>4}  median {np.median(values):5.1f}"
          f"  p90 {np.percentile(values, 90):6.1f}"
          f"  <10deg {100 * (values < 10).mean():3.0f}%"
          f"  <20deg {100 * (values < 20).mean():3.0f}%"
          f"  flipped {int((values > 120).sum()):>3}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score the keypoint and silhouette estimators separately "
                    "against the hand labels."
    )
    parser.add_argument("frames", nargs="*",
                        default=[str(project_path("data/raw"))])
    parser.add_argument("--labels", default=str(project_path("data/debug/labels")))
    parser.add_argument("--config", default=None)
    parser.add_argument("--colour", default=None, help="only this colour")
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = shape_settings(cfg)
    min_groove_coherence = policy_settings(cfg)["min_groove_coherence"]
    labels_path = Path(args.labels).resolve() / "stem_labels.json"
    if not labels_path.exists():
        print(f"No labels at {labels_path}")
        return 1
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    lookup: dict[str, Path] = {}
    for folder in args.frames:
        for candidate in sorted(Path(folder).resolve().rglob("*")):
            if candidate.suffix.lower() in IMAGE_SUFFIXES:
                lookup.setdefault(candidate.name, candidate)

    rows = []
    for name, entries in sorted(labels.items()):
        wanted = [e for e in entries if e.get(CALYX_KEY) and e.get("true_angle_deg") is not None]
        if not wanted or name not in lookup:
            continue
        frame = cv2.imread(str(lookup[name]))
        if frame is None:
            continue

        fruit = classical.find_fruit(frame, **settings)
        for entry in wanted:
            gx, gy = entry["centroid_xy"]
            match = None
            best = None
            for item in fruit:
                x1, y1, x2, y2 = item.bbox
                if not (x1 <= gx <= x2 and y1 <= gy <= y2):
                    continue
                distance = math.hypot(item.centroid[0] - gx, item.centroid[1] - gy)
                if best is None or distance < best:
                    match, best = item, distance
            if match is None:
                continue

            truth = float(entry["true_angle_deg"])
            colour = match.colour or entry.get("colour") or ""
            if args.colour and colour != args.colour:
                continue

            row = {"colour": colour, "stem_method": match.stem_method,
                   "keypoints": None, "shape": None, "fused": None,
                   "kp_axis": None, "groove_axis": None, "groove_plus_kp": None,
                   "coherence": None, "groove_kp_disagreement": None}

            # --- keypoints alone -------------------------------------------
            kp_result = None
            if match.stem_end is not None and match.blossom_end is not None:
                x1, y1, x2, y2 = match.bbox
                diagonal = math.hypot(max(1, x2 - x1), max(1, y2 - y1))
                stem = orient.Keypoint(x=match.stem_end[0], y=match.stem_end[1],
                                       confidence=float(match.stem_quality or 0.5),
                                       visible=True)
                blossom = orient.Keypoint(x=match.blossom_end[0], y=match.blossom_end[1],
                                          confidence=float(match.stem_quality or 0.5),
                                          visible=True)
                kp_result = orient.keypoint_orientation(stem, blossom, diagonal)
                if kp_result.angle_deg is not None:
                    row["keypoints"] = error_deg(kp_result.angle_deg, truth)

            # --- silhouette alone ------------------------------------------
            mask = orient.segment_fruit(
                frame, match.bbox,
                saturation_floor=settings["saturation_floor"],
                belt_hue=settings["belt_hue"],
            )
            shape_result = orient.shape_orientation(mask) if mask is not None else None
            if shape_result is not None and shape_result.angle_deg is not None:
                row["shape"] = error_deg(shape_result.angle_deg, truth)

            # --- grooves: axis only, and axis + the keypoints' flip ---------
            # The two estimators fail in different places. On green the
            # keypoints' AXIS is often badly wrong while their FLIP is mostly
            # right (18 flips in 415). The grooves are the mirror image: they
            # measure the axis directly off the fruit's own surface bands, and
            # cannot speak to the flip at all. If that holds, taking the axis
            # from one and the flip from the other beats either alone.
            #
            # groove_axis itself is stored UNGATED - the coherence-by-band
            # table further down needs the full range, including the low end,
            # to show where the floor should sit. groove_plus_kp is the
            # column that claims to say what the pipeline would actually
            # produce, and the pipeline never trusts a groove reading below
            # policy.min_groove_coherence (see PaprikaEngine._groove_axis_estimate
            # and _apply_groove_crosscheck) - so this column is gated the same
            # way, or it silently reports a fusion the runtime would never
            # perform, diluted by every smooth, low-coherence fruit whose
            # "axis" was noise. An earlier revision of this tool computed it
            # ungated, which is why the direction table used to make
            # groove_plus_kp look worse than keypoints alone even on green:
            # most of the 802 fruit sit below coherence 0.25 (see GROOVE AXIS
            # BY COHERENCE below), and every one of them was dragging this
            # average toward its own noise.
            if mask is not None:
                gx1, gy1, gx2, gy2 = match.bbox
                measured = end_on.groove_axis(frame[gy1:gy2, gx1:gx2], mask)
                if measured is not None:
                    groove, coherence = measured
                    row["coherence"] = coherence
                    row["groove_axis"] = axis_error_deg(groove, truth)
                    if (
                        coherence >= min_groove_coherence
                        and kp_result is not None
                        and kp_result.angle_deg is not None
                    ):
                        # Keep the groove axis; choose the end the keypoints
                        # lean towards.
                        options = (groove % 360.0, (groove + 180.0) % 360.0)
                        best = min(options,
                                   key=lambda a: error_deg(a, kp_result.angle_deg))
                        row["groove_plus_kp"] = error_deg(best, truth)
                        # Exactly the comparison orient.groove_crosscheck()
                        # makes at runtime (PaprikaEngine._apply_groove_crosscheck),
                        # at the same coherence floor - see DOES GROOVE
                        # DISAGREEMENT PREDICT ERROR? below for whether
                        # policy.max_groove_disagreement_deg (30 degrees,
                        # unmeasured) is actually the right place to draw it.
                        row["groove_kp_disagreement"] = axis_error_deg(
                            groove, kp_result.angle_deg
                        )
            if kp_result is not None and kp_result.angle_deg is not None:
                row["kp_axis"] = axis_error_deg(kp_result.angle_deg, truth)

            # --- what fuse() would return ----------------------------------
            if kp_result is not None:
                fused = orient.fuse(kp_result, shape_result)
                if fused.angle_deg is not None:
                    row["fused"] = error_deg(fused.angle_deg, truth)

            # How far apart the two estimators are, independent of either
            # being right. If this predicts error it is usable as a GATE - a
            # fruit the two disagree about can be sent for another look -
            # which is worth more than a marginally better average, because it
            # converts a bad placement into a flagged one.
            if kp_result is not None and kp_result.angle_deg is not None \
                    and shape_result is not None and shape_result.angle_deg is not None:
                row["disagreement"] = error_deg(
                    kp_result.angle_deg, shape_result.angle_deg
                )
            rows.append(row)

    if not rows:
        print("Nothing scored - check the frames folders and the labels path.")
        return 1

    print(f"\n{len(rows)} labelled fruit, each estimator scored independently\n")
    print("DIRECTION error (which end the stem is on counts)")
    for key in ("keypoints", "shape", "fused", "groove_plus_kp"):
        report(key, [r[key] for r in rows if r[key] is not None])
    print("\nAXIS error only (0-90; being 180 out scores zero)")
    for key in ("kp_axis", "groove_axis"):
        report(key, [r[key] for r in rows if r[key] is not None])

    for colour in sorted({r["colour"] for r in rows if r["colour"]}):
        subset = [r for r in rows if r["colour"] == colour]
        print(f"\n{colour.upper()}  ({len(subset)} fruit)")
        for key in ("keypoints", "shape", "fused", "groove_plus_kp"):
            report(key, [r[key] for r in subset if r[key] is not None])
        print("   axis only:")
        for key in ("kp_axis", "groove_axis"):
            report("   " + key, [r[key] for r in subset if r[key] is not None])

    print("\nby how the stem was found:")
    for method in sorted({r["stem_method"] for r in rows}):
        subset = [r for r in rows if r["stem_method"] == method]
        print(f"\n  stem_method = {method}  ({len(subset)} fruit)")
        for key in ("keypoints", "shape", "fused"):
            report("    " + key, [r[key] for r in subset if r[key] is not None])
        # groove_axis is reported here too, and it matters most on exactly
        # the stem_method=none rows: that is the population keypoints and
        # fused have "no measurements" for above, because there is no stem
        # for the pose model to have found in the first place. It is an AXIS
        # error (0-90), not comparable to the DIRECTION rows above it.
        report("    groove_axis", [r["groove_axis"] for r in subset if r["groove_axis"] is not None])

    print("\n" + "=" * 70)
    print("GROOVE AXIS BY COHERENCE")
    print("  Coherence is how aligned the surface bands were. A smooth fruit")
    print("  has no grooves to read and the axis means nothing there.")
    banded = [r for r in rows if r["coherence"] is not None
              and r["groove_axis"] is not None]
    if banded:
        print(f"\n  {'coherence':>14}{'n':>6}{'groove axis':>14}{'kp axis':>11}")
        for low, high in ((0.0, 0.25), (0.25, 0.35), (0.35, 0.50),
                          (0.50, 0.65), (0.65, 1.01)):
            band = [r for r in banded if low <= r["coherence"] < high]
            if not band:
                continue
            groove = np.median([r["groove_axis"] for r in band])
            kp = [r["kp_axis"] for r in band if r["kp_axis"] is not None]
            kp_median = np.median(kp) if kp else float("nan")
            print(f"  {f'{low:.2f}-{high:.2f}':>14}{len(band):>6}"
                  f"{groove:>14.1f}{kp_median:>11.1f}")
        print("\n  Where the groove axis beats the keypoint axis, the grooves")
        print("  are the better source for the DIRECTION OF THE FRUIT, and the")
        print("  keypoints need only decide which end.")

    print("\n" + "=" * 70)
    print("DOES DISAGREEMENT PREDICT ERROR?")
    print("  If the keypoint error is much worse when the two estimators")
    print("  disagree, disagreement is a usable gate: reorient instead of place.")
    both = [r for r in rows if r.get("disagreement") is not None
            and r["keypoints"] is not None]
    if both:
        print(f"\n  {'disagreement':>16}{'n':>6}{'kp median':>11}{'kp p90':>9}"
              f"{'kp >20deg':>11}{'flipped':>9}")
        for low, high in ((0, 10), (10, 20), (20, 35), (35, 60), (60, 120), (120, 181)):
            band = [r for r in both if low <= r["disagreement"] < high]
            if not band:
                continue
            errors = np.array([r["keypoints"] for r in band])
            print(f"  {f'{low}-{high} deg':>16}{len(band):>6}{np.median(errors):>11.1f}"
                  f"{np.percentile(errors, 90):>9.1f}"
                  f"{100 * (errors > 20).mean():>10.0f}%"
                  f"{int((errors > 120).sum()):>9}")
        for cut in (20, 30, 45):
            agree = [r for r in both if r["disagreement"] < cut]
            differ = [r for r in both if r["disagreement"] >= cut]
            if not differ:
                continue
            a = np.array([r["keypoints"] for r in agree])
            d = np.array([r["keypoints"] for r in differ])
            print(f"\n  gate at {cut} deg: would flag {len(differ)} of {len(both)} fruit "
                  f"({100 * len(differ) / len(both):.0f}%)")
            print(f"     kept   : median {np.median(a):5.1f}  >20deg {100*(a>20).mean():3.0f}%"
                  f"  flipped {int((a>120).sum())}")
            print(f"     flagged: median {np.median(d):5.1f}  >20deg {100*(d>20).mean():3.0f}%"
                  f"  flipped {int((d>120).sum())}")
    print("\n" + "=" * 70)
    print("DOES GROOVE DISAGREEMENT PREDICT ERROR?")
    print("  Same question as above, for the OTHER cross-check: when a")
    print("  high-coherence groove reading disagrees with the keypoint axis,")
    print("  is the keypoint more often wrong? If so, this is what")
    print("  policy.max_groove_disagreement_deg (paprika_engine.py's")
    print("  groove_crosscheck) should be set from - it currently is not,")
    print("  since it was opened at the same 30 degrees as the kp-vs-shape")
    print("  gate without this table to check it against.")
    groove_kp = [r for r in rows if r["groove_kp_disagreement"] is not None
                 and r["keypoints"] is not None]
    if groove_kp:
        print(f"\n  {'disagreement':>16}{'n':>6}{'kp median':>11}{'kp p90':>9}"
              f"{'kp >20deg':>11}{'flipped':>9}")
        for low, high in ((0, 10), (10, 20), (20, 35), (35, 60), (60, 120), (120, 181)):
            band = [r for r in groove_kp if low <= r["groove_kp_disagreement"] < high]
            if not band:
                continue
            errors = np.array([r["keypoints"] for r in band])
            print(f"  {f'{low}-{high} deg':>16}{len(band):>6}{np.median(errors):>11.1f}"
                  f"{np.percentile(errors, 90):>9.1f}"
                  f"{100 * (errors > 20).mean():>10.0f}%"
                  f"{int((errors > 120).sum()):>9}")
        for cut in (20, 30, 45):
            agree = [r for r in groove_kp if r["groove_kp_disagreement"] < cut]
            differ = [r for r in groove_kp if r["groove_kp_disagreement"] >= cut]
            if not differ:
                continue
            a = np.array([r["keypoints"] for r in agree])
            d = np.array([r["keypoints"] for r in differ])
            print(f"\n  gate at {cut} deg: would flag {len(differ)} of {len(groove_kp)} "
                  f"fruit ({100 * len(differ) / len(groove_kp):.0f}%)")
            print(f"     kept   : median {np.median(a):5.1f}  >20deg {100*(a>20).mean():3.0f}%"
                  f"  flipped {int((a>120).sum())}")
            print(f"     flagged: median {np.median(d):5.1f}  >20deg {100*(d>20).mean():3.0f}%"
                  f"  flipped {int((d>120).sum())}")
    else:
        print("\n  No fruit had both a usable keypoint angle and a groove reading")
        print("  above min_groove_coherence - nothing to measure this against yet.")

    print()
    print("If shape's median beats keypoints' on a colour, fuse() is holding it")
    print("back there: it currently lets the silhouette settle only the flip,")
    print("never the axis. If shape is worse, the present policy is right.")
    print()
    print("groove_plus_kp above is now gated at policy.min_groove_coherence, the")
    print("same floor PaprikaEngine trusts a groove reading at - it reports what")
    print("the pipeline would actually produce, not every fruit a groove_axis()")
    print("call happened to return something for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())