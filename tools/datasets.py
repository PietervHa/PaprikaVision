"""
What datasets exist, what they were built from, and whether they are intact

    python -m tools.datasets
    python -m tools.datasets --show v4
    python -m tools.datasets --verify
    python -m tools.datasets --trace models/paprika_pose.pt

Once the raw frames are gone, the dataset folders ARE the record. This reads
them back: what is there, what each was built on, and whether any of it has
changed since it was written.

Verification matters more than it sounds. A dataset folder is ordinary files -
someone can drop images in, a sync can half-copy it, an editor can rewrite a
label - and none of that announces itself. A model whose training data has
quietly changed underneath it is a model with no provenance at all, which is
the thing keeping the datasets was supposed to prevent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.utils.paths import project_path  # noqa: E402
from tools import dataset_manifest as dm  # noqa: E402


def datasets_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and (d / "train" / "images").exists())


def summarise(dataset: Path) -> None:
    manifest = dm.read(dataset)
    ok, message = dm.verify(dataset)
    mark = "ok " if ok else "!! "
    if manifest is None:
        counts = dm.counts(dataset)
        total = sum(c["fruit"] for c in counts.values())
        print(f"  {mark}{dataset.name:<24} {total:>5} fruit   (no manifest)")
        return
    counts = manifest.get("counts", {})
    total = sum(c.get("fruit", 0) for c in counts.values())
    frames = sum(c.get("frames", 0) for c in counts.values())
    parents = manifest.get("built_from", {}).get("parents", [])
    lineage = ", ".join(p.get("name", "?") for p in parents) or "-"
    print(f"  {mark}{dataset.name:<24} {frames:>4} frames {total:>5} fruit   "
          f"id {manifest.get('content_id','?')}   from: {lineage}")
    if not ok:
        print(f"       {message}")


def show(dataset: Path) -> int:
    manifest = dm.read(dataset)
    if manifest is None:
        print(f"{dataset} has no {dm.MANIFEST_NAME}.")
        counts = dm.counts(dataset)
        print(f"counts: {json.dumps(counts, indent=2)}")
        return 1
    print(json.dumps(manifest, indent=2))
    ok, message = dm.verify(dataset)
    print(f"\nintegrity: {message}")

    landmarks = manifest.get("landmarks", {})
    total = sum(landmarks.values()) or 1
    occluded = landmarks.get("occluded", 0)
    print(f"\nlandmark visibility: "
          f"{landmarks.get('visible',0)} visible, {occluded} occluded, "
          f"{landmarks.get('absent',0)} absent")
    if occluded / total < 0.05:
        print("  ! Under 5% occluded. A model trained here cannot place a")
        print("    landmark it cannot see, so standing and stem-away fruit will")
        print("    not work - see ANNOTATION_SPEC section 3.")
    return 0 if ok else 1


def trace(weights: Path, root: Path) -> int:
    """Follow a model back through the datasets it came from."""
    manifest_path = weights.with_suffix(".json")
    if not manifest_path.exists():
        print(f"No manifest beside {weights.name}.")
        print("Models trained before these were written cannot be traced - "
              "that is the situation they exist to end.")
        return 1
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"{weights.name}")
    print(f"  trained   {record.get('created','?')}  "
          f"git {record.get('git_commit','?')}")
    print(f"  from      {record.get('base_weights','?')}  "
          f"imgsz {record.get('imgsz','?')}  epochs {record.get('epochs','?')}")
    dataset = Path(record.get("dataset", ""))
    depth = 1
    while dataset and dataset.exists():
        manifest = dm.read(dataset)
        ok, message = dm.verify(dataset)
        indent = "  " + "  " * depth
        if manifest is None:
            print(f"{indent}{dataset.name}  (no manifest)")
            break
        counts = manifest.get("counts", {})
        total = sum(c.get("fruit", 0) for c in counts.values())
        print(f"{indent}{dataset.name}  {total} fruit  id "
              f"{manifest.get('content_id','?')}  [{message}]")
        parents = manifest.get("built_from", {}).get("parents", [])
        if not parents:
            break
        # Follow the first parent. Several parents are recorded but a full tree
        # is more than anyone reads; --show gives the rest.
        nxt = parents[0].get("path")
        dataset = Path(nxt) if nxt else None
        depth += 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List, inspect and verify exported datasets."
    )
    parser.add_argument("--root", default=str(project_path("data/datasets")))
    parser.add_argument("--show", metavar="NAME",
                        help="print one dataset's manifest in full")
    parser.add_argument("--verify", action="store_true",
                        help="check every dataset against its recorded content id")
    parser.add_argument("--trace", metavar="WEIGHTS",
                        help="follow a model back through its datasets")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.trace:
        return trace(Path(args.trace).resolve(), root)

    found = datasets_under(root)
    if not found:
        print(f"No datasets under {root}")
        return 1

    if args.show:
        wanted = [d for d in found if args.show in d.name]
        if not wanted:
            print(f"No dataset matching '{args.show}' under {root}")
            return 1
        return show(wanted[0])

    print(f"\ndatasets under {root}\n")
    for dataset in found:
        summarise(dataset)

    if args.verify:
        bad = [d for d in found if not dm.verify(d)[0]]
        print()
        if bad:
            print(f"{len(bad)} dataset(s) do not match their manifest:")
            for dataset in bad:
                print(f"   {dataset.name}: {dm.verify(dataset)[1]}")
            return 1
        print("all datasets intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
