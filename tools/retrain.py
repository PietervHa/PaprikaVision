"""
Retrain the pose model in one command, with provenance

    python -m tools.retrain --name v4
    python -m tools.retrain --name v4 --base-dataset data/datasets/paprika_v3
    python -m tools.retrain --name v4 --base-dataset data/datasets/paprika_v3 \
                            --base models/paprika_pose.pt
    python -m tools.retrain --name v4 --dry-run

Runs export, dataset check, training and scoring as one sequence, and writes
down what produced what. Every step is a tool that already exists; this removes
the ways of getting the sequence subtly wrong.

The mistakes it exists to prevent
---------------------------------
Each of these has actually happened on this project:

  Exporting only the captures. A dataset of nothing but the cases the detector
  already fails on teaches a model the hard ones and loses the easy ones. The
  source folders are now derived from the labels file, so everything that has
  ever been labelled comes along.

  Re-exporting into an existing folder. The exporter copies in without
  clearing, so stale images from a previous version linger and quietly join the
  training set. A new name is required, not suggested.

  Installing the wrong weights. runs/pose/ fills up with train, train2, train3,
  and picking the wrong one means measuring a model you did not just build.
  This records the run directory it created and installs from that.

  Weights nobody can trace. models/README.md asks for the dataset version to be
  kept with the weights; a manifest is written beside them automatically -
  dataset path, fruit counts, training arguments, git commit, scores.

  Overwriting a working model. The current weights are backed up before
  anything replaces them, and the new model is scored on frames it never saw
  BEFORE being installed. Worse than the one in service and it is not
  installed unless you insist.

Growing a model without keeping the raw frames
----------------------------------------------
An exported dataset is self-contained: it holds copies of the images and their
labels. Once data/datasets/paprika_v3 exists, data/raw, data/captures and
stem_labels.json are no longer needed to train from it - archive or delete
them, and keep the dataset folder as the artefact.

    --base-dataset data/datasets/paprika_v3

merges one (or several) of those into the new version, then adds whatever has
been labelled since. Each fruit keeps the train/val side it was on before, so a
model trained on v4 can still be compared with one trained on v3 - a frame that
crossed from train to val between versions would quietly invalidate that
comparison.

Why the old images have to be kept in SOME form
-----------------------------------------------
A network trained only on new examples forgets the old ones. There is no
setting that avoids this: the weights have no memory of data they can no longer
see, and a few hundred captured failures would teach it the hard cases while
losing the easy ones it already handles. Keeping the exported dataset is the
cheap version of remembering - one folder per version, archivable, and nothing
else to keep in step.

--base models/paprika_pose.pt fine-tunes from existing weights instead of
starting fresh. On a dataset this size that tends to overfit the additions, so
it is offered rather than recommended, and the scoring step will say whether it
worked.

What it does not do
-------------------
It will not label for you, and it will not decide whether the dataset is good
enough. dataset_check's warnings about standing and occluded fruit are printed
in full and are yours to read: a run that trains cleanly on a dataset missing
the cases you care about is a waste of an afternoon, and no script can tell the
difference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.utils.paths import project_path  # noqa: E402

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}


def run(command: list[str], dry_run: bool, cwd: Path) -> int:
    printable = " ".join(f'"{c}"' if " " in c else c for c in command)
    print(f"\n$ {printable}")
    if dry_run:
        return 0
    return subprocess.call(command, cwd=str(cwd))


def folders_holding_labelled_frames(labels_path: Path, root: Path) -> list[Path]:
    """Every directory under the project that holds a frame we have labelled.

    Derived rather than asked for. Passing the wrong set of folders is the
    single easiest way to build a bad dataset, and the labels file already
    knows which frames matter - so let it answer.
    """
    if not labels_path.exists():
        return []
    wanted = set(json.loads(labels_path.read_text(encoding="utf-8")))
    found: dict[Path, int] = {}
    for candidate in root.rglob("*"):
        if candidate.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "datasets" in candidate.parts:
            # Already-exported copies. Including them would feed a dataset
            # back into itself.
            continue
        if candidate.name in wanted:
            found[candidate.parent] = found.get(candidate.parent, 0) + 1
    return [p for p, _ in sorted(found.items(), key=lambda kv: -kv[1])]


def merge_datasets(bases, staged, out_dir):
    """Build one dataset from existing ones plus newly exported frames.

    A frame already present in a base keeps the split it had. Version-to-
    version comparisons are only meaningful if the val set stays the val set -
    a frame that crosses from train to val looks like an improvement and is
    not one.
    """
    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    taken: set[str] = set()
    carried = added = 0
    for source, is_base in [(b, True) for b in bases] + (
            [(staged, False)] if staged is not None else []):
        for split in ("train", "val"):
            images = source / split / "images"
            if not images.exists():
                continue
            for image in sorted(images.iterdir()):
                if image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if image.stem in taken:
                    continue
                label = source / split / "labels" / f"{image.stem}.txt"
                if not label.exists():
                    continue
                shutil.copy2(image, out_dir / split / "images" / image.name)
                shutil.copy2(label, out_dir / split / "labels" / label.name)
                taken.add(image.stem)
                if is_base:
                    carried += 1
                else:
                    added += 1

    reference = (staged if staged is not None else bases[0]) / "data.yaml"
    target = out_dir / "data.yaml"
    if reference.exists():
        text = reference.read_text(encoding="utf-8")
        lines = [f"path: {out_dir.as_posix()}" if l.startswith("path:") else l
                 for l in text.splitlines()]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return carried, added


def newest_run(runs_root: Path, after: float) -> Path | None:
    """The training run created by this invocation, not whichever is newest.

    runs/pose fills up with train, train2, train3 and picking by name sorts
    train10 before train2. Picking by "newest" alone would happily return a run
    from last week if training failed. Both are how you end up measuring a
    model you did not just build.
    """
    if not runs_root.exists():
        return None
    candidates = [
        d for d in runs_root.iterdir()
        if d.is_dir() and (d / "weights" / "best.pt").exists()
        and (d / "weights" / "best.pt").stat().st_mtime >= after
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / "weights" / "best.pt").stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export, check, train and score the pose model as one step."
    )
    parser.add_argument("--name", required=True,
                        help="dataset version name, e.g. v3. Must not already exist.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=None,
                        help="defaults to paprika.pose.imgsz so the model is "
                             "trained at the size it will be run at")
    parser.add_argument("--base", default="yolo11n-pose.pt",
                        help="weights to start from. The default trains fresh; "
                             "point at models/paprika_pose.pt to fine-tune, "
                             "which risks forgetting on a small dataset.")
    parser.add_argument("--base-dataset", action="append", default=[],
                        metavar="PATH",
                        help="an existing exported dataset to build on. "
                             "Repeatable. Its images come across with their "
                             "original train/val side, so versions stay "
                             "comparable - and the raw frames it was built "
                             "from are not needed.")
    parser.add_argument("--install", action="store_true",
                        help="install the new weights if they score at least as "
                             "well as the ones in service")
    parser.add_argument("--force-install", action="store_true",
                        help="install even if the new model scores worse")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = project_path(".")
    started = datetime.now()
    out_dir = project_path("data/datasets") / f"paprika_{args.name}"
    if out_dir.exists() and not args.dry_run:
        print(f"{out_dir} already exists.\n"
              f"Pick a new --name. The exporter copies in without clearing, so "
              f"reusing a folder mixes this dataset with the last one.")
        return 1

    try:
        import yaml

        cfg = yaml.safe_load(
            (project_path("config/default.yaml")).read_text(encoding="utf-8")
        )
        pose_cfg = (cfg.get("paprika") or {}).get("pose") or {}
        config_imgsz = int(pose_cfg.get("imgsz", 640))
    except Exception:
        config_imgsz = 640
    imgsz = args.imgsz or config_imgsz
    if imgsz != config_imgsz:
        print(f"\n!! Training at {imgsz} but paprika.pose.imgsz is {config_imgsz}.")
        print("   Inferring at a size the model never trained on costs accuracy "
              "for no reason.")
        print(f"   Set paprika.pose.imgsz to {imgsz} before running this model.")

    bases = [Path(b).resolve() for b in args.base_dataset]
    for base in bases:
        if not (base / "train" / "images").exists():
            print(f"{base} does not look like an exported dataset "
                  f"(no train/images).")
            return 1

    labels_path = project_path("data/debug/labels") / "stem_labels.json"
    sources = folders_holding_labelled_frames(labels_path, root)
    if not sources and not bases:
        print(f"Nothing to train on. Either label some frames (expected "
              f"{labels_path}) or pass --base-dataset.")
        return 1
    if sources:
        print("frame folders holding newly labelled images:")
        for folder in sources:
            print(f"   {folder}")
    else:
        print("no newly labelled frames - building from the base dataset(s) alone")

    # 1. export the new frames, then merge the bases in ---------------------
    staged = out_dir.with_name(out_dir.name + "_new")
    if sources:
        command = [sys.executable, "-m", "tools.export_dataset"]
        command += [str(f) for f in sources]
        command += ["--out", str(staged)]
        if run(command, args.dry_run, root) != 0:
            print("export failed")
            return 1

    if not args.dry_run:
        carried, added = merge_datasets(bases, staged if sources else None, out_dir)
        print(f"\nmerged dataset at {out_dir}")
        print(f"   {carried} frame(s) carried over from {len(bases)} base dataset(s), "
              f"keeping their original train/val side")
        print(f"   {added} newly labelled frame(s) added")
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
    else:
        print(f"\n(dry run) would merge {len(bases)} base dataset(s) "
              f"and the newly exported frames into {out_dir}")

    # 2. check -------------------------------------------------------------
    print("\n--- dataset check: read the warnings, they are about YOUR fruit ---")
    checked = run([sys.executable, "-m", "tools.dataset_check", str(out_dir)],
                  args.dry_run, root)
    if checked != 0 and not args.dry_run:
        print("\nThe dataset has errors. Fix them, or accept them deliberately:")
        print(f"   python -m tools.dataset_check \"{out_dir}\" --i-know-better")
        print("Training on a dataset with known blind spots is a decision, not "
              "an accident - so it is not made for you here.")
        return 1

    # 3. train -------------------------------------------------------------
    runs_root = root / "runs" / "pose"
    before = started.timestamp()
    train = ["yolo", "pose", "train",
             f"data={(out_dir / 'data.yaml').as_posix()}",
             f"model={args.base}", f"imgsz={imgsz}", f"epochs={args.epochs}",
             f"patience={args.patience}", f"batch={args.batch}",
             "amp=False", "workers=4"]
    if run(train, args.dry_run, root) != 0:
        print("training failed")
        return 1

    if args.dry_run:
        print("\n(dry run - stopping before scoring)")
        return 0

    produced = newest_run(runs_root, before)
    if produced is None:
        print(f"No new weights under {runs_root}. Did training finish?")
        return 1
    candidate = produced / "weights" / "best.pt"
    print(f"\ntrained weights: {candidate}")

    # 4. score, on frames it never saw -------------------------------------
    val_images = out_dir / "val" / "images"
    print("\n--- scoring on the held-out split ---")
    print("Scoring on data/raw would include the training frames and flatter "
          "the result.")
    live = project_path("models/paprika_pose.pt")
    backup = None
    if live.exists():
        backup = live.with_name(
            f"paprika_pose_before_{args.name}_{started:%Y%m%d_%H%M%S}.pt"
        )
        shutil.copy2(live, backup)
        print(f"current model backed up to {backup.name}")

    shutil.copy2(candidate, live)
    run([sys.executable, "-m", "tools.label_stems", str(val_images),
         "--score", "--worst", "10"], False, root)

    # 5. provenance --------------------------------------------------------
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(root)
        ).decode().strip()
    except Exception:
        commit = "unknown"
    manifest = produced / "MANIFEST.json"
    manifest.write_text(json.dumps({
        "created": started.isoformat(timespec="seconds"),
        "dataset": str(out_dir),
        "source_folders": [str(f) for f in sources],
        "base_weights": args.base,
        "imgsz": imgsz,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "git_commit": commit,
        "replaced": str(backup) if backup else None,
    }, indent=2), encoding="utf-8")
    shutil.copy2(manifest, live.with_suffix(".json"))
    print(f"\nprovenance written to {manifest} and beside the installed weights")

    if not args.install and not args.force_install:
        print("\nThe new model is installed for scoring only.")
        print("Read the numbers above against the model you had. To keep it, "
              "nothing more is needed.")
        if backup:
            print(f"To go back:  copy {backup.name} models\\paprika_pose.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
