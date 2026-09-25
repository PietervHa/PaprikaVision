"""
What a dataset folder is, and how to prove it still is that

An exported dataset is the durable artefact of this project: it holds copies of
the images and their labels, so once it exists the raw frames and the labels
file are no longer needed to train from it. That only works if a dataset can
answer three questions on its own.

    What am I?        name, when it was built, how many frames and fruit
    Where did I come from?   which datasets it was built on, and what was added
    Am I still intact?       a content id over every label file

The third is the one that earns its keep. A dataset folder is ordinary files:
someone can drop images in, a sync can half-copy it, an editor can rewrite a
label. None of that announces itself, and a model trained on a quietly altered
dataset is a model whose provenance is a lie. The content id is computed from
the sorted (name, label-hash) pairs, so any added, removed or edited label
changes it.

Lineage is recorded as the PARENT'S content id, not its path. Folders get moved
and renamed; what was actually trained on does not change. A chain of ids can
be checked; a chain of paths can only be hoped at.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

MANIFEST_NAME = "DATASET.json"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}


def content_id(dataset: Path) -> str:
    """A short id over every label file in the dataset.

    Labels rather than images, deliberately: the labels are what the model
    learns from, they are small enough to hash quickly, and an image that
    changes without its label changing is a different problem (and a rarer
    one) than a label being edited.
    """
    digest = hashlib.sha1()
    for split in ("train", "val"):
        folder = dataset / split / "labels"
        if not folder.exists():
            continue
        for label in sorted(folder.iterdir()):
            if label.suffix != ".txt":
                continue
            digest.update(f"{split}/{label.name}".encode("utf-8"))
            digest.update(hashlib.sha1(label.read_bytes()).digest())
    return digest.hexdigest()[:12]


def counts(dataset: Path) -> dict:
    result = {}
    for split in ("train", "val"):
        images = dataset / split / "images"
        labels = dataset / split / "labels"
        frames = sum(1 for p in images.iterdir()
                     if p.suffix.lower() in IMAGE_SUFFIXES) if images.exists() else 0
        fruit = 0
        if labels.exists():
            for label in labels.iterdir():
                if label.suffix == ".txt":
                    fruit += sum(1 for line in label.read_text(encoding="utf-8").splitlines()
                                 if line.strip())
        result[split] = {"frames": frames, "fruit": fruit}
    return result


def landmark_stats(dataset: Path) -> dict:
    """Visibility flags across the whole set.

    Carried in the manifest because it is the statistic that decides whether a
    dataset can teach standing fruit at all, and reading it later should not
    require re-walking every label file.
    """
    seen = {"visible": 0, "occluded": 0, "absent": 0}
    for split in ("train", "val"):
        folder = dataset / split / "labels"
        if not folder.exists():
            continue
        for label in folder.iterdir():
            if label.suffix != ".txt":
                continue
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                # class cx cy w h then (x y v) per landmark
                for index in range(5, len(parts), 3):
                    try:
                        flag = int(float(parts[index + 2]))
                    except (IndexError, ValueError):
                        continue
                    if flag == 2:
                        seen["visible"] += 1
                    elif flag == 1:
                        seen["occluded"] += 1
                    else:
                        seen["absent"] += 1
    return seen


def write(dataset: Path, name: str, parents: list[dict], added: int,
          carried: int, sources: list[str]) -> dict:
    manifest = {
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "content_id": content_id(dataset),
        "counts": counts(dataset),
        "landmarks": landmark_stats(dataset),
        "built_from": {
            "parents": parents,
            "frames_carried": carried,
            "frames_added": added,
            "new_frame_folders": sources,
        },
    }
    (dataset / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def read(dataset: Path) -> dict | None:
    path = dataset / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def verify(dataset: Path) -> tuple[bool, str]:
    """Has anything changed since the manifest was written?"""
    manifest = read(dataset)
    if manifest is None:
        return False, "no manifest - built before these were written, or edited by hand"
    recorded = manifest.get("content_id")
    actual = content_id(dataset)
    if recorded != actual:
        return False, f"content changed: manifest says {recorded}, labels hash to {actual}"
    return True, f"intact ({actual})"
