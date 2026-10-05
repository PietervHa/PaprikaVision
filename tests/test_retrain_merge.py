"""Merging a base dataset with freshly exported frames.

Two rules, pulling in different directions, and both matter:

  the LABEL comes from the fresh export, so a click corrected since the base
  was built actually reaches the new dataset. The base copy winning was a quiet
  trap - re-labelling an old frame, or running --fix-visibility over standing
  fruit, appeared to work, saved correctly, and was silently discarded at
  export because an older copy already existed.

  the train/val SIDE comes from the base. A frame crossing from train to val
  between versions looks like an improvement and is not one.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.retrain import merge_datasets

BOTH_VISIBLE = "0 0.5 0.5 0.5 0.5 0.6 0.4 2 0.4 0.6 2\n"
BLOSSOM_OCCLUDED = "0 0.5 0.5 0.5 0.5 0.6 0.4 2 0.4 0.6 1\n"


def _build(root: Path, name: str, frames: dict) -> Path:
    dataset = root / name
    image = np.full((80, 80, 3), (190, 110, 45), np.uint8)
    for split, stems in frames.items():
        (dataset / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset / split / "labels").mkdir(parents=True, exist_ok=True)
        for stem, label in stems:
            cv2.imwrite(str(dataset / split / "images" / f"{stem}.bmp"), image)
            (dataset / split / "labels" / f"{stem}.txt").write_text(label)
    (dataset / "data.yaml").write_text(
        "path: x\ntrain: train/images\nval: val/images\n"
    )
    return dataset


def _label(out: Path, stem: str):
    for split in ("train", "val"):
        path = out / split / "labels" / f"{stem}.txt"
        if path.exists():
            return split, path.read_text().strip()
    return None, None


@pytest.fixture
def merged(tmp_path):
    base = _build(tmp_path, "base", {
        "train": [("shared", BOTH_VISIBLE), ("onlybase", BOTH_VISIBLE)],
        "val": [("inval", BOTH_VISIBLE)],
    })
    # The staged export puts "inval" in TRAIN - the side must not follow it.
    staged = _build(tmp_path, "staged", {
        "train": [("shared", BLOSSOM_OCCLUDED), ("brandnew", BLOSSOM_OCCLUDED),
                  ("inval", BLOSSOM_OCCLUDED)],
    })
    out = tmp_path / "merged"
    counts = merge_datasets([base], staged, out)
    return out, counts


def test_a_relabelled_frame_takes_its_current_label(merged):
    out, _ = merged
    _, label = _label(out, "shared")
    assert label == BLOSSOM_OCCLUDED.strip(), (
        "the fix made since the base was built must reach the new dataset"
    )


def test_a_frame_only_in_the_base_is_carried_unchanged(merged):
    out, _ = merged
    split, label = _label(out, "onlybase")
    assert (split, label) == ("train", BOTH_VISIBLE.strip())


def test_a_brand_new_frame_is_added(merged):
    out, _ = merged
    split, label = _label(out, "brandnew")
    assert (split, label) == ("train", BLOSSOM_OCCLUDED.strip())


def test_the_split_still_comes_from_the_base(merged):
    """New label, base's side. A frame crossing train to val between versions
    looks like an improvement and is not one."""
    out, _ = merged
    split, label = _label(out, "inval")
    assert split == "val"
    assert label == BLOSSOM_OCCLUDED.strip()


def test_the_counts_distinguish_the_three_cases(merged):
    _, (carried, added, refreshed) = merged
    assert (carried, added, refreshed) == (1, 1, 2)


def test_a_base_with_no_new_frames_is_copied_through(tmp_path):
    base = _build(tmp_path, "base", {"train": [("a", BOTH_VISIBLE)],
                                     "val": [("b", BOTH_VISIBLE)]})
    out = tmp_path / "merged"
    carried, added, refreshed = merge_datasets([base], None, out)
    assert (carried, added, refreshed) == (2, 0, 0)
    assert _label(out, "b")[0] == "val"
