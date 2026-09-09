"""Path resolution must not depend on the working directory.

This is a regression test for a real failure: debug frames were written to
backend/data/debug/failures because a bare relative path resolved against the
working directory an IDE run configuration had chosen. Nothing errored - the
files simply appeared somewhere nobody was looking.
"""

import os
from pathlib import Path

import pytest

from backend.utils.paths import PROJECT_ROOT, project_path


def test_project_root_is_the_repository_root():
    # The markers that prove we resolved to the root and not to a package dir.
    assert (PROJECT_ROOT / "config" / "default.yaml").is_file()
    assert (PROJECT_ROOT / "backend").is_dir()
    assert (PROJECT_ROOT / "requirements.txt").is_file()


def test_relative_paths_anchor_to_the_project_root():
    assert project_path("data/debug/failures") == PROJECT_ROOT / "data/debug/failures"


def test_absolute_paths_are_left_alone():
    # Somebody who writes an absolute path means it - usually to put results on
    # another volume - so it must never be re-anchored.
    absolute = Path(PROJECT_ROOT.anchor) / "elsewhere" / "results"
    assert project_path(absolute) == absolute


@pytest.mark.parametrize("value", [None, ""])
def test_missing_value_falls_back_to_the_default(value):
    assert project_path(value, "data/results") == PROJECT_ROOT / "data/results"


def test_no_value_at_all_is_the_project_root():
    assert project_path(None) == PROJECT_ROOT


def test_resolution_is_independent_of_the_working_directory(tmp_path):
    """The actual bug. Same config value, different cwd, same answer."""
    original = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT / "backend")
        from_backend = project_path("data/debug/failures")
        os.chdir(tmp_path)
        from_elsewhere = project_path("data/debug/failures")
    finally:
        os.chdir(original)

    assert from_backend == from_elsewhere == PROJECT_ROOT / "data/debug/failures"


def test_failure_capture_directory_is_absolute_and_rooted():
    """The capture reports where it will write, so /status can answer
    "where are my frames" without knowing how the process was launched."""
    from backend.core.overlay_worker import _FailureCapture

    reported = Path(_FailureCapture({}).status()["dir"])
    assert reported.is_absolute()
    assert reported == PROJECT_ROOT / "data" / "debug" / "failures"
