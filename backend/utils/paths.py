"""
Project Paths

One definition of where the project lives, and one rule for what a relative
path means.

The rule: a relative path anywhere in the config, or anywhere in the code, is
relative to the PROJECT ROOT - never to the working directory the process
happened to be started from.

Why that rule, and why in one place
-----------------------------------
This project is started in several ways: `python -m backend.main` from the
root, a run configuration in an IDE, a service wrapper, a shortcut. Those do
not agree on the working directory. A bare `Path("data/debug/failures")`
therefore points at a different folder depending on how the machine was
launched that morning, and it fails silently - the folder is simply created
somewhere else, with no error and nothing in the log to suggest anything is
wrong. Debug frames were once written to backend/data/debug/failures for
exactly this reason while an operator watched an empty data/debug/failures and
concluded the feature was broken.

The rule itself was never in doubt - six separate modules each spelled out
`Path(__file__).resolve().parents[2]` and one of them documented it as "like
every other path in this project". What was missing was somewhere to say it
once. Six copies means the seventh author has to notice the convention by
reading unrelated files, which is how the seventh copy ends up being the one
that gets it wrong.

`parents[2]` is load-bearing: this file is backend/utils/paths.py, so two
levels up from its own directory is the repository root. Moving this file
changes that number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def project_path(value: Union[str, Path, None], default: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a configured path against the project root.

    Absolute paths are returned untouched - somebody who writes an absolute
    path in the config means it, typically to put results on another volume.
    Relative ones are anchored to PROJECT_ROOT.

    Args:
        value:   the configured path. None or empty falls back to `default`.
        default: used when `value` is missing. Empty itself means "the project
                 root", which is the only sensible reading of "no path given".

    Returns:
        An absolute Path. Nothing is created here: deciding WHERE a directory
        belongs and deciding WHETHER to create it are separate decisions, and
        a resolver that quietly makes directories is impossible to call from a
        read-only context.
    """
    candidate = value if value not in (None, "") else default
    if candidate in (None, ""):
        return PROJECT_ROOT
    path = Path(str(candidate)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
