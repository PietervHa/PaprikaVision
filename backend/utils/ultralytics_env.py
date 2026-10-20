"""
Ultralytics must never pip-install anything at runtime

Ultralytics checks its optional dependencies when a model loads or exports,
and by default installs whatever is missing - mid-run, from the internet. It
installed openvino and onnxruntime into this venv during a benchmark without
being asked. On the line that is unacceptable: a production PC changing its
own packages because a config pointed at an .onnx turns a config edit into an
unreviewed software update, and on a machine without internet it just hangs
on the retries.

The switch is the YOLO_AUTOINSTALL environment variable, read ONCE when
ultralytics.utils is imported (ultralytics 8.3.40, utils/__init__.py).
utils/checks.py then copies the value into its own namespace with
`from ultralytics.utils import AUTOINSTALL`, so setting the variable after the
import changes nothing. disable_autoinstall() therefore sets the variable and,
if Ultralytics is already loaded, patches both copies - import order cannot
undo it.

With it off, a missing package makes check_requirements log a warning and
return, and the import that follows fails with an ordinary ImportError. That
is the right failure: requirements.txt is where packages come from.
"""

from __future__ import annotations

import os
import sys


def disable_autoinstall() -> None:
    """Stop Ultralytics from pip-installing packages, whenever it is imported."""
    os.environ["YOLO_AUTOINSTALL"] = "False"
    for name in ("ultralytics.utils", "ultralytics.utils.checks"):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "AUTOINSTALL"):
            module.AUTOINSTALL = False
