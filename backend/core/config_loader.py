"""
Configuration Loader

Loads config/default.yaml (or --config <path>) into a module-level `cfg` dict.

Slimmed down from the VisionSoftwareMDE original: the OCR/Roboflow/classifier
normalisation and validation that made up most of that file described backends
this project does not have, so it was dropped rather than carried over dead.
Validation here covers only the paprika block.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

_VALID_BACKENDS = {"shape", "pose", "simulator"}
_VALID_MODES = {"paprika", "idle"}


def _load_config() -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    args, _ = parser.parse_known_args()

    config_path = Path(args.config) if args.config else _PROJECT_ROOT / "config" / "default.yaml"

    if not config_path.exists():
        log.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        return config if config is not None else {}
    except yaml.YAMLError as exc:
        log.error("Failed to parse YAML file %s: %s", config_path, exc)
        sys.exit(1)
    except Exception as exc:
        log.error("Failed to load config file %s: %s", config_path, exc)
        sys.exit(1)


def _validate(config: dict) -> dict:
    """Warn loudly about settings that would fail confusingly at runtime."""
    block = config.get("paprika")
    if not isinstance(block, dict):
        log.warning("No 'paprika' block in config; using defaults")
        config["paprika"] = {}
        return config

    backend = str(block.get("backend", "shape")).strip().lower()
    if backend not in _VALID_BACKENDS:
        log.warning(
            "paprika.backend '%s' is not one of %s; falling back to 'shape'",
            backend, sorted(_VALID_BACKENDS),
        )
        backend = "shape"
    block["backend"] = backend

    if backend == "pose":
        pose_cfg = block.get("pose") if isinstance(block.get("pose"), dict) else {}
        model_path = Path(str(pose_cfg.get("model_path", "")))
        if not model_path.is_absolute():
            model_path = _PROJECT_ROOT / model_path
        if not model_path.exists():
            # Not fatal: the detector reports "not ready" and the HMI shows it,
            # which is friendlier than refusing to boot on a machine where
            # somebody is mid-way through copying weights across.
            log.warning(
                "paprika.pose.model_path does not exist: %s - detector will report not ready",
                model_path,
            )

    mode = str(config.get("vision_mode", "paprika")).strip().lower()
    if mode not in _VALID_MODES:
        log.warning("vision_mode '%s' unknown; falling back to 'paprika'", mode)
        config["vision_mode"] = "paprika"

    config["paprika"] = block
    return config


cfg = _validate(_load_config())
