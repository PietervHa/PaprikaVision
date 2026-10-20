# Every `python -m tools.<name>` runs this first. The tools import ultralytics
# directly, and must no more pip-install packages at runtime than the app may;
# see backend/utils/ultralytics_env.py. Child processes (yolo pose train,
# label_stems from retrain) inherit the setting through the environment.
from backend.utils.ultralytics_env import disable_autoinstall

disable_autoinstall()
