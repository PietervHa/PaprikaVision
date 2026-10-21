import pytest

from backend.detection.paprika import pose_detector


@pytest.fixture(autouse=True)
def fake_keep_cuda_off(monkeypatch):
    """Record keep_cuda_off instead of patching the real torch for the session.

    The real one is process-wide and permanent by design; one test loading a
    model on "cpu" would otherwise change torch for every test after it.
    """
    state = {"calls": 0}

    def keep_cuda_off():
        state["calls"] += 1

    monkeypatch.setattr(pose_detector, "keep_cuda_off", keep_cuda_off)
    monkeypatch.setattr(pose_detector, "cuda_kept_off", lambda: state["calls"] > 0)
    return state
