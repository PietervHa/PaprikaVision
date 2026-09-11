"""What the failure capture decides is worth a frame.

Driven by a real 50-frame run: 4 of the 7 captured frames turned out to be a
correctly placed fruit beside a second fruit half out of shot. Those are the
belt working normally, not the detector failing, and they crowd out the cases
capture exists to find.
"""

import pytest

from backend.core.overlay_worker import _FailureCapture


def fruit(placement, pose):
    return {"placement": placement, "orientation": {"pose": pose}}


def scene(*detections, primary=None):
    """A result as the engine builds it: primary is one OF the detections."""
    detections = list(detections)
    if primary is None:
        primary = detections[0] if detections else None
    return {"primary": primary, "detections": detections}


def capture(**block):
    block.setdefault("save_failures", True)
    return _FailureCapture(block)


def test_placed_fruit_is_not_a_failure():
    assert capture()._first_failure(scene(fruit("place", "lying"))) is None


def test_empty_belt_is_not_a_failure():
    """An empty belt is the normal state; capturing it would save the shift."""
    assert capture()._first_failure(scene()) is None


def test_incomplete_is_ignored_by_default():
    """The frame that dominated the real run: one good fruit, one at the edge."""
    good = fruit("place", "lying")
    clipped = fruit("reject", "incomplete")
    assert capture()._first_failure(scene(good, clipped, primary=good)) is None


def test_stem_not_found_is_captured():
    found = capture()._first_failure(scene(fruit("reject", "stem_not_found")))
    assert found is not None
    assert found["orientation"]["pose"] == "stem_not_found"


@pytest.mark.parametrize("placement,pose", [
    ("reorient", "lying"),
    ("unknown", "lying"),
    ("reject", "upside_down"),
    ("reject", "stem_not_found"),
])
def test_every_other_failure_is_captured(placement, pose):
    assert capture()._first_failure(scene(fruit(placement, pose))) is not None


def test_an_ignored_pose_does_not_disqualify_the_whole_frame():
    """The case that makes this a per-fruit filter rather than a per-frame one:
    a clipped fruit next to one the stem search failed on is exactly the scene
    worth keeping, so the search must not stop at the clipped one."""
    clipped = fruit("reject", "incomplete")
    interesting = fruit("reject", "stem_not_found")
    found = capture()._first_failure(scene(clipped, interesting, primary=clipped))
    assert found is interesting


def test_primary_wins_when_it_is_itself_interesting():
    """So the filename carries the primary's verdict, not a bystander's."""
    primary = fruit("reorient", "lying")
    other = fruit("reject", "stem_not_found")
    found = capture()._first_failure(scene(primary, other, primary=primary))
    assert found is primary


def test_empty_ignore_list_captures_everything():
    clipped = fruit("reject", "incomplete")
    found = capture(ignore_poses=[])._first_failure(scene(clipped))
    assert found is clipped


def test_ignore_list_accepts_a_bare_string():
    """ignore_poses: incomplete is an easy thing to write in YAML."""
    assert capture(ignore_poses="stem_not_found")._first_failure(
        scene(fruit("reject", "stem_not_found"))
    ) is None


def test_ignore_list_is_case_insensitive():
    assert capture(ignore_poses=["INCOMPLETE"])._first_failure(
        scene(fruit("reject", "incomplete"))
    ) is None


def test_ignored_frames_are_counted_so_an_empty_folder_is_explainable():
    cap = capture()
    for _ in range(3):
        cap._first_failure(scene(fruit("reject", "incomplete")))
    assert cap.status()["ignored"] == 3
    assert cap.status()["ignore_poses"] == ["incomplete"]


def test_a_clean_frame_does_not_count_as_ignored():
    cap = capture()
    cap._first_failure(scene(fruit("place", "lying")))
    assert cap.status()["ignored"] == 0
