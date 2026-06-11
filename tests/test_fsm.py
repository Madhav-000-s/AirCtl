from airctl.classifier import ClassifiedFrame
from airctl.fsm import (
    ArmedChanged,
    FSMConfig,
    GestureActivated,
    GestureFSM,
    GestureHeld,
    GestureReleased,
)

FPS = 30.0
CFG = FSMConfig(confirm_frames=5, cooldown_ms=400, arm_hold_s=1.0,
                disarm_hold_s=1.0, arm_timeout_s=10.0)


def step(fsm, pose, t, speed=0.0, swipe=None):
    return fsm.update(ClassifiedFrame(t=t, pose=pose, swipe=swipe,
                                      speed=speed, features=None))


def run(fsm, pose, seconds, t0, speed=0.0):
    """Feed a constant pose for `seconds`; return (events, end_time)."""
    events = []
    n = int(seconds * FPS)
    for i in range(n):
        events += step(fsm, pose, t0 + i / FPS, speed)
    return events, t0 + n / FPS


def armed_fsm(t0=0.0):
    fsm = GestureFSM(CFG)
    _, t = run(fsm, "open_palm", 1.2, t0)
    assert fsm.armed
    _, t = run(fsm, None, 0.1, t)  # drop the hand so the palm hold ends
    return fsm, t


def names(events, cls):
    return [e.name for e in events if isinstance(e, cls)]


def test_open_palm_hold_arms():
    fsm = GestureFSM(CFG)
    events, _ = run(fsm, "open_palm", 1.2, 0.0)
    armed = [e for e in events if isinstance(e, ArmedChanged)]
    assert [e.armed for e in armed] == [True]
    # The arming palm itself must not fire a gesture action.
    assert names(events, GestureActivated) == []


def test_brief_palm_does_not_arm():
    fsm = GestureFSM(CFG)
    events, t = run(fsm, "open_palm", 0.5, 0.0)
    events2, _ = run(fsm, None, 0.5, t)
    assert not fsm.armed
    assert all(not isinstance(e, ArmedChanged) for e in events + events2)


def test_disarmed_pose_fires_nothing():
    fsm = GestureFSM(CFG)
    events, _ = run(fsm, "peace", 2.0, 0.0)
    assert events == []


def test_armed_pose_activates_once_with_held_and_release():
    fsm, t = armed_fsm()
    events, t = run(fsm, "peace", 0.5, t)
    assert names(events, GestureActivated) == ["peace"]
    assert len(names(events, GestureHeld)) > 0
    events, _ = run(fsm, None, 0.1, t)
    assert names(events, GestureReleased) == ["peace"]


def test_candidate_never_fires():
    """A pose held for fewer than confirm_frames must not activate."""
    fsm, t = armed_fsm()
    events = []
    for i in range(CFG.confirm_frames - 1):
        events += step(fsm, "peace", t + i / FPS)
    events += step(fsm, None, t + CFG.confirm_frames / FPS)
    assert names(events, GestureActivated) == []


def test_cooldown_blocks_immediate_retrigger():
    fsm, t = armed_fsm()
    _, t = run(fsm, "peace", 0.4, t)
    _, t = run(fsm, None, 0.1, t)           # release -> cooldown starts
    events, t = run(fsm, "peace", 0.25, t)   # inside the 400 ms cooldown
    assert names(events, GestureActivated) == []
    events, _ = run(fsm, "peace", 0.5, t)    # cooldown has expired
    assert names(events, GestureActivated) == ["peace"]


def test_fast_moving_pose_never_confirms():
    fsm, t = armed_fsm()
    events, _ = run(fsm, "peace", 1.0, t, speed=2.0)
    assert names(events, GestureActivated) == []


def test_fist_hold_disarms():
    fsm, t = armed_fsm()
    events, _ = run(fsm, "fist", 1.2, t)
    armed = [e.armed for e in events if isinstance(e, ArmedChanged)]
    assert armed == [False]
    assert not fsm.armed


def test_arm_timeout_disarms():
    fsm, t = armed_fsm()
    events, _ = run(fsm, None, CFG.arm_timeout_s + 0.5, t)
    armed = [e.armed for e in events if isinstance(e, ArmedChanged)]
    assert armed == [False]


def test_activity_refreshes_arm_timeout():
    fsm, t = armed_fsm()
    # Stay busy past the original timeout: hold a gesture at t+8s for 3s.
    _, t2 = run(fsm, None, 8.0, t)
    _, t3 = run(fsm, "peace", 3.0, t2)
    assert fsm.armed  # would have expired at t+10 without the refresh


def test_swipe_fires_only_when_armed():
    fsm = GestureFSM(CFG)
    events = step(fsm, "open_palm", 0.0, speed=2.0, swipe="swipe_right")
    assert names(events, GestureActivated) == []
    fsm2, t = armed_fsm()
    events = step(fsm2, "open_palm", t, speed=2.0, swipe="swipe_right")
    assert names(events, GestureActivated) == ["swipe_right"]
