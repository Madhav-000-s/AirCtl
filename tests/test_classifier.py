import numpy as np

from airctl.classifier import ClassifierConfig, PoseClassifier
from airctl.features import extract_features

import hands

FPS = 30.0


def feed(clf, landmarks, t):
    return clf.classify(t, extract_features(landmarks, "Right"))


def hold(clf, builder, n=10, t0=0.0, **kw):
    """Feed n identical frames; return the last ClassifiedFrame."""
    cf = None
    for i in range(n):
        cf = feed(clf, builder(**kw), t0 + i / FPS)
    return cf


def test_static_poses():
    expectations = [
        (hands.open_palm, "open_palm"),
        (hands.fist, "fist"),
        (hands.peace, "peace"),
        (hands.three, "three"),
        (hands.shaka, "shaka"),
        (hands.point, "point_up"),
    ]
    for builder, expected in expectations:
        cf = hold(PoseClassifier(), builder)
        assert cf.pose == expected, f"{builder.__name__} -> {cf.pose}"


def test_point_down():
    cf = hold(PoseClassifier(), hands.point, angle_deg=180)
    assert cf.pose == "point_down"


def test_fist_is_never_point():
    clf = PoseClassifier()
    for i in range(60):
        cf = feed(clf, hands.fist(), i / FPS)
        assert cf.pose not in ("point_up", "point_down")


def test_no_hand_gives_no_pose():
    clf = PoseClassifier()
    cf = clf.classify(0.0, None)
    assert cf.pose is None and cf.swipe is None


def test_pinch_hysteresis():
    clf = PoseClassifier(ClassifierConfig(pinch_enter=0.30, pinch_exit=0.45))
    t = [0.0]

    def at(dist):
        t[0] += 1 / FPS
        return feed(clf, hands.pinch(dist), t[0]).pose

    assert at(0.6) != "pinch"        # apart: not pinched
    assert at(0.10) == "pinch"       # below enter: pinch engages
    assert at(0.40) == "pinch"       # between thresholds: stays pinched
    assert at(0.50) != "pinch"       # above exit: releases
    assert at(0.40) != "pinch"       # between thresholds: stays released


def test_swipe_right_on_fast_palm_motion():
    clf = PoseClassifier()
    swipes = []
    # Open palm moving right at 2 image-widths/s.
    for i in range(12):
        t = i / FPS
        cf = feed(clf, hands.open_palm(center=(0.2 + 2.0 * t, 0.5)), t)
        if cf.swipe:
            swipes.append(cf.swipe)
    assert swipes == ["swipe_right"]  # exactly one, then refractory window


def test_peace_swipe_uses_two_finger_prefix():
    clf = PoseClassifier()
    swipes = []
    for i in range(12):
        t = i / FPS
        cf = feed(clf, hands.peace(center=(0.8 - 2.0 * t, 0.5)), t)
        if cf.swipe:
            swipes.append(cf.swipe)
    assert swipes == ["swipe2_left"]


def test_no_swipe_without_gating_pose():
    clf = PoseClassifier()
    for i in range(12):
        t = i / FPS
        cf = feed(clf, hands.fist(center=(0.2 + 2.0 * t, 0.5)), t)
        assert cf.swipe is None


def test_slow_motion_is_not_a_swipe():
    clf = PoseClassifier()
    for i in range(30):
        t = i / FPS
        cf = feed(clf, hands.open_palm(center=(0.3 + 0.2 * t, 0.5)), t)
        assert cf.swipe is None
