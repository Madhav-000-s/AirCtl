import numpy as np
import pytest

from airctl.features import MIDDLE_MCP, extract_features, normalize_landmarks

import hands


def states(landmarks):
    return extract_features(landmarks, "Right").finger_states


def test_open_palm_all_extended():
    assert states(hands.open_palm()) == (True, True, True, True, True)


def test_fist_none_extended():
    assert states(hands.fist()) == (False, False, False, False, False)


def test_point_index_only():
    assert states(hands.point()) == (False, True, False, False, False)


def test_peace():
    assert states(hands.peace()) == (False, True, True, False, False)


def test_shaka():
    assert states(hands.shaka()) == (True, False, False, False, True)


@pytest.mark.parametrize("scale", [0.05, 0.12, 0.25])
@pytest.mark.parametrize("center", [(0.2, 0.3), (0.5, 0.5), (0.8, 0.7)])
def test_scale_and_translation_invariance(scale, center):
    assert states(hands.peace(scale=scale, center=center)) == \
        (False, True, True, False, False)


@pytest.mark.parametrize("angle", [-40, -15, 0, 20, 45])
def test_rotation_invariance(angle):
    assert states(hands.three(angle_deg=angle)) == (False, True, True, True, False)


def test_normalization_orients_middle_mcp_up():
    for angle in (0, 30, -60):
        norm = normalize_landmarks(hands.open_palm(angle_deg=angle))
        np.testing.assert_allclose(norm[MIDDLE_MCP, :2], [0.0, -1.0], atol=1e-9)


def test_pinch_distance_is_scale_invariant():
    d_small = extract_features(hands.pinch(0.2, scale=0.06), "Right").pinch_index
    d_large = extract_features(hands.pinch(0.2, scale=0.2), "Right").pinch_index
    assert d_small == pytest.approx(0.2, abs=1e-6)
    assert d_large == pytest.approx(0.2, abs=1e-6)


def test_index_dir_y_negative_when_pointing_up():
    f = extract_features(hands.point(), "Right")
    assert f.index_dir_y < 0  # image y grows downward
