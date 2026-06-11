from pathlib import Path

import pytest

from airctl.actions import build_action_map
from airctl.config import load_config

DEFAULT_TOML = (Path(__file__).parent.parent / "src" / "airctl"
                / "config.default.toml")


def test_bundled_default_config_parses_and_builds():
    cfg = load_config(DEFAULT_TOML)
    assert cfg.camera.backend == "dshow"
    assert cfg.fsm.confirm_frames == 5
    assert cfg.classifier.pinch_enter == pytest.approx(0.30)
    actions = build_action_map(cfg.mapping)
    # The full v1 vocabulary from the design doc must be mapped.
    for gesture in ("peace", "point_up", "point_down", "shaka", "pinch",
                    "pinch2", "swipe_left", "swipe_right",
                    "swipe2_left", "swipe2_right", "three"):
        assert gesture in actions, f"{gesture} missing from default mapping"


def test_missing_sections_fall_back_to_defaults(tmp_path):
    p = tmp_path / "minimal.toml"
    p.write_text('[modes.default]\n"peace" = { action = "media.play_pause" }\n')
    cfg = load_config(p)
    assert cfg.camera.width == 640
    assert cfg.fsm.cooldown_ms == 400
    assert list(cfg.mapping) == ["peace"]


def test_malformed_toml_raises(tmp_path):
    p = tmp_path / "broken.toml"
    p.write_text("[camera\ndevice = ???")
    with pytest.raises(Exception):
        load_config(p)


def test_bad_action_in_mapping_raises_on_build(tmp_path):
    p = tmp_path / "badaction.toml"
    p.write_text('[modes.default]\n"fist" = { action = "nope.nope" }\n')
    cfg = load_config(p)  # parses fine ...
    with pytest.raises(ValueError):  # ... but must be rejected before use
        build_action_map(cfg.mapping)
