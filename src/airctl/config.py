"""TOML configuration: load, defaults, and watchdog hot-reload (§6.1).

Load order: explicit --config path, else %APPDATA%\\airctl\\config.toml
(created from the bundled default on first run). A failed reload keeps the
previous configuration running — a typo while editing must never kill the app.
"""

from __future__ import annotations

import logging
import shutil
import threading
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Callable

from .capture import CameraConfig
from .classifier import ClassifierConfig
from .fsm import FSMConfig
from .paths import user_config_path

log = logging.getLogger(__name__)


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    fsm: FSMConfig = field(default_factory=FSMConfig)
    min_hand_size: float = 0.12
    mapping: dict[str, dict] = field(default_factory=dict)


def _default_config_text() -> str:
    return (resources.files("airctl") / "config.default.toml").read_text(
        encoding="utf-8")


def ensure_user_config() -> Path:
    path = user_config_path()
    if not path.exists():
        with resources.as_file(
            resources.files("airctl") / "config.default.toml"
        ) as src:
            shutil.copy(src, path)
        log.info("created default config at %s", path)
    return path


def load_config(path: Path) -> AppConfig:
    """Parse and validate a config file. Raises on malformed input."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    cam = data.get("camera", {})
    camera = CameraConfig(
        device=int(cam.get("device", 0)),
        backend=str(cam.get("backend", "dshow")),
        width=int(cam.get("width", 640)),
        height=int(cam.get("height", 480)),
        fps=int(cam.get("fps", 30)),
    )

    rec = data.get("recognition", {})
    classifier = ClassifierConfig(
        pinch_enter=float(rec.get("pinch_enter", 0.30)),
        pinch_exit=float(rec.get("pinch_exit", 0.45)),
        swipe_velocity=float(rec.get("swipe_velocity", 1.2)),
    )
    fsm = FSMConfig(
        confirm_frames=int(rec.get("confirm_frames", 5)),
        cooldown_ms=int(rec.get("cooldown_ms", 400)),
        arm_hold_s=float(rec.get("arm_hold_s", 1.0)),
        disarm_hold_s=float(rec.get("arm_hold_s", 1.0)),
        arm_timeout_s=float(rec.get("arm_timeout_s", 10.0)),
        max_activation_speed=float(rec.get("max_activation_speed", 0.8)),
    )

    modes = data.get("modes", {})
    mapping = modes.get("default", {})
    if not isinstance(mapping, dict):
        raise ValueError("[modes.default] must be a table")

    return AppConfig(
        camera=camera,
        classifier=classifier,
        fsm=fsm,
        min_hand_size=float(rec.get("min_hand_size", 0.12)),
        mapping=mapping,
    )


class ConfigWatcher:
    """Debounced file watcher; calls ``on_change`` off the watchdog thread."""

    def __init__(self, path: Path, on_change: Callable[[], None],
                 debounce_s: float = 0.3) -> None:
        self._path = path.resolve()
        self._on_change = on_change
        self._debounce_s = debounce_s
        self._timer: threading.Timer | None = None
        self._observer = None

    def start(self) -> None:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        watcher = self

        class Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if Path(str(event.src_path)).resolve() == watcher._path:
                    watcher._schedule()

            on_created = on_modified

        self._observer = Observer()
        self._observer.schedule(Handler(), str(self._path.parent))
        self._observer.daemon = True
        self._observer.start()

    def _schedule(self) -> None:
        # Editors fire several events per save; act once, after the dust
        # settles.
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce_s, self._on_change)
        self._timer.daemon = True
        self._timer.start()

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        if self._observer is not None:
            self._observer.stop()
