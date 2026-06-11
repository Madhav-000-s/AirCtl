"""Dispatcher thread (§3.1): consumes FSM events from a queue and runs
actions against the OS backends.

Decoupled from inference via the queue so a slow action (a shell command,
a COM hiccup) can never stall the latency-critical path. This thread owns
every COM object in the process — CoInitialize is called here and the audio
backend is constructed here, never shared across threads. That single rule
prevents a whole class of intermittent COM crashes.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

from .actions import Action
from .backends.base import Backends
from .fsm import (
    ArmedChanged,
    Event,
    GestureActivated,
    GestureHeld,
    GestureReleased,
)

log = logging.getLogger(__name__)

_STOP = object()


class Dispatcher(threading.Thread):
    def __init__(
        self,
        make_backends: Callable[[], Backends],
        action_map: dict[str, Action],
        on_armed_changed: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__(name="airctl-dispatch", daemon=True)
        self._make_backends = make_backends
        self._action_map = action_map
        self._on_armed_changed = on_armed_changed
        self.events: queue.Queue = queue.Queue(maxsize=256)

    def submit(self, event: Event) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:  # dispatcher wedged; dropping is the safe choice
            log.warning("event queue full, dropping %s", event)

    def stop(self) -> None:
        self.events.put(_STOP)

    def set_action_map(self, action_map: dict[str, Action]) -> None:
        """Hot-reload entry point; attribute swap is atomic in CPython."""
        self._action_map = action_map

    def run(self) -> None:
        try:
            import comtypes
            comtypes.CoInitialize()
        except ImportError:  # non-Windows test environments
            pass
        backends = self._make_backends()

        while True:
            event = self.events.get()
            if event is _STOP:
                break
            try:
                self._handle(event, backends)
            except Exception:
                log.exception("action failed for %s", event)

    def _handle(self, event: Event, backends: Backends) -> None:
        if isinstance(event, ArmedChanged):
            log.info("armed: %s", event.armed)
            if self._on_armed_changed:
                self._on_armed_changed(event.armed)
            return

        action = self._action_map.get(event.name)  # type: ignore[attr-defined]
        if action is None:
            return
        if isinstance(event, GestureActivated):
            log.info("gesture: %s", event.name)
            action.on_activate(event, backends)
        elif isinstance(event, GestureHeld):
            action.on_hold(event, backends)
        elif isinstance(event, GestureReleased):
            action.on_release(event, backends)
