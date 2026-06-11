"""Master volume via Windows Core Audio (pycaw) — design doc §5.1.

The IAudioEndpointVolume COM object is persistent, so each call is
sub-millisecond — what makes continuous pinch-volume viable. COM rules:
this backend must only ever be touched from the dispatcher thread, which
calls CoInitialize before creating it.

Device changes (headphones plugged in) surface as COM errors; the fix is to
drop the cached interface and re-acquire the new default endpoint.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class CoreAudioBackend:
    def __init__(self) -> None:
        self._volume = None

    def _iface(self):
        if self._volume is None:
            from pycaw.pycaw import AudioUtilities

            # Modern pycaw: GetSpeakers() returns an AudioDevice wrapper whose
            # EndpointVolume property activates IAudioEndpointVolume for us.
            self._volume = AudioUtilities.GetSpeakers().EndpointVolume
        return self._volume

    def _call(self, fn):
        """Run fn against the endpoint, re-acquiring it once on COM failure
        (default audio device changed, e.g. headphones plugged in)."""
        from comtypes import COMError  # not an OSError subclass

        try:
            return fn(self._iface())
        except (OSError, COMError):
            log.info("audio endpoint lost; re-acquiring default device")
            self._volume = None
            return fn(self._iface())

    def get_volume(self) -> float:
        return float(self._call(lambda v: v.GetMasterVolumeLevelScalar()))

    def set_volume(self, level: float) -> None:
        level = min(1.0, max(0.0, level))
        self._call(lambda v: v.SetMasterVolumeLevelScalar(level, None))

    def step_volume(self, delta: float) -> None:
        self.set_volume(self.get_volume() + delta)

    def mute_toggle(self) -> None:
        def toggle(v):
            v.SetMute(0 if v.GetMute() else 1, None)
        self._call(toggle)
