"""Camera capture thread with a 1-slot latest-wins frame buffer (§3.1).

The capture loop never blocks on downstream consumers: it overwrites the
single buffer slot every frame, and the inference thread always reads the
freshest frame. A queue here would accumulate backlog whenever inference
hiccups and latency would grow without bound — latest-wins is non-negotiable
per the design doc.

Pause support releases the camera entirely (the webcam LED turns off), which
is the privacy behavior a tray "pause" toggle should have.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

CAMERA_PRIVACY_HINT = (
    "Could not open the camera. If it is connected, check Windows Settings -> "
    "Privacy & security -> Camera and allow desktop apps to access it."
)


@dataclass
class CameraConfig:
    device: int = 0
    backend: str = "dshow"   # "dshow" (recommended on Windows) or "msmf"
    width: int = 640
    height: int = 480
    fps: int = 30


class FrameSlot:
    """Single-slot, latest-wins frame exchange between two threads."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._t: float = 0.0
        self._seq = 0

    def put(self, frame: np.ndarray, t: float) -> None:
        with self._cond:
            self._frame = frame
            self._t = t
            self._seq += 1
            self._cond.notify_all()

    def get(self, last_seq: int, timeout: float = 0.5):
        """Block until a frame newer than ``last_seq`` arrives.

        Returns (frame, t, seq) or None on timeout.
        """
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            if self._seq <= last_seq or self._frame is None:
                return None
            return self._frame, self._t, self._seq


class CaptureThread(threading.Thread):
    def __init__(self, cfg: CameraConfig, slot: FrameSlot) -> None:
        super().__init__(name="airctl-capture", daemon=True)
        self.cfg = cfg
        self.slot = slot
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self.open_failed = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()  # unblock a paused loop so it can exit

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._resume.clear()
        else:
            self._resume.set()

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    def _open(self) -> cv2.VideoCapture | None:
        api = cv2.CAP_DSHOW if self.cfg.backend == "dshow" else cv2.CAP_MSMF
        cap = cv2.VideoCapture(self.cfg.device, api)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        return cap

    def run(self) -> None:
        cap: cv2.VideoCapture | None = None
        try:
            while not self._stop.is_set():
                if not self._resume.is_set():
                    if cap is not None:
                        cap.release()  # turn the camera (and its LED) off
                        cap = None
                        log.info("capture paused, camera released")
                    self._resume.wait(timeout=0.5)
                    continue

                if cap is None:
                    cap = self._open()
                    if cap is None:
                        self.open_failed.set()
                        log.error(CAMERA_PRIVACY_HINT)
                        # Retry occasionally; the user may grant permission
                        # or plug a camera in without restarting the app.
                        if self._stop.wait(timeout=5.0):
                            break
                        continue
                    self.open_failed.clear()
                    log.info("camera %d open (%s)", self.cfg.device,
                             self.cfg.backend)

                ok, frame = cap.read()
                if not ok:
                    log.warning("frame read failed; reopening camera")
                    cap.release()
                    cap = None
                    time.sleep(0.5)
                    continue

                # Mirror to selfie view so moving your hand right moves it
                # right on screen; everything downstream assumes this.
                self.slot.put(cv2.flip(frame, 1), time.monotonic())
        finally:
            if cap is not None:
                cap.release()
