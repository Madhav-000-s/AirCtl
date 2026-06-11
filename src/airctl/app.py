"""Application wiring: threads, config hot-reload, lifecycle (§3.1).

Threads:
  capture    camera -> 1-slot latest-wins buffer
  inference  landmarker -> filter -> features -> classifier -> FSM (latency path)
  dispatch   action queue -> OS backends (owns all COM objects)
  main       pystray tray icon (blocking run loop)
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from pathlib import Path

from .actions import build_action_map
from .backends.audio import CoreAudioBackend
from .backends.base import Backends
from .backends.custom import ShellBackend
from .backends.input import SendInputBackend
from .backends.media import MediaKeyBackend
from .backends.window import Win32WindowBackend
from .capture import CAMERA_PRIVACY_HINT, CaptureThread, FrameSlot
from .classifier import PoseClassifier
from .config import ConfigWatcher, ensure_user_config, load_config
from .dispatcher import Dispatcher
from .features import extract_features
from .filters import OneEuroFilter
from .fsm import GestureFSM
from .landmarker import HandLandmarker

log = logging.getLogger(__name__)

# Adaptive power saving (§6.3.7): with no hand in view, sample at a low
# "sentinel" rate instead of full fps.
IDLE_AFTER_S = 5.0
IDLE_FPS = 5.0


def make_real_backends() -> Backends:
    input_backend = SendInputBackend()
    return Backends(
        audio=CoreAudioBackend(),
        input=input_backend,
        window=Win32WindowBackend(input_backend),
        media=MediaKeyBackend(input_backend),
        custom=ShellBackend(),
    )


class App:
    def __init__(
        self,
        config_path: Path | None = None,
        preview: bool = False,
        debug_poses: bool = False,
        use_tray: bool = True,
    ) -> None:
        self.config_path = config_path or ensure_user_config()
        self.cfg = load_config(self.config_path)
        action_map = build_action_map(self.cfg.mapping)  # fail fast on typos

        self.preview = preview
        self.debug_poses = debug_poses
        self._stop = threading.Event()

        self.slot = FrameSlot()
        self.capture = CaptureThread(self.cfg.camera, self.slot)
        self.dispatcher = Dispatcher(
            make_backends=make_real_backends,
            action_map=action_map,
            on_armed_changed=self._on_armed_changed,
        )
        self._inference = threading.Thread(
            target=self._inference_loop, name="airctl-inference", daemon=True)
        self._watcher = ConfigWatcher(self.config_path, self._reload_config)

        # Created on the inference thread; kept for config hot-swap.
        self._classifier: PoseClassifier | None = None
        self._fsm: GestureFSM | None = None

        self.tray = None
        if use_tray:
            from .tray import Tray
            self.tray = Tray(on_pause_toggle=self.capture.set_paused,
                             on_quit=self.stop)

    # -- lifecycle -----------------------------------------------------------

    def run(self) -> None:
        log.info("AirCtl starting (config: %s)", self.config_path)
        self.capture.start()
        self.dispatcher.start()
        self._inference.start()
        self._watcher.start()
        try:
            if self.tray is not None:
                self.tray.run()  # blocks until Quit
            else:
                while not self._stop.wait(timeout=0.5):
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._watcher.stop()
        self.capture.stop()
        self.dispatcher.stop()
        if self.tray is not None:
            self.tray.stop()
        self.capture.join(timeout=3)
        self.dispatcher.join(timeout=3)
        log.info("AirCtl stopped")

    # -- config hot-reload (watchdog thread) ----------------------------------

    def _reload_config(self) -> None:
        try:
            cfg = load_config(self.config_path)
            action_map = build_action_map(cfg.mapping)
        except Exception as e:
            log.error("config reload rejected, keeping previous config: %s", e)
            return
        self.cfg = cfg
        self.dispatcher.set_action_map(action_map)
        if self._classifier is not None:
            self._classifier.cfg = cfg.classifier
        if self._fsm is not None:
            self._fsm.cfg = cfg.fsm
        log.info("config reloaded (camera changes need a restart)")

    def _on_armed_changed(self, armed: bool) -> None:
        if self.tray is not None:
            self.tray.set_armed(armed)

    # -- inference thread ------------------------------------------------------

    def _inference_loop(self) -> None:
        try:
            landmarker = HandLandmarker(min_hand_size=self.cfg.min_hand_size)
        except Exception:
            log.exception("failed to initialize hand landmarker")
            self.stop()
            return

        smoother = OneEuroFilter(min_cutoff=1.0, beta=0.05)
        self._classifier = PoseClassifier(self.cfg.classifier)
        self._fsm = GestureFSM(self.cfg.fsm)

        last_seq = 0
        last_hand_t = time.monotonic()
        last_processed = 0.0
        infer_times: deque[float] = deque(maxlen=300)
        last_stats_t = time.monotonic()
        last_debug_line = ""

        while not self._stop.is_set():
            got = self.slot.get(last_seq, timeout=0.5)
            if got is None:
                if self.capture.open_failed.is_set():
                    time.sleep(0.5)
                continue
            frame, t, last_seq = got

            # Sentinel sampling while idle: no hand for a while -> ~5 fps.
            if (t - last_hand_t > IDLE_AFTER_S
                    and t - last_processed < 1.0 / IDLE_FPS):
                continue
            last_processed = t

            t0 = time.perf_counter()
            detection = landmarker.detect(frame, t)
            infer_times.append((time.perf_counter() - t0) * 1000)

            if detection is not None:
                raw_landmarks, handedness = detection
                landmarks = smoother(t, raw_landmarks)
                features = extract_features(landmarks, handedness)
                last_hand_t = t
            else:
                smoother.reset()
                features = None

            cf = self._classifier.classify(t, features)
            for event in self._fsm.update(cf):
                self.dispatcher.submit(event)

            if self.debug_poses:
                line = self._debug_line(cf)
                if line != last_debug_line:
                    print(line, flush=True)
                    last_debug_line = line
                if time.monotonic() - last_stats_t > 2.0 and infer_times:
                    qs = statistics.quantiles(infer_times, n=20)
                    print(f"  [latency] inference p50={qs[9]:.1f}ms "
                          f"p95={qs[18]:.1f}ms over {len(infer_times)} frames",
                          flush=True)
                    last_stats_t = time.monotonic()

            if self.preview:
                self._draw_preview(frame, cf)

        landmarker.close()

    def _debug_line(self, cf) -> str:
        armed = "ARMED" if self._fsm and self._fsm.armed else "disarmed"
        if cf.features is None:
            return f"[{armed}] no hand"
        bits = "".join("1" if s else "0" for s in cf.features.finger_states)
        parts = [f"[{armed}] pose={cf.pose or '-':<10} fingers={bits}",
                 f"pinch={cf.features.pinch_index:.2f}",
                 f"palm={'cam' if cf.features.palm_facing else 'away'}",
                 f"speed={cf.speed:.2f}"]
        if cf.swipe:
            parts.append(f"SWIPE={cf.swipe}")
        return " ".join(parts)

    def _draw_preview(self, frame, cf) -> None:
        import cv2

        h, w = frame.shape[:2]
        if cf.features is not None:
            for x, y, _ in cf.features.landmarks:
                cv2.circle(frame, (int(x * w), int(y * h)), 3,
                           (0, 255, 0), -1)
        armed = self._fsm is not None and self._fsm.armed
        status = f"{'ARMED' if armed else 'disarmed'}  {cf.pose or ''}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if armed else (0, 165, 255), 2)
        cv2.imshow("AirCtl preview (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop()


__all__ = ["App", "make_real_backends", "CAMERA_PRIVACY_HINT"]
