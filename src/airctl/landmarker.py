"""MediaPipe Tasks HandLandmarker wrapper (VIDEO running mode).

Uses the modern Tasks API (not legacy ``mp.solutions``). The .task model file
is downloaded once into %APPDATA%\\airctl\\models on first run.

Also applies the background-hand gate from design-doc open question #4: a
detected hand whose bounding box is tiny is probably someone walking past
behind you, not the user — ignore it.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np

from .paths import models_dir

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"


def ensure_model() -> Path:
    """Return the local model path, downloading it on first run."""
    path = models_dir() / MODEL_FILENAME
    if path.exists() and path.stat().st_size > 0:
        return path
    log.info("downloading hand landmark model (one-time, ~8 MB) ...")
    print(f"AirCtl: downloading hand landmark model to {path} ...", flush=True)
    tmp = path.with_suffix(".download")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the MediaPipe hand model ({e}). "
            f"If you are offline, download it manually from\n  {MODEL_URL}\n"
            f"and save it as\n  {path}"
        ) from e
    return path


class HandLandmarker:
    """Per-frame landmark extraction. Owns the MediaPipe graph.

    Must be created and called from a single thread (the inference thread).
    """

    def __init__(self, min_hand_size: float = 0.12, num_hands: int = 1) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self._mp = mp
        model_path = ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.min_hand_size = min_hand_size
        self._last_ts_ms = -1

    def close(self) -> None:
        self._landmarker.close()

    def detect(self, bgr_frame: np.ndarray, t: float) -> tuple[np.ndarray, str] | None:
        """Return (landmarks (21,3), handedness) for the first valid hand.

        ``t`` is a monotonic timestamp in seconds; VIDEO mode requires
        strictly increasing integer-millisecond timestamps.
        """
        ts_ms = int(t * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)

        for lm_list, handed in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float64)
            # Background-hand gate: bounding-box diagonal in image units.
            span = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
            if float(np.hypot(*span)) < self.min_hand_size:
                continue
            label = handed[0].category_name if handed else "Right"
            return pts, label
        return None
