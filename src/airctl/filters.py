"""One Euro filter for landmark smoothing.

Reference: Casiez, Roussel, Vogel — "1€ Filter: A Simple Speed-based Low-pass
Filter for Noisy Input in Interactive Systems" (CHI 2012).

Low lag during fast motion, strong smoothing when still — the right trade-off
for continuous controls like pinch-volume (design doc §4.2). Operates on whole
landmark arrays at once so one filter instance smooths all 21 points.
"""

from __future__ import annotations

import math

import numpy as np


def _smoothing_factor(t_e: float, cutoff: float | np.ndarray) -> float | np.ndarray:
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


class OneEuroFilter:
    """Vectorized One Euro filter over an ndarray signal (e.g. shape (21, 3))."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._t_prev: float | None = None
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def reset(self) -> None:
        """Forget filter state (call when the hand is lost/reacquired)."""
        self._t_prev = None
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._t_prev is None or self._x_prev is None:
            self._t_prev = t
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x.copy()

        t_e = t - self._t_prev
        if t_e <= 0:
            return self._x_prev.copy()

        # Derivative estimate, smoothed with a fixed cutoff.
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Signal smoothing with a speed-adaptive cutoff.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._t_prev = t
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat.copy()
