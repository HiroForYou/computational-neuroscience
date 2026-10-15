"""Shared validation for simulation inputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def time_grid(duration_ms: float, dt_ms: float) -> NDArray[np.float64]:
    """Return a time grid that includes both zero and the requested endpoint."""
    if not np.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("duration_ms must be a finite positive number")
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be a finite positive number")

    intervals = round(duration_ms / dt_ms)
    if intervals < 1 or not np.isclose(
        intervals * dt_ms, duration_ms, rtol=1e-9, atol=1e-12
    ):
        raise ValueError("duration_ms must be an integer multiple of dt_ms")
    return np.linspace(0.0, duration_ms, intervals + 1)


def current_trace(
    current: ArrayLike, sample_count: int, *, name: str
) -> NDArray[np.float64]:
    """Broadcast a scalar current or validate a current value for each sample."""
    trace = np.asarray(current, dtype=float)
    if trace.ndim == 0:
        trace = np.full(sample_count, float(trace))
    if trace.shape != (sample_count,):
        raise ValueError(f"{name} must be a scalar or have one value per time sample")
    if not np.all(np.isfinite(trace)):
        raise ValueError(f"{name} must contain only finite values")
    return trace


def current_pulse(
    times_ms: ArrayLike,
    amplitude: float,
    start_ms: float,
    stop_ms: float,
) -> NDArray[np.float64]:
    """Create a rectangular current pulse on the half-open interval [start, stop)."""
    times = np.asarray(times_ms, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("times_ms must be a one-dimensional finite array")
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    if not np.isfinite(start_ms) or not np.isfinite(stop_ms) or stop_ms <= start_ms:
        raise ValueError("pulse stop_ms must be greater than start_ms")
    return np.where((times >= start_ms) & (times < stop_ms), amplitude, 0.0)
