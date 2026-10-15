"""Leaky integrate-and-fire neuron with a fixed-step Euler integrator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import current_trace, time_grid


@dataclass(frozen=True)
class LIFResult:
    """Voltage trace and spike times for a leaky integrate-and-fire simulation."""

    time_ms: NDArray[np.float64]
    voltage_mv: NDArray[np.float64]
    spike_times_ms: NDArray[np.float64]


def simulate_lif(
    current_na: ArrayLike,
    *,
    duration_ms: float,
    dt_ms: float = 0.1,
    tau_m_ms: float = 20.0,
    resistance_mohm: float = 10.0,
    resting_mv: float = -65.0,
    threshold_mv: float = -50.0,
    reset_mv: float = -65.0,
    refractory_ms: float = 2.0,
) -> LIFResult:
    """Simulate a current-driven LIF neuron.

    The subthreshold equation is ``tau_m * dV/dt = -(V - V_rest) + R * I``.
    Current is in nA, resistance in MOhm, and voltage in mV. A spike is recorded
    when the threshold is crossed, then the voltage is reset immediately.
    """
    time = time_grid(duration_ms, dt_ms)
    current = current_trace(current_na, time.size, name="current_na")
    scalar_parameters = (
        tau_m_ms,
        resistance_mohm,
        resting_mv,
        threshold_mv,
        reset_mv,
        refractory_ms,
    )
    if not all(np.isfinite(value) for value in scalar_parameters):
        raise ValueError("model parameters must be finite")
    if tau_m_ms <= 0 or resistance_mohm <= 0 or refractory_ms < 0:
        raise ValueError("tau_m_ms and resistance_mohm must be positive; refractory_ms nonnegative")
    if reset_mv >= threshold_mv:
        raise ValueError("reset_mv must be below threshold_mv")

    voltage = np.empty(time.size, dtype=float)
    voltage[0] = resting_mv
    spike_times: list[float] = []
    refractory_steps = int(np.ceil(refractory_ms / dt_ms))
    next_available_step = 1

    for index in range(time.size - 1):
        if index < next_available_step:
            voltage[index + 1] = reset_mv
            continue

        derivative = (
            -(voltage[index] - resting_mv) + resistance_mohm * current[index]
        ) / tau_m_ms
        candidate = voltage[index] + dt_ms * derivative
        if candidate >= threshold_mv:
            voltage[index + 1] = reset_mv
            spike_times.append(float(time[index + 1]))
            next_available_step = index + 1 + refractory_steps
        else:
            voltage[index + 1] = candidate

    return LIFResult(
        time_ms=time,
        voltage_mv=voltage,
        spike_times_ms=np.asarray(spike_times, dtype=float),
    )


def lif_firing_rate(
    current_na: ArrayLike,
    *,
    tau_m_ms: float = 20.0,
    resistance_mohm: float = 10.0,
    resting_mv: float = -65.0,
    threshold_mv: float = -50.0,
    reset_mv: float = -65.0,
    refractory_ms: float = 2.0,
) -> NDArray[np.float64]:
    """Return the analytic LIF firing rate (spikes/s) for constant currents.

    Above rheobase, the interspike interval is
    ``refractory + tau_m * ln((R*I + V_rest - V_reset) / (R*I + V_rest - V_th))``;
    at or below rheobase the rate is zero.
    """
    current = np.asarray(current_na, dtype=float)
    steady_state = resting_mv + resistance_mohm * current
    rate = np.zeros_like(current)
    firing = steady_state > threshold_mv
    period_ms = refractory_ms + tau_m_ms * np.log(
        (steady_state[firing] - reset_mv) / (steady_state[firing] - threshold_mv)
    )
    rate[firing] = 1000.0 / period_ms
    return rate
