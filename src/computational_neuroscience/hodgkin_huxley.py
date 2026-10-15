"""Classic squid-axon Hodgkin-Huxley model with a fixed-step Euler integrator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import current_trace, time_grid


@dataclass(frozen=True)
class HodgkinHuxleyResult:
    """Membrane voltage, gating variables, and applied current."""

    time_ms: NDArray[np.float64]
    voltage_mv: NDArray[np.float64]
    m: NDArray[np.float64]
    h: NDArray[np.float64]
    n: NDArray[np.float64]
    current_ua_cm2: NDArray[np.float64]


def _x_over_one_minus_exp_negative_x(x: float) -> float:
    """Evaluate x / (1 - exp(-x)) stably, including its limit at zero."""
    if abs(x) < 1e-7:
        return 1.0 + x / 2.0
    return x / -np.expm1(-x)


def _rates(voltage_mv: float) -> tuple[float, float, float, float, float, float]:
    """Return alpha/beta rates for m, h, and n at a membrane voltage."""
    alpha_m = _x_over_one_minus_exp_negative_x((voltage_mv + 40.0) / 10.0)
    alpha_n = 0.1 * _x_over_one_minus_exp_negative_x(
        (voltage_mv + 55.0) / 10.0
    )
    beta_m = 4.0 * np.exp(-(voltage_mv + 65.0) / 18.0)
    beta_h = 1.0 / (1.0 + np.exp(-(voltage_mv + 35.0) / 10.0))
    alpha_h = 0.07 * np.exp(-(voltage_mv + 65.0) / 20.0)
    beta_n = 0.125 * np.exp(-(voltage_mv + 65.0) / 80.0)
    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n


def _steady_state(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def simulate_hodgkin_huxley(
    current_ua_cm2: ArrayLike,
    *,
    duration_ms: float,
    dt_ms: float = 0.01,
    initial_voltage_mv: float = -65.0,
    capacitance_uf_cm2: float = 1.0,
    sodium_conductance_ms_cm2: float = 120.0,
    potassium_conductance_ms_cm2: float = 36.0,
    leak_conductance_ms_cm2: float = 0.3,
    sodium_reversal_mv: float = 50.0,
    potassium_reversal_mv: float = -77.0,
    leak_reversal_mv: float = -54.4,
) -> HodgkinHuxleyResult:
    """Simulate the classic Hodgkin-Huxley equations.

    Voltage is in mV, time in ms, current density in uA/cm^2, conductance
    density in mS/cm^2, and capacitance in uF/cm^2. Gating variables start at
    their steady state for ``initial_voltage_mv``.
    """
    time = time_grid(duration_ms, dt_ms)
    current = current_trace(
        current_ua_cm2, time.size, name="current_ua_cm2"
    )
    model_parameters = (
        initial_voltage_mv,
        capacitance_uf_cm2,
        sodium_conductance_ms_cm2,
        potassium_conductance_ms_cm2,
        leak_conductance_ms_cm2,
        sodium_reversal_mv,
        potassium_reversal_mv,
        leak_reversal_mv,
    )
    if not all(np.isfinite(value) for value in model_parameters):
        raise ValueError("model parameters must be finite")
    if capacitance_uf_cm2 <= 0 or min(
        sodium_conductance_ms_cm2,
        potassium_conductance_ms_cm2,
        leak_conductance_ms_cm2,
    ) < 0:
        raise ValueError("capacitance must be positive and conductances nonnegative")

    voltage = np.empty(time.size, dtype=float)
    m = np.empty(time.size, dtype=float)
    h = np.empty(time.size, dtype=float)
    n = np.empty(time.size, dtype=float)
    voltage[0] = initial_voltage_mv
    rates = _rates(initial_voltage_mv)
    m[0] = _steady_state(rates[0], rates[1])
    h[0] = _steady_state(rates[2], rates[3])
    n[0] = _steady_state(rates[4], rates[5])

    for index in range(time.size - 1):
        sodium_current = (
            sodium_conductance_ms_cm2
            * m[index] ** 3
            * h[index]
            * (voltage[index] - sodium_reversal_mv)
        )
        potassium_current = (
            potassium_conductance_ms_cm2
            * n[index] ** 4
            * (voltage[index] - potassium_reversal_mv)
        )
        leak_current = leak_conductance_ms_cm2 * (
            voltage[index] - leak_reversal_mv
        )
        voltage[index + 1] = voltage[index] + dt_ms * (
            current[index]
            - sodium_current
            - potassium_current
            - leak_current
        ) / capacitance_uf_cm2

        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = _rates(
            voltage[index]
        )
        m[index + 1] = m[index] + dt_ms * (
            alpha_m * (1.0 - m[index]) - beta_m * m[index]
        )
        h[index + 1] = h[index] + dt_ms * (
            alpha_h * (1.0 - h[index]) - beta_h * h[index]
        )
        n[index + 1] = n[index] + dt_ms * (
            alpha_n * (1.0 - n[index]) - beta_n * n[index]
        )

    return HodgkinHuxleyResult(
        time_ms=time,
        voltage_mv=voltage,
        m=m,
        h=h,
        n=n,
        current_ua_cm2=current,
    )
