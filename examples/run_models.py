"""Run the roadmap's starter experiments and save their arrays under outputs/."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from computational_neuroscience import (
    current_pulse,
    fit_linear_model,
    fit_pca,
    fit_poisson_glm,
    simulate_hodgkin_huxley,
    simulate_lif,
    train_hopfield,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output_dir = Path(
        os.environ.get("COMPUTATIONAL_NEUROSCIENCE_OUTPUT_DIR", ROOT / "outputs")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    lif = simulate_lif(2.0, duration_ms=100.0)
    np.savez(
        output_dir / "lif.npz",
        time_ms=lif.time_ms,
        voltage_mv=lif.voltage_mv,
        spike_times_ms=lif.spike_times_ms,
    )
    print(f"LIF: {lif.spike_times_ms.size} spikes in 100 ms")

    hh_time = np.arange(0.0, 30.0 + 0.01, 0.01)
    hh_current = current_pulse(hh_time, 10.0, 5.0, 20.0)
    hh = simulate_hodgkin_huxley(
        hh_current, duration_ms=30.0, dt_ms=0.01
    )
    np.savez(
        output_dir / "hodgkin_huxley.npz",
        time_ms=hh.time_ms,
        voltage_mv=hh.voltage_mv,
        m=hh.m,
        h=hh.h,
        n=hh.n,
        current_ua_cm2=hh.current_ua_cm2,
    )
    print(f"Hodgkin-Huxley: peak voltage {hh.voltage_mv.max():.1f} mV")

    memory = np.array([1, -1, 1, -1, -1, 1, -1, 1], dtype=float)
    corrupted = memory.copy()
    corrupted[2] *= -1
    recall = train_hopfield(memory).recall(corrupted, seed=7)
    print(
        "Hopfield: recovered corrupted pattern = "
        f"{np.array_equal(recall.state, memory)}; energy {recall.energy_history[0]:.2f} "
        f"-> {recall.energy_history[-1]:.2f}"
    )

    stimulus = np.linspace(-1.0, 1.0, 30)
    response = 0.5 + 2.0 * stimulus
    model = fit_linear_model(stimulus[:, None], response)
    activity = np.column_stack((response, 2.0 * response, -response))
    pca = fit_pca(activity, n_components=1)
    rng = np.random.default_rng(11)
    counts = rng.poisson(np.exp(0.4 + 0.7 * stimulus))
    poisson_model = fit_poisson_glm(stimulus[:, None], counts)
    print(
        f"Analysis: linear fit R^2={model.r_squared:.3f}; "
        f"first PCA component explains {pca.explained_variance_ratio[0]:.1%}; "
        f"Poisson GLM converged={poisson_model.converged}"
    )
    print(f"Simulation arrays saved in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
