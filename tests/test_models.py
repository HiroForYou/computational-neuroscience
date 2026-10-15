import numpy as np
import pytest

from computational_neuroscience import current_pulse, lif_firing_rate, simulate_hodgkin_huxley, simulate_lif
from computational_neuroscience.analysis import (
    fit_linear_model,
    fit_pca,
    fit_poisson_glm,
)
from computational_neuroscience.hopfield import train_hopfield


def test_current_pulse_uses_half_open_interval() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(current_pulse(times, 4.0, 1.0, 3.0), [0.0, 4.0, 4.0, 0.0])


def test_lif_is_quiet_without_input_and_fires_with_suprathreshold_input() -> None:
    quiet = simulate_lif(0.0, duration_ms=100.0, dt_ms=0.1)
    driven = simulate_lif(2.0, duration_ms=100.0, dt_ms=0.1)

    assert quiet.spike_times_ms.size == 0
    assert driven.spike_times_ms.size > 0
    assert np.all(driven.voltage_mv < -50.0)
    assert driven.time_ms[-1] == pytest.approx(100.0)


def test_lif_rejects_invalid_time_grid() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        simulate_lif(0.0, duration_ms=1.0, dt_ms=0.3)


def test_hodgkin_huxley_stays_finite_and_gates_remain_bounded() -> None:
    time = np.arange(0.0, 30.0 + 0.01, 0.01)
    current = current_pulse(time, 10.0, 5.0, 20.0)
    result = simulate_hodgkin_huxley(current, duration_ms=30.0, dt_ms=0.01)

    assert np.all(np.isfinite(result.voltage_mv))
    assert np.max(result.voltage_mv) > 0.0
    for gate in (result.m, result.h, result.n):
        assert np.all((gate >= 0.0) & (gate <= 1.0))


def test_hodgkin_huxley_voltage_dependent_rates_are_finite_at_singularities() -> None:
    for voltage in (-55.0, -40.0):
        result = simulate_hodgkin_huxley(
            0.0, duration_ms=0.1, dt_ms=0.01, initial_voltage_mv=voltage
        )
        assert np.all(np.isfinite(result.voltage_mv))


def test_hopfield_recovers_noiseless_stored_pattern_and_energy_does_not_increase() -> None:
    pattern = np.array([1, -1, 1, -1, -1, 1, -1, 1])
    network = train_hopfield(pattern)
    recall = network.recall(pattern, seed=7)

    assert np.array_equal(recall.state, pattern)
    assert np.all(np.diff(recall.energy_history) <= 1e-12)
    assert recall.converged


def test_hopfield_recovers_a_corrupted_pattern() -> None:
    pattern = np.array([1, -1, 1, -1, -1, 1, -1, 1])
    noisy = pattern.copy()
    noisy[2] *= -1
    recall = train_hopfield(pattern).recall(noisy, seed=3)

    assert np.array_equal(recall.state, pattern)


def test_hopfield_rejects_non_bipolar_patterns() -> None:
    with pytest.raises(ValueError, match=r"-1 or \+1"):
        train_hopfield([[0, 1, -1]])


def test_linear_model_recovers_a_known_relationship() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    targets = 2.0 + 3.0 * features[:, 0]
    model = fit_linear_model(features, targets)

    assert model.intercept == pytest.approx(2.0)
    assert model.coefficients[0] == pytest.approx(3.0)
    assert model.r_squared == pytest.approx(1.0)
    assert np.allclose(model.predict(features), targets)


def test_pca_captures_a_rank_one_population_signal() -> None:
    latent = np.linspace(-2.0, 2.0, 20)
    observations = np.column_stack((latent, 2.0 * latent, -latent))
    pca = fit_pca(observations, n_components=1)

    assert pca.explained_variance_ratio[0] == pytest.approx(1.0)
    assert pca.transform(observations).shape == (20, 1)


def test_poisson_glm_fits_a_count_response() -> None:
    rng = np.random.default_rng(11)
    features = np.linspace(-1.0, 1.0, 500)[:, None]
    true_rate = np.exp(0.4 + 0.7 * features[:, 0])
    counts = rng.poisson(true_rate)
    model = fit_poisson_glm(features, counts)

    assert model.converged
    assert model.intercept == pytest.approx(0.4, abs=0.15)
    assert model.coefficients[0] == pytest.approx(0.7, abs=0.2)
    assert np.all(model.predict_rate(features) > 0.0)


def test_poisson_glm_rejects_noninteger_counts() -> None:
    with pytest.raises(ValueError, match="nonnegative integers"):
        fit_poisson_glm([[0.0], [1.0]], [0.0, 0.5])


def test_lif_firing_rate_matches_simulation_and_rheobase() -> None:
    assert lif_firing_rate(1.5) == 0.0
    simulated = simulate_lif(2.5, duration_ms=1000.0, dt_ms=0.01).spike_times_ms
    simulated_rate = 1000.0 / np.mean(np.diff(simulated))
    assert simulated_rate == pytest.approx(float(lif_firing_rate(2.5)), rel=0.01)
