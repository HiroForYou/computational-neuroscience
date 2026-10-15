"""Dense binary Hopfield network with Hebbian learning and asynchronous recall."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class HopfieldNetwork:
    """A bipolar Hopfield network with a symmetric, zero-diagonal weight matrix."""

    weights: NDArray[np.float64]

    @property
    def neuron_count(self) -> int:
        return self.weights.shape[0]

    def energy(self, state: ArrayLike) -> float:
        """Return E = -0.5 * s.T @ W @ s for a bipolar network state."""
        state_array = _validate_state(state, self.neuron_count)
        return float(-0.5 * state_array @ self.weights @ state_array)

    def recall(
        self,
        initial_state: ArrayLike,
        *,
        max_sweeps: int = 100,
        seed: int = 0,
    ) -> "RecallResult":
        """Update one randomly ordered neuron at a time until a sweep is stable."""
        state = _validate_state(initial_state, self.neuron_count).copy()
        if max_sweeps < 1:
            raise ValueError("max_sweeps must be at least one")
        rng = np.random.default_rng(seed)
        energy_history = [self.energy(state)]
        converged = False

        for _ in range(max_sweeps):
            changed = False
            for neuron in rng.permutation(self.neuron_count):
                activation = float(self.weights[neuron] @ state)
                updated = 1.0 if activation > 0 else -1.0 if activation < 0 else state[neuron]
                if updated != state[neuron]:
                    state[neuron] = updated
                    changed = True
            energy_history.append(self.energy(state))
            if not changed:
                converged = True
                break

        return RecallResult(
            state=state,
            energy_history=np.asarray(energy_history, dtype=float),
            converged=converged,
        )


@dataclass(frozen=True)
class RecallResult:
    """Final recalled pattern and energy after each asynchronous sweep."""

    state: NDArray[np.float64]
    energy_history: NDArray[np.float64]
    converged: bool


def _validate_state(state: ArrayLike, neuron_count: int) -> NDArray[np.float64]:
    state_array = np.asarray(state, dtype=float)
    if state_array.shape != (neuron_count,):
        raise ValueError(f"state must have shape ({neuron_count},)")
    if not np.all(np.isin(state_array, (-1.0, 1.0))):
        raise ValueError("state values must be -1 or +1")
    return state_array


def train_hopfield(patterns: ArrayLike) -> HopfieldNetwork:
    """Learn bipolar patterns with Hebbian weights and a zero diagonal."""
    pattern_array = np.asarray(patterns, dtype=float)
    if pattern_array.ndim == 1:
        pattern_array = pattern_array[np.newaxis, :]
    if pattern_array.ndim != 2 or pattern_array.shape[1] < 2:
        raise ValueError("patterns must have shape (pattern_count, neuron_count >= 2)")
    if pattern_array.shape[0] < 1:
        raise ValueError("at least one pattern is required")
    if not np.all(np.isin(pattern_array, (-1.0, 1.0))):
        raise ValueError("pattern values must be -1 or +1")

    neuron_count = pattern_array.shape[1]
    weights = pattern_array.T @ pattern_array / neuron_count
    np.fill_diagonal(weights, 0.0)
    return HopfieldNetwork(weights=weights)
