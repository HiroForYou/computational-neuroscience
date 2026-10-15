"""Small analysis tools for model fitting and neural population activity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class LinearModel:
    """Least-squares linear model with an explicit intercept."""

    coefficients: NDArray[np.float64]
    intercept: float
    r_squared: float

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        feature_array = _feature_matrix(features)
        if feature_array.shape[1] != self.coefficients.size:
            raise ValueError("feature count does not match fitted model")
        return feature_array @ self.coefficients + self.intercept


@dataclass(frozen=True)
class PCAResult:
    """Centered principal components and their explained-variance ratios."""

    mean: NDArray[np.float64]
    components: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]

    def transform(self, observations: ArrayLike) -> NDArray[np.float64]:
        observation_array = _feature_matrix(observations)
        if observation_array.shape[1] != self.mean.size:
            raise ValueError("feature count does not match fitted PCA")
        return (observation_array - self.mean) @ self.components.T


@dataclass(frozen=True)
class PoissonGLM:
    """Poisson generalized linear model with a log link and explicit intercept."""

    coefficients: NDArray[np.float64]
    intercept: float
    converged: bool
    iterations: int

    def predict_rate(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return the expected count for each feature row."""
        feature_array = _feature_matrix(features)
        if feature_array.shape[1] != self.coefficients.size:
            raise ValueError("feature count does not match fitted model")
        linear_predictor = feature_array @ self.coefficients + self.intercept
        return np.exp(np.clip(linear_predictor, -30.0, 30.0))


def _feature_matrix(features: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix


def fit_linear_model(features: ArrayLike, targets: ArrayLike) -> LinearModel:
    """Fit ordinary least squares and report the in-sample coefficient of determination."""
    feature_array = _feature_matrix(features)
    target_array = np.asarray(targets, dtype=float)
    if target_array.ndim != 1 or target_array.shape[0] != feature_array.shape[0]:
        raise ValueError("targets must be one-dimensional with one value per sample")
    if not np.all(np.isfinite(target_array)):
        raise ValueError("targets must contain only finite values")

    design = np.column_stack((np.ones(feature_array.shape[0]), feature_array))
    parameters, _, _, _ = np.linalg.lstsq(design, target_array, rcond=None)
    prediction = design @ parameters
    residual_sum = float(np.sum((target_array - prediction) ** 2))
    total_sum = float(np.sum((target_array - target_array.mean()) ** 2))
    r_squared = 1.0 if total_sum == 0 and residual_sum == 0 else (
        0.0 if total_sum == 0 else 1.0 - residual_sum / total_sum
    )
    return LinearModel(
        coefficients=parameters[1:],
        intercept=float(parameters[0]),
        r_squared=r_squared,
    )


def fit_pca(observations: ArrayLike, n_components: int) -> PCAResult:
    """Fit PCA by SVD to rows-as-observations, without feature scaling."""
    observation_array = _feature_matrix(observations)
    maximum_components = min(
        observation_array.shape[0], observation_array.shape[1]
    )
    if not 1 <= n_components <= maximum_components:
        raise ValueError(f"n_components must be between 1 and {maximum_components}")

    mean = observation_array.mean(axis=0)
    centered = observation_array - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    total_variance = float(variance.sum())
    ratios = (
        variance[:n_components] / total_variance
        if total_variance > 0
        else np.zeros(n_components, dtype=float)
    )
    return PCAResult(
        mean=mean,
        components=right_vectors[:n_components],
        explained_variance_ratio=ratios,
    )


def fit_poisson_glm(
    features: ArrayLike,
    counts: ArrayLike,
    *,
    l2_penalty: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> PoissonGLM:
    """Fit a Poisson GLM with log link using damped Newton/IRLS updates.

    ``counts`` must contain nonnegative integer observations. ``l2_penalty``
    regularizes feature coefficients but not the intercept.
    """
    feature_array = _feature_matrix(features)
    count_array = np.asarray(counts, dtype=float)
    if count_array.ndim != 1 or count_array.size != feature_array.shape[0]:
        raise ValueError("counts must be one-dimensional with one value per sample")
    if (
        not np.all(np.isfinite(count_array))
        or np.any(count_array < 0)
        or not np.all(count_array == np.floor(count_array))
    ):
        raise ValueError("counts must be finite nonnegative integers")
    if not np.isfinite(l2_penalty) or l2_penalty < 0:
        raise ValueError("l2_penalty must be finite and nonnegative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    design = np.column_stack((np.ones(feature_array.shape[0]), feature_array))
    penalty = np.diag(np.r_[0.0, np.full(feature_array.shape[1], l2_penalty)])
    parameters = np.zeros(design.shape[1], dtype=float)
    parameters[0] = np.log(max(float(count_array.mean()), 1e-6))

    def log_likelihood(candidate: NDArray[np.float64]) -> float:
        eta = np.clip(design @ candidate, -30.0, 30.0)
        return float(
            np.sum(count_array * eta - np.exp(eta))
            - 0.5 * candidate @ penalty @ candidate
        )

    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        eta = np.clip(design @ parameters, -30.0, 30.0)
        rate = np.exp(eta)
        gradient = design.T @ (count_array - rate) - penalty @ parameters
        hessian = design.T @ (rate[:, None] * design) + penalty
        step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        current_likelihood = log_likelihood(parameters)
        scale = 1.0
        candidate = parameters + step
        while scale > 1e-8 and log_likelihood(candidate) < current_likelihood:
            scale *= 0.5
            candidate = parameters + scale * step
        if log_likelihood(candidate) < current_likelihood:
            break
        parameters = candidate
        if np.linalg.norm(scale * step) <= tolerance * (
            1.0 + np.linalg.norm(parameters)
        ):
            converged = True
            break

    return PoissonGLM(
        coefficients=parameters[1:],
        intercept=float(parameters[0]),
        converged=converged,
        iterations=iterations,
    )
