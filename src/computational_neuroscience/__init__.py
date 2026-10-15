"""Small computational neuroscience models for the self-study roadmap."""

from ._validation import current_pulse
from .hodgkin_huxley import HodgkinHuxleyResult, simulate_hodgkin_huxley
from .hopfield import HopfieldNetwork, train_hopfield
from .lif import LIFResult, lif_firing_rate, simulate_lif
from .analysis import (
    LinearModel,
    PCAResult,
    PoissonGLM,
    fit_linear_model,
    fit_pca,
    fit_poisson_glm,
)

__all__ = [
    "HodgkinHuxleyResult",
    "HopfieldNetwork",
    "LinearModel",
    "LIFResult",
    "PCAResult",
    "PoissonGLM",
    "current_pulse",
    "fit_linear_model",
    "fit_pca",
    "fit_poisson_glm",
    "lif_firing_rate",
    "simulate_hodgkin_huxley",
    "simulate_lif",
    "train_hopfield",
]
