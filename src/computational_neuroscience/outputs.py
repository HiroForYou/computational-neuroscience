"""Output helpers for the weekly scripts and notebooks (requires matplotlib)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np


def in_notebook() -> bool:
    """Return True when running inside a Jupyter kernel."""
    return "ipykernel" in sys.modules


if not in_notebook() and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_DIR_VARIABLE = "COMPUTATIONAL_NEUROSCIENCE_OUTPUT_DIR"


def project_root(start: Path | None = None) -> Path:
    """Return the nearest folder that contains pyproject.toml, else the start folder."""
    current = (start or Path.cwd()).resolve()
    for folder in (current, *current.parents):
        if (folder / "pyproject.toml").is_file():
            return folder
    return current


def output_dir(week: int) -> Path:
    """Create and return the output folder for one weekly exercise.

    By default this is ``weeks/week_XX_<topic>/outputs/``, next to the week's script and
    notebook. If the environment variable is set, outputs go to ``<variable>/week_XX/``.
    """
    if OUTPUT_DIR_VARIABLE in os.environ:
        destination = Path(os.environ[OUTPUT_DIR_VARIABLE]) / f"week_{week:02d}"
    else:
        root = project_root()
        week_folders = sorted((root / "weeks").glob(f"week_{week:02d}_*/"))
        destination = (week_folders[0] / "outputs") if week_folders else root / "outputs" / f"week_{week:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _report_saved(destination: Path) -> None:
    """Print a saved path relative to the project, so notebooks do not embed local paths."""
    try:
        shown = destination.resolve().relative_to(project_root())
    except ValueError:
        shown = destination
    print(f"Saved: {shown.as_posix()}")


def save_figure(week: int, figure: plt.Figure, filename: str) -> None:
    """Save a PNG in the week's output folder; keep it open for inline display in notebooks."""
    destination = output_dir(week) / filename
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    if not in_notebook():
        plt.close(figure)
    _report_saved(destination)


def save_data(week: int, filename: str, **arrays: np.ndarray) -> None:
    """Save named arrays as a compressed NumPy archive."""
    destination = output_dir(week) / filename
    np.savez_compressed(destination, **arrays)
    _report_saved(destination)


def save_text(week: int, filename: str, text: str) -> None:
    """Save a UTF-8 text report in the week's output folder."""
    destination = output_dir(week) / filename
    destination.write_text(text, encoding="utf-8")
    _report_saved(destination)
