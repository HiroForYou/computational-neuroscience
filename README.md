# Computational Neuroscience

**English** · [Español](README.es.md)

An independent, hands-on roadmap for learning computational neuroscience. The goal is to connect biological intuition, mathematical models, and executable experiments: understand a model, implement or explore it, and use simulations to test what it explains.

The project includes interactive browser modules and a small Python package with runnable examples and tests. It is still in progress: the roadmap is a proposed learning path, not a claim that every topic is already covered.

## Learning roadmap

Suggested pace: **18 fully planned weeks at 5–8 hours per week** (about 90–144 hours). This is a self-study estimate, not an official course duration. Each week's objectives, practice, and deliverable are in the [week-by-week roadmap](docs/ruta-de-aprendizaje.md).

| Weeks | Topic |
|---|---|
| 1–6 | Python and math, LIF, Hodgkin–Huxley, numerical integration, synapses, and linear systems |
| 7–11 | Dynamical systems, population models, Hopfield networks, stochastic processes, and experiment design |
| 12–16 | Model fitting, Poisson GLMs, encoding/decoding, cross-validation, PCA, and signal processing |
| 17–18 | Specialization and a reproducible capstone |

The full [self-study roadmap](docs/ruta-de-aprendizaje.md) provides the study tasks, practice, and a deliverable for every week, plus checkpoints and official Stanford and Neuromatch Academy resources.

## What is implemented

| Layer | Contents |
|---|---|
| Python package (`src/`) | LIF (with analytic f-I curve), Hodgkin–Huxley, Hopfield, linear fit, Poisson GLM, PCA, `current_pulse` helper |
| Weeks (`weeks/week_XX_<topic>/`) | One folder per week (1–18) with the script (theory, equations, figure, `assert`-based validation), its executed notebook, and its `outputs/`; synaptic filtering, Wilson–Cowan, Poisson spikes, cross-validation, signal processing, and Bayesian decision live only here |
| Interactive modules (`interactive/`) | Eight browser modules: LIF, Hodgkin–Huxley, synapses, 2D linear systems, Wilson–Cowan, Hopfield, Poisson spike trains, and Bayesian decisions |
| Not covered yet | Synaptic plasticity/STDP, reinforcement learning, real experimental datasets |

## Current modules

**Try them online:** [hiroforyou.github.io/computational-neuroscience/interactive/](https://hiroforyou.github.io/computational-neuroscience/interactive/) — every module runs in the browser, published with GitHub Pages from `interactive/`.

| Module | What you can explore |
|---|---|
| [LIF neuron: integrate, fire, and reset](https://hiroforyou.github.io/computational-neuroscience/interactive/lif.html) | Change current, noise, and membrane parameters; compare simulated and analytic f–I curves; see rheobase, refractory saturation, and noise-driven irregular firing (ISI histogram, CV) |
| [Hodgkin–Huxley: the neuron as a dynamical system](https://hiroforyou.github.io/computational-neuroscience/interactive/hodgkin-huxley.html) | Adjust stimulus and conductances; explore action potentials, channel gating, ionic currents, phase space, and firing-rate behavior |
| [Synapses: temporal summation and coincidence detection](https://hiroforyou.github.io/computational-neuroscience/interactive/sinapsis.html) | Vary synaptic and membrane time constants, burst interval, and weight; compare the simulated peak with the geometric-series formula; see when summation makes a postsynaptic LIF fire |
| [2D linear systems: fixed points and stability](https://hiroforyou.github.io/computational-neuroscience/interactive/sistemas-lineales.html) | Edit a 2×2 matrix or click the trace–determinant plane; launch trajectories in the phase plane; see eigenvalues, eigenvectors, nullclines, and the classification (focus, node, saddle, center) |
| [Wilson–Cowan: excitation, inhibition, and oscillations](https://hiroforyou.github.io/computational-neuroscience/interactive/wilson-cowan.html) | Tune input and E/I connections; find and classify fixed points on the nullclines; explore low activity, limit-cycle oscillations, bistability, and the bifurcation diagram |
| [Hopfield network: memory as an energy landscape](https://hiroforyou.github.io/computational-neuroscience/interactive/hopfield.html) | Store and recall patterns; add noise; inspect network energy, overlaps, synaptic weights, update rules, and memory capacity |
| [Poisson spike trains: randomness and variability](https://hiroforyou.github.io/computational-neuroscience/interactive/poisson.html) | Change rate, window, repetitions, and refractory period; compare count and ISI histograms with Poisson theory; Fano factor, CV, and how the rate estimate converges with more trials |
| [Bayesian decision: perceiving with noise](https://hiroforyou.github.io/computational-neuroscience/interactive/decision-bayesiana.html) | Change noise, separation, and prior; drag the decision boundary; see error areas, the posterior, and simulated versus theoretical accuracy (Φ(d′/2)) |

### Screenshots

Initial state of each module (light theme). Click an image to open the module.

<table>
  <tr>
    <td width="50%" align="center">
      <a href="https://hiroforyou.github.io/computational-neuroscience/interactive/lif.html"><img src="docs/images/interactive-lif.webp" alt="LIF neuron" width="100%"></a><br>
      <sub><b>LIF neuron</b></sub>
    </td>
    <td width="50%" align="center">
      <a href="https://hiroforyou.github.io/computational-neuroscience/interactive/hodgkin-huxley.html"><img src="docs/images/interactive-hodgkin-huxley.webp" alt="Hodgkin–Huxley" width="100%"></a><br>
      <sub><b>Hodgkin–Huxley</b></sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <a href="https://hiroforyou.github.io/computational-neuroscience/interactive/sinapsis.html"><img src="docs/images/interactive-sinapsis.webp" alt="Synapses" width="100%"></a><br>
      <sub><b>Synapses</b></sub>
    </td>
    <td width="50%" align="center">
      <a href="https://hiroforyou.github.io/computational-neuroscience/interactive/sistemas-lineales.html"><img src="docs/images/interactive-sistemas-lineales.webp" alt="2D linear systems" width="100%"></a><br>
      <sub><b>2D linear systems</b></sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <a href="https://hiroforyou.github.io/computational-neuroscience/interactive/wilson-cowan.html"><img src="docs/images/interactive-wilson-cowan.webp" alt="Wilson–Cowan" width="100%"></a><br>
      <sub><b>Wilson–Cowan</b></sub>
    </td>
    <td width="50%" align="center">
      <a href="https://hiroforyou.github.io/computational-neuroscience/interactive/hopfield.html"><img src="docs/images/interactive-hopfield.webp" alt="Hopfield network" width="100%"></a><br>
      <sub><b>Hopfield network</b></sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <a href="https://hiroforyou.github.io/computational-neuroscience/interactive/poisson.html"><img src="docs/images/interactive-poisson.webp" alt="Poisson spike trains" width="100%"></a><br>
      <sub><b>Poisson spike trains</b></sub>
    </td>
    <td width="50%" align="center">
      <a href="https://hiroforyou.github.io/computational-neuroscience/interactive/decision-bayesiana.html"><img src="docs/images/interactive-decision-bayesiana.webp" alt="Bayesian decision" width="100%"></a><br>
      <sub><b>Bayesian decision</b></sub>
    </td>
  </tr>
</table>

## Getting started

The interactive browser modules need no installation: use the [online version](https://hiroforyou.github.io/computational-neuroscience/interactive/) or open any HTML file in `interactive/` in a modern browser.

To run the Python experiments (Python 3.10+):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
python examples/run_models.py
```

`run_models.py` saves LIF and Hodgkin–Huxley traces as NumPy `.npz` files in the ignored `outputs/` folder, and prints Hopfield recall and basic analysis results. Install only `.[examples]` if you do not need the test suite.

Run the script for any roadmap week with:

```powershell
python weeks/run_weekly.py --list
python weeks/run_weekly.py 1
```

Each week saves its figures/data/report in its own `weeks/week_XX_<topic>/outputs/` folder, which Git ignores (set `COMPUTATIONAL_NEUROSCIENCE_OUTPUT_DIR` to write them elsewhere). The [week-by-week roadmap](docs/ruta-de-aprendizaje.md) maps each script to its learning goal.

### Notebooks

Each week folder contains the script and its notebook side by side:

```text
weeks/week_03_lif_currents/
├── week_03_lif_currents.py      # source: edit here
├── week_03_lif_currents.ipynb   # generated and executed (theory, equations, figures)
└── outputs/                     # .png, .npz, reports (ignored by Git)
```

Open the notebook (VS Code, JupyterLab, or GitHub) to read the theory and see the figures without running anything. The notebook is generated from the script's `# %%` cells:

```powershell
python tools/convert_to_notebooks.py --execute            # regenerate all 18, with figures
python tools/convert_from_notebooks.py weeks/week_03_lif_currents/week_03_lif_currents.ipynb   # bring notebook edits back to the script
```

Edit the script (or edit the notebook and convert it back), then regenerate. A test fails if a notebook is out of date.

## Study loop

1. Read the model explanation and identify its assumptions.
2. Predict how changing a parameter will affect the system.
3. Run the experiment and compare the result with your prediction.
4. Explain the result in your own words and note what the model leaves out.
5. Record code, parameters, units, and limitations so the result can be reproduced.

## Project structure

```text
computational-neuroscience/
├── interactive/                     # Browser modules (no installation)
│   ├── index.html                   # Landing page for GitHub Pages
│   ├── lif.html                     # Week 3: LIF neuron and f–I curve
│   ├── hodgkin-huxley.html          # Weeks 4–5
│   ├── sinapsis.html                # Week 6: temporal summation
│   ├── sistemas-lineales.html       # Week 7: fixed points and stability
│   ├── wilson-cowan.html            # Week 8: E–I populations
│   ├── hopfield.html                # Week 9
│   ├── poisson.html                 # Week 10: spike-train variability
│   └── decision-bayesiana.html      # Week 17: Bayesian decision
├── src/computational_neuroscience/  # Python package
│   ├── lif.py                       # Leaky integrate-and-fire and analytic f-I curve
│   ├── hodgkin_huxley.py            # Hodgkin–Huxley (explicit Euler)
│   ├── hopfield.py                  # Hebbian training and asynchronous recall
│   ├── analysis.py                  # Linear fit, Poisson GLM, PCA
│   ├── outputs.py                   # Save figures/data for scripts and notebooks (matplotlib)
│   └── _validation.py               # Time grids, current traces, current_pulse
├── examples/
│   └── run_models.py                # Starter experiments with every model
├── weeks/
│   ├── run_weekly.py                # Runner: python weeks/run_weekly.py <week>
│   ├── week_01_python_reproducibility/  # script .py + notebook .ipynb + outputs/
│   ├── …
│   └── week_18_capstone/
├── tools/                           # convert_to_notebooks.py / convert_from_notebooks.py
├── tests/
│   ├── test_models.py               # Numerical behavior and input validation
│   └── test_weekly_scripts.py       # Every weekly script runs/validates; notebooks are in sync
├── docs/
│   ├── ruta-de-aprendizaje.md       # Week-by-week roadmap and course links
│   └── images/                      # README screenshots of the interactive modules
├── .github/workflows/pages.yml      # Publishes interactive/ to GitHub Pages
├── pyproject.toml
├── README.md / README.es.md
└── .gitignore
```

## Language

The current interactive modules and detailed roadmap are in Spanish. The repository README is available in [Spanish](README.es.md) and English.

## Scope

This is a self-study project, not a formal course or a substitute for primary literature. Models are simplified representations of biological systems; conclusions should be interpreted in light of each model's assumptions.
