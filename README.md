# Computational Neuroscience

**English** · [Español](README.es.md)

An independent, hands-on roadmap for learning computational neuroscience. The goal is to connect biological intuition, mathematical models, and executable experiments: understand a model, implement or explore it, and use simulations to test what it explains.

The project is in progress. It currently contains two interactive, self-contained learning modules; the roadmap below is a proposed path, not a claim that every topic is already covered.

## Learning roadmap

| Stage | Topic | Focus |
|---|---|---|
| 1 | Foundations | Review the mathematics, physics, and Python needed to describe neurons and neural systems |
| 2 | Neuron models | Membrane potentials, ion channels, action potentials, and the Hodgkin–Huxley model |
| 3 | Neural dynamics | Excitability, phase planes, firing patterns, and simplified neuron models |
| 4 | Networks and memory | Recurrent networks, attractors, associative memory, and the Hopfield model |
| 5 | Learning and plasticity | Synaptic learning rules, neural plasticity, and learning in networks |
| 6 | From models to data | Analyze neural signals, compare models with observations, and document limitations |

Each stage will grow through explanations, small experiments, and code as the project develops.

## Current modules

| Module | What you can explore |
|---|---|
| [Hodgkin–Huxley: the neuron as a dynamical system](<Hodgkin–Huxley_ la neurona como sistema dinámico.html>) | Adjust stimulus and conductances; explore action potentials, channel gating, ionic currents, phase space, and firing-rate behavior |
| [Hopfield network: memory as an energy landscape](<Red de Hopfield_ memoria como paisaje de energía.html>) | Store and recall patterns; add noise; inspect network energy, overlaps, synaptic weights, update rules, and memory capacity |

## Getting started

There is no setup or build step for the current modules. Open either HTML file in a modern web browser and interact with the controls. The simulations run in the browser.

## Suggested study loop

1. Read the model explanation and identify its assumptions.
2. Make a prediction about how changing a parameter will affect the system.
3. Run an experiment and compare the result with your prediction.
4. Explain the result in your own words and note what the model leaves out.
5. When a topic calls for it, reproduce the experiment in code and keep the work reproducible.

## Project structure

```text
computational-neuroscience/
├── Hodgkin–Huxley_ la neurona como sistema dinámico.html
├── Red de Hopfield_ memoria como paisaje de energía.html
├── README.md
├── README.es.md
└── .gitignore
```

## Language

The current interactive modules are in Spanish. The repository README is available in [Spanish](README.es.md) and English.

## Scope

This is a self-study project, not a formal course or a substitute for primary literature. Models are simplified representations of biological systems; conclusions should be interpreted in light of each model's assumptions.
