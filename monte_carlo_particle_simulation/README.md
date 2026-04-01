# Monte Carlo Simulation of Particle Transport and Decay

This project implements a high-performance **Monte Carlo simulation framework** for modeling stochastic particle motion (random walk/diffusion) and probabilistic particle decay. The underlying logic uses highly vectorized NumPy arrays rather than Object-Oriented paradigms for individual particles, allowing simulation of massive sets (10k+ particles) at rapid speeds.

## Problem Statement

In particle physics, radiation transport, and detector science (such as research conducted at CERN), analytical solutions to transport equations are often impossible to solve directly due to complex geometries or intertwined stochastic processes (e.g., scattering and spontaneous radioactive decay). **Monte Carlo simulation** solves this by simulating the probabilistic interactions of millions of individual particles, allowing the bulk macroscopic properties (mean free paths, energy deposition maps) to emerge through the Law of Large Numbers. 

This project distills that capability down to a pure Python framework, showing how such problems are approached computationally.

## Methodology

### 1. Particle Transport (Random Walk)
Particles are propagated in continuous space ($2D$ or $3D$) using a random walk driven by a Wiener process. At each timestep $\Delta t$, the positional displacement follows a normal distribution:
$$ \Delta x \sim \mathcal{N}(0, \sigma^2 \Delta t) $$
This mimics perfect diffusion. Boundary conditions can be configured as **reflective** (e.g., a contained gas chamber) or **absorbing** (e.g., a detector wall or biological shielding).

### 2. Particle Decay
Particles exhibit memoryless radioactive decay defined by half-life or decay constant ($\lambda$). The probability of any given particle decaying in a step $\Delta t$ is computed via:
$$ P(\text{decay}) = 1 - e^{-\lambda \Delta t} $$
A random binomial draw against this probability is computationally vectorized across all currently "alive" particles in the simulation each step.

## Real-World Relevance

Tools like GEANT4, FLUKA, and MCNP handle precisely these mathematics to design particle accelerators, build shielding for nuclear reactors, and plan radiation dosages for cancer treatments. This codebase simulates the core random-walk-with-decay aspect using Pythonic data science paradigms, focusing heavily on NumPy vectorization instead of low-level C++ iteration.

## Features

- **Vectorized Engine**: Replaces slow `for` loops globally by managing the particle population via N-dimensional `numpy` arrays.
- **Physical Boundary Checking**: Fast implementation of absorbing and reflective geometries.
- **Analytical Overlays**: Plots simulated survival counts against continuous $N_0 e^{-\lambda t}$ theory, and calculates empirical Half-Lives.
- **MSD Analysis**: Verifies random walk fidelity by ensuring Mean Squared Displacement grows linearly with time: $\langle x^2 \rangle \propto t$.

## Running the Project

To set up the environment:
```bash
python -m venv .venv
source .venv/Scripts/activate # On Windows use .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option 1: Live Interactive App
We provide an interactive **Streamlit** dashboard to instantly visualize parameter changes.
```bash
streamlit run app.py
```

### Option 2: Jupyter Notebook
A documented tutorial notebook is provided for pedagogical use or deeper data extraction.
```bash
jupyter lab notebooks/demo_simulation.ipynb
```
*(The raw notebooks contain rich explanations detailing the correlation of generated curves with theoretic formulas).*
