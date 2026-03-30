## Project Overview (portfolio-ready)

### What is this project?
A teaching-focused scientific workflow for polymer formulation where machine learning filters theoretical candidates before they reach the laboratory. It shows how to encode composition/process physics, learn surrogate models from synthetic-yet-realistic data, and drive an experiment loop that mimics lab validation.

### What does it demonstrate?
- Data generation that respects composition sums (~100 wt%), rule-of-mixtures trends, and temperature/process effects observed in real labs.
- Surrogate modeling under data scarcity, contrasting linear baselines against Random Forests while tracking error bars.
- Constrained candidate generation plus Pareto-style optimization so only the most promising formulations are forwarded to lab benches.
- An iterative train ? propose ? simulate ? retrain loop that mirrors how engineers tighten hypotheses between theory and experiment.

### How to read it
1. `notebooks/01_data_generation.ipynb`: physics-informed sampling, validation plots, and lab-style sanity checks.
2. `notebooks/02_modeling.ipynb`: surrogate comparisons, learning curves, and uncertainty discussion.
3. `notebooks/03_optimization.ipynb`: candidate generation, UCB ranking, Pareto front snapshots, and "virtual lab" updates.
