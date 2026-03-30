## Notebooks (technical storyline)

### 01 - Data Generation & Validation
**Purpose**
- Introduce the polymer-formulation problem with composition constraints.
- Justify the use of physics-informed synthetic data and verify its properties.
- Document checks on rule of mixtures, correlations, and differences between N=500 vs. N=40 scenarios.

**What to look for**
- Histograms and composition sums (~100 wt%) as basic sanity checks.
- Expected trade-offs (filler → strength, plasticizer → elongation) and their influence on variability.
- A brief *Key Takeaways* section that links to the modeling notebook.

### 02 - Modeling (Full vs. Low-Data)
**Purpose**
- Compare a linear baseline against a Random Forest surrogate.
- Analyze how performance changes when moving from ull_500 to low_data_40.
- Use tree dispersion as a lightweight proxy for uncertainty.

**What to look for**
- Metric tables (RMSE/R²) per dataset split.
- Discussion on when a non-linear surrogate pays off in engineering workflows.

### 03 - Optimization (Experimental Design)
**Purpose**
- Implement constrained candidate generation to propose feasible formulations.
- Evaluate single- and multi-objective functions with reference to the Pareto front.
- Show how the uncertainty term (kappa/UCB) drives exploration vs. exploitation and how to close an iterative loop.

**What to look for**
- Differences among kappa = 0/1/2 when selecting candidates.
- Evolution of the 	rain -> propose -> simulate -> retrain loop when exploration is prioritized.
