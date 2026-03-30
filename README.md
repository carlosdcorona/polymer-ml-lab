# ML-Guided Polymer Discovery Loop

## Scientific Premise
This repository reframes the project as a didactic scientific study: we use machine learning to shrink the theoretical search space of polymer formulations before committing to lab experiments. Synthetic-yet-physical data emulate the hypotheses an R&D team would draft on paper; surrogate models score those hypotheses, and an optimization loop proposes only the candidates most likely to survive real laboratory validation.

## Learning Focus
- **From theory to lab:** encode physical constraints (mass balance, process limits) and let ML surface which formulations deserve experimental budget.
- **Transparent methodology:** every step—data generation, modeling, optimization—is documented so a reviewer can follow the reasoning end to end.
- **Reproducible insights:** notebooks double as lab books, showing how virtual experiments evolve and which recommendations emerge.

## Repository Map
- `notebooks/`
  - `01_data_generation.ipynb`: builds the physics-informed synthetic dataset and sanity-checks it against expected trends.
  - `02_modeling.ipynb`: compares linear baselines vs. Random Forest surrogates under full-data and low-data regimes, highlighting uncertainty-aware metrics.
  - `03_optimization.ipynb`: turns trained models into a candidate generation engine (single- and multi-objective) that mimics lab iteration.
- `src/data_generation.py`: reusable module with the formulation bounds, sampler, and lab-simulation helper so experiments can be scripted outside notebooks.
- `scripts/strip_notebook_outputs.py`: utility to keep notebooks lightweight before version control.
- `docs/01_project_overview.md`: high-level storyline for reviewers.
- `requirements.txt`: minimal dependencies (NumPy, pandas, scikit-learn, matplotlib, seaborn).
- `data/`: optional cache for generated datasets (ignored by git by default).

## Scientific Workflow
1. **Define the theoretical sandbox:** a constrained formulation space (polymer base, additives, processing variables) that encodes domain heuristics.
2. **Generate pseudo-lab data:** draw thousands of formulations that satisfy ~100 wt% and process bounds, project them through physics-inspired equations, and obtain target properties with controlled noise.
3. **Train surrogate models:** quantify how well each target (tensile strength, elongation, thermal resistance) can be predicted with little data; keep track of uncertainty to avoid overconfident recommendations.
4. **Prioritize lab experiments:** use constrained candidate generation + Upper Confidence Bound (UCB) scoring to suggest formulations most likely to outperform current bests while still exploring novel regions.
5. **Iterate:** simulate the measurement of proposed candidates, append them to the dataset, retrain, and observe how the Pareto front tightens—mirroring a real design-make-test cycle.

## Running the Project
1. **Set up the environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
2. **Reproduce the virtual experiments**
   - Launch Jupyter Lab or VS Code notebooks.
   - Execute notebooks in numerical order (`01` ? `03`). Each notebook ends with a "Lab Notes" section pointing to the next step.
3. **Automate data generation (optional)**
   ```bash
   python -m src.data_generation  # use functions to script custom datasets
   ```
4. **Log insights for your portfolio**
   - Export plots/tables from each notebook (Jupyter `File > Export Notebook As > HTML`).
   - Capture summary metrics (model MAE, Pareto front coordinates) to reference in your GitHub README or case study.

## Expected Results & Insights
- **Physics-consistent trends:** filler increases tensile strength but hurts elongation; plasticizer does the opposite; temperature shows a bell-shaped optimum.
- **Modeling takeaway:** Random Forest surrogates stay robust even when data is throttled to 20–30 samples, illustrating why ML can pre-filter hypotheses before lab work.
- **Optimization payoff:** the UCB-driven loop narrows the Pareto front in ~3 iterations, so only the top ~5% of theoretical candidates would be sent to the lab for confirmation.

## Limitations & Next Steps
- Synthetic data approximates reality; once true lab data arrives, plug it into the same notebooks or swap surrogates for Gaussian Processes.
- Current optimization uses heuristic search—future work can integrate Bayesian Optimization or evolutionary algorithms for better sample efficiency.
- Add domain-specific constraints (cost, aging, rheology) as additional targets to make the lab recommendations production-ready.

By following this structure you can present the repository as a complete scientific narrative: theory ? ML-assisted screening ? lab-ready recommendations.
