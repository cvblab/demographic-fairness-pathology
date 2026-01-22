# Evaluating Fairness
This repository contains the source code for the paper '-'.

## Structure

- `src/`: Source code python scripts.
- `results/`: Output directory structure.
  - `tables/`
    - `exp0/`: Baseline Performance (AUC) and Fairness (DeltaEO) tables (Experiment 0).
    - `exp1/`: Demographic Leakage Analysis results (Experiment 1).
    - `exp2/`: Mitigation Experiment results (Experiment 2).
    - `counts/`: Sample counts in LaTeX format.
  - `plots/`
    - `correlation/`: Plots showing Leakage vs Fairness, and Performance vs Fairness.
    - `classifiers/`: Boxplots of fairness gaps by classifier algorithm.
    - `mitigation/`: Bar plots comparing mitigation strategies.

## Prerequisites

- **Data**: The code assumes data is located at `../data`.
- **Dependencies**: Python 3.8+, pandas, numpy, scikit-learn, seaborn, matplotlib, fairlearn, umap-learn.

## Running the Pipeline

### 1. Main Experiment Loop
Run the main pipeline to generate all intermediate results (Tables and Experiment Logs). This script handles data loading, preprocessing, Exp0 (Baseline), Exp1 (Leakage), and Exp2 (Mitigation).

```bash
cd src
python main.py
```
*Note: This may take a significant amount of time depending on the dataset size and number of repeats.*

### 2. Analysis and Plot Generation
After `main.py` has completed and populated `results/tables/exp0`, `exp1`, and `exp2`, run the following scripts to generate the final figures and summary tables.

#### Correlation Analysis (Leakage vs Fairness)
```bash
python correlation.py
```
Output: `results/plots/correlation/correlation_leakage_csv_eo.png`

#### Performance vs Unfairness Trade-offs
```bash
python find_tradeoffs.py
```
Output: `results/plots/correlation/plot_perf_vs_bias.png`

#### Methodology Comparison (Classifiers)
```bash
python analyze_classifiers.py
```
Output: `results/plots/classifiers/classifier_fairness_boxplot.png`

#### Mitigation Results Visualization
```bash
python generate_plots.py
```
Output: `results/plots/mitigation/final_mitigation_plot_Balanced.png`

## Configuration

Modify `src/config.py` to adjust:
- `RESULTS_DIR`: Output location (Default: `/workspace/src/clean/results`)
- `TASKS_CONFIG`: Tasks, biases, and groups definitions.
- `N_REPEATS`: Number of bootstrap repeats.
