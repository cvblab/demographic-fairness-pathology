# Fairness Multimodal Evaluation via Neural Text-Proxies

This repository contains the codebase for evaluating biases, multimodal fusion, and cross-cohort transfer learning in skin pathology representations. It focuses on integrating Whole Slide Images (WSI) and Pathology Reports, robustly evaluating performance under missing text conditions via AI-generated proxies, and enforcing demographic fairness using variational adversarial alignment.

## Directory Structure

- `metadata/`: Contains all JSON and CSV files with dataset demographics and splits.
- `features_*/`: Pre-extracted visual embeddings (CTransPath / Swin-T).
- `output_histogpt_reports_*/`: HistoGPT generated reports and their PathologyBERT embeddings.
- `src/`: Core pipeline execution scripts.
  - `old/`: Archive of legacy data preparation and diagnostic scripts.
- `FOMENT/`: Centralized output directory for all metrics, LaTeX tables, and plots.

## Prerequisites & Installation

To run this pipeline, you will need Python 3.9+ and the following core libraries.

```bash
pip install torch torchvision transformers scikit-learn pandas numpy matplotlib seaborn scipy
```

Make sure you are executing the scripts from the `/workspace/src` directory.

## Pipeline Execution

The pipeline is organized in sequential steps, evaluating isolated modalities, baseline multimodal approaches, proposed adversarial alignment, and cross-cohort generalization.

### Step 0: Modality Validation
```bash
python step0.py
```
Evaluates purely the intrinsic diagnostic signal of real vs. AI-generated text modalities on the perfectly paired subset (samples that have a real report available). This establishes the reliability of the generated text proxy. Outputs to `FOMENT/step0_text_validation/`.

### Step 1: Baseline Linear Probes
```bash
python step1.py
```
Evaluates straightforward unimodal linear probes for vision-only and text-only representations. Computes standard metrics (AUROC, Balanced Accuracy, F1-Score) and demographic leakage (predictability of Sex/Age). Outputs to `FOMENT/step1_results_<Dataset>/`.

### Step 2: Main Multimodal & Fairness Evaluation (Multiclass)
```bash
python step2.py
```
This comprehensive script tests multimodal fusion techniques (Concatenation, Cross-Attention, Late Fusion) alongside fairness baselines (Reweighting, GroupDRO, Adversarial Debiasing) and our proposed **Variational Adversarial Alignment** on all datasets (AI4SKIN, ASSIST, TCGA-SKCM). It calculates:
- Performance (AUROC, Bal. Acc, F1)
- Intersectional Fairness ($\Delta$EO, Demographic Parity)
- Uncertainty Quantification (Mutual Information, Predictive Entropy via MC Dropout)
- Out-Of-Distribution (OOD) generalizability on zero-context data.

Outputs, bootstrapped confidence intervals, and LaTeX tables are stored in `FOMENT/step2_results_<Dataset>/`.

> **Note**: For Binary classification tasks (e.g., Benign vs Malignant), run the parallel script:
> ```bash
> python step2m.py
> ```

### Step 3: Cross-Cohort Zero-Shot Transfer
```bash
python step3_cross_cohort.py
```
Pre-trains the adversarial alignment model on TCGA-SKCM and transfers the aligned representations zero-shot to AI4SKIN and ASSIST. This evaluates whether the alignment generalizes to out-of-distribution centers enforcing fairness and retaining accuracy. Outputs to `FOMENT/step3_cross_cohort_results/`.

### Step 4: Generative Proxy Ablation
```bash
python step4_proxy_ablation.py
```
Simulates extreme degradation in the AI-generated reports (via Gaussian Noise and Feature Masking) to measure the robustness of the multimodal fusion. Demonstrates that variational alignment maintains fairness and predictive accuracy without overfitting to brittle generative artifacts.

### Plotting
```bash
python plot_figures.py
```
Generates all publication-ready visualizations leveraging the `FOMENT/` CSV artifacts, including:
- Performance vs. Fairness Pareto Frontiers.
- Target cohort evaluation subgroup bar charts.
- Cross-cohort transfer heatmaps.
- Proxy degradation robustness curves (Gaussian Noise and Feature Masking).
