# Complete results: all methods, tasks and metrics

Complete results of every evaluated configuration on the six tasks (AI4SKIN, ASSIST and TCGA-SKCM; multiclass and binary), with CTransPath features and HistoGPT reports unless the method name states otherwise. These files replace the full result tables of the supplementary material of the manuscript "When Do Pathology Reports Help Whole-Slide Classifiers? Accuracy and Fairness under Test-Time Report Substitution".

One file per cohort and task (`<cohort>_<task>.csv`), plus `all_methods_all_tasks.csv` with all rows.

| Column | Meaning |
|---|---|
| `cohort`, `task` | Cohort and task (`multiclass` or `binary`) |
| `method` | Configuration (codes and definitions as in Table 2 of the manuscript; det./diag./low-rank: alignment covariance form) |
| `real_report_at_test_time` | `yes` for references that use the real-report summary at training and test time (not deployable) |
| `auroc`, `auroc_ci_low`, `auroc_ci_high` | Macro AUROC on pooled out-of-fold predictions, with 95% patient-level bootstrap interval |
| `balanced_accuracy`, `macro_f1` | Class with the highest predicted probability (binary: threshold 0.5); no threshold tuning |
| `epochs` | Mean number of epochs selected by early stopping (empty for linear probes and ABMIL) |
| `eopp_gap_sex`, `eopp_gap_age` | Equal-opportunity gap (Eq. 8 of the manuscript); point estimates |
| `eodds_gap_sex` | Equalised-odds gap by sex; point estimate |

Empty fairness cells: metric not defined (fewer than five positive slides in a group).
