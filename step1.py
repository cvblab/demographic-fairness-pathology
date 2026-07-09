import os
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from scipy.spatial.distance import cosine
import warnings

from data_utils import get_combined_dataset
from evaluation_utils import (
    compute_auroc, expected_calibration_error, calculate_delta_eo, 
    run_demographic_probe, pad_proba_columns, bootstrap_fairness_metrics, 
    compute_p_values, get_intersectional_groups, plot_metrics_with_pvalues,
    plot_fairness_metrics_step1
)

warnings.filterwarnings("ignore")

# Removed custom bootstrap_metrics

def run_evaluation(dataset_name, train_csvs, test_csv, device, tokenizer, model, ai4skin_json, assist_json, tcga_df=None):
    print(f"\n{'='*50}\nStarting Evaluation for {dataset_name}\n{'='*50}")
    
    X_real_train, X_gen_train, X_vis_train, y_train, age_train, sex_train, center_train = get_combined_dataset(
        train_csvs, device, tokenizer, model, ai4skin_json, assist_json, tcga_df, dataset_filter=dataset_name)
    
    X_real_test, X_gen_test, X_vis_test, y_test, age_test, sex_test, center_test = get_combined_dataset(
        [test_csv], device, tokenizer, model, ai4skin_json, assist_json, tcga_df, dataset_filter=dataset_name)
    
    if len(y_train) == 0 or len(y_test) == 0:
        print(f"Skipping {dataset_name} due to lack of data.")
        return

    # Text Similarity Metrics
    similarities = []
    for i in range(len(X_real_test)):
        n_r = np.linalg.norm(X_real_test[i])
        n_g = np.linalg.norm(X_gen_test[i])
        if n_r > 0 and n_g > 0:
            sim = 1 - cosine(X_real_test[i], X_gen_test[i])
            similarities.append(sim)
            
    sim_mean = np.mean(similarities) if similarities else 0.0
    print(f"\n--- Text Similarity Metrics ---")
    print(f"Average Cosine Similarity (Real vs Generated) on Test Set: {sim_mean:.4f}")
    
    num_classes = int(np.max(np.concatenate([y_train, y_test]))) + 1
    print(f"\nDetected {num_classes} total classes.")

    if num_classes < 2:
        print("Not enough classes for ML evaluation. Skipping ML training.")
        return

    median_age = 60
    age_group_train = np.where(pd.isna(age_train), 'Unknown', np.where(age_train >= median_age, 'Older', 'Younger'))
    age_group_test = np.where(pd.isna(age_test), 'Unknown', np.where(age_test >= median_age, 'Older', 'Younger'))

    models = {
        'Real': {'X_tr': X_real_train, 'X_te': X_real_test, 'name': 'Text-LR (Real)'},
        'Gen': {'X_tr': X_gen_train, 'X_te': X_gen_test, 'name': 'Text-LR (Gen)'},
        'Vis': {'X_tr': X_vis_train, 'X_te': X_vis_test, 'name': 'Vision-LR (CTransPath)'},
        'MultiReal': {'X_tr': np.hstack((X_real_train, X_vis_train)), 'X_te': np.hstack((X_real_test, X_vis_test)), 'name': 'Multimodal-LR (Real+Vis)'},
        'MultiGen': {'X_tr': np.hstack((X_gen_train, X_vis_train)), 'X_te': np.hstack((X_gen_test, X_vis_test)), 'name': 'Multimodal-LR (Gen+Vis)'}
    }
    
    y_prob_dict = {}
    y_pred_dict = {}
    eval_metrics = {}

    for key, info in models.items():
        m_name = info['name']
        print(f"\nTraining Logistic Regression on {m_name}...")
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(info['X_tr'], y_train)
        
        y_pred = clf.predict(info['X_te'])
        y_prob = pad_proba_columns(clf.predict_proba(info['X_te']), clf.classes_, num_classes)
        
        y_prob_dict[m_name] = y_prob
        y_pred_dict[m_name] = y_pred
        
        eval_metrics[m_name] = {
            'auroc': compute_auroc(y_test, y_prob, num_classes),
            'bacc': balanced_accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'acc': accuracy_score(y_test, y_pred),
            'ece': expected_calibration_error(y_test, y_prob),
            'delta_eo_sex': calculate_delta_eo(y_test, y_pred, sex_test),
            'delta_eo_age': calculate_delta_eo(y_test, y_pred, age_group_test),
            'probe_auroc_sex': run_demographic_probe(info['X_tr'], info['X_te'], sex_train, sex_test),
            'probe_auroc_age': run_demographic_probe(info['X_tr'], info['X_te'], age_group_train, age_group_test)
        }
    
    # Statistical Significance using Bootstrapping
    print("Running bootstrap evaluation (1000 iterations) for statistical significance...")
    ci_results, boot_results = bootstrap_fairness_metrics(y_test, y_prob_dict, y_pred_dict, num_classes, n_iterations=1000, sex_attr=sex_test, age_attr=age_group_test)
    
    print("\n--- Computing p-values vs Vision-LR (CTransPath) ---")
    p_values = compute_p_values(boot_results, baseline_model="Vision-LR (CTransPath)")
    
    print(f"\n[Test Set N Sizes for {dataset_name}]")
    print("Sex:\n", pd.Series(sex_test).value_counts().to_string())
    print("Age:\n", pd.Series(age_group_test).value_counts().to_string())
    
    # Print Table
    print("\n--- Classification & Fairness Metrics (Test Set) ---")
    # Identify best values
    best_vals = {
        'auroc': max(eval_metrics[m['name']]['auroc'] for m in models.values() if not pd.isna(eval_metrics[m['name']]['auroc'])),
        'f1': max(eval_metrics[m['name']]['f1'] for m in models.values() if not pd.isna(eval_metrics[m['name']]['f1'])),
        'bacc': max(eval_metrics[m['name']]['bacc'] for m in models.values() if not pd.isna(eval_metrics[m['name']]['bacc'])),
        'ece': min(eval_metrics[m['name']]['ece'] for m in models.values() if not pd.isna(eval_metrics[m['name']]['ece'])),
        'delta_eo_sex': min(eval_metrics[m['name']]['delta_eo_sex'] for m in models.values() if not pd.isna(eval_metrics[m['name']]['delta_eo_sex'])),
        'delta_eo_age': min(eval_metrics[m['name']]['delta_eo_age'] for m in models.values() if not pd.isna(eval_metrics[m['name']]['delta_eo_age'])),
        'probe_auroc_sex': min(eval_metrics[m['name']]['probe_auroc_sex'] for m in models.values() if not pd.isna(eval_metrics[m['name']]['probe_auroc_sex'])),
        'probe_auroc_age': min(eval_metrics[m['name']]['probe_auroc_age'] for m in models.values() if not pd.isna(eval_metrics[m['name']]['probe_auroc_age']))
    }

    def fmt(val, ci_l, ci_u, best_val):
        if pd.isna(val): return "NaN"
        s = f"{val:.3f}"
        if ci_l is not None and ci_u is not None:
            s += f" ({ci_l:.3f} - {ci_u:.3f})"
        if val == best_val:
            return f"**{s}**"
        return s

    def get_stars(p):
        if pd.isna(p): return ""
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""

    def fmt_with_p(val, ci_l, ci_u, best_val, p_val):
        s = fmt(val, ci_l, ci_u, best_val)
        stars = get_stars(p_val)
        if stars: s += f" {stars}"
        return s

    results_list = []
    for key, info in models.items():
        m_name = info['name']
        em = eval_metrics[m_name]
        st = ci_results.get(m_name, {})
        pv = p_values.get(m_name, {})
        
        results_list.append({
            "Model": m_name,
            "AUROC": fmt_with_p(em['auroc'], st.get('auroc', [None, None])[0], st.get('auroc', [None, None])[1], best_vals['auroc'], pv.get('auroc', np.nan)),
            "F1-Score": fmt_with_p(em['f1'], st.get('f1', [None, None])[0], st.get('f1', [None, None])[1], best_vals['f1'], pv.get('f1', np.nan)),
            "Bal Acc": fmt_with_p(em['bacc'], st.get('bacc', [None, None])[0], st.get('bacc', [None, None])[1], best_vals['bacc'], pv.get('bacc', np.nan)),
            "ECE (↓)": fmt_with_p(em['ece'], st.get('ece', [None, None])[0], st.get('ece', [None, None])[1], best_vals['ece'], pv.get('ece', np.nan)),
            "DeltaEO (Sex)": fmt_with_p(em['delta_eo_sex'], st.get('delta_eo_sex', [None, None])[0], st.get('delta_eo_sex', [None, None])[1], best_vals['delta_eo_sex'], pv.get('delta_eo_sex', np.nan)),
            "DeltaEO (Age)": fmt_with_p(em['delta_eo_age'], st.get('delta_eo_age', [None, None])[0], st.get('delta_eo_age', [None, None])[1], best_vals['delta_eo_age'], pv.get('delta_eo_age', np.nan)),
            "Probe (Sex)": fmt(em['probe_auroc_sex'], None, None, best_vals['probe_auroc_sex']),
            "Probe (Age)": fmt(em['probe_auroc_age'], None, None, best_vals['probe_auroc_age']),
        })
        
    df_results = pd.DataFrame(results_list)
    print(df_results.to_markdown(index=False))
    
    out_dir = f"/workspace/src/FOMENT/step1_results_{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    df_results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    
    if p_values:
        df_p_values = pd.DataFrame(p_values).T
        df_p_values.index.name = "Model"
        df_p_values.to_csv(os.path.join(out_dir, "p_values_vs_Vision-LR.csv"))
    with open(os.path.join(out_dir, "results.tex"), "w") as f:
        f.write("\\begin{table}[h!]\\centering\\Large\n")
        f.write(df_results.to_latex(index=False))
        f.write("\\end{table}")
        
    print("\n--- Generating Metric Plot with p-values ---")
    plot_metrics_with_pvalues(eval_metrics, p_values, "Vision-LR (CTransPath)", dataset_name, out_dir)
    
    print("\n--- Generating Fairness Metric Plot ---")
    plot_fairness_metrics_step1(eval_metrics, dataset_name, out_dir)

def main():
    print("Loading demographic metadata...")
    with open("/workspace/src/metadata/raw/AI4SKIN.json", "r") as f:
        ai4skin_json = json.load(f)
    with open("/workspace/src/metadata/raw/ASSIST.json", "r") as f:
        assist_json = json.load(f)
    tcga_df = pd.read_csv('/workspace/src/metadata/raw/tcga_reports_cleaned.csv', usecols=['project_id', 'slide_id', 'sex', 'overall_survival'])
        
    print("Initializing PathologyBERT model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("tsantos/PathologyBERT")
    model = AutoModel.from_pretrained("tsantos/PathologyBERT").to(device)
    model.eval()

    # Evaluate AI4SKIN
    ai4skin_train = ["/workspace/src/metadata/ai4skin_train.csv", "/workspace/src/metadata/ai4skin_val.csv"]
    ai4skin_test = "/workspace/src/metadata/ai4skin_test.csv"
    run_evaluation("AI4SKIN", ai4skin_train, ai4skin_test, device, tokenizer, model, ai4skin_json, assist_json, tcga_df)
    
    # Evaluate ASSIST
    assist_train = ["/workspace/src/metadata/assist_train.csv", "/workspace/src/metadata/assist_val.csv"]
    assist_test = "/workspace/src/metadata/assist_test.csv"
    run_evaluation("ASSIST", assist_train, assist_test, device, tokenizer, model, ai4skin_json, assist_json, tcga_df)
    
    # Evaluate TCGA-SKCM
    tcga_train = ["/workspace/src/metadata/tcga_train.csv", "/workspace/src/metadata/tcga_val.csv"]
    tcga_test = "/workspace/src/metadata/tcga_test.csv"
    run_evaluation("TCGA-SKCM", tcga_train, tcga_test, device, tokenizer, model, ai4skin_json, assist_json, tcga_df)

if __name__ == "__main__":
    main()
