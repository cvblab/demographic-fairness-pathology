import os
import json
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

from data_utils import load_data
from evaluation_utils import (
    compute_auroc, get_intersectional_groups, calculate_demographic_parity, calculate_equalized_odds,
    calculate_subgroup_metrics, pad_proba_columns, expected_calibration_error, calculate_delta_eo,
    run_demographic_probe, bootstrap_fairness_metrics, compute_p_values
)
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from step2 import train_adversarial_multimodal, extract_adv_fused

def robust_compute_class_weight(y, num_classes):
    from sklearn.utils.class_weight import compute_class_weight
    unique_y = np.unique(y)
    cw = compute_class_weight('balanced', classes=unique_y, y=y)
    cw_dict = dict(zip(unique_y, cw))
    return np.array([cw_dict.get(i, 1.0) for i in range(num_classes)])

def main():
    print("Loading demographic metadata...")
    with open("/workspace/src/metadata/raw/AI4SKIN.json", "r") as f: ai4skin_json = json.load(f)
    with open("/workspace/src/metadata/raw/ASSIST.json", "r") as f: assist_json = json.load(f)
    tcga_df = pd.read_csv('/workspace/src/metadata/raw/tcga_reports_cleaned.csv', usecols=['project_id', 'slide_id', 'sex'])
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("tsantos/PathologyBERT")
    model = AutoModel.from_pretrained("tsantos/PathologyBERT").to(device)
    model.eval()
    
    def load_combined_fold(dataset_name, csv_path):
        X_r, X_g, X_v, y, a, s, c = load_data(csv_path, device, tokenizer, model, ai4skin_json, assist_json, tcga_df, dataset_filter=dataset_name)
        return X_r, X_g, X_v, y, a, s, c

    print("\n[Phase 1] Loading TCGA-SKCM for Pre-training...")
    X_r_tc_tr, X_g_tc_tr, X_v_tc_tr, y_tc_tr, a_tc_tr, s_tc_tr, c_tc_tr = load_combined_fold("TCGA-SKCM", "/workspace/src/metadata/tcga_train.csv")
    
    num_classes_tcga = int(np.max(y_tc_tr)) + 1
    cw_tcga = robust_compute_class_weight(y_tc_tr, num_classes_tcga)
    
    def get_age_group(a):
        if pd.isna(a): return 'Unknown'
        elif a < 40: return 'Young'
        elif a > 60: return 'Old'
        else: return 'Middle'
        
    adv_group_sex_tr = list(s_tc_tr)
    adv_group_age_tr = [get_age_group(a) for a in a_tc_tr]
    adv_group_center_tr = list(c_tc_tr)
    
    le_adv_sex = LabelEncoder()
    le_adv_age = LabelEncoder()
    le_adv_center = LabelEncoder()
    from collections import Counter
    group_counts_sex = Counter(adv_group_sex_tr)
    group_counts_age = Counter(adv_group_age_tr)
    group_counts_center = Counter(adv_group_center_tr)
    
    valid_adv_sex_tr = [i for i, g in enumerate(adv_group_sex_tr) if 'Unknown' not in g and 'unknown' not in str(g) and 'N/A' not in str(g) and group_counts_sex[g] >= 5]
    valid_adv_age_tr = [i for i, g in enumerate(adv_group_age_tr) if 'Unknown' not in g and 'unknown' not in str(g) and 'N/A' not in str(g) and group_counts_age[g] >= 5]
    valid_adv_center_tr = [i for i, g in enumerate(adv_group_center_tr) if 'Unknown' not in g and 'unknown' not in str(g) and 'N/A' not in str(g) and group_counts_center[g] >= 5]
    
    le_adv_sex.fit([adv_group_sex_tr[i] for i in valid_adv_sex_tr])
    le_adv_age.fit([adv_group_age_tr[i] for i in valid_adv_age_tr])
    le_adv_center.fit([adv_group_center_tr[i] for i in valid_adv_center_tr])
    
    a_train_sex = np.zeros(len(adv_group_sex_tr), dtype=int)
    mask_tr_sex = np.isin(adv_group_sex_tr, le_adv_sex.classes_)
    a_train_sex[mask_tr_sex] = le_adv_sex.transform([adv_group_sex_tr[i] for i in range(len(adv_group_sex_tr)) if mask_tr_sex[i]])
    
    a_train_age = np.zeros(len(adv_group_age_tr), dtype=int)
    mask_tr_age = np.isin(adv_group_age_tr, le_adv_age.classes_)
    a_train_age[mask_tr_age] = le_adv_age.transform([adv_group_age_tr[i] for i in range(len(adv_group_age_tr)) if mask_tr_age[i]])
    
    a_train_center = np.zeros(len(adv_group_center_tr), dtype=int)
    mask_tr_center = np.isin(adv_group_center_tr, le_adv_center.classes_)
    a_train_center[mask_tr_center] = le_adv_center.transform([adv_group_center_tr[i] for i in range(len(adv_group_center_tr)) if mask_tr_center[i]])
    
    print("\n--- [Phase 1] Pre-training on Source Cohort (TCGA-SKCM) ---")
    tcga_models = []
    for seed in range(5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        align_net, clf_net = train_adversarial_multimodal(
            X_v_tc_tr, X_r_tc_tr, X_g_tc_tr, y_tc_tr,
            a_train_sex, mask_tr_sex, len(le_adv_sex.classes_),
            a_train_age, mask_tr_age, len(le_adv_age.classes_),
            a_train_center, mask_tr_center, len(le_adv_center.classes_),
            num_classes=num_classes_tcga, class_weights=cw_tcga, epochs=100, lr=1e-3, beta=1.0, lambda_contra=0.1, cov_type='low_rank', device=device
        )
        tcga_models.append((align_net, clf_net))
        
    print("Pre-training Complete.")
    
    # 1.1 TCGA-SKCM Validation
    X_r_tc_te, X_g_tc_te, X_v_tc_te, y_tc_te, a_tc_te, s_tc_te, c_tc_te = load_combined_fold("TCGA-SKCM", "/workspace/src/metadata/tcga_test.csv")
    
    all_probs_tcga = []
    for align_net, clf_net in tcga_models:
        align_net.eval()
        clf_net.eval()
        with torch.no_grad():
            mu_te, logvar_te, factor_te = align_net(torch.tensor(X_g_tc_te, dtype=torch.float32).to(device))
            if align_net.cov_type == 'deterministic':
                e_tilde_te = mu_te
            elif align_net.cov_type == 'diagonal':
                std_te = torch.exp(0.5 * torch.clamp(logvar_te, -10, 10))
                e_tilde_te = mu_te + torch.randn_like(std_te) * std_te
            elif align_net.cov_type == 'low_rank':
                import torch.distributions as dist
                cov_diag = torch.exp(torch.clamp(logvar_te, -10, 10))
                mvn = dist.LowRankMultivariateNormal(mu_te, factor_te, cov_diag)
                e_tilde_te = mvn.rsample()
            clf_logits, _ = clf_net(torch.tensor(X_v_tc_te, dtype=torch.float32).to(device), e_tilde_te)
            all_probs_tcga.append(torch.nn.functional.softmax(clf_logits, dim=1).cpu().numpy())
            
    probs_tcga = np.mean(all_probs_tcga, axis=0)
    preds_tcga = np.argmax(probs_tcga, axis=1)
        
    prob_tcga_eval = probs_tcga[:, 1] if num_classes_tcga == 2 else probs_tcga
    tcga_auroc = compute_auroc(y_tc_te, prob_tcga_eval, num_classes_tcga)
    tcga_inter_te = get_intersectional_groups(a_tc_te, s_tc_te)
    tcga_eo = calculate_equalized_odds(y_tc_te, preds_tcga, tcga_inter_te)
    print(f"\n[Phase 1 Validation] TCGA-SKCM Test Set -> AUROC: {tcga_auroc:.3f} | Eq. Odds: {tcga_eo:.3f}")
    
    out_dir = f"/workspace/src/FOMENT/step3_cross_cohort_results"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    for ds_name, ds_prefix in [("AI4SKIN", "ai4skin"), ("ASSIST", "assist")]:
        print(f"\n[Phase 2] Transfer Learning on {ds_name}...")
        X_r_tr, X_g_tr, X_v_tr, y_tr, a_tr, s_tr, _ = load_combined_fold(ds_name, f"/workspace/src/metadata/{ds_prefix}_train.csv")
        X_r_te, X_g_te, X_v_te, y_te, a_te, s_te, _ = load_combined_fold(ds_name, f"/workspace/src/metadata/{ds_prefix}_test.csv")
        print(f"\n[Test Set N Sizes for {ds_name}]")
        print("Sex:\n", pd.Series(s_te).value_counts().to_string())
        print("Age:\n", pd.Series(a_te).value_counts().to_string())
        
        num_classes = int(max(np.max(y_tr), np.max(y_te))) + 1
        
        y_prob_dict = {}
        y_pred_dict = {}
        eval_metrics = {}
        
        def add_eval_baseline(model_name, X_train_ft, X_test_ft):
            all_probs = []
            for seed in range(5):
                lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight='balanced', random_state=seed)
                lr.fit(X_train_ft, y_tr)
                probs = pad_proba_columns(lr.predict_proba(X_test_ft), lr.classes_, num_classes)
                all_probs.append(probs)
            probs_aggr = np.mean(all_probs, axis=0)
            preds_aggr = np.argmax(probs_aggr, axis=1)
            
            y_prob_dict[model_name] = probs_aggr
            y_pred_dict[model_name] = preds_aggr
            
            prob_eval = probs_aggr[:, 1] if num_classes == 2 else probs_aggr
            auroc = compute_auroc(y_te, prob_eval, num_classes)
            bacc = balanced_accuracy_score(y_te, preds_aggr)
            f1 = f1_score(y_te, preds_aggr, average='weighted')
            ece = expected_calibration_error(y_te, probs_aggr)
            delta_eo_sex = calculate_delta_eo(y_te, preds_aggr, s_te)
            
            a_te_bin = np.array([0 if x < 65 else 1 for x in a_te])
            a_tr_bin = np.array([0 if x < 65 else 1 for x in a_tr])
            delta_eo_age = calculate_delta_eo(y_te, preds_aggr, a_te_bin)
            
            probe_sex = run_demographic_probe(X_train_ft, X_test_ft, s_tr, s_te)
            probe_age = run_demographic_probe(X_train_ft, X_test_ft, a_tr_bin, a_te_bin)
            
            eval_metrics[model_name] = {
                'auroc': auroc, 'bacc': bacc, 'f1': f1, 'ece': ece,
                'delta_eo_sex': delta_eo_sex, 'delta_eo_age': delta_eo_age,
                'probe_auroc_sex': probe_sex, 'probe_auroc_age': probe_age
            }
        add_eval_baseline("Vision-Only (Linear)", X_v_tr, X_v_te)
        add_eval_baseline("Text-Only Real (Linear)", X_r_tr, X_r_te)
        add_eval_baseline("Text-Only Gen (Linear)", X_g_tr, X_g_te)
        
        print("  Extracting TCGA-aligned pure representations across 5 seeds...")
        def extract_pure_fused(align, X_v, X_t, num_samples=10):
            align.eval()
            img_t = torch.tensor(X_v, dtype=torch.float32).to(device)
            e_t = torch.tensor(X_t, dtype=torch.float32).to(device)
            with torch.no_grad():
                mu, logvar, factor = align(e_t)
            all_seta = []
            with torch.no_grad():
                for _ in range(num_samples):
                    if align.cov_type == 'deterministic': e_tilde = mu
                    elif align.cov_type == 'diagonal':
                        std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))
                        e_tilde = mu + torch.randn_like(std) * std
                    elif align.cov_type == 'low_rank':
                        import torch.distributions as dist
                        cov_diag = torch.exp(torch.clamp(logvar, -10, 10))
                        mvn = dist.LowRankMultivariateNormal(mu, factor, cov_diag)
                        e_tilde = mvn.rsample()
                    fused = torch.cat([img_t, e_tilde], dim=1).cpu().numpy()
                    all_seta.append(fused)
            return np.mean(all_seta, axis=0)

        def add_eval_transfer(model_name, X_r_tr, X_r_te):
            all_probs = []
            for seed, (align_net, clf_net) in enumerate(tcga_models):
                s_eta_r_tr = extract_pure_fused(align_net, X_v_tr, X_r_tr, 5)
                s_eta_r_te = extract_pure_fused(align_net, X_v_te, X_r_te, 10)
                
                lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight='balanced', random_state=seed)
                lr.fit(s_eta_r_tr, y_tr)
                probs = pad_proba_columns(lr.predict_proba(s_eta_r_te), lr.classes_, num_classes)
                all_probs.append(probs)
                
            probs_aggr = np.mean(all_probs, axis=0)
            preds_aggr = np.argmax(probs_aggr, axis=1)
            
            y_prob_dict[model_name] = probs_aggr
            y_pred_dict[model_name] = preds_aggr
            
            prob_eval = probs_aggr[:, 1] if num_classes == 2 else probs_aggr
            auroc = compute_auroc(y_te, prob_eval, num_classes)
            bacc = balanced_accuracy_score(y_te, preds_aggr)
            f1 = f1_score(y_te, preds_aggr, average='weighted')
            ece = expected_calibration_error(y_te, probs_aggr)
            delta_eo_sex = calculate_delta_eo(y_te, preds_aggr, s_te)
            
            a_te_bin = np.array([0 if x < 65 else 1 for x in a_te])
            a_tr_bin = np.array([0 if x < 65 else 1 for x in a_tr])
            delta_eo_age = calculate_delta_eo(y_te, preds_aggr, a_te_bin)
            
            probe_sex = run_demographic_probe(s_eta_r_tr, s_eta_r_te, s_tr, s_te)
            probe_age = run_demographic_probe(s_eta_r_tr, s_eta_r_te, a_tr_bin, a_te_bin)
            
            eval_metrics[model_name] = {
                'auroc': auroc, 'bacc': bacc, 'f1': f1, 'ece': ece,
                'delta_eo_sex': delta_eo_sex, 'delta_eo_age': delta_eo_age,
                'probe_auroc_sex': probe_sex, 'probe_auroc_age': probe_age
            }

        add_eval_transfer("Transfer Adv Align (Real+Vis)", X_r_tr, X_r_te)
        add_eval_transfer("Transfer Adv Align (Gen+Vis)", X_g_tr, X_g_te)
        
        # Run Bootstrapping
        print(f"  Running bootstrap evaluation (1000 iterations) for {ds_name}...")
        a_te_bin_test = np.array([0 if x < 65 else 1 for x in a_te])
        ci_results, boot_results = bootstrap_fairness_metrics(y_te, y_prob_dict, y_pred_dict, num_classes, n_iterations=1000, sex_attr=s_te, age_attr=a_te_bin_test)
        
        p_values = compute_p_values(boot_results, baseline_model="Vision-Only (Linear)")
        
        # Print the detailed table
        print(f"\n--- Classification & Fairness Metrics ({ds_name} Test Set) ---")
        models_list = ["Vision-Only (Linear)", "Text-Only Real (Linear)", "Text-Only Gen (Linear)", "Transfer Adv Align (Real+Vis)", "Transfer Adv Align (Gen+Vis)"]
        
        # Identify best values
        best_vals = {
            'auroc': max(eval_metrics[m]['auroc'] for m in models_list if not pd.isna(eval_metrics[m]['auroc'])),
            'f1': max(eval_metrics[m]['f1'] for m in models_list if not pd.isna(eval_metrics[m]['f1'])),
            'bacc': max(eval_metrics[m]['bacc'] for m in models_list if not pd.isna(eval_metrics[m]['bacc'])),
            'ece': min(eval_metrics[m]['ece'] for m in models_list if not pd.isna(eval_metrics[m]['ece'])),
            'delta_eo_sex': min(eval_metrics[m]['delta_eo_sex'] for m in models_list if not pd.isna(eval_metrics[m]['delta_eo_sex'])),
            'delta_eo_age': min(eval_metrics[m]['delta_eo_age'] for m in models_list if not pd.isna(eval_metrics[m]['delta_eo_age'])),
            'probe_auroc_sex': min(eval_metrics[m]['probe_auroc_sex'] for m in models_list if not pd.isna(eval_metrics[m]['probe_auroc_sex'])),
            'probe_auroc_age': min(eval_metrics[m]['probe_auroc_age'] for m in models_list if not pd.isna(eval_metrics[m]['probe_auroc_age']))
        }

        def get_stars(p):
            if pd.isna(p): return ""
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            return ""

        def fmt_with_p(val, ci_l, ci_u, best_val, p_val, is_higher):
            if pd.isna(val): return "NaN"
            s = f"{val:.3f}"
            if ci_l is not None and ci_u is not None:
                s += f" ({ci_l:.3f} - {ci_u:.3f})"
            is_best = (val >= best_val) if is_higher else (val <= best_val)
            if is_best:
                s = f"**{s}**"
            stars = get_stars(p_val)
            if stars: s += f" {stars}"
            return s

        for m_name in models_list:
            em = eval_metrics[m_name]
            st = ci_results.get(m_name, {})
            pv = p_values.get(m_name, {})
            row = {
                "Model": m_name,
                "AUROC": fmt_with_p(em['auroc'], st.get('auroc', [None, None])[0], st.get('auroc', [None, None])[1], best_vals['auroc'], pv.get('auroc', np.nan), True),
                "F1-Score": fmt_with_p(em['f1'], st.get('f1', [None, None])[0], st.get('f1', [None, None])[1], best_vals['f1'], pv.get('f1', np.nan), True),
                "Bal Acc": fmt_with_p(em['bacc'], st.get('bacc', [None, None])[0], st.get('bacc', [None, None])[1], best_vals['bacc'], pv.get('bacc', np.nan), True),
                "ECE": fmt_with_p(em['ece'], st.get('ece', [None, None])[0], st.get('ece', [None, None])[1], best_vals['ece'], pv.get('ece', np.nan), False),
                "DeltaEO (Sex)": fmt_with_p(em['delta_eo_sex'], st.get('delta_eo_sex', [None, None])[0], st.get('delta_eo_sex', [None, None])[1], best_vals['delta_eo_sex'], pv.get('delta_eo_sex', np.nan), False),
                "DeltaEO (Age)": fmt_with_p(em['delta_eo_age'], st.get('delta_eo_age', [None, None])[0], st.get('delta_eo_age', [None, None])[1], best_vals['delta_eo_age'], pv.get('delta_eo_age', np.nan), False),
                "Probe (Sex)": fmt_with_p(em['probe_auroc_sex'], None, None, best_vals['probe_auroc_sex'], np.nan, False),
                "Probe (Age)": fmt_with_p(em['probe_auroc_age'], None, None, best_vals['probe_auroc_age'], np.nan, False),
            }
            results.append(row)
            print(f"  {m_name:30} | AUC: {em['auroc']:.3f} | F1: {em['f1']:.3f} | ECE: {em['ece']:.3f} | EO(Sex): {em['delta_eo_sex']:.3f}")
            
    final_df = pd.DataFrame(results)
    os.makedirs(out_dir, exist_ok=True)
    final_df.to_csv(f"{out_dir}/transfer_metrics_full.csv", index=False)
    print(f"\nSaved cross-cohort transfer artifacts to {out_dir}/transfer_metrics_full.csv")

if __name__ == "__main__":
    main()
