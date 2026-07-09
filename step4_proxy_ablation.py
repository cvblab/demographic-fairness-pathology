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
from evaluation_utils import compute_auroc, get_intersectional_groups, calculate_delta_eo, pad_proba_columns
from step2 import train_adversarial_multimodal

def robust_compute_class_weight(y, num_classes):
    from sklearn.utils.class_weight import compute_class_weight
    unique_y = np.unique(y)
    cw = compute_class_weight('balanced', classes=unique_y, y=y)
    cw_dict = dict(zip(unique_y, cw))
    return np.array([cw_dict.get(i, 1.0) for i in range(num_classes)])

def extract_pure_fused(align, X_v, X_t, num_samples=10, device='cpu'):
    align.eval()
    img_t = torch.tensor(X_v, dtype=torch.float32).to(device)
    e_t = torch.tensor(X_t, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, logvar, factor = align(e_t)
    all_seta = []
    with torch.no_grad():
        for _ in range(num_samples):
            if align.cov_type == 'deterministic':
                e_tilde = mu
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

def inject_noise(X, noise_scale=0.0, mask_prob=0.0):
    X_noisy = np.copy(X)
    if noise_scale > 0.0:
        X_noisy += np.random.normal(0, noise_scale, X_noisy.shape)
    if mask_prob > 0.0:
        mask = np.random.binomial(1, mask_prob, X_noisy.shape)
        X_noisy[mask == 1] = 0.0
    return X_noisy

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

    print("\n[Phase 1] Pre-training on TCGA-SKCM...")
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
    
    torch.manual_seed(42)
    np.random.seed(42)
    align_net, clf_net = train_adversarial_multimodal(
        X_v_tc_tr, X_r_tc_tr, X_g_tc_tr, y_tc_tr,
        a_train_sex, mask_tr_sex, len(le_adv_sex.classes_),
        a_train_age, mask_tr_age, len(le_adv_age.classes_),
        a_train_center, mask_tr_center, len(le_adv_center.classes_),
        num_classes=num_classes_tcga, class_weights=cw_tcga, epochs=60, lr=1e-3, beta=1.0, lambda_contra=0.1, cov_type='diagonal', device=device
    )
    print("Pre-training Complete.")
    
    out_dir = f"/workspace/src/FOMENT/ablation_results"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    # ---------------------------------------------------------
    # Ablation settings
    noise_scales = [0.0, 0.5, 1.0, 2.0, 3.0]
    mask_probs = [0.0, 0.25, 0.50, 0.75, 0.90]
    
    for ds_name, ds_prefix in [("AI4SKIN", "ai4skin"), ("ASSIST", "assist")]:
        print(f"\n[Phase 2] Ablation on {ds_name}...")
        X_r_tr, X_g_tr, X_v_tr, y_tr, a_tr, s_tr, _ = load_combined_fold(ds_name, f"/workspace/src/metadata/{ds_prefix}_train.csv")
        X_r_te, X_g_te, X_v_te, y_te, a_te, s_te, _ = load_combined_fold(ds_name, f"/workspace/src/metadata/{ds_prefix}_test.csv")
        
        num_classes = int(max(np.max(y_tr), np.max(y_te))) + 1
        
        # 1. Evaluate Gaussian Noise on X_g_te
        for ns in noise_scales:
            X_g_te_noisy = inject_noise(X_g_te, noise_scale=ns, mask_prob=0.0)
            s_eta_g_tr = extract_pure_fused(align_net, X_v_tr, X_g_tr, 5, device=device)
            s_eta_g_te = extract_pure_fused(align_net, X_v_te, X_g_te_noisy, 10, device=device)
            
            lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight='balanced')
            lr.fit(s_eta_g_tr, y_tr)
            probs = pad_proba_columns(lr.predict_proba(s_eta_g_te), lr.classes_, num_classes)
            preds = lr.predict(s_eta_g_te)
            
            prob_eval = probs[:, 1] if num_classes == 2 else probs
            auroc = compute_auroc(y_te, prob_eval, num_classes)
            delta_eo = calculate_delta_eo(y_te, preds, s_te)
            
            results.append({
                "Target Cohort": ds_name,
                "Ablation Type": "Gaussian Noise",
                "Intensity": ns,
                "AUROC": auroc,
                "DeltaEO (Sex)": delta_eo
            })
            print(f"  [{ds_name}] Noise scale: {ns:.2f} | AUC: {auroc:.3f} | EO(Sex): {delta_eo:.3f}")

        # 2. Evaluate Masking on X_g_te
        for mp in mask_probs:
            if mp == 0.0: continue # Already tested as noise 0.0
            X_g_te_masked = inject_noise(X_g_te, noise_scale=0.0, mask_prob=mp)
            s_eta_g_tr = extract_pure_fused(align_net, X_v_tr, X_g_tr, 5, device=device)
            s_eta_g_te = extract_pure_fused(align_net, X_v_te, X_g_te_masked, 10, device=device)
            
            lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight='balanced')
            lr.fit(s_eta_g_tr, y_tr)
            probs = pad_proba_columns(lr.predict_proba(s_eta_g_te), lr.classes_, num_classes)
            preds = lr.predict(s_eta_g_te)
            
            prob_eval = probs[:, 1] if num_classes == 2 else probs
            auroc = compute_auroc(y_te, prob_eval, num_classes)
            delta_eo = calculate_delta_eo(y_te, preds, s_te)
            
            results.append({
                "Target Cohort": ds_name,
                "Ablation Type": "Feature Masking",
                "Intensity": mp,
                "AUROC": auroc,
                "DeltaEO (Sex)": delta_eo
            })
            print(f"  [{ds_name}] Mask prob: {mp:.2f} | AUC: {auroc:.3f} | EO(Sex): {delta_eo:.3f}")

    final_df = pd.DataFrame(results)
    final_df.to_csv(f"{out_dir}/proxy_ablation.csv", index=False)
    print(f"\nSaved proxy ablation results to {out_dir}/proxy_ablation.csv")

if __name__ == "__main__":
    main()
