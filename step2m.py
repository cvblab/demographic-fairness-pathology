import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn

from multimodal_model import IntermediateConcatenationModel, IntermediateCrossAttentionModel, LateFusionModel, train_model, predict_with_uq, MultimodalDataset
from data_utils import get_combined_dataset
from evaluation_utils import (
    compute_auroc, run_demographic_probe, calculate_delta_eo,
    get_intersectional_groups, calculate_demographic_parity, calculate_equalized_odds,
    calculate_subgroup_metrics, plot_reliability_diagrams, bootstrap_fairness_metrics,
    expected_calibration_error, pad_proba_columns, compute_p_values
)
from fairness_baselines import train_reweighting, train_adversarial_debiasing, train_group_dro, predict_mlp

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class AlignmentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, cov_type='diagonal', rank=10):
        super().__init__()
        self.cov_type = cov_type
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, input_dim)
        if cov_type in ['diagonal', 'low_rank']:
            self.fc_logvar = nn.Linear(hidden_dim, input_dim)
        if cov_type == 'low_rank':
            self.fc_factor = nn.Linear(hidden_dim, input_dim * rank)
            self.rank = rank
            
    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        if self.cov_type == 'deterministic': return mu, None, None
        logvar = self.fc_logvar(h)
        if self.cov_type == 'diagonal': return mu, logvar, None
        factor = self.fc_factor(h).view(-1, mu.size(1), self.rank)
        return mu, logvar, factor

class ClassifierModel(nn.Module):
    def __init__(self, img_dim, text_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.ln_img = nn.LayerNorm(img_dim)
        self.ln_text = nn.LayerNorm(text_dim)
        self.fc1 = nn.Linear(img_dim + text_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.text_to_img_proj = nn.Linear(text_dim, img_dim) if text_dim != img_dim else nn.Identity()
    def forward(self, img, text):
        img_n = self.ln_img(img)
        text_n = self.ln_text(text)
        x = torch.cat([img_n, text_n], dim=1)
        s_eta = F.relu(self.fc1(x))
        out = self.fc2(s_eta)
        return out, s_eta

class AdversaryModel(nn.Module):
    def __init__(self, hidden_dim, num_sensitive_classes):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_sensitive_classes)
    def forward(self, s_eta, alpha=1.0):
        s_eta_rev = GradientReversalLayer.apply(s_eta, alpha)
        h = F.relu(self.fc1(s_eta_rev))
        return self.fc2(h)

def train_adversarial_multimodal(X_tr_img, X_tr_text_real, X_tr_text_syn, y_train,
                                 a_train_sex, mask_tr_sex, num_sensitive_classes_sex,
                                 a_train_age, mask_tr_age, num_sensitive_classes_age,
                                 a_train_center, mask_tr_center, num_sensitive_classes_center,
                                 num_classes, class_weights,
                                 epochs=100, lr=1e-3, beta=1.0, lambda_contra=0.1, cov_type='diagonal', device='cpu'):
    import torch.distributions as dist
    from torch.utils.data import TensorDataset, DataLoader
    img_dim = X_tr_img.shape[1]
    text_dim = X_tr_text_real.shape[1]
    
    align_net = AlignmentModel(text_dim, cov_type=cov_type).to(device)
    clf_net = ClassifierModel(img_dim, text_dim, num_classes).to(device)
    adv_net_sex = AdversaryModel(256, num_sensitive_classes_sex).to(device)
    adv_net_age = AdversaryModel(256, num_sensitive_classes_age).to(device)
    adv_net_center = AdversaryModel(256, num_sensitive_classes_center).to(device)
    
    optimizer_align = torch.optim.Adam(align_net.parameters(), lr=lr)
    optimizer_clf = torch.optim.Adam(clf_net.parameters(), lr=lr)
    optimizer_adv_sex = torch.optim.Adam(adv_net_sex.parameters(), lr=lr)
    optimizer_adv_age = torch.optim.Adam(adv_net_age.parameters(), lr=lr)
    optimizer_adv_center = torch.optim.Adam(adv_net_center.parameters(), lr=lr)
    
    dataset = TensorDataset(
        torch.tensor(X_tr_img, dtype=torch.float32),
        torch.tensor(X_tr_text_real, dtype=torch.float32),
        torch.tensor(X_tr_text_syn, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(a_train_sex, dtype=torch.long),
        torch.tensor(mask_tr_sex, dtype=torch.bool),
        torch.tensor(a_train_age, dtype=torch.long),
        torch.tensor(mask_tr_age, dtype=torch.bool),
        torch.tensor(a_train_center, dtype=torch.long),
        torch.tensor(mask_tr_center, dtype=torch.bool)
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    cw_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    total_steps = epochs * len(dataloader)
    
    for epoch in range(epochs):
        align_net.train()
        clf_net.train()
        adv_net_sex.train()
        adv_net_age.train()
        adv_net_center.train()
        
        for batch_idx, (img_t, e_R_t, e_G_t, y_t, a_t_sex, mask_t_sex, a_t_age, mask_t_age, a_t_center, mask_t_center) in enumerate(dataloader):
            img_t, e_R_t, e_G_t = img_t.to(device), e_R_t.to(device), e_G_t.to(device)
            y_t = y_t.to(device)
            a_t_sex, mask_t_sex = a_t_sex.to(device), mask_t_sex.to(device)
            a_t_age, mask_t_age = a_t_age.to(device), mask_t_age.to(device)
            a_t_center, mask_t_center = a_t_center.to(device), mask_t_center.to(device)
            
            mu, logvar, factor = align_net(e_G_t)
            
            if cov_type == 'deterministic':
                nll_loss = F.mse_loss(mu, e_R_t)
                e_tilde = mu
            elif cov_type == 'diagonal':
                logvar_clamped = torch.clamp(logvar, min=-10, max=10)
                nll_loss = 0.5 * torch.mean((e_R_t - mu)**2 / torch.exp(logvar_clamped) + logvar_clamped)
                std = torch.exp(0.5 * logvar_clamped)
                eps = torch.randn_like(std)
                e_tilde = mu + eps * std
            elif cov_type == 'low_rank':
                logvar_clamped = torch.clamp(logvar, min=-10, max=10)
                cov_diag = torch.exp(logvar_clamped)
                mvn = dist.LowRankMultivariateNormal(mu, factor, cov_diag)
                nll_loss = -mvn.log_prob(e_R_t).mean()
                e_tilde = mvn.rsample()
                
            clf_logits, s_eta = clf_net(img_t, e_tilde)
            ce_loss = F.cross_entropy(clf_logits, y_t, weight=cw_t)
            
            if lambda_contra > 0.0:
                e_tilde_proj = clf_net.text_to_img_proj(e_tilde)
                contra_loss = 1.0 - F.cosine_similarity(img_t, e_tilde_proj, dim=-1).mean()
            else:
                contra_loss = torch.tensor(0.0, device=device)
                
            global_step = epoch * len(dataloader) + batch_idx
            p = float(global_step) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            adv_logits_sex = adv_net_sex(s_eta, alpha=alpha)
            if mask_t_sex.sum() > 0: adv_loss_sex = F.cross_entropy(adv_logits_sex[mask_t_sex], a_t_sex[mask_t_sex])
            else: adv_loss_sex = torch.tensor(0.0, device=device)
            
            adv_logits_age = adv_net_age(s_eta, alpha=alpha)
            if mask_t_age.sum() > 0: adv_loss_age = F.cross_entropy(adv_logits_age[mask_t_age], a_t_age[mask_t_age])
            else: adv_loss_age = torch.tensor(0.0, device=device)
            
            adv_logits_center = adv_net_center(s_eta, alpha=alpha)
            if mask_t_center.sum() > 0: adv_loss_center = F.cross_entropy(adv_logits_center[mask_t_center], a_t_center[mask_t_center])
            else: adv_loss_center = torch.tensor(0.0, device=device)
                
            total_loss = nll_loss + ce_loss + beta * (adv_loss_sex + adv_loss_age + adv_loss_center) + lambda_contra * contra_loss
            
            optimizer_align.zero_grad()
            optimizer_clf.zero_grad()
            optimizer_adv_sex.zero_grad()
            optimizer_adv_age.zero_grad()
            optimizer_adv_center.zero_grad()
            total_loss.backward()
            optimizer_align.step()
            optimizer_clf.step()
            optimizer_adv_sex.step()
            optimizer_adv_age.step()
            optimizer_adv_center.step()
            
    align_net.eval()
    clf_net.eval()
    return align_net, clf_net

def extract_adv_fused(align, clf, X_v, X_t, num_samples=10, device='cpu'):
    img_t = torch.tensor(X_v, dtype=torch.float32).to(device)
    e_G_t = torch.tensor(X_t, dtype=torch.float32).to(device)
    mu, logvar, factor = align(e_G_t)
    
    all_probs = []
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
                
            clf_logits, s_eta = clf(img_t, e_tilde)
            probs = F.softmax(clf_logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_seta.append(s_eta.cpu().numpy())
            
    mean_probs = np.mean(all_probs, axis=0)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=1)
    expected_entropy = np.mean([-np.sum(p * np.log(p + 1e-12), axis=1) for p in all_probs], axis=0)
    mutual_info = entropy - expected_entropy
    mean_seta = np.mean(all_seta, axis=0)
    
    aggr_list = []
    for i in range(len(X_v)):
        aggr_list.append({
            'mean_probs': mean_probs[i],
            'prediction': np.argmax(mean_probs[i]),
            'entropy': entropy[i],
            'mutual_info': mutual_info[i],
            'fused_emb': mean_seta[i]
        })
    return mean_probs, mean_seta, aggr_list

def robust_compute_class_weight(y, num_classes):
    unique_y = np.unique(y)
    cw = compute_class_weight('balanced', classes=unique_y, y=y)
    cw_dict = dict(zip(unique_y, cw))
    return np.array([cw_dict.get(i, 1.0) for i in range(num_classes)])

def run_evaluation(dataset_name, train_csv, val_csv, test_csv, device, tokenizer, model, ai4skin_json, assist_json, tcga_df=None):
    print(f"\n{'='*50}\nStarting Evaluation for {dataset_name}\n{'='*50}")
    
    out_dir = f"/workspace/src/FOMENT/step2_results_{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    X_real_tr, X_gen_tr, X_vis_tr, y_tr, age_tr, sex_tr, center_tr = get_combined_dataset(
        [train_csv], device, tokenizer, model, ai4skin_json, assist_json, tcga_df, dataset_filter=dataset_name)
    X_real_va, X_gen_va, X_vis_va, y_va, age_va, sex_va, center_va = get_combined_dataset(
        [val_csv], device, tokenizer, model, ai4skin_json, assist_json, tcga_df, dataset_filter=dataset_name)
    X_real_te, X_gen_te, X_vis_te, y_te, age_te, sex_te, center_te = get_combined_dataset(
        [test_csv], device, tokenizer, model, ai4skin_json, assist_json, tcga_df, dataset_filter=dataset_name)
        
    if len(y_tr) == 0 or len(y_te) == 0:
        print(f"Skipping {dataset_name} due to lack of data.")
        return


    
    # Label Encoding for multiclass
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # Fit on all possible labels to avoid unseen labels
    all_y = np.concatenate([y_tr, y_va, y_te])
    le.fit(all_y)
    
    y_tr = le.transform(y_tr)
    y_va = le.transform(y_va)
    y_te = le.transform(y_te)
    
    # Save the encoder for HUSC later
    trained_models = {'label_encoder': le}

    num_classes = len(le.classes_)

    inter_tr = get_intersectional_groups(age_tr, sex_tr)
    inter_te = get_intersectional_groups(age_te, sex_te)
    
    age_te_discrete = np.array(['Unknown' if pd.isna(a) else ('Older' if a >= 60 else 'Younger') for a in age_te])
    
    print(f"\n[Test Set N Sizes for {dataset_name}]")
    print("Sex:\n", pd.Series(sex_te).value_counts().to_string())
    print("Age:\n", pd.Series(age_te).value_counts().to_string())
    
    results = []
    y_prob_dict = {}
    y_pred_dict = {}
    emb_dict = {}
    
    def add_evaluation(name, probs, preds, fused_emb_tr, fused_emb_te):
        if num_classes == 2 and probs.ndim == 2:
            prob_eval = probs[:, 1]
        else:
            prob_eval = probs
            
        auroc = compute_auroc(y_te, prob_eval, num_classes)
        eo = calculate_equalized_odds(y_te, preds, inter_te)
        dp = calculate_demographic_parity(preds, inter_te)
        delta_eo = calculate_delta_eo(y_te, preds, inter_te)
        
        sub_metrics = calculate_subgroup_metrics(y_te, probs, preds, inter_te, num_classes)
        
        y_prob_dict[name] = probs
        y_pred_dict[name] = preds
        emb_dict[name] = (fused_emb_tr, fused_emb_te)
        
        f1 = f1_score(y_te, preds, average='macro' if num_classes > 2 else 'binary')
        bal_acc = balanced_accuracy_score(y_te, preds)
        
        results.append({
            "Model": name,
            "AUROC": auroc,
            "Bal Acc": bal_acc,
            "F1-Score": f1,
            "Worst AUC": sub_metrics['worst_auc'],
            "Macro AUC": sub_metrics['macro_auc'],
            "Eq. Odds (Sex)": calculate_equalized_odds(y_te, preds, sex_te),
            "Dem. Parity (Sex)": calculate_demographic_parity(preds, sex_te),
            "Delta EO (Sex)": calculate_delta_eo(y_te, preds, sex_te),
            "Delta EO (Age)": calculate_delta_eo(y_te, preds, age_te_discrete)
        })
        print(f"{name:30} | AUC: {auroc:.3f} | Worst AUC: {sub_metrics['worst_auc']:.3f} | Eq. Odds: {eo:.3f}")

    # keep the label encoder if it exists
    if 'trained_models' not in locals():
        trained_models = {}
    print("\n--- Evaluating Linear Baselines (Vision & Text) ---")
    for name, X_tr, X_te in [("Vision-Only", X_vis_tr, X_vis_te), 
                             ("Text-Only (Real)", X_real_tr, X_real_te), 
                             ("Text-Only (Gen)", X_gen_tr, X_gen_te),
                             ("Multimodal-LR (Real+Vis)", np.hstack((X_vis_tr, X_real_tr)), np.hstack((X_vis_te, X_real_te))),
                             ("Multimodal-LR (Gen+Vis)", np.hstack((X_vis_tr, X_gen_tr)), np.hstack((X_vis_te, X_gen_te)))]:
        lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight='balanced')
        lr.fit(X_tr, y_tr)
        trained_models[name] = lr
        probs = pad_proba_columns(lr.predict_proba(X_te), lr.classes_, num_classes)
        preds = lr.predict(X_te)
        add_evaluation(name, probs, preds, X_tr, X_te)

    print("\n--- Evaluating Fairness Baselines (Vision) ---")
    # Reweighting is a linear model (no nonlinear hidden layer), so its
    # representation-leakage probe necessarily uses the raw vision input X_vis;
    # this is reported as such and is NOT expected to differ from Vision-Only's
    # leakage numbers, since reweighting only reweights the *loss*, not the
    # representation. Adv Debiasing and GroupDRO, in contrast, learn a nonlinear
    # penultimate representation, which IS what should be probed for leakage.
    lr_rew = train_reweighting(X_vis_tr, y_tr, inter_tr, num_classes)
    probs_rew = pad_proba_columns(lr_rew.predict_proba(X_vis_te), lr_rew.classes_, num_classes)
    add_evaluation("Reweighting", probs_rew, lr_rew.predict(X_vis_te), X_vis_tr, X_vis_te)
    
    def get_ensemble_mlp(train_fn, X_tr_input, X_te_input):
        all_probs_te, all_probs_tr, all_h_te, all_h_tr = [], [], [], []
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            mlp = train_fn()
            probs, h_te = predict_mlp(mlp, X_te_input, return_embeddings=True)
            probs_tr, h_tr = predict_mlp(mlp, X_tr_input, return_embeddings=True)
            all_probs_te.append(probs)
            all_probs_tr.append(probs_tr)
            all_h_te.append(h_te)
            all_h_tr.append(h_tr)
        return np.mean(all_probs_te, axis=0), np.mean(all_h_te, axis=0), np.mean(all_h_tr, axis=0)
        
    probs_adv, h_adv_te, h_adv_tr = get_ensemble_mlp(
        lambda: train_adversarial_debiasing(X_vis_tr, y_tr, inter_tr, num_classes, X_val=X_vis_va, y_val=y_va),
        X_vis_tr, X_vis_te
    )
    add_evaluation("Adv Debiasing (Vision)", probs_adv, np.argmax(probs_adv, axis=1), h_adv_tr, h_adv_te)
    
    probs_dro, h_dro_te, h_dro_tr = get_ensemble_mlp(
        lambda: train_group_dro(X_vis_tr, y_tr, inter_tr, num_classes, X_val=X_vis_va, y_val=y_va),
        X_vis_tr, X_vis_te
    )
    add_evaluation("GroupDRO (Vision)", probs_dro, np.argmax(probs_dro, axis=1), h_dro_tr, h_dro_te)

    # Multimodal Fairness Baselines
    X_mm_tr, X_mm_va, X_mm_te = np.hstack((X_vis_tr, X_real_tr)), np.hstack((X_vis_va, X_real_va)), np.hstack((X_vis_te, X_real_te))
    
    probs_adv_mm, h_adv_te_mm, h_adv_tr_mm = get_ensemble_mlp(
        lambda: train_adversarial_debiasing(X_mm_tr, y_tr, inter_tr, num_classes, X_val=X_mm_va, y_val=y_va),
        X_mm_tr, X_mm_te
    )
    add_evaluation("Adv Debiasing (Multimodal)", probs_adv_mm, np.argmax(probs_adv_mm, axis=1), h_adv_tr_mm, h_adv_te_mm)
    
    # Store deterministic uncertainty for Adv Debiasing (Multimodal)
    uq_results = {}
    adv_mm_entropy = -np.sum(probs_adv_mm * np.log(probs_adv_mm + 1e-12), axis=1)
    uq_results["Adv Debiasing (Multimodal)"] = [
        {"prediction": np.argmax(probs_adv_mm[i]), "label": y_te[i], "entropy": adv_mm_entropy[i], "mutual_info": 0.0}
        for i in range(len(y_te))
    ]
    
    probs_dro_mm, h_dro_te_mm, h_dro_tr_mm = get_ensemble_mlp(
        lambda: train_group_dro(X_mm_tr, y_tr, inter_tr, num_classes, X_val=X_mm_va, y_val=y_va),
        X_mm_tr, X_mm_te
    )
    add_evaluation("GroupDRO (Multimodal)", probs_dro_mm, np.argmax(probs_dro_mm, axis=1), h_dro_tr_mm, h_dro_te_mm)

    print("\n--- Evaluating Multimodal Models ---")
    
    for text_type, X_txt_tr, X_txt_va, X_txt_te in [("Real", X_real_tr, X_real_va, X_real_te), ("Gen", X_gen_tr, X_gen_va, X_gen_te)]:
        train_dataset = MultimodalDataset(X_txt_tr, X_vis_tr, y_tr, None)
        val_dataset = MultimodalDataset(X_txt_va, X_vis_va, y_va, None)
        test_dataset = MultimodalDataset(X_txt_te, X_vis_te, y_te, None)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        text_dim = X_txt_tr.shape[1]
        image_dim = X_vis_tr.shape[1]
        
        for mm_name, MMClass in [
            (f"Multimodal Concat ({text_type})", IntermediateConcatenationModel),
            (f"Multimodal CrossAttn ({text_type})", IntermediateCrossAttentionModel),
            (f"Multimodal LateFusion ({text_type})", LateFusionModel)
        ]:
            te_preds_list = []
            tr_preds_list = []
            for seed in range(5):
                torch.manual_seed(seed)
                np.random.seed(seed)
                cw_mm = robust_compute_class_weight(y_tr, num_classes)
                mm_model = MMClass(text_dim=text_dim, image_dim=image_dim, num_classes=num_classes)
                mm_model = train_model(mm_model, train_loader, val_dataloader=val_loader, epochs=50, device=device, class_weights=cw_mm)
                te_preds = predict_with_uq(mm_model, test_loader, num_mc_samples=30, device=device)
                tr_preds = predict_with_uq(mm_model, DataLoader(train_dataset, batch_size=32), num_mc_samples=5, device=device)
                te_preds_list.append(te_preds)
                tr_preds_list.append(tr_preds)
                
            def aggregate_preds(preds_list):
                aggr = []
                for i in range(len(preds_list[0])):
                    res = {}
                    for k in preds_list[0][i].keys():
                        if k in ['mean_probs', 'entropy', 'mutual_info', 'variance', 'fused_emb']:
                            res[k] = np.mean([p[i][k] for p in preds_list], axis=0)
                        elif k == 'prediction':
                            res[k] = np.argmax(np.mean([p[i]['mean_probs'] for p in preds_list], axis=0))
                        else:
                            res[k] = preds_list[0][i][k]
                    aggr.append(res)
                return aggr
                
            te_preds_aggr = aggregate_preds(te_preds_list)
            tr_preds_aggr = aggregate_preds(tr_preds_list)
            
            probs = np.array([p['mean_probs'] for p in te_preds_aggr])
            preds = np.array([p['prediction'] for p in te_preds_aggr])
            X_fused_te = np.array([p['fused_emb'] for p in te_preds_aggr])
            X_fused_tr = np.array([p['fused_emb'] for p in tr_preds_aggr])
            
            uq_results[mm_name] = te_preds_aggr
            add_evaluation(mm_name, probs, preds, X_fused_tr, X_fused_te)

    print("\n--- Evaluating Proposal Covariance Ablations ---")
    # Form intersectional group of sex, age, and center for the adversary
    # We will just map age back to age_groups first for intersection
    def get_age_group(a):
        if pd.isna(a): return 'Unknown'
        elif a < 40: return 'Young'
        elif a > 60: return 'Old'
        else: return 'Middle'
    
    adv_group_sex_tr = list(sex_tr)
    adv_group_age_tr = [get_age_group(a) for a in age_tr]
    adv_group_center_tr = list(center_tr)
    
    le_adv_sex = LabelEncoder()
    le_adv_age = LabelEncoder()
    le_adv_center = LabelEncoder()
    from collections import Counter
    group_counts_sex = Counter(adv_group_sex_tr)
    group_counts_age = Counter(adv_group_age_tr)
    group_counts_center = Counter(adv_group_center_tr)
    
    # Only keep groups with at least 5 samples to prevent adversarial memorization
    valid_adv_sex_tr = [i for i, g in enumerate(adv_group_sex_tr) if 'Unknown' not in g and 'unknown' not in str(g) and 'N/A' not in str(g) and group_counts_sex[g] >= 5]
    valid_adv_age_tr = [i for i, g in enumerate(adv_group_age_tr) if 'Unknown' not in g and 'unknown' not in str(g) and 'N/A' not in str(g) and group_counts_age[g] >= 5]
    valid_adv_center_tr = [i for i, g in enumerate(adv_group_center_tr) if 'Unknown' not in g and 'unknown' not in str(g) and 'N/A' not in str(g) and group_counts_center[g] >= 5]
    
    if len(valid_adv_sex_tr) > 0 and len(valid_adv_age_tr) > 0 and len(valid_adv_center_tr) > 0:
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
        
        cw = robust_compute_class_weight(y_tr, num_classes)
        
        for cov in ['deterministic', 'diagonal', 'low_rank']:
            all_probs_te, all_probs_tr, all_h_te, all_h_tr = [], [], [], []
            all_te_preds_adv = []
            adv_models = []
            for seed in range(5):
                torch.manual_seed(seed)
                np.random.seed(seed)
                align_net, clf_net = train_adversarial_multimodal(
                    X_vis_tr, X_real_tr, X_gen_tr, y_tr,
                    a_train_sex, mask_tr_sex, len(le_adv_sex.classes_),
                    a_train_age, mask_tr_age, len(le_adv_age.classes_),
                    a_train_center, mask_tr_center, len(le_adv_center.classes_),
                    num_classes=num_classes, class_weights=cw, epochs=100, lr=1e-3, beta=1.0, lambda_contra=0.1, cov_type=cov, device=device
                )
                adv_models.append((align_net, clf_net))
                mean_probs_tr, X_fused_tr_adv, aggr_tr_list = extract_adv_fused(align_net, clf_net, X_vis_tr, X_gen_tr, 5, device=device)
                mean_probs_te, X_fused_te_adv, aggr_te_list = extract_adv_fused(align_net, clf_net, X_vis_te, X_gen_te, 30, device=device)
                
                all_probs_te.append(mean_probs_te)
                all_probs_tr.append(mean_probs_tr)
                all_h_te.append(X_fused_te_adv)
                all_h_tr.append(X_fused_tr_adv)
                all_te_preds_adv.append(aggr_te_list)
                
            aggr_probs_te = np.mean(all_probs_te, axis=0)
            aggr_probs_tr = np.mean(all_probs_tr, axis=0)
            aggr_h_te = np.mean(all_h_te, axis=0)
            aggr_h_tr = np.mean(all_h_tr, axis=0)
            
            # Aggregate uncertainty metrics across seeds
            aggr_entropy = np.mean([[p['entropy'] for p in seed_preds] for seed_preds in all_te_preds_adv], axis=0)
            aggr_mi = np.mean([[p['mutual_info'] for p in seed_preds] for seed_preds in all_te_preds_adv], axis=0)
            
            preds_adv = np.argmax(aggr_probs_te, axis=1)
            
            model_name = f"Adv Align ({cov})"
            add_evaluation(model_name, aggr_probs_te, preds_adv, aggr_h_tr, aggr_h_te)
            trained_models[model_name] = adv_models
            
            uq_results[model_name] = [
                {"prediction": preds_adv[i], "label": y_te[i], "entropy": aggr_entropy[i], "mutual_info": aggr_mi[i]}
                for i in range(len(y_te))
            ]
            
    print("\n--- Bootstrapping Fairness Metrics (N=1000) ---")
    ci_results, boot_results = bootstrap_fairness_metrics(y_te, y_prob_dict, y_pred_dict, num_classes, n_iterations=1000, sex_attr=sex_te, age_attr=age_te_discrete)
    
    print("\n--- Computing p-values vs Vision-Only ---")
    p_values = compute_p_values(boot_results, baseline_model="Vision-Only")
    if p_values:
        df_p_values = pd.DataFrame(p_values).T
        df_p_values.index.name = "Model"
        df_p_values.to_csv(f"{out_dir}/p_values_vs_Vision-Only.csv")

    # Compile Final Results
    final_df = pd.DataFrame(results)
    
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
            s = f"\\textbf{{{s}}}"
        stars = get_stars(p_val)
        if stars: s += f" {stars}"
        return s

    best_vals = {col: (final_df[col].max() if col in ['AUROC', 'Worst AUC', 'Macro AUC'] else final_df[col].min()) for col in final_df.columns if col != 'Model'}
    
    final_formatted = []
    for i, row in final_df.iterrows():
        m_name = row['Model']
        st = ci_results.get(m_name, {})
        pv = p_values.get(m_name, {})
        
        formatted_row = {"Model": m_name}
        for col, col_key in [('AUROC', 'auroc'), ('Bal Acc', 'bacc'), ('F1-Score', 'f1'), ('Worst AUC', 'worst_auc'), ('Macro AUC', 'macro_auc')]:
            val = row[col]
            if pd.isna(val): formatted_row[col] = "NaN"
            else:
                formatted_row[col] = fmt_with_p(val, st.get(col_key, [None, None])[0], st.get(col_key, [None, None])[1], best_vals[col], pv.get(col_key, np.nan), True)
                
        for col, col_key in [('Eq. Odds (Sex)', 'equalized_odds_sex'), ('Dem. Parity (Sex)', 'demographic_parity_sex'), ('Delta EO (Sex)', 'delta_eo_sex'), ('Delta EO (Age)', 'delta_eo_age')]:
            val = row[col]
            if pd.isna(val): formatted_row[col] = "NaN"
            else:
                ci_l = st.get(col_key, [None, None])[0] if col_key else None
                ci_u = st.get(col_key, [None, None])[1] if col_key else None
                p_v = pv.get(col_key, np.nan) if col_key else np.nan
                formatted_row[col] = fmt_with_p(val, ci_l, ci_u, best_vals[col], p_v, False)
                
        final_formatted.append(formatted_row)

    # We save the unformatted CSV
    final_df.to_csv(f"{out_dir}/metrics.csv", index=False)
    
    df_formatted = pd.DataFrame(final_formatted)
    print(df_formatted.to_markdown(index=False))
    
    with open(os.path.join(out_dir, "results.tex"), "w") as f:
        f.write("\\begin{table}[h!]\\centering\\Large\n")
        f.write(df_formatted.to_latex(index=False))
        f.write("\\end{table}")
    
    print("\n--- Representation Leakage Analysis ---")
    leakage_res = []
    for name, (X_tr, X_te) in emb_dict.items():
        leak_sex = run_demographic_probe(X_tr, X_te, sex_tr, sex_te)
        leak_age = run_demographic_probe(X_tr, X_te, age_tr, age_te)
        leak_inter = run_demographic_probe(X_tr, X_te, inter_tr, inter_te)
        leakage_res.append({"Model": name, "Probe Sex": leak_sex, "Probe Age": leak_age, "Probe Intersectional": leak_inter})
    leak_df = pd.DataFrame(leakage_res)
    leak_df.to_csv(f"{out_dir}/leakage.csv", index=False)
    print(leak_df.to_markdown(index=False))

    print("\n--- Uncertainty Analysis ---")
    unc_res = []
    for mm_name, preds in uq_results.items():
        entropies = np.array([p['entropy'] for p in preds])
        mis = np.array([p['mutual_info'] for p in preds])
        correct = np.array([p['prediction'] == p['label'] for p in preds])
        
        unc_res.append({
            "Model": mm_name,
            "Correct Entropy": entropies[correct].mean() if correct.sum() > 0 else np.nan,
            "Incorrect Entropy": entropies[~correct].mean() if (~correct).sum() > 0 else np.nan,
            "Correct MI": mis[correct].mean() if correct.sum() > 0 else np.nan,
            "Incorrect MI": mis[~correct].mean() if (~correct).sum() > 0 else np.nan
        })
    unc_df = pd.DataFrame(unc_res)
    unc_df.to_csv(f"{out_dir}/uncertainty.csv", index=False)
    unc_df_round = unc_df.round(3)
    with open(f"{out_dir}/uncertainty.tex", "w") as f:
        f.write("\\begin{table}[h!]\\centering\\Large\n")
        f.write(unc_df_round.to_latex(index=False))
        f.write("\\end{table}")
    print(unc_df_round.to_markdown(index=False))

    # Plotting
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=final_df, x="AUROC", y="Delta EO (Sex)", hue="Model", s=300, edgecolor='black', palette='tab20')
    plt.title(f"{dataset_name}: Fairness-Accuracy Trade-off")
    plt.xlabel("Pathology AUROC (Higher is Better)")
    plt.ylabel(r"$\Delta$EO (Sex) (Lower is Better)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/tradeoff.png", dpi=300)
    plt.close()

    plot_reliability_diagrams(y_te, y_prob_dict, inter_te, dataset_name, out_dir)
    from evaluation_utils import plot_media_figures
    plot_media_figures(y_te, y_prob_dict, emb_dict, inter_te, dataset_name, out_dir, num_classes)
    
    # === OOD HUSC EVALUATION ===
    if dataset_name in ['AI4SKIN', 'ASSIST']:
        print(f"\nEvaluating OOD Generalization on HUSC ({dataset_name})")
        ds_base = dataset_name.replace('_BINARY', '')
        df_tr = pd.read_csv(f"/workspace/src/metadata/with_HUSC/{ds_base.lower()}_train.csv")
        df_va = pd.read_csv(f"/workspace/src/metadata/with_HUSC/{ds_base.lower()}_val.csv")
        df_te = pd.read_csv(f"/workspace/src/metadata/with_HUSC/{ds_base.lower()}_test.csv")
        df_all = pd.concat([df_tr, df_va, df_te])
        df_husc = df_all[df_all['center'] == 'HUSC'].copy()
        if ds_base == 'ASSIST':
            df_husc['WSI'] = df_husc['WSI'].apply(lambda x: x.split('_')[0] if str(x).startswith('MC') else x)
            
        print(f"DEBUG: df_husc length from splits: {len(df_husc)}")
        if len(df_husc) > 0:
            tmp_csv = f"/workspace/src/metadata/tmp_husc_{ds_base}.csv"
            df_husc.to_csv(tmp_csv, index=False)
            
            _, X_gen_husc, X_vis_husc, y_husc, _, _, _ = get_combined_dataset(
                [tmp_csv], device, tokenizer, model, ai4skin_json, assist_json, tcga_df, dataset_filter=dataset_name)
                
            print(f"DEBUG: len(y_husc) from get_combined_dataset is {len(y_husc)}")
            if len(y_husc) > 0:
                le = trained_models.get('label_encoder', None)
                if le is not None:
                    # Handle unseen classes gracefully
                    known_mask = np.isin(y_husc, le.classes_)
                    y_husc[~known_mask] = le.classes_[0] # Dummy for unknown to prevent crash
                    y_husc = le.transform(y_husc)
                
                husc_results = []
                
                if 'Vision-Only' in trained_models:
                    clf = trained_models['Vision-Only']
                    y_prob = clf.predict_proba(X_vis_husc)
                    y_pred = clf.predict(X_vis_husc)
                    auc = compute_auroc(y_husc, y_prob, num_classes)
                    f1 = f1_score(y_husc, y_pred, average='macro' if num_classes > 2 else 'binary')
                    b_acc = balanced_accuracy_score(y_husc, y_pred)
                    husc_results.append({'Model': 'Vision-Only', 'AUROC': auc, 'F1-Score': f1, 'Bal Acc': b_acc})
                    
                if 'Text-Only (Gen)' in trained_models:
                    clf = trained_models['Text-Only (Gen)']
                    y_prob = clf.predict_proba(X_gen_husc)
                    y_pred = clf.predict(X_gen_husc)
                    auc = compute_auroc(y_husc, y_prob, num_classes)
                    f1 = f1_score(y_husc, y_pred, average='macro' if num_classes > 2 else 'binary')
                    b_acc = balanced_accuracy_score(y_husc, y_pred)
                    husc_results.append({'Model': 'Text-Only (Gen)', 'AUROC': auc, 'F1-Score': f1, 'Bal Acc': b_acc})
                    
                for align_variant in ['Adv Align (deterministic)', 'Adv Align (diagonal)', 'Adv Align (low_rank)']:
                    try:
                        if align_variant in trained_models:
                            all_probs_husc = []
                            for align_net, clf_net in trained_models[align_variant]:
                                mean_probs_husc, _, _ = extract_adv_fused(align_net, clf_net, X_vis_husc, X_gen_husc, 30, device=device)
                                all_probs_husc.append(mean_probs_husc)
                                
                            aggr_probs_husc = np.mean(all_probs_husc, axis=0)
                            y_pred = np.argmax(aggr_probs_husc, axis=1)
                            auc = compute_auroc(y_husc, aggr_probs_husc, num_classes)
                            f1 = f1_score(y_husc, y_pred, average='macro' if num_classes > 2 else 'binary')
                            b_acc = balanced_accuracy_score(y_husc, y_pred)
                            husc_results.append({'Model': align_variant, 'AUROC': auc, 'F1-Score': f1, 'Bal Acc': b_acc})
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"Could not evaluate {align_variant} on HUSC:", e)
                    
                if husc_results:
                    husc_df = pd.DataFrame(husc_results)
                    husc_df.to_csv(f"{out_dir}/ood_husc_metrics.csv", index=False)
                    print(f"Saved OOD HUSC results to {out_dir}/ood_husc_metrics.csv")
                    print(husc_df.to_markdown(index=False))

    print(f"\nSaved all artifacts to {out_dir}")

def main():
    print("Loading demographic metadata...")
    with open("/workspace/src/metadata/raw/AI4SKIN.json", "r") as f: ai4skin_json = json.load(f)
    with open("/workspace/src/metadata/raw/ASSIST.json", "r") as f: assist_json = json.load(f)
    tcga_df = pd.read_csv('/workspace/src/metadata/raw/tcga_reports_cleaned.csv', usecols=['project_id', 'slide_id', 'sex'])
        
    print("Initializing PathologyBERT model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("tsantos/PathologyBERT")
    model = AutoModel.from_pretrained("tsantos/PathologyBERT").to(device)
    model.eval()

    #run_evaluation("AI4SKIN", "/workspace/src/metadata/ai4skin_train.csv", "/workspace/src/metadata/ai4skin_val.csv", "/workspace/src/metadata/ai4skin_test.csv", device, tokenizer, model, ai4skin_json, assist_json, tcga_df)
    run_evaluation("ASSIST", "/workspace/src/metadata/assist_train.csv", "/workspace/src/metadata/assist_val.csv", "/workspace/src/metadata/assist_test.csv", device, tokenizer, model, ai4skin_json, assist_json, tcga_df)
    #run_evaluation("TCGA-SKCM", "/workspace/src/metadata/tcga_train.csv", "/workspace/src/metadata/tcga_val.csv", "/workspace/src/metadata/tcga_test.csv", device, tokenizer, model, ai4skin_json, assist_json, tcga_df)

if __name__ == "__main__":
    main()
