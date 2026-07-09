import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def get_intersectional_groups(age, sex, median_age=60):
    """Combines Age and Sex into intersectional subgroups."""
    groups = []
    for a, s in zip(age, sex):
        if pd.isna(a) or s in ['Unknown', 'unknown', 'N/A']:
            groups.append('Unknown')
        else:
            age_group = 'Older' if a >= median_age else 'Younger'
            groups.append(f"{s}_{age_group}")
    return np.array(groups)

def pad_proba_columns(y_prob, present_classes, num_classes):
    """Pads a (N, K) predict_proba matrix to (N, num_classes), placing each
    column from `present_classes` (e.g. clf.classes_) at its correct global
    class index and filling absent classes with zero probability.

    This is required because sklearn classifiers (and our MLP/torch models
    restricted to observed labels) only emit columns for classes seen in
    their training fold, while num_classes is computed globally over the
    full train+test label range. Without this padding, any downstream code
    that assumes y_prob.shape[1] == num_classes (e.g. one-hot Brier score
    computation, or naive column indexing) will silently misalign columns
    or raise a broadcasting error whenever a class is absent from a given
    training fold or subgroup slice.
    """
    y_prob = np.asarray(y_prob)
    if y_prob.shape[1] == num_classes:
        return y_prob
    padded = np.zeros((y_prob.shape[0], num_classes), dtype=y_prob.dtype)
    for j, c in enumerate(present_classes):
        c = int(c)
        if 0 <= c < num_classes:
            padded[:, c] = y_prob[:, j]
    return padded

def compute_auroc(y_true, y_prob, num_classes):
    """Computes the AUROC for binary or multiclass classification.
    
    Assumes y_prob has exactly num_classes columns (pad with pad_proba_columns
    upstream if the classifier was trained on a subset of classes).
    """
    if num_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 and y_prob.shape[1] > 1 else y_prob)
    else:
        aucs = []
        for c in range(num_classes):
            y_true_c = (np.array(y_true).astype(str) == str(c)).astype(int)
            if len(np.unique(y_true_c)) == 2 and c < y_prob.shape[1]:
                aucs.append(roc_auc_score(y_true_c, y_prob[:, c]))
        return np.mean(aucs) if aucs else np.nan

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Computes the Expected Calibration Error (ECE)."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def run_demographic_probe(X_train, X_test, attr_train, attr_test):
    """Trains a logistic regression model to predict a demographic attribute from embeddings."""
    from sklearn.linear_model import LogisticRegression
    valid_train = [i for i, a in enumerate(attr_train) if a not in ['Unknown', 'unknown', 'N/A'] and not pd.isna(a)]
    valid_test = [i for i, a in enumerate(attr_test) if a not in ['Unknown', 'unknown', 'N/A'] and not pd.isna(a)]
    
    if len(valid_train) == 0 or len(valid_test) == 0:
        return np.nan
        
    le = LabelEncoder()
    le.fit(attr_train[valid_train])
    y_tr = le.transform(attr_train[valid_train])
    
    valid_test = [i for i in valid_test if attr_test[i] in le.classes_]
    if len(valid_test) == 0: return np.nan
    y_te = le.transform(attr_test[valid_test])
    
    if len(np.unique(y_tr)) < 2: return np.nan
    
    lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight='balanced')
    lr.fit(X_train[valid_train], y_tr)
    y_prob = lr.predict_proba(X_test[valid_test])
    
    num_classes = len(np.unique(y_tr))
    try:
        if num_classes == 2:
            return roc_auc_score(y_te, y_prob[:, 1])
        else:
            aucs = []
            for c in range(num_classes):
                y_te_c = (y_te == c).astype(int)
                if len(np.unique(y_te_c)) == 2:
                    aucs.append(roc_auc_score(y_te_c, y_prob[:, c]))
            return np.mean(aucs) if aucs else np.nan
    except ValueError:
        return np.nan

def calculate_demographic_parity(y_pred, sensitive_attr):
    """Calculates Demographic Parity Difference (max diff in positive prediction rate)."""
    df = pd.DataFrame({'y_pred': y_pred, 'sensitive': sensitive_attr})
    rates = []
    for group in df['sensitive'].unique():
        if group == 'Unknown': continue
        group_df = df[df['sensitive'] == group]
        if not group_df.empty:
            rates.append((group_df['y_pred'] > 0).mean())
    if len(rates) >= 2:
        return max(rates) - min(rates)
    return 0.0

def calculate_equalized_odds(y_true, y_pred, sensitive_attr):
    """Calculates Equalized Odds Difference (max diff across groups for TPR and FPR)."""
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sensitive': sensitive_attr})
    classes = np.unique(y_true)
    eo_diffs = []
    
    for c in classes:
        tprs, fprs = [], []
        for group in df['sensitive'].unique():
            if group == 'Unknown': continue
            group_df = df[df['sensitive'] == group]
            pos_df = group_df[group_df['y_true'] == c]
            neg_df = group_df[group_df['y_true'] != c]
            
            tpr = (pos_df['y_pred'] == c).mean() if not pos_df.empty else np.nan
            fpr = (neg_df['y_pred'] == c).mean() if not neg_df.empty else np.nan
            
            tprs.append(tpr)
            fprs.append(fpr)
            
        tprs = [x for x in tprs if not np.isnan(x)]
        fprs = [x for x in fprs if not np.isnan(x)]
        if len(tprs) >= 2 and len(fprs) >= 2:
            eo_diffs.append(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))
            
    return np.mean(eo_diffs) if eo_diffs else 0.0

def calculate_delta_eo(y_true, y_pred, sensitive_attr):
    """Calculates Equal Opportunity Difference (Delta EO)."""
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sensitive': sensitive_attr})
    classes = np.unique(y_true)
    delta_eos = []
    
    for c in classes:
        tprs = []
        for group in df['sensitive'].unique():
            if group == 'Unknown': continue
            group_df = df[df['sensitive'] == group]
            pos_df = group_df[group_df['y_true'] == c]
            tpr = (pos_df['y_pred'] == c).mean() if not pos_df.empty else np.nan
            tprs.append(tpr)
            
        tprs = [x for x in tprs if not np.isnan(x)]
        if len(tprs) >= 2:
            delta_eos.append(max(tprs) - min(tprs))
            
    return np.mean(delta_eos) if delta_eos else 0.0

def calculate_subgroup_metrics(y_true, y_prob, y_pred, sensitive_attr, num_classes):
    """Calculates AUC, ECE, and Brier Score per subgroup, returning Worst-group and Macro AUC."""
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sensitive': sensitive_attr})
    groups = [g for g in df['sensitive'].unique() if g != 'Unknown']
    
    subgroup_aucs = []
    subgroup_eces = {}
    subgroup_briers = {}
    
    for group in groups:
        idx = (sensitive_attr == group)
        if sum(idx) == 0 or len(np.unique(y_true[idx])) < 2:
            continue
            
        try:
            auc = compute_auroc(y_true[idx], y_prob[idx], num_classes)
            subgroup_aucs.append(auc)
        except ValueError:
            pass
            
        subgroup_eces[group] = expected_calibration_error(y_true[idx], y_prob[idx])
        
        # Brier score computation
        if num_classes == 2:
            prob_1 = y_prob[idx, 1] if y_prob.shape[1] > 1 else y_prob[idx]
            subgroup_briers[group] = brier_score_loss(y_true[idx], prob_1)
        else:
            y_true_onehot = np.eye(num_classes)[y_true[idx]]
            subgroup_briers[group] = np.mean((y_prob[idx] - y_true_onehot)**2)
            
    worst_auc = min(subgroup_aucs) if subgroup_aucs else np.nan
    macro_auc = np.mean(subgroup_aucs) if subgroup_aucs else np.nan
    
    return {
        'worst_auc': worst_auc,
        'macro_auc': macro_auc,
        'subgroup_eces': subgroup_eces,
        'subgroup_briers': subgroup_briers
    }

def plot_reliability_diagrams(y_true, y_prob_dict, sensitive_attr, dataset_name, out_dir, n_bins=10):
    """Plots calibration curves per subgroup for different models."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    groups = [g for g in np.unique(sensitive_attr) if g != 'Unknown']
    num_models = len(y_prob_dict)
    
    fig, axes = plt.subplots(num_models, len(groups), figsize=(4 * len(groups), 4 * num_models), squeeze=False)
    
    for i, (model_name, y_prob) in enumerate(y_prob_dict.items()):
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            y_prob_plot = np.max(y_prob, axis=1) # Use max prob for multi-class calibration
            predictions = np.argmax(y_prob, axis=1)
        else:
            y_prob_plot = y_prob
            predictions = (y_prob > 0.5).astype(int)
            
        for j, group in enumerate(groups):
            idx = (sensitive_attr == group)
            if sum(idx) == 0:
                continue
                
            prob_g = y_prob_plot[idx]
            true_g = (predictions[idx] == y_true[idx]).astype(int)
            
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_centers = []
            bin_accuracies = []
            
            for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
                in_bin = (prob_g > bin_lower) & (prob_g <= bin_upper)
                if np.sum(in_bin) > 0:
                    bin_centers.append(np.mean(prob_g[in_bin]))
                    bin_accuracies.append(np.mean(true_g[in_bin]))
                    
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.plot(bin_centers, bin_accuracies, marker='o', label='Model')
            ax.set_title(f"{model_name}\n{group}")
            ax.set_xlabel("Mean Predicted Confidence")
            ax.set_ylabel("Fraction of Positives / Accuracy")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.5)
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_reliability_diagrams.png"), dpi=300)
    plt.close()


def bootstrap_fairness_metrics(y_true, y_prob_dict, y_pred_dict, num_classes, n_iterations=1000, sex_attr=None, age_attr=None):
    """Bootstraps fairness metrics for statistical significance testing."""
    np.random.seed(42)
    n_samples = len(y_true)
    
    boot_results = {model: {
        'auroc': [], 'bacc': [], 'f1': [], 'ece': [],
        'worst_auc': [], 'macro_auc': [],
        'delta_eo_sex': [], 'delta_eo_age': [],
        'demographic_parity_sex': [], 'equalized_odds_sex': []
    } for model in y_prob_dict.keys()}
    
    for _ in range(n_iterations):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        sex_boot = sex_attr[indices] if sex_attr is not None else None
        age_boot = age_attr[indices] if age_attr is not None else None
        
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        for model in y_prob_dict.keys():
            y_prob_boot = y_prob_dict[model][indices]
            y_pred_boot = y_pred_dict[model][indices]
            
            try:
                boot_results[model]['auroc'].append(compute_auroc(y_true_boot, y_prob_boot, num_classes))
            except:
                pass
                
            from sklearn.metrics import balanced_accuracy_score, f1_score
            boot_results[model]['bacc'].append(balanced_accuracy_score(y_true_boot, y_pred_boot))
            boot_results[model]['f1'].append(f1_score(y_true_boot, y_pred_boot, average='weighted'))
            boot_results[model]['ece'].append(expected_calibration_error(y_true_boot, y_prob_boot))
            
            if sex_boot is not None:
                boot_results[model]['delta_eo_sex'].append(calculate_delta_eo(y_true_boot, y_pred_boot, sex_boot))
                boot_results[model]['demographic_parity_sex'].append(calculate_demographic_parity(y_pred_boot, sex_boot))
                boot_results[model]['equalized_odds_sex'].append(calculate_equalized_odds(y_true_boot, y_pred_boot, sex_boot))
            
            if age_boot is not None:
                boot_results[model]['delta_eo_age'].append(calculate_delta_eo(y_true_boot, y_pred_boot, age_boot))
            
            # We use Sex for macro_auc and worst_auc if available
            sub_attr = sex_boot if sex_boot is not None else (age_boot if age_boot is not None else None)
            if sub_attr is not None:
                sub_metrics = calculate_subgroup_metrics(y_true_boot, y_prob_boot, y_pred_boot, sub_attr, num_classes)
                if not np.isnan(sub_metrics['worst_auc']):
                    boot_results[model]['worst_auc'].append(sub_metrics['worst_auc'])
                if not np.isnan(sub_metrics['macro_auc']):
                    boot_results[model]['macro_auc'].append(sub_metrics['macro_auc'])
                
    ci_results = {}
    for model in y_prob_dict.keys():
        ci_results[model] = {}
        for metric in boot_results[model].keys():
            if len(boot_results[model][metric]) > 0:
                ci_results[model][metric] = np.percentile(boot_results[model][metric], [2.5, 97.5])
            else:
                ci_results[model][metric] = (np.nan, np.nan)
                
    return ci_results, boot_results

def compute_p_values(boot_results, baseline_model):
    """Computes two-sided p-values comparing all models to a baseline using bootstrap results."""
    p_values = {}
    if baseline_model not in boot_results:
        return p_values
    
    for model in boot_results.keys():
        if model == baseline_model:
            continue
        p_values[model] = {}
        for metric in boot_results[model].keys():
            arr_model = np.array(boot_results[model][metric])
            arr_base = np.array(boot_results[baseline_model][metric])
            
            # Find common valid indices
            valid = ~(np.isnan(arr_model) | np.isnan(arr_base))
            if np.sum(valid) > 0:
                diff = arr_model[valid] - arr_base[valid]
                p = 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))
                p_values[model][metric] = p
            else:
                p_values[model][metric] = np.nan
    return p_values

def plot_media_figures(y_true, y_prob_dict, emb_dict, sensitive_attr, dataset_name, out_dir, num_classes):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    from sklearn.manifold import TSNE
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Filter out Multimodal and Adversarial models that use Gen text
    exclude_list = ["(Gen)", "Adv Align"]
    y_prob_dict = {k: v for k, v in y_prob_dict.items() if not any(excl in k for excl in exclude_list)}
    emb_dict = {k: v for k, v in emb_dict.items() if not any(excl in k for excl in exclude_list)}
    
    # ---------------------------------------------------------
    # Manuscript-ready styling
    # ---------------------------------------------------------
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 24,
        'lines.linewidth': 3
    })
    
    # Optional: if you have many models, subset them for clarity in ROC/PR
    # For now, plot all
    colors = sns.color_palette("tab10", len(y_prob_dict))
    
    # ---------------------------------------------------------
    # 1. ROC Curves (Macro-averaged)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    if num_classes == 2:
        y_true_bin = np.hstack((1-y_true_bin, y_true_bin)) # (N, 2)
        
    for i, (model_name, y_prob) in enumerate(y_prob_dict.items()):
        # Calculate macro ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for c in range(num_classes):
            if c < y_prob.shape[1] and len(np.unique(y_true_bin[:, c])) > 1:
                fpr[c], tpr[c], _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
                roc_auc[c] = auc(fpr[c], tpr[c])
        
        # Aggregate
        all_fpr = np.unique(np.concatenate([fpr[c] for c in range(num_classes) if c in fpr]))
        mean_tpr = np.zeros_like(all_fpr)
        valid_classes = [c for c in range(num_classes) if c in fpr]
        for c in valid_classes:
            mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
        mean_tpr /= len(valid_classes)
        macro_auc = auc(all_fpr, mean_tpr)
        
        plt.plot(all_fpr, mean_tpr, color=colors[i], label=f'{model_name} (AUC = {macro_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} - ROC Curves')
    plt.legend(loc="lower right", bbox_to_anchor=(1.05, 0))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_ROC_Curves.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. PR Curves (Macro-averaged)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    for i, (model_name, y_prob) in enumerate(y_prob_dict.items()):
        precision = dict()
        recall = dict()
        average_precision = dict()
        for c in range(num_classes):
            if c < y_prob.shape[1] and len(np.unique(y_true_bin[:, c])) > 1:
                precision[c], recall[c], _ = precision_recall_curve(y_true_bin[:, c], y_prob[:, c])
                average_precision[c] = average_precision_score(y_true_bin[:, c], y_prob[:, c])
        
        # Micro/Macro aggregate for plotting can be tricky, we'll plot Micro-average PR for simplicity
        if y_prob.shape[1] == num_classes:
            precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
            ap_micro = average_precision_score(y_true_bin, y_prob, average="micro")
            plt.plot(recall_micro, precision_micro, color=colors[i], label=f'{model_name} (AP = {ap_micro:.3f})')
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{dataset_name} - PR Curves (Micro)')
    plt.legend(loc="lower left", bbox_to_anchor=(1.05, 0))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_PR_Curves.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. Subgroup Performance Bar Charts
    # ---------------------------------------------------------
    groups = [g for g in np.unique(sensitive_attr) if g != 'Unknown']
    if len(groups) > 0:
        bar_data = []
        for model_name, y_prob in y_prob_dict.items():
            for group in groups:
                idx = (sensitive_attr == group)
                if sum(idx) > 0:
                    y_true_g = y_true[idx]
                    y_prob_g = y_prob[idx]
                    if len(np.unique(y_true_g)) > 1:
                        auc_val = compute_auroc(y_true_g, y_prob_g, num_classes)
                        bar_data.append({'Model': model_name, 'Group': group, 'AUROC': auc_val})
        
        if len(bar_data) > 0:
            import pandas as pd
            df_bar = pd.DataFrame(bar_data)
            df_bar.to_csv(os.path.join(out_dir, f"{dataset_name}_subgroup_metrics.csv"), index=False)
            plt.figure(figsize=(14, 8))
            sns.barplot(data=df_bar, x='Model', y='AUROC', hue='Group', palette='pastel')
            plt.title(f'{dataset_name} - Subgroup AUROC')
            plt.xticks(rotation=45, ha='right')
            plt.ylim([0.4, 1.0])
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{dataset_name}_Subgroup_AUROC.png"), dpi=300)
            plt.close()

    # ---------------------------------------------------------
    # 4. t-SNE Embeddings Visualization
    # ---------------------------------------------------------
    # We will pick one baseline (e.g. Vision-Only) and one Multimodal model to plot embeddings
    # from the `emb_dict` which contains (fused_emb_tr, fused_emb_te)
    
    models_to_plot = [m for m in emb_dict.keys() if "LateFusion" in m or "Vision" in m][:2]
    for model_name in models_to_plot:
        _, emb_te = emb_dict[model_name]
        if emb_te is None or len(emb_te) == 0: continue
        
        # Subsample if too large for t-SNE
        n_samples = min(len(emb_te), 1000)
        idx = np.random.choice(len(emb_te), n_samples, replace=False)
        emb_sub = emb_te[idx]
        y_sub = y_true[idx]
        
        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(emb_sub)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=y_sub, palette='Set1', s=100, alpha=0.7)
        plt.title(f'{dataset_name} - t-SNE: {model_name}')
        plt.xlabel('t-SNE Dim 1')
        plt.ylabel('t-SNE Dim 2')
        plt.legend(title='Class')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset_name}_tSNE_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"), dpi=300)
        plt.close()
def plot_metrics_with_pvalues(eval_metrics, p_values, baseline_model, dataset_name, out_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import numpy as np

    metrics_to_plot = ['auroc', 'f1', 'bacc']
    metric_labels = ['AUROC', 'F1-Score', 'Bal Acc']
    
    models = list(eval_metrics.keys())
    
    # Filter out models using Gen text to keep the plot clean as requested before
    exclude_list = ["(Gen)", "Adv Align"]
    models = [m for m in models if not any(excl in m for excl in exclude_list)]
    
    # Reorder models: Vision (baseline), Text-only, Multimodal
    def get_order(m):
        if "Vision" in m: return 0
        if "Text" in m: return 1
        if "Multimodal" in m: return 2
        return 3
    models = sorted(models, key=get_order)
    
    if not models:
        return
        
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        vals = [eval_metrics[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i*width - width, vals, width, label=label, edgecolor='black')
        
        for j, m in enumerate(models):
            if m == baseline_model:
                continue
            p = p_values.get(m, {}).get(metric, np.nan)
            if not np.isnan(p):
                stars = ""
                if p < 0.001: stars = "***"
                elif p < 0.01: stars = "**"
                elif p < 0.05: stars = "*"
                
                if stars:
                    h = vals[j]
                    ax.text(x[j] + i*width - width, h + 0.01, stars, ha='center', va='bottom', fontsize=18, color='red', fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title(f'{dataset_name} - Performance vs {baseline_model}\n(*p<0.05, **p<0.01, ***p<0.001)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05), ncol=3)
    ax.set_ylim([0.0, 1.1])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_metrics_pvalues.png"), dpi=300)
    plt.close()

def plot_fairness_metrics_step1(eval_metrics, dataset_name, out_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import numpy as np
    
    models = list(eval_metrics.keys())
    exclude_list = ["(Gen)", "Adv Align"]
    models = [m for m in models if not any(excl in m for excl in exclude_list)]
    
    def get_order(m):
        if "Vision" in m: return 0
        if "Text" in m: return 1
        if "Multimodal" in m: return 2
        return 3
    models = sorted(models, key=get_order)
    
    if not models:
        return
        
    metrics = ['delta_eo_sex', 'delta_eo_age', 'probe_auroc_sex', 'probe_auroc_age']
    labels = ['Delta EO (Sex)', 'Delta EO (Age)', 'Probe AUROC (Sex)', 'Probe AUROC (Age)']
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [eval_metrics[m].get(metric, 0) for m in models]
        ax.bar(x + i*width - 1.5*width, vals, width, label=label, edgecolor='black')
        
    ax.set_ylabel('Score')
    ax.set_title(f'{dataset_name} - Fairness Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05), ncol=2)
    ax.set_ylim([0.0, 1.1])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_fairness_metrics.png"), dpi=300)
    plt.close()

