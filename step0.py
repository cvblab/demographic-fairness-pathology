import os
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel
import json

from data_utils import get_combined_dataset
from evaluation_utils import compute_auroc, pad_proba_columns

def robust_compute_class_weight(y, num_classes):
    from sklearn.utils.class_weight import compute_class_weight
    unique_y = np.unique(y)
    cw = compute_class_weight('balanced', classes=unique_y, y=y)
    cw_dict = dict(zip(unique_y, cw))
    return {i: cw_dict.get(i, 1.0) for i in range(num_classes)}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "/workspace/src/FOMENT/step0_text_validation"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading demographic metadata...")
    with open("/workspace/src/metadata/raw/AI4SKIN.json", "r") as f: ai4skin_json = json.load(f)
    with open("/workspace/src/metadata/raw/ASSIST.json", "r") as f: assist_json = json.load(f)
    tcga_df = pd.read_csv('/workspace/src/metadata/raw/tcga_reports_cleaned.csv', usecols=['project_id', 'slide_id', 'sex'])
    
    print("Initializing PathologyBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("tsantos/PathologyBERT")
    model = AutoModel.from_pretrained("tsantos/PathologyBERT").to(device)
    model.eval()
    
    datasets = ["AI4SKIN", "ASSIST", "TCGA-SKCM"]
    results = []
    
    for ds in datasets:
        print(f"\n=======================================================")
        print(f" Validating Text Modality on {ds}")
        print(f"=======================================================")
        ds_prefix = ds.lower().replace("-skcm", "")
        
        # We don't pass the tokenizer/model because text embeddings are pre-extracted
        X_r_tr, X_g_tr, X_v_tr, y_tr, age_tr, sex_tr, center_tr = get_combined_dataset(
            [f"/workspace/src/metadata/{ds_prefix}_train.csv"], device, tokenizer, model, ai4skin_json, assist_json, tcga_df, dataset_filter=ds)
        
        X_r_te, X_g_te, X_v_te, y_te, age_te, sex_te, center_te = get_combined_dataset(
            [f"/workspace/src/metadata/{ds_prefix}_test.csv"], device, tokenizer, model, ai4skin_json, assist_json, tcga_df, dataset_filter=ds)
            
        num_classes = int(np.max(np.concatenate([y_tr, y_te]))) + 1
        
        # Filter for paired samples (only keep cases where real report is not missing)
        # Missing text embeddings are stored as zero vectors in our pipeline
        mask_tr = np.linalg.norm(X_r_tr, axis=1) > 1e-5
        mask_te = np.linalg.norm(X_r_te, axis=1) > 1e-5
        
        X_r_tr_paired, X_g_tr_paired, X_v_tr_paired, y_tr_paired = X_r_tr[mask_tr], X_g_tr[mask_tr], X_v_tr[mask_tr], y_tr[mask_tr]
        X_r_te_paired, X_g_te_paired, X_v_te_paired, y_te_paired = X_r_te[mask_te], X_g_te[mask_te], X_v_te[mask_te], y_te[mask_te]
        
        print(f"Total Test Samples: {len(y_te)}")
        print(f"Paired Test Samples (having real reports): {len(y_te_paired)} ({(len(y_te_paired)/len(y_te))*100:.1f}%)")
        
        if len(np.unique(y_tr_paired)) < 2 or len(np.unique(y_te_paired)) < 2:
            print("Not enough paired classes to train/evaluate. Skipping.")
            continue
            
        cw_tr = robust_compute_class_weight(y_tr_paired, num_classes)
        
        def eval_modality(name, X_train, y_train, X_test, y_test):
            lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight=cw_tr)
            lr.fit(X_train, y_train)
            
            preds = lr.predict(X_test)
            probs = pad_proba_columns(lr.predict_proba(X_test), lr.classes_, num_classes)
            
            if num_classes == 2 and probs.ndim == 2:
                prob_eval = probs[:, 1]
            else:
                prob_eval = probs
                
            try:
                auroc = compute_auroc(y_test, prob_eval, num_classes)
            except:
                auroc = np.nan
                
            bacc = balanced_accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            
            results.append({
                "Dataset": ds,
                "Model": name,
                "AUROC": auroc,
                "Bal Acc": bacc,
                "F1-Score": f1,
                "Paired N": len(y_test)
            })
            print(f"  {name:20} | AUROC: {auroc:.3f} | Bal Acc: {bacc:.3f} | F1: {f1:.3f}")

        eval_modality("Vision-Only", X_v_tr_paired, y_tr_paired, X_v_te_paired, y_te_paired)
        eval_modality("Text-Only (Real)", X_r_tr_paired, y_tr_paired, X_r_te_paired, y_te_paired)
        eval_modality("Text-Only (Gen)", X_g_tr_paired, y_tr_paired, X_g_te_paired, y_te_paired)
        
    df_res = pd.DataFrame(results)
    print("\n=======================================================")
    print(" Summary Validation Results")
    print("=======================================================")
    print(df_res.to_markdown(index=False))
    df_res.to_csv(f"{out_dir}/text_validation_metrics.csv", index=False)
    print(f"\nArtifacts saved to {out_dir}/text_validation_metrics.csv")

if __name__ == "__main__":
    main()
