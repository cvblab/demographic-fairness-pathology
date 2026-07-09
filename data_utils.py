import os
import json
import numpy as np
import pandas as pd
import torch

def get_embedding(text, device, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return emb

def load_data(split_csv, device, tokenizer, model, ai4skin_json, assist_json, tcga_df=None, dataset_filter=None):
    df = pd.read_csv(split_csv)
    X_real = []
    X_gen = []
    X_vis = []
    y = []
    age = []
    sex = []
    center = []
    
    for _, row in df.iterrows():
        wsi = row['WSI']
        label = row['GT']
        
        if wsi.startswith("AI4SKIN_") or wsi.startswith("CF") or wsi.startswith("DMF") or wsi.startswith("FXA"):
            ds = "AI4SKIN"
            raw_id = wsi.replace("AI4SKIN_", "")
            j_data = ai4skin_json
            match_id = row.get('ano', wsi)
        elif wsi.startswith("ASSIST_") or wsi.startswith("MC"):
            ds = "ASSIST"
            raw_id = wsi[len("ASSIST_"):] if wsi.startswith("ASSIST_") else wsi
            j_data = assist_json
            match_id = wsi
        elif wsi.startswith("TCGA"):
            ds = "TCGA-SKCM"
            raw_id = wsi
            if wsi.startswith("TCGA-SKCM_"):
                raw_id = wsi[len("TCGA-SKCM_"):]
            j_data = None
            match_id = raw_id
        else:
            continue
            
        if dataset_filter is not None and ds != dataset_filter:
            continue
            
        d_age = np.nan
        d_sex = 'Unknown'
        
        if ds == "TCGA-SKCM":
            d_age = row.get('age', np.nan)
            d_sex = row.get('sex', 'Unknown')
            if d_sex == 'M': d_sex = 'Male'
            elif d_sex == 'F': d_sex = 'Female'
        else:
            for item in j_data:
                if item.get("wsi_id") == match_id:
                    d_age = item.get("age", np.nan)
                    if d_age is None:
                        d_age = np.nan
                    d_sex = item.get("sex", "Unknown")
                    if d_sex is None:
                        d_sex = "Unknown"
                    if d_sex == 'M': d_sex = 'Male'
                    elif d_sex == 'F': d_sex = 'Female'
                    break
                
        if ds == "AI4SKIN":
            if "HUSC" in raw_id:
                json_path = f"/workspace/data/features_ASSIST/AI4SKIN_HUSC_reports/{wsi}_unified.json"
            else:
                json_path = f"/workspace/data/features_ASSIST/AI4SKIN_reports/{wsi}_unified.json"
            vis_path = f"/workspace/data/features_ASSIST/AI4SKIN_embeddings/{wsi}.npy"
            if not os.path.exists(vis_path):
                vis_path = f"/workspace/data/features_ASSIST/AI4SKIN_embeddings/{raw_id}.npy"

        elif ds == "ASSIST":
            if "HUSC" in raw_id or str(raw_id).startswith("MC"):
                json_path = f"/workspace/data/features_ASSIST/ASSIST_HUSC_reports/{wsi}_unified.json"
                vis_path = f"/workspace/data/features_ASSIST/ASSIST_HUSC_embeddings/{wsi}.pt"
            else:
                json_path = f"/workspace/data/features_ASSIST/ASSIST_reports/{wsi}_unified.json"
                vis_path = f"/workspace/data/features_ASSIST/ASSIST_embeddings/{wsi}.pt"
                
        else: # TCGA-SKCM
            json_path = f"/workspace/data/features_ASSIST/TCGA-SKCM_reports/{wsi}_unified.json"
            if not os.path.exists(json_path):
                json_path = f"/workspace/data/features_ASSIST/TCGA-SKCM_reports/{raw_id}_unified.json"
            vis_path_npy = f"/workspace/data/features_ASSIST/TCGA-SKCM_embeddings/{raw_id}.npy"
            vis_path_pt = f"/workspace/data/features_ASSIST/TCGA-SKCM_embeddings/{raw_id}.pt"
            vis_path = vis_path_npy if os.path.exists(vis_path_npy) else vis_path_pt
            
        # fallback for missing embeddings
        if not os.path.exists(vis_path):
            fb_wsi_pt = f"/workspace/data/features_ASSIST_HUSC_nometadata/{wsi}.pt"
            fb_wsi_npy = f"/workspace/data/features_ASSIST_HUSC_nometadata/{wsi}.npy"
            if os.path.exists(fb_wsi_pt):
                vis_path = fb_wsi_pt
            elif os.path.exists(fb_wsi_npy):
                vis_path = fb_wsi_npy
            else:
                fb_raw_pt = f"/workspace/data/features_ASSIST_HUSC_nometadata/{raw_id}.pt"
                fb_raw_npy = f"/workspace/data/features_ASSIST_HUSC_nometadata/{raw_id}.npy"
                if os.path.exists(fb_raw_pt): vis_path = fb_raw_pt
                elif os.path.exists(fb_raw_npy): vis_path = fb_raw_npy
        
        if not os.path.exists(vis_path):
            continue
            
        # Missing Modality Strategy
        real_emb = None
        gen_emb = None
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if "generated_caption" in data:
                gen_caption = data["generated_caption"]
                if isinstance(gen_caption, str) and gen_caption.strip() != "" and gen_caption != "No report available":
                    gen_emb = get_embedding(gen_caption, device, tokenizer, model)
                
            if "real_caption" in data:
                real_caption = data["real_caption"]
                if isinstance(real_caption, str) and real_caption.strip() != "" and real_caption != "No report available":
                    real_emb = get_embedding(real_caption, device, tokenizer, model)
                
        # If real text is missing, use generated text as proxy
        if real_emb is None:
            if gen_emb is not None:
                real_emb = gen_emb
            else:
                real_emb = np.zeros(768) # PathologyBERT dim
                
        if gen_emb is None:
            gen_emb = np.zeros(768)
            
        # Load Vision
        try:
            if vis_path.endswith('.npy'):
                v_emb = np.load(vis_path)
            else:
                v_emb_data = torch.load(vis_path, map_location='cpu')
                if isinstance(v_emb_data, dict) and 'features' in v_emb_data:
                    v_emb = v_emb_data['features']
                else:
                    v_emb = v_emb_data
                    
                if isinstance(v_emb, torch.Tensor):
                    v_emb = v_emb.numpy()
            if v_emb.ndim > 1:
                v_emb = v_emb.mean(axis=0)
            v_emb = v_emb.flatten()
        except:
            continue
            
        X_real.append(real_emb)
        X_gen.append(gen_emb)
        X_vis.append(v_emb)
        y.append(label)
        age.append(d_age)
        sex.append(d_sex)
        center.append(row.get('center', 'Unknown'))
        
    return np.array(X_real), np.array(X_gen), np.array(X_vis), np.array(y), np.array(age), np.array(sex), np.array(center)

def get_combined_dataset(split_csvs, device, tokenizer, model, ai4skin_json, assist_json, tcga_df=None, dataset_filter=None):
    X_r, X_g, X_v, y_all, age_all, sex_all, center_all = [], [], [], [], [], [], []
    for csv in split_csvs:
        xr, xg, xv, y, a, s, c = load_data(csv, device, tokenizer, model, ai4skin_json, assist_json, tcga_df, dataset_filter)
        if len(y) > 0:
            X_r.append(xr)
            X_g.append(xg)
            X_v.append(xv)
            y_all.append(y)
            age_all.append(a)
            sex_all.append(s)
            center_all.append(c)
    if not X_r:
        return tuple([np.array([]) for _ in range(7)])
    return np.concatenate(X_r), np.concatenate(X_g), np.concatenate(X_v), np.concatenate(y_all), np.concatenate(age_all), np.concatenate(sex_all), np.concatenate(center_all)
