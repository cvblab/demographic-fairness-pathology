import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class MultimodalDataset(Dataset):
    def __init__(self, text_embeddings, image_embeddings, labels, protected_attrs=None):
        self.use_text = text_embeddings is not None
        if self.use_text:
            self.text_embeddings = torch.tensor(np.array(text_embeddings), dtype=torch.float32)
        self.image_embeddings = torch.tensor(np.array(image_embeddings), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)
        self.protected_attrs = protected_attrs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'image': self.image_embeddings[idx],
            'label': self.labels[idx],
            'idx': idx
        }
        if self.use_text:
            item['text'] = self.text_embeddings[idx]
        return item

class LateFusionModel(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim=128, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.use_text = text_dim is not None
        
        self.fc_image = nn.Linear(image_dim, hidden_dim)
        self.classifier_image = nn.Linear(hidden_dim, num_classes)
        
        if self.use_text:
            self.fc_text = nn.Linear(text_dim, hidden_dim)
            self.classifier_text = nn.Linear(hidden_dim, num_classes)
            
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, text, image, return_embeddings=False):
        i_feat = F.relu(self.fc_image(image))
        i_feat_dropped = self.dropout(i_feat)
        logits_image = self.classifier_image(i_feat_dropped)
        
        if self.use_text and text is not None:
            t_feat = F.relu(self.fc_text(text))
            t_feat_dropped = self.dropout(t_feat)
            logits_text = self.classifier_text(t_feat_dropped)
            
            logits = (logits_image + logits_text) / 2.0
            fused_emb = torch.cat([t_feat, i_feat], dim=1) 
        else:
            logits = logits_image
            fused_emb = i_feat
            
        if return_embeddings:
            return logits, fused_emb, None
        return logits

class IntermediateConcatenationModel(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim=128, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.use_text = text_dim is not None
        
        self.fc_image = nn.Linear(image_dim, hidden_dim)
        
        if self.use_text:
            self.fc_text = nn.Linear(text_dim, hidden_dim)
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fusion = nn.Linear(hidden_dim, hidden_dim)
            
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, text, image, return_embeddings=False):
        i_feat = F.relu(self.fc_image(image))
        
        if self.use_text:
            if text is None:
                t_feat = torch.zeros(i_feat.shape[0], self.fc_text.out_features, device=i_feat.device)
            else:
                t_feat = F.relu(self.fc_text(text))
            fused = torch.cat([t_feat, i_feat], dim=1)
        else:
            fused = i_feat
            
        fused_emb = F.relu(self.fusion(fused))
        dropped_fused = self.dropout(fused_emb)
        logits = self.classifier(dropped_fused)
        
        if return_embeddings:
            return logits, fused_emb, None
        return logits

class IntermediateCrossAttentionModel(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim=128, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.use_text = text_dim is not None
        
        self.fc_image = nn.Linear(image_dim, hidden_dim)
        
        if self.use_text:
            self.fc_text = nn.Linear(text_dim, hidden_dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fusion = nn.Linear(hidden_dim, hidden_dim)
            
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, text, image, return_embeddings=False):
        i_feat = F.relu(self.fc_image(image))
        attn_weights = None
        
        if self.use_text:
            if text is None:
                t_feat = torch.zeros(i_feat.shape[0], self.fc_text.out_features, device=i_feat.device)
                fused = torch.cat([t_feat, i_feat], dim=1)
            else:
                t_feat = F.relu(self.fc_text(text))
                q = t_feat.unsqueeze(1)
                k = v = i_feat.unsqueeze(1)
                
                attn_out, attn_weights = self.cross_attn(q, k, v)
                attn_out = attn_out.squeeze(1)
                
                fused = torch.cat([t_feat, attn_out], dim=1)
        else:
            fused = i_feat
            
        fused_emb = F.relu(self.fusion(fused))
        dropped_fused = self.dropout(fused_emb)
        logits = self.classifier(dropped_fused)
        
        if return_embeddings:
            return logits, fused_emb, attn_weights
        return logits

def train_model(model, train_dataloader, val_dataloader=None, epochs=50, lr=1e-3, device='cpu', class_weights=None, patience=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cw_t = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=cw_t)
    
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            text = batch.get('text', None)
            if text is not None:
                text = text.to(device)
                if torch.rand(1).item() < 0.5:
                    text = None
            image = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(text, image)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    text = batch.get('text', None)
                    if text is not None: text = text.to(device)
                    image = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    logits = model(text, image)
                    val_loss += criterion(logits, labels).item()
                    
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model

def predict_with_uq(model, dataloader, num_mc_samples=30, device='cpu'):
    model.to(device)
    model.train() # Enable dropout
    
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            text = batch.get('text', None)
            if text is not None:
                text = text.to(device)
            image = batch['image'].to(device)
            labels = batch['label'].numpy()
            
            mc_probs = []
            fused_embs = []
            attns = []
            for _ in range(num_mc_samples):
                logits, fused_emb, attn_weights = model(text, image, return_embeddings=True)
                probs = F.softmax(logits, dim=1)
                mc_probs.append(probs.cpu().numpy())
                fused_embs.append(fused_emb.cpu().numpy())
                if attn_weights is not None:
                    attns.append(attn_weights.cpu().numpy())
            
            mc_probs = np.stack(mc_probs)
            mean_probs = np.mean(mc_probs, axis=0)
            
            # Entropy of the expected probability
            entropy_expected = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
            
            # Expected entropy of the probabilities
            entropies_mc = -np.sum(mc_probs * np.log(mc_probs + 1e-10), axis=2)
            expected_entropy = np.mean(entropies_mc, axis=0)
            
            # Mutual Information (Epistemic Uncertainty)
            mutual_info = entropy_expected - expected_entropy
            
            variance = np.var(mc_probs, axis=0)
            
            mean_emb = np.mean(np.stack(fused_embs), axis=0)
            mean_attn = np.mean(np.stack(attns), axis=0) if len(attns) > 0 else None
            
            batch_results = []
            for i in range(len(labels)):
                res = {
                    'mean_probs': mean_probs[i],
                    'entropy': entropy_expected[i],
                    'mutual_info': mutual_info[i],
                    'variance': variance[i],
                    'prediction': np.argmax(mean_probs[i]),
                    'label': labels[i],
                    'idx': batch['idx'][i].item(),
                    'fused_emb': mean_emb[i]
                }
                if mean_attn is not None:
                    res['attn_weights'] = mean_attn[i]
                batch_results.append(res)
            all_probs.extend(batch_results)
            
    return all_probs
