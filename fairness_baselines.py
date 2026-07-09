import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h), h

class AdversaryModel(nn.Module):
    def __init__(self, hidden_dim, num_sensitive_classes):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_sensitive_classes)
        
    def forward(self, x, alpha=1.0):
        x_rev = GradientReversalLayer.apply(x, alpha)
        h = F.relu(self.fc1(x_rev))
        return self.fc2(h)

def train_reweighting(X_train, y_train, sensitive_train, num_classes):
    """Trains a Logistic Regression model using sample weights based on intersectional groups."""
    valid_idx = [i for i, s in enumerate(sensitive_train) if s != 'Unknown']
    if len(valid_idx) == 0:
        valid_idx = list(range(len(y_train)))
        sample_weights = np.ones(len(y_train))
    else:
        # Create a combined class-group label to balance both
        combined_labels = [f"{y}_{s}" for y, s in zip(y_train[valid_idx], sensitive_train[valid_idx])]
        weights = compute_sample_weight('balanced', combined_labels)
        sample_weights = np.zeros(len(y_train))
        sample_weights[valid_idx] = weights
        sample_weights[np.setdiff1d(range(len(y_train)), valid_idx)] = 1.0 # default weight for unknowns

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train, sample_weight=sample_weights)
    return lr

def train_adversarial_debiasing(X_train, y_train, sensitive_train, num_classes, epochs=50, lr=1e-3, beta=1.0, X_val=None, y_val=None, patience=10):
    """Trains an MLP with a Gradient Reversal Layer adversary on the sensitive attribute."""
    from sklearn.preprocessing import LabelEncoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    valid_idx = [i for i, s in enumerate(sensitive_train) if s != 'Unknown']
    if len(valid_idx) == 0:
        print("No sensitive labels for adversarial debiasing. Returning standard MLP.")
        beta = 0.0
        
    le = LabelEncoder()
    if len(valid_idx) > 0:
        le.fit(sensitive_train[valid_idx])
        num_sensitive_classes = len(le.classes_)
    else:
        num_sensitive_classes = 2
        
    clf = MLPClassifier(X_train.shape[1], num_classes).to(device)
    adv = AdversaryModel(128, num_sensitive_classes).to(device)
    
    opt_clf = optim.Adam(clf.parameters(), lr=lr)
    opt_adv = optim.Adam(adv.parameters(), lr=lr)
    
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    
    a_train_full = np.zeros(len(y_train), dtype=int)
    mask = np.zeros(len(y_train), dtype=bool)
    if len(valid_idx) > 0:
        a_train_full[valid_idx] = le.transform(sensitive_train[valid_idx])
        mask[valid_idx] = True
    
    a_t = torch.tensor(a_train_full, dtype=torch.long).to(device)
    mask_t = torch.tensor(mask, dtype=torch.bool).to(device)
    
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(X_t, y_t, a_t, mask_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    if X_val is not None and y_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val, dtype=torch.long).to(device)
        
    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        clf.train()
        adv.train()
        for bx, by, ba, bm in loader:
            p = float(epoch) / epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            logits, h = clf(bx)
            loss_clf = F.cross_entropy(logits, by)
            
            adv_logits = adv(h, alpha)
            if bm.sum() > 0:
                loss_adv = F.cross_entropy(adv_logits[bm], ba[bm])
            else:
                loss_adv = torch.tensor(0.0).to(device)
                
            loss = loss_clf + beta * loss_adv
            
            opt_clf.zero_grad()
            opt_adv.zero_grad()
            loss.backward()
            opt_clf.step()
            opt_adv.step()
            
        if X_val is not None and y_val is not None:
            clf.eval()
            with torch.no_grad():
                val_logits, _ = clf(X_v)
                val_loss = F.cross_entropy(val_logits, y_v).item()
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu() for k, v in clf.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                    
    if best_state is not None:
        clf.load_state_dict(best_state)
            
    clf.eval()
    return clf

def train_group_dro(X_train, y_train, sensitive_train, num_classes, epochs=50, lr=1e-3, step_size=0.01, unknown_weight=0.1, X_val=None, y_val=None, patience=10):
    """Trains an MLP using Group Distributionally Robust Optimization (GroupDRO).
    
    unknown_weight: down-weighting factor applied to the loss contribution of samples
    with unknown sensitive-attribute labels (excluded from the group-robust max,
    but still included in the overall objective at reduced weight).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    valid_idx = [i for i, s in enumerate(sensitive_train) if s != 'Unknown']
    if len(valid_idx) == 0:
        clf = MLPClassifier(X_train.shape[1], num_classes).to(device)
        opt = optim.Adam(clf.parameters(), lr=lr)
        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(device)
        for _ in range(epochs):
            opt.zero_grad()
            logits, _ = clf(X_t)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            opt.step()
        clf.eval()
        return clf
        
    # GroupDRO logic
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(sensitive_train[valid_idx])
    num_groups = len(le.classes_)
    
    group_weights = torch.ones(num_groups).to(device) / num_groups
    
    clf = MLPClassifier(X_train.shape[1], num_classes).to(device)
    opt = optim.Adam(clf.parameters(), lr=lr)
    
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    
    a_train_full = np.zeros(len(y_train), dtype=int)
    mask = np.zeros(len(y_train), dtype=bool)
    a_train_full[valid_idx] = le.transform(sensitive_train[valid_idx])
    mask[valid_idx] = True
    
    a_t = torch.tensor(a_train_full, dtype=torch.long).to(device)
    mask_t = torch.tensor(mask, dtype=torch.bool).to(device)
    
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(X_t, y_t, a_t, mask_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    if X_val is not None and y_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val, dtype=torch.long).to(device)
        
    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        clf.train()
        for bx, by, ba, bm in loader:
            opt.zero_grad()
            logits, _ = clf(bx)
            
            group_losses = torch.zeros(num_groups).to(device)
            for g in range(num_groups):
                g_idx = (ba == g) & bm
                if g_idx.sum() > 0:
                    group_losses[g] = F.cross_entropy(logits[g_idx], by[g_idx])
                    
            # Update group weights via exponentiated gradient descent
            with torch.no_grad():
                group_weights = group_weights * torch.exp(step_size * group_losses)
                group_weights = group_weights / group_weights.sum()
                
            loss = torch.dot(group_weights, group_losses)
            # Add loss for unknown groups
            unk_idx = ~bm
            if unk_idx.sum() > 0:
                loss += F.cross_entropy(logits[unk_idx], by[unk_idx]) * unknown_weight
                
            loss.backward()
            opt.step()
            
        if X_val is not None and y_val is not None:
            clf.eval()
            with torch.no_grad():
                val_logits, _ = clf(X_v)
                val_loss = F.cross_entropy(val_logits, y_v).item()
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu() for k, v in clf.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                    
    if best_state is not None:
        clf.load_state_dict(best_state)
            
    clf.eval()
    return clf

def predict_mlp(model, X_test, return_embeddings=False):
    """Predicts probabilities using a trained MLP PyTorch model.
    
    If return_embeddings is True, also returns the penultimate hidden
    representation (post-ReLU, pre-dropout at eval time) so that
    representation-leakage probes evaluate what the fairness intervention
    actually learned, rather than the raw frozen input features.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits, h = model(X_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    if return_embeddings:
        return probs, h.cpu().numpy()
    return probs
