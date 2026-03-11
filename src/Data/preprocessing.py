import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np



def normalize_tp(tp, eps=1e-6):
    tp_log = torch.log1p(tp)
    mean = tp_log.mean()
    std = tp_log.std()
    return (tp_log - mean) / (std + eps), mean, std


def normalize_features(X_local, scaler=StandardScaler):
    # X_local: [B, T, N, F]
    B, T, N_local, F_local = X_local.shape
    X_scaled = torch.zeros((B, T, N_local, F_local), dtype=torch.float32)
    scalers = []

    for f in range(F_local):
        scale = scaler()
        values = X_local[..., f].reshape(-1, 1).numpy()
        scale.fit(values)
        x_scaled = torch.tensor(scale.transform(values), dtype=torch.float32).reshape(B, T, N_local)
        X_scaled[..., f] = x_scaled
        scalers.append(scale)
    return X_scaled, scalers
        

def assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise RuntimeError(f"{name} has non-finite values (nan={nan_count}, inf={inf_count})")
    
    
def fit_log1p_zscore_stats(X_train_local, y_train_local, target_col=0, eps=1e-6):
    ref = torch.cat([
        X_train_local[..., target_col].reshape(-1),
        y_train_local.reshape(-1),
    ], dim=0).clamp_min(0.0)
    ref_log = torch.log1p(ref)
    mean = ref_log.mean()
    std = ref_log.std(unbiased=False).clamp_min(eps)
    return mean, std

def apply_log1p_zscore(t, mean, std):
    return (torch.log1p(t.clamp_min(0.0)) - mean) / std


