import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np



def normalize_tp(tp, eps=1e-6):
    tp_log = torch.log1p(tp)
    mean = tp_log.mean()
    std = tp_log.std()
    return (tp_log - mean) / (std + eps), mean, std


def normalize_features(X, scaler=MinMaxScaler):
    # X: [B, T, N, F]
    #normaliza todas features com excessão do target ([...,0] de X)
    B, T, N, F = X.shape
    X_scaled = []
    scalers = {}
    
    for f in range(1,F):
        scale = scaler()
        scale.fit(X[...,f].flatten().numpy())
        x_scaled = scale.transform(X[...,f])
        X_scaled.append(x_scaled)
        scalers.append(scale)
    return torch.cat(torch.tensor(X_scaled), dim=-1), scalers
        

def assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise RuntimeError(f"{name} has non-finite values (nan={nan_count}, inf={inf_count})")