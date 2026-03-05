import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np



def normalize_tp(tp, eps=1e-6):
    tp_log = torch.log1p(tp)
    mean = tp_log.mean()
    std = tp_log.std()
    return (tp_log - mean) / (std + eps), mean, std


def normalize_features(X, scaler=MinMaxScaler()):
    # X: [T, N, F]
    T, N, F = X.shape
    X_scaled = []
    scalers = {}
    
    for f in range(F):
        scale = scaler()
        scale.fit(X[...,f].flatten().numpy())
        x_scaled = scale.transform(X[...,f])
        X_scaled.append(x_scaled)
        scalers.append(scale)
    return torch.cat(torch.tensor(X_scaled), dim=-1), scalers
        
