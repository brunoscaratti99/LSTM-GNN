import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np



def normalize_tp(tp, eps=1e-6):
    tp_log = torch.log1p(tp)
    mean = tp_log.mean()
    std = tp_log.std()
    return (tp_log - mean) / (std + eps), mean, std


def add_consecutive_dry_days_feature(X, precip_col=0, dry_threshold=0.0):
    """
    Adiciona uma nova feature com a contagem de dias consecutivos sem chuva por no.

    Args:
        X: tensor [dias, nos, features]
        precip_col: indice da feature de precipitacao em X
        dry_threshold: valor maximo para considerar um dia como seco

    Returns:
        X_aug: tensor [dias, nos, features + 1]

    Notes:
        - A contagem e feita por no, ao longo da dimensao temporal.
        - O dia atual entra na contagem se precip <= dry_threshold.
        - Dias com chuva recebem valor 0 nessa nova feature.
    """
    if not torch.is_tensor(X):
        X = torch.as_tensor(X)

    if X.ndim != 3:
        raise ValueError(f"X deve ter shape [dias, nos, features]. Recebido: {tuple(X.shape)}")

    _, N, F = X.shape
    if not (0 <= precip_col < F):
        raise ValueError(f"precip_col={precip_col} fora do intervalo para F={F}")

    precip = X[..., precip_col]
    dry_mask = precip <= dry_threshold

    dry_days = X.new_zeros((X.shape[0], N))
    zeros_row = X.new_zeros((N,))

    dry_days[0] = dry_mask[0].to(dtype=X.dtype)
    for t in range(1, X.shape[0]):
        dry_days[t] = torch.where(
            dry_mask[t],
            dry_days[t - 1] + 1,
            zeros_row,
        )

    return torch.cat([X, dry_days.unsqueeze(-1)], dim=-1)


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

def inverse_log1p_zscore(t, mean, std):
    return (torch.expm1(t * std + mean))
