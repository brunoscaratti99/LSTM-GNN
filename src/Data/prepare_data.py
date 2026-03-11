import numpy as np
import torch


def create_sliding_windows(
    X,
    y,
    window_size,
    horizon=1,
    multi_step=False
):
    """
    Transform multivariate time series into sliding windows for forecasting.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    y : array-like of shape (n_samples,)
        Target variable.
    window_size : int
        Number of past timesteps.
    horizon : int
        Forecast horizon.
    multi_step : bool
        Whether to predict multiple future steps (not implemented).

    Returns
    -------
    X_windows : ndarray of shape (n_windows, window_size, n_features)
    y_windows : ndarray of shape (n_windows,) or (n_windows, horizon)
    """
    n_samples, n_features = X.shape[0], X.shape[2]
    if multi_step:
        print("Multi-step not implemented yet")
        return None

    X_windows, y_windows = [], []

    m = n_samples - (window_size + horizon)

    for i in range(m+1):
        X_windows.append(X[i : i + window_size])
        y_windows.append(y[i + window_size:i + window_size + horizon])

    return torch.stack(X_windows, dim=-1).permute(3,0,1,2), torch.stack(y_windows, dim=-1).permute(2,0,1)


def train_split(Xs, ys, train_ratio=0.7, val_ratio=0.2):
    test_ratio = 1-train_ratio-val_ratio
    num_samples = Xs.shape[0]
    n_train = int(train_ratio * num_samples)
    n_val   = int(val_ratio * num_samples)
    
    
    X_train, y_train = Xs[:n_train], ys[:n_train]
    X_val  , y_val   = Xs[n_train:n_train+n_val], ys[n_train:n_train+n_val]
    X_test , y_test  = Xs[n_train+n_val:], ys[n_train+n_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


    
    

