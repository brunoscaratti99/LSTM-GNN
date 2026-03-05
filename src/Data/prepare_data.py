import numpy as np


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

    if multi_step:
        print("Multi-step not implemented yet")
        return None

    X_windows, y_windows = [], []

    m = len(X) - (window_size + horizon)

    for i in range(m + 1):
        X_windows.append(X[i : i + window_size])
        y_windows.append(y[i + window_size:i + window_size + horizon])

    return np.array(X_windows), np.array(y_windows)



