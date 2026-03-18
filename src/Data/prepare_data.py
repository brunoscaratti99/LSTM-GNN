import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from dateutil.relativedelta import relativedelta
import xarray as xr


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


    
def create_batchs(X_train, X_val, X_test, y_train, y_val, y_test, batch_size, device, num_workers=0):
    pin_memory = (device=='cuda')
    train_ds = TensorDataset(X_train,y_train)
    val_ds   = TensorDataset(X_val, y_val)
    test_ds  = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
     
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader

def slice_intervalos_anuais(dataset: xr.Dataset, 
                            start_date: np.datetime64, end_date: np.datetime64, 
                            months: int, days: int):

    """
    Para cada ano entre "start_date" e "end_date", 
    pegua apenas o periodo de "months" meses e "days" dias antes de "end_date", e os concatena
    """



    yearly_delta = end_date - (end_date - relativedelta(months = months, days = days))
    yearly_delta = yearly_delta.days

    print(yearly_delta)

    if yearly_delta > 364:
        raise ValueError("Cannot slice intervals bigger than a year within a year!")
    if yearly_delta < 1:
        raise ValueError("Cannot slice intervals of less than a day!")
    if months < 0 or days < 0:
        raise ValueError("Cannot have negative months or days values!")

    # Encontra o numero de anos que serao usados
    years = np.datetime64(end_date, "Y") - np.datetime64(start_date, "Y")

    # Vets aux
    start = []
    end = []

    # Calcula os intervalos de um mes pra cada ano
    for year in range(int(years)):
        start.append(np.datetime64(end_date) - relativedelta(years = year, months = 1, days = 0, hours = 0))
        end.append(np.datetime64(end_date) - relativedelta(years = year, months = 0, days = 0, hours = 0))

    # Coloca em um zip pro python gostar
    ranges = list(zip(start, end))

    #for start, end in ranges:
    #    print(start, end)

    # Faz varios slices do banco original e concatena-os todos
    dataset = xr.concat(
        [dataset.sel(time = slice(start, end)) for start, end in ranges],
        dim = "time"
    )

    # Da sort dnv, por algum motivo algo do slicing desordena os dados
    dataset = dataset.sortby("time")

    return(dataset)