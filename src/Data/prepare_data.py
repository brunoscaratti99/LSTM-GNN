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


def _empty_window_split(X, y, window_size, horizon):
    return (
        X.new_empty((0, window_size, *X.shape[1:])),
        y.new_empty((0, horizon, *y.shape[1:])),
    )


def _create_windows_for_target_interval(X, y, window_size, horizon, start_idx, end_idx, use_context=True):
    if start_idx < 0 or end_idx > X.shape[0]:
        raise ValueError("Intervalo temporal inválido para criação de janelas.")

    first_target_idx = max(start_idx, window_size) if use_context else max(start_idx + window_size, window_size)
    last_target_idx = end_idx - horizon

    if last_target_idx < first_target_idx:
        return _empty_window_split(X, y, window_size, horizon)

    X_windows = []
    y_windows = []

    for target_idx in range(first_target_idx, last_target_idx + 1):
        window_start = target_idx - window_size
        X_windows.append(X[window_start:target_idx])
        y_windows.append(y[target_idx:target_idx + horizon])

    return torch.stack(X_windows, dim=0), torch.stack(y_windows, dim=0)


def temporal_train_val_test_split(
    X,
    y,
    window_size,
    horizon=1,
    train_ratio=0.7,
    val_ratio=0.2,
    use_context=True,
):
    """
    Faz split temporal sem leakage: primeiro separa o eixo do tempo e só então
    cria as janelas para treino, validação e teste.

    Os alvos de cada split ficam em intervalos temporais disjuntos. Quando
    `use_context=True`, validação e teste podem usar o histórico imediatamente
    anterior como contexto de entrada, sem reutilizar alvos entre splits.
    """
    X = torch.as_tensor(X)
    y = torch.as_tensor(y)

    if X.ndim < 2:
        raise ValueError("X deve ter pelo menos 2 dimensões: [T, ...].")
    if y.ndim < 1:
        raise ValueError("y deve ter pelo menos 1 dimensão: [T, ...].")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y devem ter o mesmo tamanho no eixo temporal.")
    if window_size < 1:
        raise ValueError("window_size deve ser >= 1.")
    if horizon < 1:
        raise ValueError("horizon deve ser >= 1.")
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio deve estar entre 0 e 1.")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio deve estar entre 0 e 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio deve ser < 1.")

    num_steps = X.shape[0]
    min_required = window_size + horizon + 2
    if num_steps < min_required:
        raise ValueError(
            f"Série temporal muito curta para o split temporal sem leakage. "
            f"Recebido T={num_steps}, necessário pelo menos {min_required}."
        )

    train_end = int(train_ratio * num_steps)
    val_end = train_end + int(val_ratio * num_steps)

    X_train, y_train = _create_windows_for_target_interval(
        X, y, window_size, horizon, start_idx=0, end_idx=train_end, use_context=use_context
    )
    X_val, y_val = _create_windows_for_target_interval(
        X, y, window_size, horizon, start_idx=train_end, end_idx=val_end, use_context=use_context
    )
    X_test, y_test = _create_windows_for_target_interval(
        X, y, window_size, horizon, start_idx=val_end, end_idx=num_steps, use_context=use_context
    )

    if X_train.shape[0] == 0:
        raise ValueError("Split de treino ficou sem janelas. Ajuste ratios/window_size/horizon.")
    if X_val.shape[0] == 0:
        raise ValueError("Split de validação ficou sem janelas. Ajuste ratios/window_size/horizon.")
    if X_test.shape[0] == 0:
        raise ValueError("Split de teste ficou sem janelas. Ajuste ratios/window_size/horizon.")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_split(Xs, ys, train_ratio=0.7, val_ratio=0.2, window_size=None, horizon=1, use_context=True):
    Xs = torch.as_tensor(Xs)
    ys = torch.as_tensor(ys)

    if Xs.ndim == 3 and ys.ndim >= 1:
        if window_size is None:
            if train_ratio > 1 or val_ratio > 1:
                window_size = int(train_ratio)
                horizon = int(val_ratio)
                train_ratio = 0.7
                val_ratio = 0.2
            else:
                raise ValueError(
                    "Para séries cruas X [T, ...], informe window_size e horizon "
                    "ou use temporal_train_val_test_split."
                )

        return temporal_train_val_test_split(
            Xs,
            ys,
            window_size=window_size,
            horizon=horizon,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            use_context=use_context,
        )

    if Xs.ndim != 4 or ys.ndim < 2:
        raise ValueError(
            "train_split espera X já janelado [B, T, ...] ou série crua [T, ...]."
        )

    num_samples = Xs.shape[0]
    n_train = int(train_ratio * num_samples)
    n_val = int(val_ratio * num_samples)

    X_train, y_train = Xs[:n_train], ys[:n_train]
    X_val, y_val = Xs[n_train:n_train+n_val], ys[n_train:n_train+n_val]
    X_test, y_test = Xs[n_train+n_val:], ys[n_train+n_val:]

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
