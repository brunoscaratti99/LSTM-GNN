import torch
import xarray as xr
import pandas as pd
import re
import os
import numpy as np
import matplotlib.pyplot as plt

def normalize_tp(tp, eps=1e-6):
    tp_log = torch.log1p(tp)
    mean = tp_log.mean()
    std = tp_log.std()
    return (tp_log - mean) / (std + eps), mean, std




def create_sliding_window(X, period):
    T = X.shape[0]
    Xs, ys = [], []
    for t in range(T - period):
        Xs.append(X[t : t + period - 1])
        ys.append(X[t + period - 1, :, 0])
    return torch.stack(Xs), torch.stack(ys)


def adjacency_matrix(N_local, edge_index, edge_weight=None):
    A = torch.eye(N_local, device=edge_index.device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)
    for i in range(0, len(edge_index[0]), 2):
        u, v = edge_index[0][i], edge_index[0][i + 1]
        A[u, v] = edge_weight[i]
        A[v, u] = edge_weight[i]
    return A


def daily_vertical_velocity(ds, var_name, percentiles=[10]):
    da = ds[var_name]
    daily_mean = da.resample(time="1D").mean()
    daily_min = da.resample(time="1D").min()
    daily_max = da.resample(time="1D").max()
    q = [p / 100 for p in percentiles]
    daily_percentiles = da.resample(time="1D").quantile(q)

    data_vars = {
        f"{var_name}_mean": daily_mean,
        f"{var_name}_min": daily_min,
        f"{var_name}_max": daily_max,
    }
    for i, p in enumerate(percentiles):
        data_vars[f"{var_name}_p{p}"] = daily_percentiles.isel(quantile=i)
    return xr.Dataset(data_vars)


def get_vv(t1, t2, ds, stations):
    N_local = len(stations)
    T = (pd.to_datetime(t2) - pd.to_datetime(t1)).days + 1
    X = torch.zeros((T, N_local, 12), dtype=torch.float32)
    da = ds.sel(time=slice(t1, t2))

    for i, station in enumerate(stations.keys()):
        lat, lon = stations[station][0], stations[station][1]
        da_sel = da.sel(latitude=lat, longitude=lon, method="nearest")
        X[:, i, 0:3] = torch.tensor(da_sel.w_mean.values, dtype=torch.float32)
        X[:, i, 3:6] = torch.tensor(da_sel.w_max.values, dtype=torch.float32)
        X[:, i, 6:9] = torch.tensor(da_sel.w_min.values, dtype=torch.float32)
        X[:, i, 9:12] = torch.tensor(da_sel.w_p10.values, dtype=torch.float32)
    return X



def daily_temp_features(ds, day_shift_hours=0):
    # Ajuste de corte diário (se precisar alinhar com o mesmo "dia" da precipitação)
    if day_shift_hours != 0:
        ds = ds.assign_coords(time=ds.time + np.timedelta64(day_shift_hours, "h"))

    # Kelvin -> Celsius (opcional para NN, mas melhor para interpretação física)
    t2m = ds["t2m"] - 273.15
    d2m = ds["d2m"] - 273.15

    out = xr.Dataset({
        "t2m_mean": t2m.resample(time="1D").mean(),
        "t2m_min":  t2m.resample(time="1D").min(),
        "t2m_max":  t2m.resample(time="1D").max(),
        "d2m_mean": d2m.resample(time="1D").mean(),
        "d2m_min":  d2m.resample(time="1D").min(),
        "d2m_max":  d2m.resample(time="1D").max(),
    })
    return out


def get_temp(t1, t2, ds_daily, stations):
    """
    retorna tensor [T, N, 6] com medidas de 
    média de t2m
    max de t2m
    min t2m
    mean d2m
    max d2m
    min d2m
    """
    N = len(stations)
    T = (pd.to_datetime(t2) - pd.to_datetime(t1)).days + 1
    X = torch.zeros((T, N, 6), dtype=torch.float32)
    da = ds_daily.sel(time=slice(t1, t2))

    for i, st in enumerate(stations.keys()):
        lat, lon = stations[st][0], stations[st][1]
        s = da.sel(latitude=lat, longitude=lon, method="nearest")
        X[:, i, 0] = torch.tensor(s.t2m_mean.values, dtype=torch.float32)
        X[:, i, 1] = torch.tensor(s.t2m_min.values,  dtype=torch.float32)
        X[:, i, 2] = torch.tensor(s.t2m_max.values,  dtype=torch.float32)
        X[:, i, 3] = torch.tensor(s.d2m_mean.values, dtype=torch.float32)
        X[:, i, 4] = torch.tensor(s.d2m_min.values,  dtype=torch.float32)
        X[:, i, 5] = torch.tensor(s.d2m_max.values,  dtype=torch.float32)
    return X



def create_next_experiment_folder(base_path):
    os.makedirs(base_path, exist_ok=True)
    
    existing = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and d.startswith("exp_")
    ]
    
    numbers = []
    for folder in existing:
        match = re.match(r"exp_(\d+)", folder)
        if match:
            numbers.append(int(match.group(1)))
            
    next_number = max(numbers)+1 if numbers else 1
    new_folder_name = f"exp_{next_number:03d}"
    
    full_path = os.path.join(base_path, new_folder_name)
    os.makedirs(full_path)
    
    return full_path



def _infer_coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    raise ValueError(f"Nenhuma coordenada encontrada entre: {candidates}")


def daily_wind_uv_features(
    ds,
    u_var="u",
    v_var="v",
    levels=(500, 850),
    level_coord=None,
    day_shift_hours=0,
):
    # Descobre nomes de coordenadas automaticamente (ERA5 costuma usar latitude/longitude/level)
    if level_coord is None:
        level_coord = _infer_coord_name(ds, ["level", "pressure_level", "isobaricInhPa"])

    if day_shift_hours != 0:
        ds = ds.assign_coords(time=ds.time + np.timedelta64(day_shift_hours, "h"))

    # Seleciona níveis de pressão desejados
    u = ds[u_var].sel({level_coord: list(levels)})
    v = ds[v_var].sel({level_coord: list(levels)})

    out = xr.Dataset({
        "u_mean": u.resample(time="1D").mean(),
        "u_min":  u.resample(time="1D").min(),
        "u_max":  u.resample(time="1D").max(),
        "v_mean": v.resample(time="1D").mean(),
        "v_min":  v.resample(time="1D").min(),
        "v_max":  v.resample(time="1D").max(),
    })
    return out


def get_wind_uv(t1, t2, ds_daily, stations, levels=(500, 850), level_coord=None):
    """
    Saída: tensor [T, N, 12]
    Ordem dos 12 canais:
      0:2   -> u_mean  (500, 850)
      2:4   -> u_min   (500, 850)
      4:6   -> u_max   (500, 850)
      6:8   -> v_mean  (500, 850)
      8:10  -> v_min   (500, 850)
      10:12 -> v_max   (500, 850)
    """
    if level_coord is None:
        level_coord = _infer_coord_name(ds_daily, ["level", "pressure_level", "isobaricInhPa"])

    lat_name = _infer_coord_name(ds_daily, ["latitude", "lat"])
    lon_name = _infer_coord_name(ds_daily, ["longitude", "lon"])

    N = len(stations)
    T = (pd.to_datetime(t2) - pd.to_datetime(t1)).days + 1
    X = torch.zeros((T, N, 12), dtype=torch.float32)

    da = ds_daily.sel(time=slice(t1, t2)).sel({level_coord: list(levels)})

    for i, st in enumerate(stations.keys()):
        lat, lon = stations[st][0], stations[st][1]
        s = da.sel({lat_name: lat, lon_name: lon}, method="nearest")

        X[:, i, 0:2]   = torch.tensor(s.u_mean.values, dtype=torch.float32)
        X[:, i, 2:4]   = torch.tensor(s.u_min.values,  dtype=torch.float32)
        X[:, i, 4:6]   = torch.tensor(s.u_max.values,  dtype=torch.float32)
        X[:, i, 6:8]   = torch.tensor(s.v_mean.values, dtype=torch.float32)
        X[:, i, 8:10]  = torch.tensor(s.v_min.values,  dtype=torch.float32)
        X[:, i, 10:12] = torch.tensor(s.v_max.values,  dtype=torch.float32)

    return X


def safe_r2(y_true, y_pred, eps=1e-8):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + eps))


def safe_mape(y_true, y_pred, eps=1e-3):
    return (torch.abs(y_pred - y_true) / (torch.abs(y_true) + eps)).mean() * 100.0



def save_error_plots(path, train_mse, val_mse, train_mae, val_mae, train_r2, val_r2):
    metrics = [
        ("mse", train_mse, val_mse, "MSE"),
        ("mae", train_mae, val_mae, "MAE"),
        ("r2", train_r2, val_r2, "R2"),
    ]

    for metric_name, train_values, val_values, y_label in metrics:
        plt.figure(figsize=(12, 6))
        if len(train_values) > 0:
            plt.plot(train_values, label="train", linewidth=2)
        if len(val_values) > 0:
            plt.plot(val_values, label="val", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel(y_label)
        plt.title(f"{y_label} by epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{metric_name}_curve.png"), dpi=150)
        plt.close()
        
        
        
def assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise RuntimeError(f"{name} has non-finite values (nan={nan_count}, inf={inf_count})")

