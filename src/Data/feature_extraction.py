import xarray as xr
import pandas as pd
import torch
import numpy as np
import time
from math import radians, cos, sin, asin, sqrt
import os

def smart_load_dataset(path:str, variable:str):
    """
    Made to load a era5 dataset independently of the specific date range at the end of the name.
    Instead this function loads it based on variable name.
    """

    # List all files in dir
    dir_files = os.listdir(path)

    # Loop through them
    for file in dir_files:
        # If the varible string is in the name of one of the files
        if variable in file:
            # Get the full path to it
            current_dir = os.getcwd()
            full_path = os.path.join(current_dir, path, file)
            # Load it and return it
            return(xr.open_dataset(full_path))
        
    # If no files are found with this variable name, the user has made a value error
    raise(ValueError("".join(["A dataset of variable ", variable, " does not exist in path ", path])))



def haversine_km(lat1, lon1, lat2, lon2):
    #lat e lon em graus
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    R = 6371.0
    return R * c


def change_comma(frame):
  new_frame = frame.copy()
  for col in frame.columns:
    new_frame[col] = frame[col].astype(str)
    new_frame[col] = frame[col].str.replace(',','.',regex=False)
  return new_frame


def total_precipitation(data, lat, lon, time):
    """
    Calcula a precipitação total no dia time, na latitude e longitude dadas
    """
    soma = 0
    inicio_1 = pd.to_datetime(time)-pd.to_timedelta(pd.Timedelta(hours=6))
    inicio_2 = pd.to_datetime(time)+pd.to_timedelta(pd.Timedelta(hours=6))
    inicio_3 = pd.to_datetime(time)+pd.to_timedelta(pd.Timedelta(hours=18))
    soma += data.tp.sel(latitude=lat, longitude=lon, time=inicio_1, method='nearest').values[5:].sum()
    soma += data.tp.sel(latitude=lat, longitude=lon, time=inicio_2, method='nearest').values.sum()
    soma += data.tp.sel(latitude=lat, longitude=lon, time=inicio_3, method='nearest').values[:5].sum()
    return soma

def tensor_data(t1, t2, era, stations_RS):
    N = len(stations_RS)
    T = pd.to_datetime(t2)-pd.to_datetime(t1)
    T = T.days+1
    X = torch.zeros((T,N,1))
    i = 0
    for station in stations_RS.keys():
        start = time.time()
        lat, lon = stations_RS[station][0], stations_RS[station][1]
        for t in range(T): 
            X[t,i,0] = torch.tensor(total_precipitation(data=era, lat=lat, lon=lon, time=pd.to_datetime(t1)+pd.Timedelta(days=t)))

        i += 1

    return X


def get_data(dataset,lat,lon,t1, t2):
    date_sequence = pd.date_range(start=t1, end=t2,freq="D")
    rg_rea = pd.DataFrame({
        'time': pd.to_datetime(date_sequence),
        'tp'  : [total_precipitation(data=dataset,lat=lat,lon=lon,time=pd.to_datetime(t1)+pd.Timedelta(days=i)) for i in range(len(date_sequence))]
    })
    rg_rea.set_index('time')
    return rg_rea

def _wrap_lon(lon, ds_lons):
    # Ajusta longitude para 0..360 ou -180..180 conforme o dataset
    lon = float(lon)
    lmin = float(ds_lons.min())
    lmax = float(ds_lons.max())
    if lmax > 180 and lon < 0:
        lon = lon + 360
    elif lmin < 0 and lon > 180:
        lon = lon - 360
    return lon

def uv_to_dir_speed(u, v, convention="meteorological"):
    # u: leste+, v: norte+
    speed = np.hypot(u, v)
    if convention == "meteorological":
        # direção "de onde vem" (0=N, 90=E)
        direction = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    else:
        # direção "para onde vai"
        direction = (np.degrees(np.arctan2(u, v)) + 360) % 360
    return direction, speed

def era5_uv_to_tensor(
    nc_path,
    stations,
    start=None,
    end=None,
    u_var="u",
    v_var="v",
    daily_agg="mean",
    convention="meteorological",
    return_xarray=False,
):
    """
    stations: dict {name: (lat, lon)} ou DataFrame com colunas ['name','lat','lon']
    retorna: torch.Tensor [days, n_estacoes, 2] (direção, velocidade)
    """
    ds = xr.open_dataset(nc_path)

    # resolve estações
    if isinstance(stations, dict):
        names = list(stations.keys())
        lats = [stations[k][0] for k in names]
        lons = [stations[k][1] for k in names]
    else:
        names = stations["name"].tolist()
        lats = stations["lat"].tolist()
        lons = stations["lon"].tolist()

    # recorte de tempo
    if start is not None or end is not None:
        ds = ds.sel(time=slice(start, end))

    # agrega diário
    if daily_agg == "mean":
        ds_day = ds.resample(time="1D").mean()
    elif daily_agg == "sum":
        ds_day = ds.resample(time="1D").sum()
    else:
        raise ValueError("daily_agg deve ser 'mean' ou 'sum'")

    # coleta por estação
    series = []
    for lat, lon in zip(lats, lons):
        lon = _wrap_lon(lon, ds_day.longitude)
        point = ds_day[[u_var, v_var]].sel(
            latitude=lat, longitude=lon, method="nearest"
        )
        u = point[u_var].values
        v = point[v_var].values
        direction, speed = uv_to_dir_speed(u, v, convention=convention)
        # [days, 2]
        series.append(np.stack([direction, speed], axis=-1))

    # [days, n_estacoes, 2]
    X = np.stack(series, axis=1)

    if return_xarray:
        days = pd.to_datetime(ds_day.time.values)
        return xr.DataArray(
            X,
            dims=("day", "station", "feature"),
            coords={"day": days, "station": names, "feature": ["direction", "speed"]},
        )

    return torch.tensor(X, dtype=torch.float32)





def daily_vertical_velocity(ds, var_name, percentiles=(10,), time_chunk_days=30, load_into_memory=True):
    da = ds[var_name]

    # Se ainda não estiver chunkado, divide o tempo em blocos (~30 dias)
    if getattr(da.data, "chunks", None) is None:
        try:
            da = da.chunk({"time": 24 * time_chunk_days})
        except Exception:
            pass

    resampler = da.resample(time="1D")

    out = xr.Dataset({
        f"{var_name}_mean": resampler.mean(),
        f"{var_name}_min": resampler.min(),
        f"{var_name}_max": resampler.max(),
    })

    if percentiles:
        q = np.asarray(percentiles, dtype=np.float64) / 100.0
        daily_percentiles = resampler.quantile(q)

        for i, p in enumerate(percentiles):
            p_label = str(int(p)) if float(p).is_integer() else str(p).replace(".", "_")
            out[f"{var_name}_p{p_label}"] = daily_percentiles.isel(quantile=i).astype(da.dtype)

        out = out.drop_vars("quantile", errors="ignore")

    return out.load() if load_into_memory else out




def get_vv(t1, t2, ds, stations):
    N_local = len(stations)
    T = ds.time.values.shape[0]
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


def daily_specific_humidity_features(ds, var_name="q", day_shift_hours=0):
    """
    Gera estatísticas diárias (max, mean, min) de specific humidity a partir de dados horários.
    Retorna xr.Dataset com variáveis:
      - q_max, q_mean, q_min
    """
    if day_shift_hours != 0:
        ds = ds.assign_coords(time=ds.time + np.timedelta64(day_shift_hours, "h"))

    q = ds[var_name]
    out = xr.Dataset({
        "q_max":  q.resample(time="1D").max(),
        "q_mean": q.resample(time="1D").mean(),
        "q_min":  q.resample(time="1D").min(),
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
    #T = (pd.to_datetime(t2) - pd.to_datetime(t1)).days + 1
    #T = torch.tensor(ds_daily.t2m_mean.values, dtype=torch.float32).shape[0]
    T = ds_daily.time.values.shape[0]
    X = torch.zeros((T, N, 6), dtype=torch.float32)
    da = ds_daily.sel(time=slice(t1, t2))

    for i, st in enumerate(stations.keys()):
        lat, lon = stations[st][0], stations[st][1]
        s = da.sel(latitude=lat, longitude=lon, method="nearest")
        #print(torch.tensor(s.t2m_mean.values, dtype=torch.float32).shape)
        X[:, i, 0] = torch.tensor(s.t2m_mean.values, dtype=torch.float32)
        X[:, i, 1] = torch.tensor(s.t2m_min.values,  dtype=torch.float32)
        X[:, i, 2] = torch.tensor(s.t2m_max.values,  dtype=torch.float32)
        X[:, i, 3] = torch.tensor(s.d2m_mean.values, dtype=torch.float32)
        X[:, i, 4] = torch.tensor(s.d2m_min.values,  dtype=torch.float32)
        X[:, i, 5] = torch.tensor(s.d2m_max.values,  dtype=torch.float32)
    return X


def get_specific_humidity(t1, t2, ds_daily, stations):
    """
    Retorna tensor [T, N, 3] com specific humidity diário por estação.
    Ordem dos canais: [max, mean, min]
    """
    lat_name = _infer_coord_name(ds_daily, ["latitude", "lat"])
    lon_name = _infer_coord_name(ds_daily, ["longitude", "lon"])

    N = len(stations)
    T = (pd.to_datetime(t2) - pd.to_datetime(t1)).days + 1
    X = torch.zeros((T, N, 6), dtype=torch.float32)

    ds = ds_daily.sel(time=slice(t1, t2))

    for i, st in enumerate(stations.keys()):
        lat, lon = stations[st][0], stations[st][1]
        lon = _wrap_lon(lon, ds[lon_name])
        s = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")

        X[:, i, 0:2] = torch.tensor(s.q_max.values,  dtype=torch.float32)
        X[:, i, 2:4] = torch.tensor(s.q_mean.values, dtype=torch.float32)
        X[:, i, 4:6] = torch.tensor(s.q_min.values,  dtype=torch.float32)

    return X


def era5_specific_humidity_tensor(nc_path, stations, start=None, end=None, var_name="q", day_shift_hours=0):
    """
    Lê ERA5 de specific humidity horário e retorna tensor diário [T, N, 3]
    com canais [max, mean, min].
    """
    ds = xr.open_dataset(nc_path)
    if start is not None or end is not None:
        ds = ds.sel(time=slice(start, end))

    ds_daily = daily_specific_humidity_features(ds, var_name=var_name, day_shift_hours=day_shift_hours)
    if start is None:
        start = str(pd.to_datetime(ds_daily.time.values[0]).date())
    if end is None:
        end = str(pd.to_datetime(ds_daily.time.values[-1]).date())

    return get_specific_humidity(start, end, ds_daily, stations)



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

def _infer_coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    raise ValueError(f"Nenhuma coordenada encontrada entre: {candidates}")


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





def forecast_steps_to_daily_precip(data, var_name="tp", lat=None, lon=None, to_mm=False):
    """
    Converte precipitação prevista em (time, step) para precipitação diária
    usando o tempo válido `time + step`.

    Exemplo:
    - time: 06h e 18h
    - step: 1..12 horas

    Cada previsão é reposicionada no seu horário real e depois somada por dia.
    """
    tp = data[var_name] if isinstance(data, xr.Dataset) else data

    if lat is not None and lon is not None:
        tp = tp.sel(latitude=lat, longitude=lon, method="nearest")

    if "time" not in tp.dims or "step" not in tp.dims:
        raise ValueError("A precipitação precisa ter dimensões 'time' e 'step'.")

    valid_time = tp.time + tp.step
    tp = tp.assign_coords(valid_time=(("time", "step"), valid_time.data))

    tp_hourly = tp.stack(sample=("time", "step")).reset_index("sample", drop=True)
    tp_hourly = tp_hourly.assign_coords(time=("sample", tp_hourly.valid_time.data))
    tp_hourly = tp_hourly.swap_dims({"sample": "time"}).sortby("time")
    tp_hourly = tp_hourly.drop_vars("valid_time").groupby("time").last()

    tp_daily = tp_hourly.resample(time="1D").sum()

    if to_mm:
        tp_daily = tp_daily * 1000

    return tp_daily.to_dataset(name=var_name)


def era5_daily_precip(data, lat, lon):
    return forecast_steps_to_daily_precip(data, lat=lat, lon=lon)

def day_index(dataset, start_date, index):
    real_date = pd.to_datetime(start_date)+pd.Timedelta(days=index)
    return real_date


def station_dictionary(catalogo, UF='RS'):
    latlon = {}
    catalogo = change_comma(catalogo)
    if UF=='ALL':
        for name in catalogo.DC_NOME:
            aux_df = catalogo.loc[catalogo['DC_NOME']==name]
            latlon[name] = [aux_df.VL_LATITUDE.values[0], aux_df.VL_LONGITUDE.values[0]]
        return latlon
    else:
        catalogo = catalogo.loc[catalogo['SG_ESTADO']==UF]
        for name in catalogo.DC_NOME:
            aux_df = catalogo.loc[catalogo['DC_NOME']==name]
            latlon[name] = [aux_df.VL_LATITUDE.values[0],aux_df.VL_LONGITUDE.values[0]]
    return latlon



def station_era(era, inmet, lat, lon):
    latlon = era.sel(latitude=lat, longitude=lon, method='nearest')
    return latlon

def tensor_data_old(t1, t2, era, stations_RS):
    N = len(stations_RS)
    T = pd.to_datetime(t2)-pd.to_datetime(t1)
    T = T.days+1
    X = torch.zeros((T,N,1))
    i = 0
    for station in stations_RS.keys():
        start = time.time()
        lat, lon = stations_RS[station][0], stations_RS[station][1]
        for t in range(T): 
            X[t,i,0] = torch.tensor(total_precipitation(data=era, lat=lat, lon=lon, time=pd.to_datetime(t1)+pd.Timedelta(days=t)))

        i += 1

    return X




def era5_daily_precip_all(data):
    return forecast_steps_to_daily_precip(data, to_mm=True)


def daily_precip_dataset_to_tensor(tp_daily, stations, var_name="tp"):
    """
    Converte um dataset diário de precipitação em grade para tensor [dias, N],
    onde N é o número de estações/nós em `stations`.
    """
    da = tp_daily[var_name] if isinstance(tp_daily, xr.Dataset) else tp_daily

    if "time" not in da.dims:
        raise ValueError("O dataset diário precisa ter a dimensão 'time'.")

    lat_name = _infer_coord_name(da, ["latitude", "lat"])
    lon_name = _infer_coord_name(da, ["longitude", "lon"])

    station_names = list(stations.keys())
    lat_stations = [float(stations[name][0]) for name in station_names]
    lon_stations = [_wrap_lon(float(stations[name][1]), da[lon_name]) for name in station_names]

    lat_da = xr.DataArray(lat_stations, dims="station", coords={"station": station_names})
    lon_da = xr.DataArray(lon_stations, dims="station", coords={"station": station_names})

    tp_stations = da.sel({lat_name: lat_da, lon_name: lon_da}, method="nearest").transpose("time", "station")
    return torch.tensor(tp_stations.values, dtype=torch.float32)



def tensor_data_precip(data, t1, t2, stations):
    tp_daily = era5_daily_precip_all(data).sel(time=slice(t1, t2))
    return daily_precip_dataset_to_tensor(tp_daily, stations)
