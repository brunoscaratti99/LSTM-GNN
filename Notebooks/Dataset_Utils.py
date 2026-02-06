import xarray as xr
import numpy as np
import pandas as pd
import os as os
import time as time
import torch
from math import radians, sin, cos, asin,sqrt

def reset_weights(model):
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()

def change_comma(frame):
  new_frame = frame.copy()
  for col in frame.columns:
    new_frame[col] = frame[col].astype(str)
    new_frame[col] = frame[col].str.replace(',','.',regex=False)
  return new_frame

#================================================================
#================================================================
#================================================================

def format_path(path):
    """""Formats the path string in order to avoid conflicts."""
    if path[-1]!='/':
        path = path + '/'

    if not os.path.exists(path):
        os.makedirs(path)
    return path

#================================================================
#================================================================
#================================================================

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
    #print(soma.shape)
    #for i in range(6):
    #    sum += data.sel(latitude=lat, longitude=lon, time=inicio_1, step=pd.Timedelta(hours=i+6), method='nearest').tp.values
    #    sum += data.sel(latitude=lat, longitude=lon, time=inicio_2, step=pd.Timedelta(hours=i+12), method='nearest').tp.values
    return soma

#================================================================
#================================================================
#================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    #lat e lon em graus
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    R = 6371.0
    return R * c

#================================================================
#================================================================
#================================================================

import time
#Paralelizar para acelerar leitura de dados(???)
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
            #if X[t,i,0].isnan():
            #    print("DADO NAN")
        i += 1
        #print("delta=", time.time()-start)
    return X

#================================================================
#================================================================
#================================================================

def get_data(dataset,lat,lon,t1, t2):
    date_sequence = pd.date_range(start=t1, end=t2,freq="D")
    rg_rea = pd.DataFrame({
        'time': pd.to_datetime(date_sequence),
        'tp'  : [total_precipitation(data=dataset,lat=lat,lon=lon,time=pd.to_datetime(t1)+pd.Timedelta(days=i)) for i in range(len(date_sequence))]
    })
    rg_rea.set_index('time')
    return rg_rea

#================================================================
#================================================================
#================================================================

def day_index(dataset, start_date, index):
    real_date = pd.to_datetime(start_date)+pd.Timedelta(days=index)
    return real_date

#================================================================
#================================================================
#================================================================

def grafo_distancias(stations, criterion=120):
    E_1, E_2 = [], []
    edge_weight = []
    i,j = 0, 0
    #print(len(stations.keys()))
    for station_v in stations.keys():
        for station_u in stations.keys():
            lat_v, lon_v = float(stations[station_v][0]), float(stations[station_v][1])
            lat_u, lon_u = float(stations[station_u][0]), float(stations[station_u][1])
            dist_km = haversine_km(lat_v, lon_v, lat_u, lon_u)
            if dist_km<=criterion and i!=j:
                E_1.append(i)
                E_2.append(j)
                E_1.append(j)
                E_2.append(i)
                edge_weight.append(1/dist_km)
                edge_weight.append(1/dist_km)
            i+=1
        i=0
        j+=1
    edge_index = torch.tensor([E_1,E_2], dtype=torch.long)
    return edge_index, edge_weight

#================================================================
#================================================================
#================================================================

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

#================================================================
#================================================================
#================================================================

def station_era(era, inmet, lat, lon):
    latlon = era.sel(latitude=lat, longitude=lon, method='nearest')
    return latlon

#================================================================
#================================================================
#================================================================

def split_sequence(sequence, sequence2, n_steps_in, lead_time):
    X, y = [], []
    m = len(sequence)-lead_time
    for i in range(m):
        end_ix = i+n_steps_in
        
        if end_ix>m:
            break
        seq_x, seq_y = sequence[i:end_ix,0:], sequence2[end_ix+lead_time-1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

#================================================================
#================================================================
#================================================================

def prepare_data(X, y, num_features):
    dim_1                        = X.shape[0]
    dim_2                        = X.shape[1]
    dim_y                        = y.shape[0]
    
    X                            = X.flatten()
    y                            = y.flatten()
    
    X                            = X.reshape((dim_1, dim_2, num_features))
    y                            = y.reshape((dim_y, 1, 1))
    return X,y 
